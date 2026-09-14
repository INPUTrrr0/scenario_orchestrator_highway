#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_port/actuation.py — policy demand -> carla.VehicleControl.

Extracted from `carla_ego.py` so that both backends actuate through the *same*
controller. `carla_ego.py` binds it to `v4/drivev2.py`'s limits and
`carla_highway/` binds it to the highway ego's; neither re-derives it.

That matters more than the saved lines. `ego_driver.py` already argues that
re-deriving a controller "would silently change the numbers while still calling
them the policy's" — the same trap applies across two ports of one control law,
so the law lives in one place and the vehicle limits are parameters.

Everything from SPAWN_MAX_S through `AccelerationTracker` is also copied,
unchanged, into the intersection port
(`third_party/orchestration/carla_port/actuation.py`) and osc2runner
(`osc2carla/backend/actuation.py`): every method turns an IDM acceleration into
pedals with the same law. Change them together.

Nothing here imports a script layer, so it is safe from either side.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from .carla_api import carla

#: The spawn phase (`SpawnGear`): the longest it lasts.
SPAWN_MAX_S = 2.0
#: Without engine telemetry (a test double, the local simulator) the engine is
#: revved for this long instead of until it reaches the gear's rpm.
SPAWN_REV_S = 0.5
#: How long the engaged gear is held with a manual shift before the automatic
#: gearbox carries on from it.
SPAWN_GEAR_HOLD_S = 0.25
#: The engine is at the gear's rpm from this fraction of it.
SPAWN_RPM_MATCH = 0.97
#: An acceleration demand below this is a hard brake, which the gear waits out.
SPAWN_BRAKING_MPS2 = -1.0
#: The throttle that holds a speed against drag and engine braking, which the
#: controller starts from when the spawn phase hands over. What
#: `AccelerationTracker` settled on at 4-14 m/s on an empty Town04 road: Audi TT
#: 0.38-0.41, Lincoln MKZ 0.39-0.43, Audi A2 0.51-0.56, Tesla Model 3 0.21-0.50.
HOLD_THROTTLE = 0.35


def _horizontal_velocity(vehicle) -> Optional[Tuple[float, float]]:
    try:
        v = vehicle.get_velocity()
        return float(v.x), float(v.y)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def horizontal_speed(vehicle, fallback: float = 0.0) -> float:
    """The vehicle's speed over the ground. Vertical motion is left out: a car
    settling onto the road from its spawn height is not rolling."""
    h = _horizontal_velocity(vehicle)
    return math.hypot(*h) if h is not None else float(fallback)


def _engine_rpm(vehicle) -> Optional[float]:
    try:
        return float(vehicle.get_telemetry_data().engine_rpm)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


class SpawnGear:
    """Bring a vehicle that spawns rolling into gear without a jolt.

    A CARLA vehicle given a velocity at spawn rolls in neutral with its engine
    stopped, and its automatic gearbox engages first gear on its own about two
    seconds later, at road speed. The clutch then drags the engine up through
    the lowest ratio: measured on an empty Town04 road, the MKZ went from 7.1
    to 5.2 m/s in 0.2 s, and at the part throttle an IDM demand maps to its
    engine never reached the up-shift point again, so it stayed in first at
    5 m/s while IDM asked for +1.3 m/s^2.

    Engaging any gear syncs the engine to the wheels, so what matters is the
    engine's speed at that moment: below the gear's rpm the car loses speed,
    above it the car lurches forward. Revving in neutral for a fixed 0.5 s
    missed both ways (an Audi TT's engine was near 1600 rpm for a gear that
    wanted 1160, an Audi A2's near 1000 for 1850), and engaging at once while
    braking added the engine's drag to the brakes: spawned at 13 m/s under
    IDM-B, the TT was at 7.2 m/s after a second where IDM's own speed profile
    is at 9.45.

    So a car spawned rolling goes through a spawn phase:

      rev      the engine revs in neutral until it is at the rpm of the gear
               the car's speed calls for (`choose`; the rpm is read from
               `get_telemetry_data`, or it is revved for SPAWN_REV_S without
               it). An ACCELERATION policy's car follows the demand
               kinematically meanwhile -- its velocity is the demand
               integrated, set every step -- and the gear waits out a hard
               brake (below SPAWN_BRAKING_MPS2): engaging while braking hard
               left the car up to 0.62 m/s off IDM's profile instead of 0.37.
               A PEDAL policy's car is not driven: it rolls in neutral with the
               policy's brake applied. At most SPAWN_MAX_S either way.
      engage   the gear goes in and is held with a manual shift for
               SPAWN_GEAR_HOLD_S; the controller takes over.

    Mean |v - v_ideal| over the first 2 s, against IDM's own speed profile from
    the same spawn speed, IDM-A/B/C: TT spawned at 13 m/s, 1.24-2.06 m/s
    engaging at once and 0.03-0.09 with the spawn phase; A2 at 7.76 m/s,
    0.45-1.46 revving for a fixed 0.5 s or engaging at once, and 0.03-0.07.

    A car below walking pace -- spawned at rest, or settling onto the road with
    only vertical speed -- is left to the gearbox, which launches from rest
    normally; so is one whose physics control has no gear table (the offline
    test double, the local simulator).
    """

    def __init__(self, rev_s: float = SPAWN_REV_S,
                 hold_s: float = SPAWN_GEAR_HOLD_S,
                 max_s: float = SPAWN_MAX_S):
        self.rev_s = float(rev_s)
        self.hold_s = float(hold_s)
        self.max_s = float(max_s)
        self.plan: Optional[dict] = None
        self._phase: Optional[str] = None       # None, "rev", "hold", "done"
        self._calls = 0
        self._t = 0.0
        self._v = 0.0
        self._dir: Optional[Tuple[float, float]] = None

    @staticmethod
    def choose(vehicle, speed: float) -> Optional[dict]:
        """{gear, engine_rpm} for `speed` in m/s, or None to leave it alone."""
        if speed < 1.0:
            return None
        try:
            pc = vehicle.get_physics_control()
            gears = list(pc.forward_gears)
            final = float(pc.final_ratio)
            max_rpm = float(pc.max_rpm)
            radius_m = max(float(w.radius) for w in pc.wheels) / 100.0  # cm
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if not gears or radius_m <= 0.0 or max_rpm <= 0.0:
            return None
        wheel_rpm = speed / (2.0 * math.pi * radius_m) * 60.0
        kept, lowest_fitting = None, None
        for number, g in enumerate(gears, start=1):
            rpm = wheel_rpm * final * float(g.ratio)
            if rpm <= float(g.up_ratio) * max_rpm:
                if lowest_fitting is None:
                    lowest_fitting = (number, rpm)
                if rpm >= float(g.down_ratio) * max_rpm:
                    kept = (number, rpm)
        pick = kept or lowest_fitting or (len(gears), wheel_rpm * final
                                          * float(gears[-1].ratio))
        return {"gear": int(pick[0]), "engine_rpm": round(pick[1], 1),
                "speed_mps": round(float(speed), 3)}

    def step(self, vehicle, speed: float, dt: float,
             accel: Optional[float] = None, brake: float = 0.0
             ) -> Tuple[Optional[Tuple[float, float]], dict]:
        """This step's (pedals, VehicleControl kwargs).

        `accel` is an acceleration policy's demand in m/s^2; None for a pedal
        policy, whose `brake` is applied while the engine revs. `pedals` is
        (throttle, brake) while the spawn phase drives the car -- use it
        instead of the controller, and `reset(prime=True)` the controller --
        and None otherwise. The kwargs are the manual shift while there is
        one. `speed` is read only when the vehicle cannot report a velocity.
        """
        self._calls += 1
        if self._phase is None:
            h = _horizontal_velocity(vehicle)
            s = math.hypot(*h) if h is not None else float(speed)
            plan = self.choose(vehicle, s)
            if plan is None:
                if s < 1.0 and self._calls < 3:
                    return None, {}     # a spawn velocity can land a tick late
                self._phase = "done"
                return None, {}
            self.plan = dict(plan, spawn_speed_mps=round(s, 3),
                             kinematic=accel is not None)
            self._dir = (h[0] / s, h[1] / s) if h is not None else None
            self._v, self._t, self._phase = s, 0.0, "rev"
        if self._phase == "done":
            return None, {}
        t = self._t
        self._t += dt
        if self._phase == "rev":
            return self._rev(vehicle, dt, t, accel, brake)
        if t < self.hold_s - 1e-9:
            return None, {"manual_gear_shift": True, "gear": self.plan["gear"]}
        self._phase = "done"
        return None, {}

    def _rev(self, vehicle, dt: float, t: float, accel: Optional[float],
             brake: float) -> Tuple[Optional[Tuple[float, float]], dict]:
        kinematic = accel is not None
        if kinematic:
            self._v = max(0.0, self._v + accel * dt)
            speed = self._v
        else:
            speed = horizontal_speed(vehicle, self._v)
        if speed < 1.0:
            # stopped in neutral: the gearbox launches it from rest
            self.plan["released_at_mps"] = round(speed, 3)
            self._phase = "done"
            return (0.0, 1.0 if kinematic else float(brake)), {}
        if kinematic:
            self._drive(vehicle)
        target = self.choose(vehicle, speed) or self.plan
        rpm = _engine_rpm(vehicle)
        ready = (rpm >= SPAWN_RPM_MATCH * target["engine_rpm"] if rpm is not None
                 else t >= self.rev_s - 1e-9)
        hard_brake = kinematic and accel < SPAWN_BRAKING_MPS2
        if (ready and not hard_brake) or t >= self.max_s - 1e-9:
            self.plan.update(gear=target["gear"], engine_rpm=target["engine_rpm"],
                             speed_mps=round(speed, 3), spawn_phase_s=round(t, 3),
                             rpm_matched=bool(rpm is not None and ready))
            self._phase, self._t = "hold", 0.0
            return None, {"manual_gear_shift": True, "gear": target["gear"]}
        throttle = 1.0 if rpm is None or rpm < target["engine_rpm"] else 0.0
        return ((throttle, 0.0 if kinematic else float(brake)),
                {"manual_gear_shift": True, "gear": 0})

    def _drive(self, vehicle) -> None:
        """Set the kinematic velocity, along the car's heading."""
        if self._dir is None:
            return
        try:
            v = vehicle.get_velocity()
            vehicle.set_target_velocity(type(v)(self._dir[0] * self._v,
                                                self._dir[1] * self._v, 0.0))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return
        try:
            f = vehicle.get_transform().get_forward_vector()
            n = math.hypot(f.x, f.y)
            if n > 1e-6:
                self._dir = (f.x / n, f.y / n)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


class AccelerationTracker:
    """Pedals that make a CARLA vehicle realise a commanded acceleration.

    For a policy whose action is an acceleration (`third_party/idm`):

      u = demand feedforward + kp * lag + ki * integral(lag),   u in [-1, 1]

    `lag` is the gap between a reference speed, which integrates the demand
    step by step, and the measured speed. The feedforward is the demand's own
    pedal (throttle a/3, brake -a/5: the open-loop map this replaced). The
    integral learns what the feedforward leaves out, mostly the throttle that
    merely holds a speed (HOLD_THROTTLE).

    Why each piece, from runs on an empty Town04 road:
      * The reference. The PID this replaced targeted `v + a * 0.5 s`, which
        caps the error at half a second of demand however far behind the car
        falls: it realised +0.62 of +1.26 m/s^2 and sagged to 4 m/s at a target
        of 8. With the reference: +0.94 of +0.94.
      * No derivative. On the measured speed it answered the car's own jolts
        (a gear change, the last metre of a stop at -11 m/s^2) with full
        throttle; on the error it spiked with every change of demand.
      * Throttle and brake in every regime. In first or second gear a CARLA
        car's engine alone brakes it at 2-6 m/s^2 with the throttle closed
        (Audi TT, Audi A2, Tesla Model 3), so a mild braking demand needs
        throttle. A version that never pedalled against the demand, and whose
        braking reference only ever asked for more brake, braked for a slower
        car ahead so much harder than asked that it closed to 10.2 m where
        IDM's own profile closes to 8.3; now 8.6. A stop behind a standing
        obstacle ends 2.05 m from it where IDM's profile ends at 2.01.
      * The reference restarts from the measured speed whenever the demand
        changes regime (accelerating / holding / braking), so a lag built in
        one cannot kick the next. The regimes have hysteresis (entered past
        `enter_mps2`, left inside `exit_mps2`): IDM's demand near its desired
        speed sits right at a single threshold, and restarting there every
        few steps kept the integral from ever settling.
      * kp 1.0: at 0.5 the mean speed error over that stop was 0.19 m/s for
        the first 2 s and 0.13 after, at 1.0 it is 0.07 and 0.06.
        `reset(prime=True)` starts the integral at HOLD_THROTTLE: from zero, a
        car just put into gear lost about 1 m/s to engine braking before the
        lag built up.
      * Anti-windup: the integral stops while the output is pinned.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.3,
                 throttle_per_mps2: float = 1.0 / 3.0,
                 brake_per_mps2: float = 1.0 / 5.0,
                 integral_limit: float = 2.5, lag_cap: float = 2.0,
                 v_max: float = 40.0, enter_mps2: float = 0.3,
                 exit_mps2: float = 0.1, brake_deadband: float = 0.12):
        self.kp, self.ki = float(kp), float(ki)
        self.throttle_per_mps2 = float(throttle_per_mps2)
        self.brake_per_mps2 = float(brake_per_mps2)
        self.integral_limit = float(integral_limit)
        self.lag_cap = float(lag_cap)
        self.v_max = float(v_max)
        self.enter_mps2 = float(enter_mps2)
        self.exit_mps2 = float(exit_mps2)
        self.brake_deadband = float(brake_deadband)
        self.v_ref: Optional[float] = None
        self.integral = 0.0
        self._regime: Optional[int] = None

    def reset(self, prime: bool = False) -> None:
        """Start over; `prime` starts the integral at HOLD_THROTTLE."""
        self.v_ref = None
        self.integral = HOLD_THROTTLE / self.ki if prime and self.ki > 0.0 else 0.0
        self._regime = None

    def step(self, accel: float, v: float, dt: float) -> Tuple[float, float]:
        """(throttle, brake) for this step; `self.v_ref` is the reference."""
        regime = self._regime_for(accel)
        if self.v_ref is None or regime != self._regime:
            ref = v
        else:
            ref = self.v_ref + accel * dt
        self._regime = regime
        self.v_ref = max(0.0, v - self.lag_cap,
                         min(self.v_max, v + self.lag_cap, ref))
        if self.v_ref <= 0.05 and accel <= 0.0:
            return 0.0, 1.0                   # stopped and asked to stay stopped
        lag = self.v_ref - v
        ff = (accel * self.throttle_per_mps2 if accel >= 0.0
              else accel * self.brake_per_mps2)
        u = ff + self.kp * lag + self.ki * self.integral
        pinned = (u >= 1.0 and lag > 0.0) or (u <= -1.0 and lag < 0.0)
        if not pinned:
            self.integral = max(-self.integral_limit,
                                min(self.integral_limit, self.integral + lag * dt))
        u = max(-1.0, min(1.0, u))
        if u >= 0.0:
            return u, 0.0
        # a small negative effort is engine braking, not a brake application
        return 0.0, (0.0 if -u < self.brake_deadband else -u)

    def _regime_for(self, accel: float) -> int:
        """+1 accelerating, -1 braking, 0 holding, with hysteresis."""
        was = self._regime or 0
        if was > 0 and accel >= self.exit_mps2:
            return 1
        if was < 0 and accel <= -self.exit_mps2:
            return -1
        if accel > self.enter_mps2:
            return 1
        if accel < -self.enter_mps2:
            return -1
        return 0


class CarlaEgoActuator:
    """Turns a policy's (throttle, steer) into a carla.VehicleControl.

    Steering needs two corrections, and both belong here rather than in the
    policy:

      scale  a policy's steer is normalized by ITS max road-wheel angle
             (`delta_max`); `VehicleControl.steer` is normalized by the
             vehicle's OWN max steer angle, read off its physics control.
      sign   script headings run counter-clockwise and CARLA yaw runs clockwise
             (`yaw = theta - heading` in both map frames), so a positive script
             steer — turning left — is a NEGATIVE CARLA steer. This is the same
             handedness flip the pose conversion makes; it just has to be made
             again for a rate rather than an angle.

    The longitudinal half is `AccelerationTracker`, fed the acceleration behind
    the policy's normalized throttle, after `SpawnGear`'s spawn phase.

    `policy` is duck-typed: it needs `commanded_accel(throttle)` and `ego.v`.
    """

    def __init__(self, vehicle, max_steer_deg: Optional[float] = None,
                 delta_max: float = math.radians(32),
                 v_max: float = 18.0):
        self.vehicle = vehicle
        self.spawn_gear = SpawnGear()
        self.tracker = AccelerationTracker(v_max=v_max)
        self.max_steer_deg = max_steer_deg or self._read_max_steer(vehicle)
        self.delta_max = float(delta_max)
        self.v_max = float(v_max)
        self.steer_scale = -math.degrees(self.delta_max) / max(self.max_steer_deg,
                                                               1e-6)

    @staticmethod
    def _read_max_steer(vehicle, fallback: float = 70.0) -> float:
        try:
            wheels = vehicle.get_physics_control().wheels
        except (RuntimeError, AttributeError):
            return fallback
        fronts = [w.max_steer_angle for w in wheels
                  if getattr(w, "max_steer_angle", 0.0) > 1e-6]
        return max(fronts) if fronts else fallback

    def control(self, policy, throttle_cmd: float, steer_cmd: float, dt: float):
        """The VehicleControl for one step, plus the reference speed used."""
        accel = policy.commanded_accel(throttle_cmd)
        pedals, gear = self.spawn_gear.step(self.vehicle, policy.ego.v, dt, accel)
        if pedals is not None:              # the spawn phase drives the car
            self.tracker.reset(prime=True)
            throttle, brake = pedals
        else:
            throttle, brake = self.tracker.step(accel, policy.ego.v, dt)
        steer = max(-1.0, min(1.0, steer_cmd * self.steer_scale))
        return carla.VehicleControl(throttle=float(throttle), steer=float(steer),
                                    brake=float(brake), **gear), self.tracker.v_ref
