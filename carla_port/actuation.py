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

Nothing here imports a script layer, so it is safe from either side.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from .carla_api import carla

# How far ahead IDM's commanded acceleration is projected to get a target speed
# for the PID to chase. NOT the control step: at 60 Hz one step of a 3 m/s^2
# demand is 0.05 m/s, and a 0.05 m/s error is a two-percent throttle, which does
# not move a stopped car — so a standing ego whose IDM wants to go stays stopped
# forever. Projecting over a short horizon instead gives the controller the
# authority to pull away, and costs nothing at speed because IDM's demand goes
# to zero as the ego approaches its desired speed and the target converges on
# the measurement.
ACCEL_HORIZON = 0.5      # seconds

#: How long the spawn gear is held with a manual shift before the automatic
#: gearbox carries on from it.
SPAWN_GEAR_HOLD_S = 0.25


class SpawnGear:
    """Put a vehicle that spawns rolling into the gear its speed calls for.

    A CARLA vehicle given a velocity at spawn rolls in neutral, and its
    automatic gearbox engages first gear on its own about two seconds later, at
    road speed. The clutch then spins the idling engine up through the lowest
    ratio: measured on an empty Town04 road, the MKZ went from 7.1 to 5.2 m/s in
    0.2 s, and at the part throttle an IDM demand maps to its engine never
    reached the up-shift point again, so it stayed in first at 5 m/s while IDM
    asked for +1.3 m/s^2. That looked like the ego braking for no reason.

    Engaged at spawn instead, in the highest gear whose engine speed the
    gearbox would keep (between its down- and up-shift points), the engine
    spins up through a tall ratio, so the car loses little speed doing it, and
    the gearbox shifts normally from there. A vehicle below walking pace, or
    one whose physics control carries no gear table (the offline test double),
    is left to the gearbox.
    """

    def __init__(self, hold_s: float = SPAWN_GEAR_HOLD_S):
        self.hold_s = float(hold_s)
        self.plan: Optional[dict] = None
        self._decided = False
        self._held = 0.0

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

    def control_kwargs(self, vehicle, speed: float, dt: float) -> dict:
        """Extra `carla.VehicleControl` arguments for this step: a manual shift
        into the spawn gear for the first `hold_s`, nothing afterwards."""
        if not self._decided:
            self._decided = True
            self.plan = self.choose(vehicle, speed)
        if self.plan is None or self._held >= self.hold_s - 1e-9:
            return {}
        self._held += dt
        return {"manual_gear_shift": True, "gear": self.plan["gear"]}


class LongitudinalPID:
    """Speed-tracking PID, the classic CARLA longitudinal controller.

    IDM is an acceleration law, and a CARLA vehicle takes throttle/brake, not
    acceleration. IDM's commanded accel becomes a target speed (see
    ACCEL_HORIZON) and this PID closes the loop on the speed error, so the IDM
    law is unchanged and only the actuation is new.
    """

    def __init__(self, kp: float = 0.45, ki: float = 0.08, kd: float = 0.10,
                 integral_clamp: float = 8.0, brake_deadband: float = 0.12):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_clamp = integral_clamp
        self.brake_deadband = brake_deadband
        self._integral = 0.0
        self._prev_err = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_err = 0.0

    def step(self, v_target: float, v: float, dt: float) -> Tuple[float, float]:
        """(throttle, brake), each in [0, 1]."""
        err = v_target - v
        self._integral = max(-self.integral_clamp,
                            min(self.integral_clamp, self._integral + err * dt))
        deriv = (err - self._prev_err) / dt if dt > 1e-9 else 0.0
        self._prev_err = err
        u = self.kp * err + self.ki * self._integral + self.kd * deriv
        if v_target <= 0.05:                  # a full stop is a brake command,
            return 0.0, 1.0                   # not a very small throttle
        if u >= 0.0:
            return min(1.0, u), 0.0
        # small negative demand is engine braking, not a brake application
        return 0.0, (0.0 if -u < self.brake_deadband else min(1.0, -u))


class AccelerationTracker:
    """Pedals that make a CARLA vehicle realise a commanded acceleration.

    For a policy whose action is an acceleration (`third_party/idm`):

      u = demand feedforward + kp * lag + ki * integral(lag)

    `lag` is the gap between a reference speed, which integrates the demand
    step by step, and the measured speed. The feedforward is the demand's own
    pedal (throttle a/3, brake -a/5: the open-loop map this replaces). The
    integral learns what the feedforward leaves out, mostly the throttle that
    merely holds a speed against drag -- 0.43-0.45 for the MKZ at 8-13 m/s.

    Why each piece, from runs on an empty Town04 road:
      * The reference. Targeting `v + a * ACCEL_HORIZON` instead, as
        `CarlaEgoActuator` does, caps the error at half a second of demand
        however far behind the car falls: it realised +0.62 of +1.26 m/s^2 and
        sagged to 4 m/s at a target of 8. With the reference: +0.94 of +0.94.
      * No derivative. On the measured speed it answered the car's own jolts
        (a gear change, the last metre of a stop at -11 m/s^2) with full
        throttle; on the error it spiked with every change of demand.
      * The output never pedals against the demand: a car slowing faster than
        asked (engine braking in a low gear) is not given throttle, nor one
        pulling away harder given brake. Within the band it may: that holds a
        speed.
      * The reference restarts from the measured speed whenever the demand
        changes regime (accelerating / holding / braking), so a lag built in
        one cannot kick the next. The regimes have hysteresis (entered past
        `enter_mps2`, left inside `exit_mps2`): IDM's demand near its desired
        speed sits right at a single threshold, and restarting there every
        few steps kept the integral from ever settling.
      * Accelerating, the lag works both ways; one-sided, it let a relaunch
        from standstill run at +4.5 m/s^2 when +1.5 was asked. Braking, it only
        ever asks for more brake: a car that has braked harder than asked
        keeps no credit, which had left it coasting at -1 m/s^2 toward a stop
        it was told to make at -3.
      * Anti-windup: the integral stops while the output is pinned.
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.3,
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

    def reset(self) -> None:
        self.v_ref = None
        self.integral = 0.0
        self._regime = None

    def step(self, accel: float, v: float, dt: float) -> Tuple[float, float]:
        """(throttle, brake) for this step; `self.v_ref` is the reference."""
        regime = self._regime_for(accel)
        if self.v_ref is None or regime != self._regime:
            ref = v
        else:
            ref = self.v_ref + accel * dt
        self._regime = regime
        if regime < 0:
            ref = min(ref, v)
        self.v_ref = max(0.0, v - self.lag_cap,
                         min(self.v_max, v + self.lag_cap, ref))
        if self.v_ref <= 0.05 and accel <= 0.0:
            return 0.0, 1.0                   # stopped and asked to stay stopped
        lag = self.v_ref - v
        ff = (accel * self.throttle_per_mps2 if accel >= 0.0
              else accel * self.brake_per_mps2)
        u = ff + self.kp * lag + self.ki * self.integral
        lo = 0.0 if regime > 0 else -1.0
        hi = 0.0 if regime < 0 else 1.0
        pinned = (u >= hi and lag > 0.0) or (u <= lo and lag < 0.0)
        if not pinned:
            self.integral = max(-self.integral_limit,
                                min(self.integral_limit, self.integral + lag * dt))
        u = max(lo, min(hi, u))
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

    `policy` is duck-typed: it needs `commanded_accel(throttle)` and `ego.v`.
    """

    def __init__(self, vehicle, pid: Optional[LongitudinalPID] = None,
                 max_steer_deg: Optional[float] = None,
                 delta_max: float = math.radians(32),
                 v_max: float = 18.0,
                 accel_horizon: float = ACCEL_HORIZON):
        self.vehicle = vehicle
        self.spawn_gear = SpawnGear()
        self.pid = pid or LongitudinalPID()
        self.max_steer_deg = max_steer_deg or self._read_max_steer(vehicle)
        self.delta_max = float(delta_max)
        self.v_max = float(v_max)
        self.accel_horizon = float(accel_horizon)
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
        """The VehicleControl for one step, plus the target speed used."""
        accel = policy.commanded_accel(throttle_cmd)
        v_target = max(0.0, min(self.v_max, policy.ego.v + accel * self.accel_horizon))
        throttle, brake = self.pid.step(v_target, policy.ego.v, dt)
        steer = max(-1.0, min(1.0, steer_cmd * self.steer_scale))
        gear = self.spawn_gear.control_kwargs(self.vehicle, policy.ego.v, dt)
        return carla.VehicleControl(throttle=float(throttle), steer=float(steer),
                                    brake=float(brake), **gear), v_target
