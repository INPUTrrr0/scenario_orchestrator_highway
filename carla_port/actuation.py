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
        return carla.VehicleControl(throttle=float(throttle), steer=float(steer),
                                    brake=float(brake)), v_target
