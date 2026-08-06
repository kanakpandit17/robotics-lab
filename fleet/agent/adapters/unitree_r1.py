"""Unitree R1 -- the proof that adding a robot does not touch the scheduler.

Nothing above this file changes to support it. The scheduler still calls
`robot.execute("PickObject")`; what differs is only what that means here, and
that difference is entirely contained in `capabilities()`.

The R1 is a useful second robot precisely because it is *unlike* MAiRA. It has
no gripper, so `PickObject` is genuinely unsupported and the adapter says so via
UNSUPPORTED rather than pretending. It walks, so it has `MoveTo`, which MAiRA
does not. A scheduler that reads `capabilities` off the bus handles both without
a special case; one that assumes every robot is an arm does not.

Bindings are marked SDK-BINDING and match the existing code in this repository:
gestures go through `G1ArmActionClient` on the `arm` service, locomotion through
`rt/arm_sdk` or the wireless-controller channel.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fleetcore.errors import ErrorCode, RobotError

from .base import CancelToken, RobotAdapter

log = logging.getLogger(__name__)

# Gesture names as the robot itself reports them via GetActionList.
GESTURES = {
    "Wave": "wave_above_head",
    "RaiseHand": "right_hand_up",
    "ShakeHand": "shake_hand",
    "ArmsDown": "release_arm",
    "HighFive": "high_five",
}


class UnitreeR1Adapter(RobotAdapter):
    model = "Unitree R1"

    def __init__(self, robot_id: str, config: dict | None = None):
        super().__init__(robot_id, config)
        self._arm: Any = None
        self._simulate = bool(self.config.get("simulate", False))

    def connect(self) -> None:
        if self._simulate:
            from .simulator import StubArm
            self._arm = StubArm(self.model)
            self._connected = True
            return
        try:
            # SDK-BINDING: the DDS path this repository already uses.
            #   from r1vision.unitree_dds import init_channel
            #   from unitree_sdk2py.g1.arm.g1_arm_action_client import (
            #       G1ArmActionClient)
            #   init_channel(self.config.get("iface", "auto"))
            #   self._arm = G1ArmActionClient(); self._arm.Init()
            raise ImportError(
                "Unitree binding not filled in -- see SDK-BINDING in "
                "agent/adapters/unitree_r1.py"
            )
        except Exception as exc:
            self._connected = False
            raise RobotError(ErrorCode.SDK_UNAVAILABLE,
                             f"cannot reach R1: {exc}") from exc

    def disconnect(self) -> None:
        self._arm = None
        self._connected = False

    def capabilities(self) -> dict[str, Callable[..., Any]]:
        # No PickObject and no PlaceObject: this robot has no gripper, and
        # advertising a capability it cannot honour would push the failure from
        # planning time to motion time.
        caps: dict[str, Callable[..., Any]] = {
            "Gesture": self._gesture,
            "ArmsDown": self._arms_down,
            "MoveHome": self._arms_down,
        }
        if self.config.get("walking_enabled"):
            caps["MoveTo"] = self._move_to
        return caps

    def _gesture(self, params: dict, cancel: CancelToken) -> dict:
        name = params.get("name", "Wave")
        action = GESTURES.get(name)
        if action is None:
            raise RobotError(ErrorCode.INVALID_PARAMS,
                             f"unknown gesture {name!r}",
                             {"known": sorted(GESTURES)})
        # SDK-BINDING: code = self._arm.ExecuteAction(ACTIONS[action])
        #   A non-zero code here is the robot refusing -- 7406 means a built-in
        #   mode owns the joints, which is NOT_READY rather than SDK_ERROR.
        return {"gesture": name, "action": action}

    def _arms_down(self, params: dict, cancel: CancelToken) -> dict:
        return {"gesture": "ArmsDown"}

    def _move_to(self, params: dict, cancel: CancelToken) -> dict:
        raise RobotError(ErrorCode.UNSUPPORTED,
                         "walking is not enabled on this agent")

    def stop(self) -> None:
        try:
            if self._arm is not None:
                pass  # SDK-BINDING: zero velocity, then release_arm
        except Exception:
            log.exception("R1 stop raised")
