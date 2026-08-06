"""A robot that isn't there.

`StubArm` stands in for a vendor SDK so the adapters above it run unchanged.
`SimulatorAdapter` is a whole fake robot for developing the scheduler.

Motions take real wall-clock time and are interruptible, because the bugs worth
catching before hardware -- a Stop that arrives mid-motion, a second command
while the first is running, a timeout -- only appear when the robot is slow.
An instant fake would pass tests the real system fails.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

from fleetcore.errors import ErrorCode, RobotError

from .base import CancelToken, RobotAdapter

log = logging.getLogger(__name__)


class StubArm:
    """Minimal stand-in for a vendor SDK object."""

    def __init__(self, model: str = "stub", move_time: float = 1.5,
                 failure_rate: float = 0.0):
        self.model = model
        self.move_time = move_time
        self.failure_rate = failure_rate
        self.holding: str | None = None
        self.pose = "unknown"
        self._stopped = False

    def _travel(self, seconds: float, cancel: CancelToken) -> None:
        """Sleep in slices so Stop lands mid-motion, as it would in reality."""
        self._stopped = False
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stopped:
                raise RobotError(ErrorCode.CANCELLED, "stopped mid-motion")
            if cancel.wait(0.05):
                raise RobotError(ErrorCode.CANCELLED, "superseded by Stop")
        if self.failure_rate and random.random() < self.failure_rate:
            raise RuntimeError("simulated planning failure")

    def move_home(self, speed: float = 0.2, cancel: CancelToken | None = None):
        self._travel(self.move_time / max(speed, .01) * .2, cancel or CancelToken())
        self.pose = "home"

    def pick(self, pose, speed: float = 0.2, cancel: CancelToken | None = None):
        self._travel(self.move_time / max(speed, .01) * .2, cancel or CancelToken())
        self.pose, self.holding = str(pose), str(pose)

    def place(self, pose, speed: float = 0.2, cancel: CancelToken | None = None):
        self._travel(self.move_time / max(speed, .01) * .2, cancel or CancelToken())
        self.pose, self.holding = str(pose), None

    def pause(self):
        """The real adapter's Stop calls pause(), so the stub must have it."""
        self.stop()

    def unpause(self):
        self._paused = False

    def stop(self):
        self._stopped = True

    def reset(self):
        self._stopped = False

    def close(self):
        pass

    def telemetry(self) -> dict:
        return {"pose": self.pose, "holding": self.holding}


class SimulatorAdapter(RobotAdapter):
    """A complete fake robot, for building the scheduler without hardware."""

    model = "Simulator"

    def __init__(self, robot_id: str, config: dict | None = None):
        super().__init__(robot_id, config)
        cfg = self.config
        self._arm = StubArm(
            model=cfg.get("model", "sim"),
            move_time=float(cfg.get("move_time", 1.5)),
            failure_rate=float(cfg.get("failure_rate", 0.0)),
        )
        self._estopped = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def capabilities(self) -> dict[str, Callable[..., Any]]:
        return {
            "MoveHome": self._move_home,
            "PickObject": self._pick,
            "PlaceObject": self._place,
            "StartAssembly": self._assembly,
            "ResetError": self._reset,
        }

    def _guard(self):
        if self._estopped:
            raise RobotError(ErrorCode.ESTOP, "stopped; send ResetError first")

    def _move_home(self, params: dict, cancel: CancelToken) -> dict:
        self._guard()
        self._arm.move_home(params.get("speed", 0.2), cancel)
        return {"pose": "home"}

    def _pick(self, params: dict, cancel: CancelToken) -> dict:
        self._guard()
        target = params.get("object") or params.get("target")
        if not target:
            raise RobotError(ErrorCode.INVALID_PARAMS, "PickObject needs 'object'")
        self._arm.pick(target, params.get("speed", 0.2), cancel)
        return {"picked": target, "holding": True}

    def _place(self, params: dict, cancel: CancelToken) -> dict:
        self._guard()
        target = params.get("location") or params.get("target") or "bin"
        self._arm.place(target, params.get("speed", 0.2), cancel)
        return {"placed_at": target, "holding": False}

    def _assembly(self, params: dict, cancel: CancelToken) -> dict:
        self._guard()
        steps = ["MoveHome", "PickObject", "PlaceObject"]
        for step in steps:
            cancel.raise_if_cancelled()
            self.execute(step, {"object": "part", "location": "jig"}, cancel)
        return {"routine": params.get("routine", "default"),
                "steps_completed": steps}

    def _reset(self, params: dict, cancel: CancelToken) -> dict:
        self._estopped = False
        self._arm.reset()
        return {"estop": False}

    def stop(self) -> None:
        self._estopped = True
        self._arm.stop()

    def telemetry(self) -> dict:
        return {"estopped": self._estopped, **self._arm.telemetry()}
