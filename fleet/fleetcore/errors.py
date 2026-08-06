"""Error taxonomy shared by both sides of the bus.

The scheduler has to make decisions from failures without knowing anything about
NEURA, Unitree, or whatever comes next, so errors carry a stable machine-readable
code. `retryable` is the field a scheduler actually branches on: BUSY and
SDK_UNAVAILABLE are worth trying again, UNSUPPORTED and INVALID_PARAMS never
will be, and ESTOP needs a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ErrorCode(str, Enum):
    # --- the command never started -----------------------------------------
    UNSUPPORTED = "UNSUPPORTED"          # this robot has no such capability
    INVALID_PARAMS = "INVALID_PARAMS"    # capability exists, arguments do not fit
    BUSY = "BUSY"                        # another task holds the robot
    NOT_READY = "NOT_READY"              # needs homing, calibration, enable
    ESTOP = "ESTOP"                      # latched stop; needs an explicit reset

    # --- the command started and did not finish ----------------------------
    SDK_ERROR = "SDK_ERROR"              # the vendor SDK raised
    MOTION_FAILED = "MOTION_FAILED"      # planning or execution failure
    TIMEOUT = "TIMEOUT"                  # exceeded the deadline
    CANCELLED = "CANCELLED"              # superseded by Stop

    # --- infrastructure -----------------------------------------------------
    SDK_UNAVAILABLE = "SDK_UNAVAILABLE"  # agent cannot reach the robot at all
    TRANSPORT = "TRANSPORT"              # bus failure
    INTERNAL = "INTERNAL"                # a bug; should never be routine


# Trying these again may work. Everything else will fail the same way twice.
RETRYABLE = frozenset({
    ErrorCode.BUSY,
    ErrorCode.SDK_UNAVAILABLE,
    ErrorCode.TRANSPORT,
    ErrorCode.TIMEOUT,
})


@dataclass(frozen=True)
class RobotError(Exception):
    """A failure expressed in fleet terms rather than vendor terms."""

    code: ErrorCode
    message: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "detail": self.detail,
        }

    @staticmethod
    def from_dict(data: dict) -> "RobotError":
        try:
            code = ErrorCode(data.get("code", "INTERNAL"))
        except ValueError:
            code = ErrorCode.INTERNAL
        return RobotError(
            code=code,
            message=data.get("message", ""),
            detail=data.get("detail", {}) or {},
        )
