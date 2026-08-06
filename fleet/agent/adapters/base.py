"""The seam between fleet vocabulary and a vendor SDK.

This is the only file a new robot needs. `PickBattery` means one thing to the
scheduler and something entirely different to MAiRA, LARA and an R1; an adapter
is where that translation lives, and it is the reason adding a robot never
touches the scheduler.

Contract
--------
* `capabilities()` is advertised on the bus, so the scheduler can ask what a
  robot can do rather than being compiled with the answer.
* `execute()` runs one command to completion and raises `RobotError` on
  failure. It is called on a worker thread and must honour `cancel`.
* `stop()` is called from a *different* thread, while `execute` is still
  running. It must be safe to call at any time, including when idle.

Why capabilities are declared, not discovered by reflection
-----------------------------------------------------------
An explicit table means an adapter can expose `PickObject` while the underlying
SDK calls it `grasp_at_pose`, and it keeps a typo in a method name from silently
becoming an unsupported command at runtime.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable

from fleetcore.errors import ErrorCode, RobotError

log = logging.getLogger(__name__)


class CancelToken:
    """Cooperative cancellation, checked by long-running adapter code.

    A motion cannot be interrupted from outside without the SDK's help, so
    adapters poll this between steps. `raise_if_cancelled` is the one-liner for
    the top of every loop.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, seconds: float) -> bool:
        """Sleep, but wake immediately on cancel. Returns True if cancelled."""
        return self._event.wait(seconds)

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise RobotError(ErrorCode.CANCELLED, "superseded by Stop")


class RobotAdapter(ABC):
    """Base class for every robot the fleet can talk to."""

    model: str = "unknown"

    def __init__(self, robot_id: str, config: dict | None = None):
        self.robot_id = robot_id
        self.config = config or {}
        self._connected = False

    # ------------------------------------------------------------- lifecycle
    @abstractmethod
    def connect(self) -> None:
        """Open the vendor SDK. Raise RobotError(SDK_UNAVAILABLE) if it fails."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the SDK. Must not raise."""

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------ capability
    @abstractmethod
    def capabilities(self) -> dict[str, Callable[..., Any]]:
        """Command name -> bound method. Advertised to the whole fleet."""

    def supports(self, command: str) -> bool:
        return command in self.capabilities()

    # ------------------------------------------------------------- execution
    def execute(self, command: str, params: dict, cancel: CancelToken) -> dict:
        """Run one command. Returns a JSON-safe result dict."""
        handler = self.capabilities().get(command)
        if handler is None:
            raise RobotError(
                ErrorCode.UNSUPPORTED,
                f"{self.model} does not implement {command!r}",
                {"capabilities": sorted(self.capabilities())},
            )
        if not self._connected:
            raise RobotError(ErrorCode.SDK_UNAVAILABLE,
                             f"{self.model} SDK is not connected")
        try:
            result = handler(params=params, cancel=cancel)
        except RobotError:
            raise                       # already in fleet vocabulary
        except TypeError as exc:
            # A signature mismatch is the adapter's bug, but the most common
            # cause is a caller passing parameters this command does not take.
            raise RobotError(ErrorCode.INVALID_PARAMS, str(exc)) from exc
        except Exception as exc:
            # Anything the vendor SDK throws becomes one fleet error, so the
            # scheduler never imports a NEURA or Unitree exception type.
            raise RobotError(
                ErrorCode.SDK_ERROR,
                f"{type(exc).__name__}: {exc}",
                {"command": command},
            ) from exc
        return result if isinstance(result, dict) else {"value": result}

    @abstractmethod
    def stop(self) -> None:
        """Halt motion now. Called from another thread; must be reentrant."""

    # ------------------------------------------------------------------ state
    def telemetry(self) -> dict:
        """Optional extras merged into the published state. Never raises."""
        return {}
