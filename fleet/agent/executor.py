"""Single-flight task execution with preemptible Stop.

One robot, one arm, one task at a time. The executor enforces that and owns the
robot's state machine.

Why Stop is not a queued command
--------------------------------
If Stop went through the same path as PickObject it would wait behind the motion
it is meant to interrupt, which makes it worthless. Stop is handled on the
transport thread: it sets the cancel token and calls the adapter's `stop()`
directly, while the worker thread is still inside the SDK call. Everything else
runs on the worker.

That split is the reason `RobotAdapter.stop()` is documented as reentrant and
callable from another thread -- it is the one method with that requirement.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from fleetcore.envelope import Command, RobotState
from fleetcore.errors import ErrorCode, RobotError

from .adapters.base import CancelToken, RobotAdapter

log = logging.getLogger(__name__)


@dataclass
class TaskResult:
    command_id: str
    command: str
    ok: bool
    result: dict
    error: RobotError | None
    duration_s: float


class Executor:
    """Runs one command at a time and tracks what the robot is doing."""

    def __init__(self, adapter: RobotAdapter,
                 on_state_change: Callable[[], None] | None = None):
        self.adapter = adapter
        self._on_state_change = on_state_change or (lambda: None)

        self._lock = threading.Lock()
        self._cancel = CancelToken()
        self._thread: threading.Thread | None = None

        self._state = RobotState.STARTING
        self.current_task: str | None = None
        self.current_command_id: str | None = None
        self.last_error: RobotError | None = None

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> RobotState:
        return self._state

    def _set_state(self, state: RobotState) -> None:
        if state != self._state:
            log.info("state %s -> %s", self._state.value, state.value)
            self._state = state
            self._on_state_change()

    def set_ready(self, ready: bool) -> None:
        """Called by the agent as the SDK connects or drops."""
        if not ready:
            self._set_state(RobotState.STARTING)
        elif self._state is RobotState.STARTING:
            self._set_state(RobotState.IDLE)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -------------------------------------------------------------- dispatch
    def submit(self, cmd: Command,
               on_done: Callable[[TaskResult], None]) -> RobotError | None:
        """Start a command. Returns an error if it was refused outright.

        Refusal here is the ACCEPTED/REJECTED decision: it happens before
        anything moves, so the caller can answer the scheduler immediately.
        """
        with self._lock:
            if self.busy:
                return RobotError(
                    ErrorCode.BUSY,
                    f"{self.current_task} is still running",
                    {"current_task": self.current_task,
                     "current_command_id": self.current_command_id},
                )
            if self._state is RobotState.ESTOP and cmd.command != "ResetError":
                return RobotError(
                    ErrorCode.ESTOP,
                    "robot is stopped; send ResetError to clear it",
                )
            if not self.adapter.connected:
                return RobotError(ErrorCode.SDK_UNAVAILABLE,
                                  "robot SDK is not connected")
            if not self.adapter.supports(cmd.command):
                return RobotError(
                    ErrorCode.UNSUPPORTED,
                    f"{self.adapter.model} does not implement {cmd.command!r}",
                    {"capabilities": sorted(self.adapter.capabilities())},
                )

            self._cancel.reset()
            self.current_task = cmd.command
            self.current_command_id = cmd.command_id
            self._set_state(RobotState.BUSY)
            self._thread = threading.Thread(
                target=self._run, args=(cmd, on_done),
                name=f"task-{cmd.command_id}", daemon=True)
            self._thread.start()
        return None

    def _run(self, cmd: Command, on_done: Callable[[TaskResult], None]) -> None:
        started = time.monotonic()
        error: RobotError | None = None
        result: dict = {}

        # The deadline is enforced by a timer rather than by the worker, because
        # the worker is blocked inside the SDK and cannot check the clock.
        timer = threading.Timer(max(1.0, cmd.timeout_s), self._on_timeout,
                                args=(cmd.command_id,))
        timer.daemon = True
        timer.start()
        try:
            result = self.adapter.execute(cmd.command, cmd.params, self._cancel)
        except RobotError as exc:
            error = exc
        except Exception as exc:                       # pragma: no cover
            log.exception("executor caught a non-RobotError")
            error = RobotError(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")
        finally:
            timer.cancel()

        duration = time.monotonic() - started
        with self._lock:
            self.current_task = None
            self.current_command_id = None
            self.last_error = error
            if error is None:
                self._set_state(RobotState.IDLE)
            elif error.code is ErrorCode.ESTOP or self._cancel.cancelled:
                # A cancelled task means Stop was pressed; stay latched until
                # somebody explicitly clears it.
                self._set_state(RobotState.ESTOP)
            else:
                self._set_state(RobotState.ERROR)

        if error:
            log.warning("%s failed after %.1fs: %s", cmd.command, duration, error)
        else:
            log.info("%s completed in %.1fs", cmd.command, duration)

        try:
            on_done(TaskResult(cmd.command_id, cmd.command, error is None,
                               result, error, duration))
        except Exception:
            log.exception("completion callback failed")

    def _on_timeout(self, command_id: str) -> None:
        with self._lock:
            if self.current_command_id != command_id:
                return          # already finished; nothing to interrupt
        log.warning("command %s exceeded its deadline; stopping", command_id)
        self._cancel.cancel()
        try:
            self.adapter.stop()
        except Exception:
            log.exception("adapter.stop() raised during timeout")

    # ------------------------------------------------------------------ stop
    def stop(self, latch: bool = True) -> dict:
        """Interrupt whatever is running. Safe when idle.

        Runs on the caller's thread -- deliberately not queued, so it can reach
        the SDK while the worker is still blocked inside it.
        """
        was_busy = self.busy
        task = self.current_task
        self._cancel.cancel()
        try:
            self.adapter.stop()
        except Exception:
            log.exception("adapter.stop() raised; latching state anyway")
        if latch:
            with self._lock:
                self._set_state(RobotState.ESTOP)
        return {"was_busy": was_busy, "interrupted_task": task}

    def reset(self) -> None:
        """Leave ESTOP. Only ever called from an explicit ResetError command."""
        self._cancel.reset()
        with self._lock:
            self.last_error = None
            self._set_state(
                RobotState.IDLE if self.adapter.connected else RobotState.STARTING)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._cancel.cancel()
        try:
            self.adapter.stop()
        except Exception:
            pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
