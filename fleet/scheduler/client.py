"""What the scheduler holds. One object per robot, no vendor knowledge.

    robot = fleet["maira-01"]
    robot.execute("PickObject", object="battery")

`execute` blocks until the robot finishes and raises `RobotError` if it did not.
`execute_async` returns a handle for running several robots at once. Both go
over the same request/reply pattern; the difference is only who waits.

There is no NEURA, ROS, or DDS anywhere in this file or anything it imports.
That is the point of the architecture: this module cannot accidentally acquire
a dependency on a robot's SDK, because it has no way to reach one.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from fleetcore import topics
from fleetcore.envelope import (Command, Presence, Reply, ReplyStatus,
                                RobotState, StateMessage, new_id)
from fleetcore.errors import ErrorCode, RobotError
from fleetcore.transport import Transport

log = logging.getLogger(__name__)


class Pending:
    """A command in flight. Waits for the terminal reply, not the ack."""

    def __init__(self, command_id: str, command: str, timeout_s: float):
        self.command_id = command_id
        self.command = command
        self.timeout_s = timeout_s
        self.accepted = threading.Event()
        self.finished = threading.Event()
        self.reply: Reply | None = None
        self.sent_at = time.monotonic()

    def on_reply(self, reply: Reply) -> None:
        self.reply = reply
        if reply.status is ReplyStatus.ACCEPTED:
            self.accepted.set()
            return
        self.accepted.set()
        self.finished.set()

    def wait(self, timeout: float | None = None) -> Reply:
        total = timeout if timeout is not None else self.timeout_s
        # A generous margin over the agent's own deadline: the agent should be
        # the one to time a command out and say so, because it can also stop
        # the robot. The client giving up first would leave a task running with
        # nobody watching it.
        if not self.finished.wait(total + 5.0):
            raise RobotError(
                ErrorCode.TIMEOUT,
                f"no reply for {self.command} within {total + 5.0:.0f}s",
                {"command_id": self.command_id},
            )
        assert self.reply is not None
        if self.reply.status is ReplyStatus.COMPLETED:
            return self.reply
        err = (RobotError.from_dict(self.reply.error) if self.reply.error
               else RobotError(ErrorCode.INTERNAL, "failed without an error body"))
        raise err


class RobotClient:
    """A single robot, as the scheduler sees it."""

    def __init__(self, robot_id: str, transport: Transport, client_id: str):
        self.robot_id = robot_id
        self._transport = transport
        self._client_id = client_id
        self._pending: dict[str, Pending] = {}
        self._lock = threading.Lock()

        self.state: RobotState = RobotState.OFFLINE
        self.model = "unknown"
        self.capabilities: list[str] = []
        self.current_task: str | None = None
        self.last_error: dict | None = None
        self.online = False
        self._state_ts = 0.0
        self._listeners: list[Callable[[StateMessage], None]] = []

    # ------------------------------------------------------------- public API
    def execute(self, command: str, timeout_s: float = 120.0, **params):
        """Send a command and block until the robot finishes it."""
        return self.execute_async(command, timeout_s=timeout_s, **params).wait()

    def execute_async(self, command: str, timeout_s: float = 120.0,
                      **params) -> Pending:
        """Send a command and return immediately, for parallel fleet work."""
        if not self.online:
            raise RobotError(
                ErrorCode.TRANSPORT,
                f"{self.robot_id} is offline",
                {"last_seen": self._state_ts},
            )
        # Checked locally so an unsupported command fails instantly and
        # identically whether or not the robot happens to be reachable.
        if self.capabilities and command not in self.capabilities:
            raise RobotError(
                ErrorCode.UNSUPPORTED,
                f"{self.robot_id} ({self.model}) cannot {command!r}",
                {"capabilities": self.capabilities},
            )

        cmd = Command(command=command, params=params, timeout_s=timeout_s,
                      command_id=new_id())
        cmd.reply_to = topics.reply(self._client_id, cmd.command_id)
        pending = Pending(cmd.command_id, command, timeout_s)
        with self._lock:
            self._pending[cmd.command_id] = pending
        log.info("-> %s %s %s", self.robot_id, command, params or "")
        self._transport.publish(topics.cmd(self.robot_id), cmd.to_bytes(), qos=1)
        return pending

    def stop(self, latch: bool = True) -> Reply:
        """Interrupt whatever the robot is doing. Works while BUSY."""
        return self.execute_async("Stop", timeout_s=10.0, latch=latch).wait(10.0)

    def get_state(self) -> dict:
        return self.execute("GetRobotState", timeout_s=10.0).result

    def reset_error(self) -> Reply:
        return self.execute_async("ResetError", timeout_s=15.0).wait()

    @property
    def idle(self) -> bool:
        return self.online and self.state is RobotState.IDLE

    def wait_idle(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.idle:
                return True
            time.sleep(0.1)
        return False

    def on_state_change(self, fn: Callable[[StateMessage], None]) -> None:
        self._listeners.append(fn)

    def supports(self, command: str) -> bool:
        return command in self.capabilities

    # ------------------------------------------------------------- bus events
    def _handle_state(self, msg: StateMessage) -> None:
        self.state = msg.state
        self.model = msg.model
        self.capabilities = msg.capabilities
        self.current_task = msg.current_task
        self.last_error = msg.last_error
        self._state_ts = msg.ts
        for fn in list(self._listeners):
            try:
                fn(msg)
            except Exception:
                log.exception("state listener failed")

    def _handle_presence(self, msg: Presence) -> None:
        if msg.online != self.online:
            log.info("%s is %s", self.robot_id,
                     "online" if msg.online else "OFFLINE")
        self.online = msg.online
        if not msg.online:
            self.state = RobotState.OFFLINE
            # Nothing will ever answer these now; fail them rather than let the
            # scheduler block until timeout on a robot that has gone away.
            self._fail_all(RobotError(
                ErrorCode.TRANSPORT, f"{self.robot_id} went offline mid-command"))

    def _handle_reply(self, reply: Reply) -> None:
        with self._lock:
            pending = self._pending.get(reply.command_id)
            if pending and reply.terminal:
                self._pending.pop(reply.command_id, None)
        if pending is None:
            log.debug("reply for unknown command %s", reply.command_id)
            return
        pending.on_reply(reply)

    def _fail_all(self, error: RobotError) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for p in pending:
            p.reply = Reply(command_id=p.command_id, status=ReplyStatus.FAILED,
                            robot_id=self.robot_id, error=error.to_dict())
            p.accepted.set()
            p.finished.set()
