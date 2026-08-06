"""The Robot Edge Agent. Runs on the workstation, next to the SDK.

It is the only process that touches vendor code, and it exposes exactly one
thing to the network: a command topic and some status. The laptop cannot reach
the SDK even if it wanted to -- there is no route to it, only messages.

Responsibilities, in order of importance:

1. Never let a bad message reach the robot. Unknown commands, wrong parameters
   and schema mismatches are refused at the edge.
2. Make Stop work. It bypasses the queue entirely (see `executor.py`).
3. Tell the truth about state, including when the agent itself has died --
   which is what the broker's Last Will is for.
4. Survive the network going away without dropping the robot.

Reconnection policy
-------------------
Losing the broker does *not* stop a running motion. Aborting a half-finished
pick because wifi blinked would be worse than finishing it, and the robot is
still physically fine. What it does do is refuse *new* work while disconnected
(there is no one to answer), and re-publish full state on reconnect so the
scheduler resynchronises without asking.

If you would rather a disconnected robot halt, set `stop_on_disconnect_s` in
the config; that is a safety judgement about your cell, not a default anyone
else should make for you.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

from fleetcore import topics
from fleetcore.envelope import (SCHEMA_VERSION, Command, Presence, Reply,
                                ReplyStatus, RobotState, StateMessage)
from fleetcore.errors import ErrorCode, RobotError
from fleetcore.transport import Transport

from .adapters.base import RobotAdapter
from .executor import Executor, TaskResult

log = logging.getLogger(__name__)

AGENT_VERSION = "1.0.0"

# Commands the agent answers itself. They must work when the robot is busy,
# broken or in ESTOP, which is precisely when you need them most.
CONTROL_COMMANDS = {"Stop", "GetRobotState", "Ping"}


class _SeenCommands:
    """Bounded memory of recent command ids, for at-least-once delivery.

    MQTT QoS 1 redelivers. Without this, one retransmitted `PickObject` is two
    picks. Replies are cached alongside so a duplicate gets the original answer
    rather than a BUSY it did not earn.
    """

    def __init__(self, capacity: int = 512):
        self._items: OrderedDict[str, Reply | None] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def seen(self, command_id: str) -> bool:
        with self._lock:
            return command_id in self._items

    def add(self, command_id: str) -> None:
        with self._lock:
            self._items[command_id] = None
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def record_reply(self, command_id: str, reply: Reply) -> None:
        with self._lock:
            if command_id in self._items:
                self._items[command_id] = reply

    def last_reply(self, command_id: str) -> Reply | None:
        with self._lock:
            return self._items.get(command_id)


class EdgeAgent:
    def __init__(self, robot_id: str, adapter: RobotAdapter,
                 transport: Transport, config: dict | None = None):
        self.robot_id = robot_id
        self.adapter = adapter
        self.transport = transport
        self.config = config or {}

        self.executor = Executor(adapter, on_state_change=self.publish_state)
        self._seen = _SeenCommands()
        self._stop_flag = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_disconnect: float | None = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        log.info("edge agent %s starting (%s)", self.robot_id, self.adapter.model)
        self.transport.connect()
        self.transport.subscribe(topics.cmd(self.robot_id), self._on_command)

        # Presence first: say we are here before claiming any state.
        self._publish_presence(True)
        self.publish_state()

        self._spawn(self._sdk_supervisor, "sdk-supervisor")
        self._spawn(self._state_heartbeat, "state-heartbeat")

    def stop(self) -> None:
        log.info("edge agent stopping")
        self._stop_flag.set()
        try:
            self.executor.shutdown()
        finally:
            # A clean shutdown publishes its own offline marker; the will covers
            # the unclean case.
            self._publish_presence(False)
            self.adapter.disconnect()
            time.sleep(0.2)          # let the last publishes flush
            self.transport.disconnect()
        for t in self._threads:
            t.join(timeout=2.0)

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop_flag.wait(0.5):
                pass
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.stop()

    def _spawn(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    # -------------------------------------------------------- SDK supervision
    def _sdk_supervisor(self) -> None:
        """Keep the vendor SDK connected, with backoff.

        The SDK is the thing most likely to be unavailable at boot -- the
        workstation may come up before the robot does -- so the agent starts
        anyway, reports STARTING, and keeps trying. An agent that exited on a
        failed SDK connect would need a human to restart it every morning.
        """
        delay = 1.0
        while not self._stop_flag.is_set():
            if not self.adapter.connected:
                try:
                    self.adapter.connect()
                    log.info("SDK connected")
                    delay = 1.0
                    self.executor.set_ready(True)
                    self.publish_state()
                except RobotError as exc:
                    self.executor.set_ready(False)
                    log.warning("SDK connect failed: %s (retry in %.0fs)",
                                exc, delay)
                    self.publish_state()
                    self._stop_flag.wait(delay)
                    delay = min(delay * 2, 30.0)
                    continue
            self._stop_flag.wait(2.0)

    def _state_heartbeat(self) -> None:
        """Periodic state, so a stale retained message is detectable by age."""
        period = float(self.config.get("state_period_s", 5.0))
        grace = self.config.get("stop_on_disconnect_s")
        while not self._stop_flag.wait(period):
            connected = getattr(self.transport, "connected", True)
            if connected:
                if self._last_disconnect is not None:
                    log.info("bus reconnected; republishing state")
                    self._last_disconnect = None
                    self._publish_presence(True)
                self.publish_state()
            else:
                if self._last_disconnect is None:
                    self._last_disconnect = time.monotonic()
                    log.warning("bus disconnected; refusing new work")
                elif grace and time.monotonic() - self._last_disconnect > float(grace):
                    if self.executor.busy:
                        log.error("bus down for %ss with a task running -- "
                                  "stopping the robot", grace)
                        self.executor.stop()

    # ------------------------------------------------------------- publishing
    def _publish_presence(self, online: bool) -> None:
        self.transport.publish(
            topics.presence(self.robot_id),
            Presence(self.robot_id, online).to_bytes(),
            qos=1, retain=True)

    def publish_state(self) -> None:
        try:
            caps = sorted(self.adapter.capabilities()) if self.adapter.connected else []
        except Exception:
            caps = []
        msg = StateMessage(
            robot_id=self.robot_id,
            model=self.adapter.model,
            state=self.executor.state,
            capabilities=sorted(set(caps) | CONTROL_COMMANDS),
            current_task=self.executor.current_task,
            current_command_id=self.executor.current_command_id,
            last_error=(self.executor.last_error.to_dict()
                        if self.executor.last_error else None),
            sdk_connected=self.adapter.connected,
            telemetry=self._safe_telemetry(),
            agent_version=AGENT_VERSION,
        )
        self.transport.publish(topics.state(self.robot_id), msg.to_bytes(),
                               qos=1, retain=True)

    def _reply(self, cmd: Command, status: ReplyStatus,
               result: dict | None = None, error: RobotError | None = None) -> None:
        if not cmd.reply_to:
            return
        reply = Reply(command_id=cmd.command_id, status=status,
                      robot_id=self.robot_id, result=result or {},
                      error=error.to_dict() if error else None)
        self._seen.record_reply(cmd.command_id, reply)
        self.transport.publish(cmd.reply_to, reply.to_bytes(), qos=1)

    # ---------------------------------------------------------- command entry
    def _on_command(self, _topic: str, payload: bytes) -> None:
        """Runs on the transport thread. Must return quickly."""
        try:
            cmd = Command.from_bytes(payload)
        except Exception as exc:
            log.error("undecodable command dropped: %s", exc)
            return

        if cmd.v != SCHEMA_VERSION:
            log.error("schema mismatch: agent speaks v%s, got v%s",
                      SCHEMA_VERSION, cmd.v)
            self._reply(cmd, ReplyStatus.REJECTED, error=RobotError(
                ErrorCode.INVALID_PARAMS,
                f"schema v{cmd.v} not supported; agent speaks v{SCHEMA_VERSION}"))
            return

        # Duplicate delivery: answer with the original outcome, do not re-run.
        if self._seen.seen(cmd.command_id):
            previous = self._seen.last_reply(cmd.command_id)
            log.info("duplicate command %s ignored", cmd.command_id)
            if previous and cmd.reply_to:
                self.transport.publish(cmd.reply_to, previous.to_bytes(), qos=1)
            return
        self._seen.add(cmd.command_id)

        log.info("command %s %s %s", cmd.command_id, cmd.command, cmd.params or "")

        if cmd.command in CONTROL_COMMANDS:
            self._handle_control(cmd)
            return

        if cmd.command == "ResetError":
            self.executor.reset()

        error = self.executor.submit(
            cmd, on_done=lambda res: self._on_task_done(cmd, res))
        if error is not None:
            self._reply(cmd, ReplyStatus.REJECTED, error=error)
            self.publish_state()
            return
        self._reply(cmd, ReplyStatus.ACCEPTED, {"command": cmd.command})

    def _handle_control(self, cmd: Command) -> None:
        """Answered inline, never queued -- these must work while BUSY."""
        if cmd.command == "Stop":
            result = self.executor.stop(latch=bool(cmd.params.get("latch", True)))
            self._reply(cmd, ReplyStatus.COMPLETED, result)
            self.publish_state()
        elif cmd.command == "Ping":
            self._reply(cmd, ReplyStatus.COMPLETED,
                        {"agent_version": AGENT_VERSION, "ts": time.time()})
        elif cmd.command == "GetRobotState":
            self.publish_state()
            self._reply(cmd, ReplyStatus.COMPLETED, {
                "state": self.executor.state.value,
                "current_task": self.executor.current_task,
                "sdk_connected": self.adapter.connected,
                "telemetry": self._safe_telemetry(),
            })

    def _on_task_done(self, cmd: Command, res: TaskResult) -> None:
        if res.ok:
            self._reply(cmd, ReplyStatus.COMPLETED,
                        {**res.result, "duration_s": round(res.duration_s, 3)})
        else:
            self._reply(cmd, ReplyStatus.FAILED, error=res.error)
        self.transport.publish(topics.event(self.robot_id), Reply(
            command_id=res.command_id,
            status=ReplyStatus.COMPLETED if res.ok else ReplyStatus.FAILED,
            robot_id=self.robot_id,
            result={"command": res.command,
                    "duration_s": round(res.duration_s, 3)},
            error=res.error.to_dict() if res.error else None).to_bytes(), qos=1)
        self.publish_state()

    def _safe_telemetry(self) -> dict:
        try:
            return self.adapter.telemetry()
        except Exception:
            log.exception("telemetry raised")
            return {}
