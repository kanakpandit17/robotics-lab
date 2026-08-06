"""The fleet: every robot on the bus, discovered rather than configured.

The scheduler never holds a list of IP addresses. It subscribes to wildcard
topics and robots appear as their agents announce themselves, which is what
makes "add a Unitree R1 later" a deployment step rather than a code change.

    fleet = Fleet(transport)
    fleet.start()
    fleet.wait_for("maira-01", timeout=10)
    fleet["maira-01"].execute("PickObject", object="battery")

Capability routing is the other half of that promise: `any_capable("Gesture")`
finds a robot that can do the thing without the caller knowing which robots
exist or what they are.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Iterator

from fleetcore import topics
from fleetcore.envelope import Presence, Reply, RobotState, StateMessage
from fleetcore.errors import ErrorCode, RobotError
from fleetcore.transport import Transport

from .client import RobotClient

log = logging.getLogger(__name__)


def default_client_id() -> str:
    return f"sched-{socket.gethostname().split('.')[0]}-{int(time.time()) % 100000}"


class Fleet:
    def __init__(self, transport: Transport, client_id: str | None = None):
        self.transport = transport
        self.client_id = client_id or default_client_id()
        self._robots: dict[str, RobotClient] = {}
        self._lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.transport.connect()
        # Replies are addressed to this client id, so two schedulers on the same
        # bus never see each other's answers.
        self.transport.subscribe(topics.reply_wildcard(self.client_id),
                                 self._on_reply)
        self.transport.subscribe(topics.all_states(), self._on_state)
        self.transport.subscribe(topics.all_presence(), self._on_presence)
        log.info("fleet listening as %s", self.client_id)

    def close(self) -> None:
        self.transport.disconnect()
        self._started = False

    def __enter__(self) -> "Fleet":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- accessors
    def __getitem__(self, robot_id: str) -> RobotClient:
        with self._lock:
            if robot_id not in self._robots:
                raise KeyError(
                    f"no robot {robot_id!r} on the bus; known: "
                    f"{sorted(self._robots)}")
            return self._robots[robot_id]

    def __contains__(self, robot_id: str) -> bool:
        with self._lock:
            return robot_id in self._robots

    def __iter__(self) -> Iterator[RobotClient]:
        with self._lock:
            return iter(list(self._robots.values()))

    @property
    def robots(self) -> list[RobotClient]:
        with self._lock:
            return list(self._robots.values())

    def online(self) -> list[RobotClient]:
        return [r for r in self.robots if r.online]

    def wait_for(self, robot_id: str, timeout: float = 15.0) -> RobotClient:
        """Block until a robot is on the bus and reporting itself online."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                robot = self._robots.get(robot_id)
            if robot is not None and robot.online and robot.capabilities:
                return robot
            time.sleep(0.05)
        raise RobotError(
            ErrorCode.TRANSPORT,
            f"{robot_id} did not appear within {timeout:.0f}s",
            {"seen": sorted(r.robot_id for r in self.robots)},
        )

    # ------------------------------------------------------- capability routing
    def capable_of(self, command: str) -> list[RobotClient]:
        return [r for r in self.robots if r.online and r.supports(command)]

    def any_capable(self, command: str) -> RobotClient:
        """Pick an idle robot that can run this command.

        Idle robots first, so a fleet with one busy and one free arm routes to
        the free one instead of queueing behind work already in progress.
        """
        candidates = self.capable_of(command)
        if not candidates:
            raise RobotError(
                ErrorCode.UNSUPPORTED,
                f"no online robot can {command!r}",
                {"online": [r.robot_id for r in self.online()]},
            )
        return next((r for r in candidates if r.idle), candidates[0])

    def stop_all(self) -> dict[str, str]:
        """Fleet-wide halt. Best effort -- one failure must not block the rest."""
        results = {}
        for robot in self.online():
            try:
                robot.stop()
                results[robot.robot_id] = "stopped"
            except Exception as exc:
                log.error("could not stop %s: %s", robot.robot_id, exc)
                results[robot.robot_id] = f"FAILED: {exc}"
        return results

    # ------------------------------------------------------------ bus routing
    def _robot(self, robot_id: str) -> RobotClient:
        with self._lock:
            robot = self._robots.get(robot_id)
            if robot is None:
                robot = RobotClient(robot_id, self.transport, self.client_id)
                self._robots[robot_id] = robot
                log.info("discovered robot %s", robot_id)
            return robot

    def _on_state(self, topic: str, payload: bytes) -> None:
        try:
            msg = StateMessage.from_bytes(payload)
        except Exception:
            log.exception("bad state message on %s", topic)
            return
        self._robot(msg.robot_id)._handle_state(msg)

    def _on_presence(self, topic: str, payload: bytes) -> None:
        try:
            msg = Presence.from_bytes(payload)
        except Exception:
            log.exception("bad presence message on %s", topic)
            return
        self._robot(msg.robot_id)._handle_presence(msg)

    def _on_reply(self, topic: str, payload: bytes) -> None:
        try:
            reply = Reply.from_bytes(payload)
        except Exception:
            log.exception("bad reply on %s", topic)
            return
        # Replies name their robot, so routing does not depend on topic shape.
        target = reply.robot_id
        with self._lock:
            robot = self._robots.get(target)
        if robot is not None:
            robot._handle_reply(reply)
            return
        for r in self.robots:
            r._handle_reply(reply)
