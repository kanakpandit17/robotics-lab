#!/usr/bin/env python3
"""End-to-end test of the whole stack over the loopback broker.

    python -m tests.test_stack

Runs a real scheduler against a real edge agent in one process. The transport is
swapped for the in-process broker; everything else -- envelopes, executor, state
machine, retained messages, wills -- is the production code path.

The cases worth having are the awkward ones: Stop landing mid-motion, a second
command arriving while the first runs, a duplicate delivery, and the workstation
dying without warning.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fleetcore import topics
from fleetcore.envelope import Presence, RobotState
from fleetcore.errors import ErrorCode, RobotError
from fleetcore.transport import LoopbackBroker, LoopbackTransport

from agent.adapters.simulator import SimulatorAdapter
from agent.edge_agent import EdgeAgent
from scheduler.fleet import Fleet

PASS, FAIL = "  PASS", "  FAIL"
results: list[tuple[bool, str]] = []


def check(cond: bool, label: str) -> None:
    results.append((bool(cond), label))
    print(f"{PASS if cond else FAIL}  {label}")


def build(robot_id="sim-01", move_time=0.6):
    broker = LoopbackBroker()
    will = (topics.presence(robot_id), Presence(robot_id, False).to_bytes())
    agent_tp = LoopbackTransport(broker, f"agent-{robot_id}", will=will)
    adapter = SimulatorAdapter(robot_id, {"move_time": move_time})
    agent = EdgeAgent(robot_id, adapter, agent_tp, {"state_period_s": 0.2})
    agent.start()

    fleet = Fleet(LoopbackTransport(broker, "sched-test"), client_id="sched-test")
    fleet.start()
    return broker, agent, agent_tp, fleet


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)-7s %(name)-20s %(message)s")

    print("\n1. discovery and capability advertisement")
    broker, agent, agent_tp, fleet = build()
    robot = fleet.wait_for("sim-01", timeout=5)
    check(robot.online, "robot announced itself online")
    check(robot.model == "Simulator", f"model discovered: {robot.model}")
    check("PickObject" in robot.capabilities, "capabilities advertised")
    check("Stop" in robot.capabilities, "control commands advertised")
    check(robot.wait_idle(5), "reaches IDLE once the SDK connects")

    print("\n2. a command runs to completion")
    reply = robot.execute("PickObject", object="battery")
    check(reply.result.get("picked") == "battery", "result returned to caller")
    check(robot.wait_idle(3), "returns to IDLE afterwards")

    print("\n3. unsupported command is refused without moving")
    try:
        robot.execute("Fly")
        check(False, "should have raised")
    except RobotError as exc:
        check(exc.code is ErrorCode.UNSUPPORTED, f"UNSUPPORTED: {exc}")
        check(not exc.retryable, "marked non-retryable")

    print("\n4. bad parameters are refused")
    try:
        robot.execute("PickObject")           # missing 'object'
        check(False, "should have raised")
    except RobotError as exc:
        check(exc.code is ErrorCode.INVALID_PARAMS, f"INVALID_PARAMS: {exc}")

    print("\n5. one task at a time")
    pending = robot.execute_async("PickObject", object="widget")
    pending.accepted.wait(2)
    try:
        robot.execute("MoveHome", timeout_s=5)
        check(False, "second command should have been rejected")
    except RobotError as exc:
        check(exc.code is ErrorCode.BUSY, f"BUSY while running: {exc}")
        check(exc.retryable, "BUSY is retryable")
    pending.wait()
    check(True, "first command still completed")

    print("\n6. Stop preempts a running motion")
    robot.wait_idle(3)
    pending = robot.execute_async("PickObject", object="slow", timeout_s=30)
    pending.accepted.wait(2)
    time.sleep(0.15)
    t0 = time.monotonic()
    stop_reply = robot.stop()
    elapsed = time.monotonic() - t0
    check(elapsed < 1.0, f"Stop answered while BUSY in {elapsed*1000:.0f}ms")
    check(stop_reply.result.get("was_busy") is True, "Stop reports it interrupted")
    try:
        pending.wait(5)
        check(False, "interrupted task should have failed")
    except RobotError as exc:
        check(exc.code is ErrorCode.CANCELLED, f"task cancelled: {exc}")

    print("\n7. ESTOP latches until explicitly reset")
    time.sleep(0.3)
    check(robot.state is RobotState.ESTOP, f"latched in ESTOP (is {robot.state.value})")
    try:
        robot.execute("MoveHome")
        check(False, "motion should be refused while latched")
    except RobotError as exc:
        check(exc.code is ErrorCode.ESTOP, f"refused: {exc}")
    robot.reset_error()
    time.sleep(0.3)
    check(robot.state is RobotState.IDLE, f"cleared to IDLE (is {robot.state.value})")
    robot.execute("MoveHome")
    check(True, "motion works again after reset")

    print("\n8. duplicate delivery does not run the command twice")
    robot.wait_idle(3)
    from fleetcore.envelope import Command
    cmd = Command(command="PickObject", params={"object": "dup"},
                  reply_to=topics.reply("sched-test", "dup-test"),
                  command_id="dup-test", timeout_s=10)
    calls = {"n": 0}
    original = agent.adapter._pick

    def counting(params, cancel):
        calls["n"] += 1
        return original(params, cancel)

    agent.adapter._pick = counting
    raw = cmd.to_bytes()
    agent._on_command(topics.cmd("sim-01"), raw)
    agent._on_command(topics.cmd("sim-01"), raw)     # redelivery
    time.sleep(1.2)
    check(calls["n"] == 1, f"adapter invoked exactly once (was {calls['n']})")
    agent.adapter._pick = original

    print("\n9. the workstation dies without warning")
    robot.wait_idle(3)
    pending = robot.execute_async("PickObject", object="doomed", timeout_s=30)
    pending.accepted.wait(2)
    agent_tp.kill()                      # power cut: no clean disconnect
    time.sleep(0.3)
    check(not robot.online, "broker's will marked the robot offline")
    check(robot.state is RobotState.OFFLINE, "scheduler sees OFFLINE")
    try:
        pending.wait(3)
        check(False, "in-flight command should have failed")
    except RobotError as exc:
        check(exc.code is ErrorCode.TRANSPORT,
              f"in-flight command failed fast: {exc}")
    try:
        robot.execute("MoveHome")
        check(False, "should refuse to send to an offline robot")
    except RobotError as exc:
        check(exc.code is ErrorCode.TRANSPORT, f"refused while offline: {exc}")

    print("\n10. a second, different robot needs no scheduler change")
    broker2, agent2, tp2, fleet2 = build("arm-02", move_time=0.3)
    will3 = (topics.presence("r1-99"), Presence("r1-99", False).to_bytes())
    tp3 = LoopbackTransport(broker2, "agent-r1-99", will=will3)

    class GestureOnly(SimulatorAdapter):
        model = "Unitree R1"

        def capabilities(self):
            return {"Gesture": self._gesture_impl, "MoveHome": self._move_home}

        def _gesture_impl(self, params, cancel):
            return {"gesture": params.get("name", "Wave")}

    agent3 = EdgeAgent("r1-99", GestureOnly("r1-99", {"move_time": 0.2}),
                       tp3, {"state_period_s": 0.2})
    agent3.start()

    fleet2.wait_for("arm-02", timeout=5)
    r1 = fleet2.wait_for("r1-99", timeout=5)
    check(len(fleet2.robots) == 2, f"fleet sees {len(fleet2.robots)} robots")
    check(r1.model == "Unitree R1", "heterogeneous model discovered")

    # Capability routing: the scheduler never names a robot.
    picker = fleet2.any_capable("PickObject")
    check(picker.robot_id == "arm-02", f"PickObject routed to {picker.robot_id}")
    waver = fleet2.any_capable("Gesture")
    check(waver.robot_id == "r1-99", f"Gesture routed to {waver.robot_id}")
    check(not r1.supports("PickObject"), "R1 correctly reports no gripper")
    res = waver.execute("Gesture", name="Wave")
    check(res.result.get("gesture") == "Wave", "gesture executed on the R1")

    print("\n11. fleet-wide stop")
    out = fleet2.stop_all()
    check(all(v == "stopped" for v in out.values()), f"stop_all: {out}")

    for a in (agent, agent2, agent3):
        try:
            a.stop()
        except Exception:
            pass

    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    print(f"\n{'='*58}\n  {passed}/{total} checks passed\n{'='*58}")
    if passed != total:
        for ok, label in results:
            if not ok:
                print(f"  FAILED: {label}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
