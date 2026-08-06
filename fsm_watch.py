"""Watch the R1's locomotion FSM while you drive it with the handheld remote.

This is the diagnostic for the long-standing 127: `sport` SetVelocity is
rejected, arm ExecuteAction returns 7406, and the working theory has been that
a built-in mode owns the joints. Until now the FSM was unreadable -- the topic
carrying it is typed `unitree_hg::SportModeState_`, which unitree_sdk2py 1.0.1
does not ship, so it looked silent. r1vision/hg_types.py supplies the type.

    .venv/bin/python fsm_watch.py

Then work the remote and watch which id each action selects:

    L2+UP   standing preparation
    R2+A    operation control (balance handed to the robot)
    sticks  walk
    L2+Y    zero torque
    L2+B    (held 5 s) damping / emergency

Every transition is timestamped, so you end with a map of button -> FSM id.
The one worth finding is the id the robot occupies while walking under the
remote: that is the state an external SetVelocity has to be compatible with,
and comparing it against the id present when a command is refused is the
shortest path to understanding 127.

Position is shown alongside, from `rt/odommodestate`. That field is drifting
IMU integration rather than leg odometry -- a 483 s recording wandered 2.5 m
while the robot was walked between two stations -- so it is here to confirm
that it stays broken across FSM states, not because it can be trusted.
"""

import math
import sys
import time
from datetime import datetime

from teach_path import RawTerminal


def main(argv):
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as Odom

    from r1vision.hg_types import SportModeState_ as Fsm
    from r1vision.unitree_dds import default_peers, init_channel, robot_iface

    init_channel(robot_iface("auto"), peers=default_peers())

    state = {}
    ChannelSubscriber("rt/sportmodestate", Fsm).Init(
        lambda m: state.__setitem__("fsm", m), 10)
    ChannelSubscriber("rt/odommodestate", Odom).Init(
        lambda m: state.__setitem__("odom", m), 10)

    print("waiting for rt/sportmodestate ...", end="", flush=True)
    t0 = time.monotonic()
    while "fsm" not in state and time.monotonic() - t0 < 10:
        time.sleep(0.1)
    if "fsm" not in state:
        print(" nothing. Robot powered on?")
        return 1
    print(" ok\n")
    print("  Work the remote. Transitions are logged. Press q to finish.\n")
    print(f"  {'time':>8}  {'fsm_id':>7} {'mode':>5} {'task':>5} "
          f"{'t_task':>7}   position")
    print("  " + "-" * 62)

    history = []
    last_key = None
    t0 = time.monotonic()
    start_pos = None

    with RawTerminal() as term:
        while True:
            time.sleep(0.05)
            fsm = state.get("fsm")
            odom = state.get("odom")
            if fsm is None:
                continue
            now = time.monotonic() - t0

            pos = (0.0, 0.0)
            if odom is not None:
                pos = (odom.position[0], odom.position[1])
                if start_pos is None:
                    start_pos = pos
            moved = (math.hypot(pos[0] - start_pos[0], pos[1] - start_pos[1])
                     if start_pos else 0.0)

            key = (fsm.fsm_id, fsm.fsm_mode, fsm.task_id)
            if key != last_key:
                last_key = key
                stamp = datetime.now().strftime("%H:%M:%S")
                line = (f"  {stamp}  {fsm.fsm_id:>7} {fsm.fsm_mode:>5} "
                        f"{fsm.task_id:>5} {fsm.task_time:>7.2f}   "
                        f"({pos[0]:+.2f},{pos[1]:+.2f})")
                history.append((now, key, moved))
                sys.stdout.write("\r" + line + " " * 12 + "\n")
                sys.stdout.flush()

            sys.stdout.write(
                f"\r   live  fsm={fsm.fsm_id:<5} mode={fsm.fsm_mode:<3} "
                f"task={fsm.task_id:<3} t={fsm.task_time:6.2f}  "
                f"pos=({pos[0]:+.2f},{pos[1]:+.2f}) moved={moved:5.2f}m   ")
            sys.stdout.flush()

            if term.key() in ("q", "\x03"):
                break

    print("\n\n" + "=" * 66)
    ids = sorted({h[1][0] for h in history})
    print(f"  FSM ids seen: {ids}")
    print(f"  transitions:  {len(history)}")
    if start_pos:
        fin = state["odom"].position
        print(f"  position moved {math.hypot(fin[0]-start_pos[0], fin[1]-start_pos[1]):.2f} m"
              f" over {time.monotonic()-t0:.0f} s")
    print("=" * 66)
    if len(ids) <= 1:
        print("\n  One FSM id the whole time, with mode/task/task_time pinned\n"
              "  at zero. If the remote was used, this topic is publishing a\n"
              "  constant and is not the live locomotion state on this\n"
              "  firmware. The remote's commands travel on\n"
              "  rt/wirelesscontroller -- sniff that instead (wireless_probe).\n"
              "\n  Do NOT try L2+Y to force a transition: it is zero-torque,\n"
              "  and a standing robot collapses. L2+B held 5 s is the safe\n"
              "  stop -- it damps down rather than dropping.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
