#!/usr/bin/env python3
"""Drive the R1 from the keyboard. Run this before trying it from the headset.

    python r1_walk.py --dry-run     # no robot; prove the plumbing works
    python r1_walk.py               # for real

Why a keyboard version exists at all
------------------------------------
The point of this file is that you are standing next to the robot looking at it,
with your hand on the terminal, when it takes its first commanded step. In the
headset you cannot see the room, cannot see whether a leg is about to catch, and
cannot find the keyboard. Establish that the robot moves the way you expect and
that stopping works from here first.

Hold-to-walk
------------
Holding a key down makes the terminal repeat it, and each repeat refreshes the
command. Let go and the repeats stop, so the watchdog in `locomotion.py` zeroes
the velocity within a fraction of a second. That is the same deadman the
headset's grip button provides, and it means there is no key you can press that
leaves the robot walking.

Keys
----
    w / s      forward / back            a / d      strafe left / right
    q / e      turn left / right         space      stop
    1          start (stand and balance, FSM 500)
    2          stand up
    0          DAMP -- joints release, the robot will sag. Support it first.
    x          quit (stops the robot on the way out)
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cyclonedds",))

from r1vision.locomotion import BACKENDS, LocoDriver  # noqa: E402

# How long one keypress stays live. Terminal auto-repeat is ~30 ms once it gets
# going, but the first repeat lags by ~500 ms, so anything shorter than that
# makes a held key stutter.
KEY_HOLD = 0.6

MOVES = {
    "w": (1.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0),
    "a": (0.0, 1.0, 0.0),
    "d": (0.0, -1.0, 0.0),
    "q": (0.0, 0.0, 1.0),
    "e": (0.0, 0.0, -1.0),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="auto",
                    help="interface facing the robot (ethernet)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="scale on the caps in locomotion.py (0.5 = half)")
    ap.add_argument("--backend", default="wireless", choices=BACKENDS,
                    help="sport = LocoClient.SetVelocity (returns 127 on this "
                         "R1); wireless = publish to rt/wirelesscontroller, the "
                         "channel the handheld remote drives")
    ap.add_argument("--loco-service", default="auto",
                    help="sport backend only: loco, sport, or auto to ask the "
                         "robot which one answers")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full control path but never talk to a robot")
    args = ap.parse_args()

    driver = LocoDriver(iface=args.iface, scale=args.speed, dry_run=args.dry_run,
                        service=args.loco_service, backend=args.backend)

    print(__doc__.split("Keys\n----\n")[1])
    print(f"  limits: {driver.max_vx:.2f} m/s fwd, {driver.max_vy:.2f} m/s side, "
          f"{driver.max_w:.2f} rad/s yaw")
    if args.dry_run:
        print("  DRY RUN -- nothing is sent to a robot\n")
    else:
        print("\n  Clear the space around the robot. Ctrl-C stops it.\n")

    driver.start()

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    held: tuple[float, float, float] = (0.0, 0.0, 0.0)
    held_at = 0.0
    status_at = 0.0

    # A run that commanded nothing and a run the robot ignored look identical
    # on the status line, which overwrites itself and keeps no history. Count
    # what actually happened so the exit summary can tell them apart.
    keys_seen = 0
    moves_seen = 0
    peak = (0.0, 0.0, 0.0)

    try:
        tty.setcbreak(fd)
        while True:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1).lower()
                keys_seen += 1

                if key == "x" or key == "\x03":
                    break
                if key == " ":
                    held, held_at = (0.0, 0.0, 0.0), 0.0
                    driver.stop()
                    print("\r  stop" + " " * 50)
                elif key in MOVES:
                    # Echo the transition, not every auto-repeat. Without this
                    # the only sign a key registered is a number on a line that
                    # overwrites itself, which is easy to miss while you are
                    # watching the robot rather than the screen.
                    if held == (0.0, 0.0, 0.0):
                        moves_seen += 1
                        print(f"\r  key {key!r} -> commanding" + " " * 40)
                    held, held_at = MOVES[key], time.monotonic()
                elif key == "1":
                    print(f"\r  start: {driver.fsm('start')[1]}" + " " * 30)
                elif key == "2":
                    print(f"\r  stand_up: {driver.fsm('stand_up')[1]}" + " " * 30)
                elif key == "0":
                    print("\r  DAMP -- the robot will sag. Press 0 again to confirm."
                          + " " * 10)
                    if select.select([sys.stdin], [], [], 3.0)[0] and \
                            sys.stdin.read(1) == "0":
                        print(f"\r  damp: {driver.fsm('damp')[1]}" + " " * 30)
                    else:
                        print("\r  damp cancelled" + " " * 30)

            # A key that has not repeated recently is a key that was let go.
            if held_at and time.monotonic() - held_at > KEY_HOLD:
                held, held_at = (0.0, 0.0, 0.0), 0.0

            vx, vy, w = driver.command(*held)
            peak = (max(peak[0], abs(vx)), max(peak[1], abs(vy)),
                    max(peak[2], abs(w)))

            now = time.monotonic()
            if now - status_at > 0.1:
                status_at = now
                flag = "" if driver.error is None else f"  ERR {driver.error}"
                print(f"\r  vx {vx:+.2f}  vy {vy:+.2f}  yaw {w:+.2f}   "
                      f"sent {driver.sent}{flag}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print("\n  stopping robot ...")
        driver.close()
        print("  done.\n")
        print(f"  keys received     {keys_seen}")
        print(f"  move commands     {moves_seen}")
        print(f"  peak commanded    vx {peak[0]:+.2f}  vy {peak[1]:+.2f}  "
              f"yaw {peak[2]:+.2f}")
        print(f"  frames published  {driver.sent}")
        if moves_seen == 0:
            print("\n  No movement key was ever registered, so the robot was\n"
                  "  correctly told to stand still the whole time. This run\n"
                  "  says nothing about whether it obeys -- press and hold w\n"
                  "  for about a second to actually command a step.")
        elif peak[0] == 0.0 and peak[1] == 0.0 and peak[2] == 0.0:
            print("\n  Keys registered but every command clamped to zero.\n"
                  "  Check --speed.")
        else:
            print("\n  Commands were issued and published. If the robot did\n"
                  "  not move, the wireless channel is reaching the bus but\n"
                  "  not this robot's controller -- say so and we look at\n"
                  "  whether the real remote must be switched off first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
