#!/usr/bin/env python3
"""Test bilateral arm teleop from the keyboard, before ever using a headset.

    python arm_teleop_cli.py --script              # no robot, no terminal
    python arm_teleop_cli.py --script --dry-run     # (--script always
                                                      #  implies dry-run; see
                                                      #  below)
    python arm_teleop_cli.py --dry-run              # interactive, no robot
    python arm_teleop_cli.py                        # interactive, for real

Why this exists
---------------
Same reasoning as `r1_walk.py`: you should see the arm move, with your hand on
the keyboard, before a VR headset takes away your view of the room. This
drives `r1vision.arm_teleop.ArmTeleopDriver` the same way the headset would --
a stream of synthetic controller positions and a grip flag -- so it is the
same code path as `./vr.sh --teleop`, just fed by keys instead of WebXR.

Two ways to run it
------------------
`--script` runs a short scripted engage / move / release sequence with no
keyboard needed at all, and always uses a simulated arm (`--dry-run` is
implied) -- a scripted motion should never reach real hardware unattended.
This is the one to run first, and the one that works on any OS: it needs no
raw-terminal support.

Without `--script`, this is the interactive keyboard version, matching
`r1_walk.py`'s controls and countdown discipline. It needs a real terminal
(macOS/Linux `termios`), like the rest of this repo.

Keys (interactive mode)
------------------------
    w/s        target forward / back (+x / -x)
    a/d        target left / right (+y / -y)
    r/f        target up / down (+z / -z)
    g          toggle grip (engage / release the clutch)
    h          switch which hand these keys drive
    x          quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("numpy",))

import numpy as np  # noqa: E402

from r1vision.arm_teleop import HANDS, ArmTeleopDriver  # noqa: E402

# Metres per key tap while held, and identity orientation (no wrist-roll
# input from this harness -- it only exercises the position path).
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
KEY_HOLD = 0.6   # matches r1_walk.py's terminal auto-repeat allowance

# Metres per second, applied while a key counts as held, scaled by real
# elapsed time -- not a fixed step per loop tick. An earlier version added a
# fixed 1 cm on every ~50 Hz loop tick for up to KEY_HOLD after any detected
# keystroke, which works out to roughly 1 m/s: in a workspace that is only a
# few centimetres across (see TELEOP.md), a single tap that lingered inside
# that window could push the target a metre away. Confirmed on hardware: it
# drove the shoulder to within ~45 deg of its mechanical limit. This is
# deliberately slow -- a full second of holding covers only 3 cm.
NUDGE_RATE_M_S = 0.03
# Hard ceiling on the accumulated target itself, independent of anything
# downstream (AXIS_MAP, clamp_reachable): defense in depth so a stuck key or
# a long session cannot silently build up an absurd target again.
MAX_TARGET_NORM_M = 0.30


def run_script(driver: ArmTeleopDriver, hand: str) -> int:
    """No keyboard, no robot: proves command -> clutch -> IK -> send end to
    end with a scripted sequence, and checks the results make sense."""
    print(f"  scripted smoke test on the '{hand}' hand (simulated arm)\n")
    driver.start()

    def tick(pos, grip, seconds=0.3):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            driver.command(hand, pos, IDENTITY_QUAT, grip)
            time.sleep(driver.period)

    origin = np.array([0.0, 0.0, 0.0])
    tick(origin, grip=False)   # settle: driver should be shadowing live angles
    before = driver.status()
    print(f"  before engage : {before['solved_deg'][hand]} deg  "
         f"weight={before['weight']}")

    # 5 cm, which the R1's arm can actually strike from its rest pose. The
    # workspace is smaller than it looks: the segments are 88 mm and 117 mm,
    # so a 14 cm reach from rest already lands outside it and the residual
    # would swamp what this test is trying to measure.
    move = (0.05, 0.02, 0.0)
    tick(origin, grip=True)             # engage: should not move yet (delta=0)
    tick(origin + move, grip=True, seconds=0.6)   # move
    moved = driver.status()
    print(f"  after move    : {moved['solved_deg'][hand]} deg  "
         f"weight={moved['weight']}  engaged={moved['engaged'][hand]}  "
         f"ik_residual={moved['ik_err_mm'][hand]} mm")

    tick(origin + move, grip=False)   # release: should freeze
    frozen = driver.status()
    print(f"  after release : {frozen['solved_deg'][hand]} deg  "
         f"weight={frozen['weight']}  engaged={frozen['engaged'][hand]}")

    driver.close()

    ok = True
    if before["solved_deg"][hand] is None:
        print("\n  FAIL: never solved an initial pose")
        ok = False
    if moved["solved_deg"][hand] == before["solved_deg"][hand]:
        print("\n  FAIL: commanding a 10cm move produced no change")
        ok = False
    if frozen["solved_deg"][hand] != moved["solved_deg"][hand]:
        print("\n  FAIL: releasing the clutch moved the arm instead of "
             "freezing it")
        ok = False
    if moved["weight"] <= 0.0:
        print("\n  FAIL: weight never ramped up while engaged")
        ok = False

    print("\n  " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


class _Keyboard:
    """Non-blocking single-key reads on Windows and on macOS/Linux.

    The original version was termios-only, which meant the one safety step
    that exists precisely so a human watches the robot while driving it --
    keyboard control before the headset goes on -- could not run on Windows
    at all. `msvcrt` is the stdlib equivalent there.
    """

    def __enter__(self):
        self.posix = False
        try:
            import termios
            import tty

            self._termios, self._tty = termios, tty
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.posix = True
        except ImportError:
            import msvcrt

            self._msvcrt = msvcrt
        return self

    def key(self):
        """The key pressed, or None. Never blocks."""
        if self.posix:
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
            return None
        if self._msvcrt.kbhit():
            ch = self._msvcrt.getch()
            try:
                return ch.decode("utf-8", "ignore").lower()
            except Exception:
                return None
        return None

    def __exit__(self, *exc):
        if self.posix:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN,
                                    self._saved)
        return False


def run_interactive(driver: ArmTeleopDriver, hand: str) -> int:
    print(__doc__.split("Keys (interactive mode)\n------------------------\n")[1])
    if driver.dry_run:
        print("  DRY RUN -- nothing is sent to a robot\n")
    else:
        print("\n  Clear the space around both arms. Ctrl-C stops it.\n")

    driver.start()

    pos = {h: np.array([0.0, 0.0, 0.0]) for h in HANDS}
    grip = {h: False for h in HANDS}
    held_keys: dict[str, float] = {}
    active_hand = hand
    last_tick = time.monotonic()
    status_at = 0.0

    # Unit directions, pre-AXIS_MAP, not scaled or robot-frame: the actual
    # step size is applied below as NUDGE_RATE_M_S * dt. arm_teleop.py's
    # shared control loop always applies AXIS_MAP (written for real WebXR
    # controller coordinates) to whatever position it receives, so this
    # harness emits vectors in that same pre-transform space for its
    # w/a/s/d/r/f labels to come out meaning what they say. Found by testing:
    # an earlier version emitted post-transform (robot-frame) vectors
    # directly, so AXIS_MAP was applied a second time on top of them and "w"
    # (meant as forward) drove the arm backward instead.
    moves = {
        "w": (0, 0, -1), "s": (0, 0, 1),
        "a": (1, 0, 0), "d": (-1, 0, 0),
        "r": (0, 1, 0), "f": (0, -1, 0),
    }

    try:
        with _Keyboard() as kb:
            while True:
                key = kb.key()
                if key is not None:
                    if key in ("x", "\x03"):
                        break
                    if key == "g":
                        grip[active_hand] = not grip[active_hand]
                        print(f"\r  {active_hand}: grip "
                             f"{'ENGAGED' if grip[active_hand] else 'released'}"
                             + " " * 30)
                    elif key == "h":
                        active_hand = ("left" if active_hand == "right"
                                       else "right")
                        print(f"\r  now driving: {active_hand}" + " " * 40)
                    elif key in moves:
                        held_keys[key] = time.monotonic()

                now = time.monotonic()
                dt = now - last_tick
                last_tick = now
                for k, at in list(held_keys.items()):
                    if now - at > KEY_HOLD:
                        del held_keys[k]
                    else:
                        pos[active_hand] = (pos[active_hand]
                                           + np.array(moves[k]) * NUDGE_RATE_M_S * dt)
                # Defense in depth: even if the rate/timing above is ever
                # wrong again, the accumulated target itself cannot silently
                # grow past a sane bound.
                n = np.linalg.norm(pos[active_hand])
                if n > MAX_TARGET_NORM_M:
                    pos[active_hand] = pos[active_hand] * (MAX_TARGET_NORM_M / n)

                for h in HANDS:
                    driver.command(h, pos[h], IDENTITY_QUAT, grip[h])

                if now - status_at > 0.1:
                    status_at = now
                    st = driver.status()
                    flag = "" if driver.error is None else f"  ERR {driver.error}"
                    print(f"\r  [{active_hand}] target "
                         f"{pos[active_hand].round(3)}  "
                         f"angles {st['solved_deg'][active_hand]} deg  "
                         f"weight {st['weight']:.2f}  sent {st['sent']}{flag}   ",
                         end="", flush=True)
                # The key reader never blocks, so without this the loop spins
                # a core flat out for no benefit.
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  releasing and stopping ...")
        driver.close()
        print("  done.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="en6")
    ap.add_argument("--hand", choices=HANDS, default="right",
                    help="which hand the keyboard drives first (interactive mode)")
    ap.add_argument("--both", action="store_true",
                    help="drive both arms (default: the one --hand names, "
                         "for a simpler first test)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full control path but never talk to a robot")
    ap.add_argument("--script", action="store_true",
                    help="no keyboard, no robot: scripted smoke test (implies --dry-run)")
    args = ap.parse_args()

    hands = HANDS if args.both else (args.hand,)
    dry_run = args.dry_run or args.script
    driver = ArmTeleopDriver(iface=args.iface, hands=hands, scale=args.scale,
                             dry_run=dry_run)

    if args.script:
        return run_script(driver, args.hand)
    return run_interactive(driver, args.hand)


if __name__ == "__main__":
    sys.exit(main())
