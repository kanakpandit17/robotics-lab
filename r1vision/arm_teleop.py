"""Continuous bilateral arm teleoperation, streamed from VR controller poses.

Shape copied deliberately from `locomotion.py`'s `LocoDriver`: one background
thread owns the DDS connection and publishes at a fixed rate from whatever
HTTP handler threads last recorded, with a watchdog that freezes a hand if
the headset goes quiet, independent of the clutch (grip button) release.

The clutch model
-----------------
Grip newly pressed on a hand -> capture that controller's current position
and twist, and that arm's *current* end-effector position (via forward
kinematics on its live measured angles). While held, the target is
`ee_at_engage + scale * (controller_now - controller_at_engage)` -- a
relative offset, not an absolute mapping from headset space to robot space,
so the controller's physical position in the room never has to line up with
the arm's. Grip released -> freeze: keep commanding the last solved angles,
do not recompute from a stale pose, do not drop weight -- same "let go = stop,
not fall" rule `locomotion.py` uses for walking. Re-gripping starts a new
reference, like picking a mouse up and putting it down somewhere else.

Why one weight for two arms
----------------------------
`r1_arm.py`'s `WEIGHT_SLOT` blends *the whole* `rt/arm_sdk` frame, not one arm
at a time -- there is no such thing as "right arm engaged, left arm still
under the robot's own control" at the DDS level. So every cycle publishes
**both** arms' target angles regardless of which hand is actually clutched;
the un-clutched arm holds still only because its target is refreshed to
"wherever it currently is" every cycle it is not being driven. A frame that
forgets to do that for the idle arm is a frame that lets it drift, not one
that leaves it alone -- see TELEOP.md.

What is verified vs. assumed
-----------------------------
The control loop, watchdog, clutch math and step/limit clamps are exercised
end-to-end by `arm_teleop_cli.py --script`. Link geometry, joint axes and
joint limits are no longer guesses -- they are parsed from the robot's own
URDF by `arm_kinematics`. What remains unverified is `AXIS_MAP` (which way
the operator's hand motion maps onto the robot's frame) and, more
importantly, whether `rt/arm_sdk` moves this robot's arms at all: nothing
here has yet been run against the hardware.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from . import arm_kinematics

# Duplicated from r1_arm.py rather than imported, so this module (and
# arm_teleop_cli.py --dry-run) stays importable without cyclonedds installed
# -- the same reason quest_stream.py lazy-imports arm_actions instead of
# importing it at the top. Keep these in sync with r1_arm.py by hand; they
# are measured constants, not expected to change.
CONTROL_HZ = 100
RIGHT_ARM = [22, 23, 24, 25, 26]   # shoulder pitch/roll/yaw, elbow, wrist roll
LEFT_ARM = [15, 16, 17, 18, 19]    # same order
JOINTS = {"right": RIGHT_ARM, "left": LEFT_ARM}
HANDS = ("left", "right")

# No mirroring table: kinematics for both arms are read straight from the
# URDF, which carries each side's real offsets, axes and (asymmetric) limits.
# An earlier version guessed that the left arm was the right arm with two
# signs flipped; the URDF makes that guess unnecessary.

# WebXR controller space: +X right, +Y up, -Z forward (away from the user).
# arm_kinematics space: +Z up (hanging = -Z), and X/Y are otherwise only
# defined by this mapping. First guess: pushing the controller away from you
# moves the target forward (+X), right moves it -Y, up moves it +Z. Verify
# this is actually intuitive on hardware before trusting it; edit this one
# matrix if an axis feels backwards or swapped -- one tested place to audit,
# same reasoning `locomotion.shape()` gives for centralising its own tuning.
# Row 2 was [1, 0, 0], which mirrored left/right: WebXR +X is the operator's
# right, and it was being mapped to robot +Y. The URDF settles which way that
# points -- the right arm's wrist sits at y = -0.139 and the left at +0.139,
# so +Y is the robot's LEFT (standard x-forward / y-left / z-up). Moving your
# hand right therefore drove the arm left. The comment above this matrix used
# to assert "right moves it +Y", which is simply not this robot's convention;
# the assertion, not the robot, was wrong. Found by checking the mapping
# against the URDF (see axis_check.py) rather than from inside the headset,
# where a mirrored axis is indistinguishable from teleop just working badly.
AXIS_MAP = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

# Joint limits come from the URDF via arm_kinematics -- they are markedly
# asymmetric on this robot (right shoulder roll runs -142..+13 deg), which no
# hand-written placeholder would have got right.

# Expressed per SECOND, not per cycle, and multiplied by the measured cycle
# time. The loop does not actually hold 100 Hz on Windows -- timer granularity
# puts it nearer 45-65 Hz -- and a per-cycle limit would quietly make the arm
# move at half the intended speed there, or faster on a machine that sustains
# the full rate. A safety limit that changes with the host's clock behaviour
# is not a safety limit.
MAX_SLEW_DEG_PER_S = 40.0
# s; a hand this stale while gripped freezes. Raised from 0.3 against a
# measurement, not a guess: the headset page polls at 30 Hz, but over a phone
# hotspot also carrying the MJPEG video stream the requests actually land at
# 5-13 Hz, and stall to 1-3 Hz under load. At 0.3 s every such stall tripped
# the watchdog and froze the arm mid-motion -- which from inside the headset
# is indistinguishable from the teleop simply not following you, and is a
# strong suspect for the original "it doesn't follow what I'm doing" report.
# 0.6 s clears the observed stalls while still stopping well inside a second.
# This is affordable now in a way it was not before, because losing the
# headset no longer depends on this timer alone: releaseTeleop() fires on XR
# visibility loss, and MAX_CONTINUOUS_ENGAGE_S bounds a stuck grip from the
# robot side. If the link is ever fixed (wired, or video off), put it back.
WATCHDOG = 0.6
# Robot-side backstop, independent of the browser: bounds how long one
# continuous grip can hold authority regardless of what the client does or
# fails to do. Confirmed on hardware that the browser side has real gaps --
# taking the Quest off does not necessarily end the session or fire any page
# event, so a controller left resting with something against its grip button
# kept driving the arm for close to a minute with no software path to stop
# it, until the robot's ethernet was physically unplugged. The client-side
# fixes (xrSession visibilitychange -> release, in xr_page.py) close that
# specific gap, but this exists so a *future* client bug of the same shape
# cannot reproduce it: the robot itself will not honour a grip forever no
# matter what it is told. Re-engaging (a fresh grip transition) resets it.
MAX_CONTINUOUS_ENGAGE_S = 20.0
# How long panic_stop() forces every hand's grip to False, no matter what
# input keeps arriving. NOT one-shot -- an earlier version cleared weight for
# a single cycle only, and a test with a synthetic held grip caught it
# re-engaging on the very next cycle from one leftover in-flight request. That
# is not a synthetic-only concern: the real client keeps polling for a moment
# after estop() fires and before the XR session actually finishes ending, so
# a still-gripped hand can genuinely send one more grip=True after panic_stop()
# is called. A bounded lockout window, not a single flag, is what actually
# guarantees the stop -- same reasoning as MAX_CONTINUOUS_ENGAGE_S's `capped`.
PANIC_LOCKOUT_S = 1.5
# Both cut down from 3.0/2.0 after a live test: with the old values, letting
# go (or the headset losing track, or exiting VR) left the arm visibly
# sagging under gravity for up to ~5 s as stiffness eased out -- easy to
# mistake for the arm still doing something on its own. Shorter values still
# avoid a hard instant cutoff (the thing RELEASE_GRACE/WEIGHT_RAMP_S exist to
# prevent) while settling in ~1.3 s instead.
RELEASE_GRACE = 0.8              # s after the last engaged hand releases
                                 # before weight ramps back to 0
WEIGHT_RAMP_S = 0.5              # matches r1_arm.py's engage ramp duration


def _twist_angle(quat_wxyz, axis=(0.0, 0.0, 1.0)) -> float:
    """Rotation of `quat_wxyz` about `axis`, via swing-twist decomposition.

    Used as the wrist-roll signal from a controller's orientation: the twist
    about a fixed axis is well defined even though the controller's full
    orientation is not being tracked as a target pose.
    """
    w, x, y, z = quat_wxyz
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    s = float(np.dot((x, y, z), n))
    return 2.0 * math.atan2(s, w)


def _clamp_limits(angles, kin) -> tuple[float, ...]:
    """Clamp to the URDF's real limits for that arm."""
    return tuple(float(v) for v in np.clip(np.asarray(angles, dtype=float),
                                           kin.lower, kin.upper))


def _clamp_step(prev: tuple[float, ...], target: tuple[float, ...],
                max_step: float) -> tuple[float, ...]:
    out = []
    for p, t in zip(prev, target):
        d = t - p
        if d > max_step:
            d = max_step
        elif d < -max_step:
            d = -max_step
        out.append(p + d)
    return tuple(out)


class _DryArm:
    """Stands in for `r1_lowcmd_arm.LowCmdArm` so the path runs with no robot.

    `angles()` mirrors back whatever was last `send()`t, converged instantly
    -- enough to exercise clutch math, step clamping and the send loop
    without pretending to simulate real actuator dynamics.
    """

    def __init__(self) -> None:
        self._angles = [0.0] * 35

    def connect(self, timeout: float = 8.0) -> bool:
        return True

    def angles(self) -> list[float]:
        return list(self._angles)

    def send(self, targets: dict[int, float], kp_scale: float = 1.0) -> None:
        # A real joint at kp_scale 0 is free and would not track; mirroring
        # that keeps the dry run honest about what release actually does.
        if kp_scale <= 0.0:
            return
        for idx, q in targets.items():
            self._angles[idx] = q


class ArmTeleopDriver:
    """Owns the arm_sdk connection and one send loop for both arms."""

    def __init__(self, iface: str = "en6", hands: tuple[str, ...] = HANDS,
                scale: float = 1.0,
                max_slew_deg_s: float = MAX_SLEW_DEG_PER_S,
                watchdog: float = WATCHDOG, release_grace: float = RELEASE_GRACE,
                weight_ramp_s: float = WEIGHT_RAMP_S,
                max_continuous_engage_s: float = MAX_CONTINUOUS_ENGAGE_S,
                dry_run: bool = False):
        for h in hands:
            if h not in HANDS:
                raise ValueError(f"unknown hand {h!r}, must be one of {HANDS}")
        self.iface = iface
        self.hands = tuple(hands)
        self.scale = scale
        self.max_slew = math.radians(max_slew_deg_s)   # rad/s
        self.watchdog = watchdog
        self.release_grace = release_grace
        self.weight_ramp_s = weight_ramp_s
        self.max_continuous_engage = max_continuous_engage_s
        self.dry_run = dry_run
        self.period = 1.0 / CONTROL_HZ

        self._client = None
        self._lock = threading.Lock()
        self._input: dict[str, dict] = {}   # hand -> {pos, quat, grip, at}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._closing = False
        self._panic_until = 0.0   # monotonic timestamp; 0.0 = not panicking

        # Display-only state for status(), written by the loop thread and
        # read (under lock) by HTTP handler threads.
        self._weight = 0.0
        self._engaged = {h: False for h in self.hands}
        self._solved_deg = {h: None for h in self.hands}
        # Residual from the IK solve: how far the arm's reachable pose is from
        # where the operator's hand actually is. Large means they have reached
        # outside the workspace, which on the HUD reads as "the arm stopped
        # following me" -- worth showing rather than hiding.
        self._ik_err = {h: 0.0 for h in self.hands}

        self.error: str | None = None
        self.sent = 0

    # ---------------------------------------------------------------- connect
    def connect(self):
        if self._client is not None:
            return self._client
        if self.dry_run:
            self._client = _DryArm()
            self._client.connect()
            return self._client
        # rt/lowcmd, not rt/arm_sdk: arm_sdk frames are accepted by this robot
        # and silently discarded, because its blend-weight slot 29 is the R1's
        # HEAD_PITCH joint rather than a spare. Measured on hardware -- a 20 deg
        # commanded sweep produced 0.00 deg of motion via arm_sdk, and moved the
        # arm immediately via lowcmd.
        from r1_lowcmd_arm import LowCmdArm   # lazy: keeps --dry-run free of cyclonedds

        arm = LowCmdArm(self.iface)
        if not arm.connect():
            raise RuntimeError(
                f"no rt/lowstate received on {self.iface}; is this ethernet "
                "on the robot's 192.168.123.x segment?")
        self._client = arm
        return arm

    # ------------------------------------------------------------------ input
    def command(self, hand: str, pos_xyz, quat_wxyz, grip: bool) -> None:
        """Record the latest controller sample for `hand`. Cheap; called from
        HTTP handler threads. The control loop does all the actual solving."""
        if hand not in HANDS:
            raise ValueError(f"unknown hand {hand!r}")
        with self._lock:
            self._input[hand] = {
                "pos": np.asarray(pos_xyz, dtype=float),
                "quat": tuple(float(v) for v in quat_wxyz),
                "grip": bool(grip),
                "at": time.monotonic(),
            }

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Force weight to 0 before the thread exits, not just idle release."""
        self._closing = True
        time.sleep(self.weight_ramp_s + 4 * self.period)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def panic_stop(self) -> None:
        """Immediate release, not close()'s graceful ramp -- for an operator
        -facing STOP: every hand is forced ungripped for PANIC_LOCKOUT_S,
        not just this cycle -- see that constant for why a single-cycle
        version was not enough. The thread and DDS connection stay up; this
        hands the arm back NOW, it does not tear the server down. See
        quest_stream.py's /estop, which pairs this with re-selecting the
        robot's own `ai` mode -- clearing weight alone stops OUR authority
        but leaves nothing holding the arm if the robot's own controller
        was never in charge to begin with."""
        with self._lock:
            self._panic_until = time.monotonic() + PANIC_LOCKOUT_S
            self._input = {}

    # ------------------------------------------------------------- send loop
    def _loop(self) -> None:
        try:
            arm = self.connect()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"  arm teleop unavailable: {self.error}")
            return
        print(f"  arm teleop: hands {self.hands} at {CONTROL_HZ} Hz "
             f"(scale {self.scale:.2f}, iface {self.iface})")

        solved: dict[str, tuple[float, ...] | None] = {h: None for h in self.hands}
        clutch: dict[str, dict | None] = {h: None for h in self.hands}
        stale_logged: dict[str, bool] = {h: False for h in self.hands}
        engaged_since: dict[str, float | None] = {h: None for h in self.hands}
        max_engage_logged: dict[str, bool] = {h: False for h in self.hands}
        capped: dict[str, bool] = {h: False for h in self.hands}
        weight = 0.0
        last_active_at = 0.0
        prev_cycle = time.monotonic()

        while not self._stop.is_set():
            cycle = time.monotonic()
            # Measured, not assumed. A stalled cycle must not be allowed to
            # authorise a correspondingly huge joint step, so this is capped
            # at 4x the nominal period -- after a long stall the arm resumes
            # at its normal speed rather than lunging to catch up.
            dt = min(cycle - prev_cycle, 4 * self.period)
            prev_cycle = cycle
            max_step = self.max_slew * dt
            try:
                angles = arm.angles()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(self.period)
                continue

            targets: dict[int, float] = {}
            any_engaged = False
            with self._lock:
                panic_active = cycle < self._panic_until

            for hand in self.hands:
                joints = JOINTS[hand]
                kin = arm_kinematics.chain(hand)

                with self._lock:
                    inp = self._input.get(hand)

                grip = False
                raw_grip = bool(inp["grip"]) if inp is not None else False
                if inp is not None:
                    age = cycle - inp["at"]
                    fresh = age <= self.watchdog
                    grip = inp["grip"] and fresh
                    if inp["grip"] and not fresh:
                        if not stale_logged[hand]:
                            stale_logged[hand] = True
                            print(f"  teleop watchdog: {hand} hand quiet for "
                                 f"{age:.1f}s, freezing that arm")
                    else:
                        stale_logged[hand] = False

                # panic_stop(): forced false at the source, for every hand,
                # regardless of what raw_grip says, for the full lockout
                # window -- not a one-shot override further down. See
                # PANIC_LOCKOUT_S for why a single cycle was not enough.
                if panic_active:
                    grip = False
                    raw_grip = False

                # Robot-side cap on continuous authority (MAX_CONTINUOUS_
                # ENGAGE_S), independent of the client. Once tripped, the hand
                # stays refused -- not just for the one cycle that trips it --
                # until the RAW input genuinely reports grip=False at least
                # once. Checking only `clutch[hand] is None` here was not
                # enough: forcing grip False for a single cycle sets exactly
                # that, so a client still sending grip=True every cycle (a
                # stuck button, or the bug this exists for) walked straight
                # back in as a "fresh" engagement on the very next cycle --
                # caught by testing, not assumed. A genuine release-and-regrip
                # clears it immediately; a continuously-true signal cannot.
                if capped[hand]:
                    if not raw_grip:
                        capped[hand] = False
                    else:
                        grip = False

                if grip and clutch[hand] is None:
                    engaged_since[hand] = cycle
                    max_engage_logged[hand] = False
                elif not grip:
                    engaged_since[hand] = None
                elif (engaged_since[hand] is not None
                      and cycle - engaged_since[hand] > self.max_continuous_engage):
                    if not max_engage_logged[hand]:
                        max_engage_logged[hand] = True
                        print(f"  teleop: {hand} hand held past "
                             f"{self.max_continuous_engage:.0f}s continuous, "
                             "forcing release (let go and re-grip to continue)")
                    grip = False
                    capped[hand] = True

                if not grip and clutch[hand] is not None:
                    clutch[hand] = None   # just released this cycle

                if solved[hand] is None or (not grip and clutch[hand] is None
                                            and weight < 1e-3):
                    # No authority yet (weight ~0): shadow the robot's own
                    # controller so the first engagement starts from where
                    # the arm actually is, not from a stale zero.
                    solved[hand] = tuple(angles[j] for j in joints)

                if grip:
                    any_engaged = True
                    if clutch[hand] is None:
                        clutch[hand] = {
                            "ctrl0": inp["pos"].copy(),
                            "twist0": _twist_angle(inp["quat"]),
                            "ee0": kin.fk(solved[hand]),
                            "wrist0": solved[hand][4],
                        }
                    c = clutch[hand]
                    delta_robot = AXIS_MAP @ (inp["pos"] - c["ctrl0"])
                    target = kin.clamp_reachable(c["ee0"] + self.scale * delta_robot)
                    wrist_roll = c["wrist0"] + (_twist_angle(inp["quat"]) - c["twist0"])

                    # Seeded from the pose we are already holding, so
                    # successive solves stay continuous and cheap -- and the
                    # arm follows the hand rather than jumping between equally
                    # valid elbow configurations.
                    #
                    # Deliberately given a small iteration budget and no
                    # random restarts: this is a real-time loop, and the slew
                    # clamp below caps motion at MAX_SLEW_DEG_PER_S
                    # regardless of how precisely the target was solved. Only
                    # the direction has to be right; the solve converges
                    # across cycles as the arm slews. The full multi-start
                    # solver costs ~75 ms on a hard target, which would drop
                    # the loop from 100 Hz to 13 Hz -- slow enough that a
                    # released clutch keeps moving for another cycle.
                    q_sol, ik_err = kin.ik(target, q_init=solved[hand],
                                          wrist_roll=wrist_roll,
                                          iters=12, restarts=0)
                    self._ik_err[hand] = ik_err
                    solved[hand] = _clamp_step(
                        solved[hand], _clamp_limits(q_sol, kin), max_step)

                for idx, q in zip(joints, solved[hand]):
                    targets[idx] = q

                with self._lock:
                    self._engaged[hand] = grip
                    self._solved_deg[hand] = tuple(
                        round(math.degrees(a), 1) for a in solved[hand])

            if any_engaged and not self._closing:
                last_active_at = cycle

            # Every hand's grip was already forced False above for the whole
            # panic_active window, so clutch/engaged/capped are already
            # clearing through the normal per-hand release path -- this only
            # needs to pin weight itself at 0 immediately, rather than let it
            # follow the normal graceful release_grace + weight_ramp_s decay
            # (up to ~1.3s), which is fine for a deliberate release but too
            # slow for what an emergency stop is for.
            if panic_active:
                weight = 0.0
                last_active_at = 0.0

            target_weight = (
                0.0 if self._closing else
                (1.0 if (cycle - last_active_at) < self.release_grace else 0.0))
            step = self.period / max(self.weight_ramp_s, 1e-6)
            if weight < target_weight:
                weight = min(target_weight, weight + step)
            elif weight > target_weight:
                weight = max(target_weight, weight - step)

            try:
                # Fully released: send an empty frame so every joint is
                # disabled rather than commanded at zero stiffness. Under
                # rt/lowcmd an unset motor_cmd entry defaults to q=0, so
                # commanding the arm with no stiffness is not the same as not
                # commanding it -- only the empty frame truly hands the arm
                # back.
                if weight <= 1e-3:
                    arm.send({})
                else:
                    arm.send(targets, kp_scale=weight)
                self.sent += 1
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

            with self._lock:
                self._weight = weight

            remaining = self.period - (time.monotonic() - cycle)
            if remaining > 0:
                self._stop.wait(remaining)

    # ------------------------------------------------------------------ state
    def status(self) -> dict:
        with self._lock:
            controller = {}
            for h in self.hands:
                inp = self._input.get(h)
                controller[h] = None if inp is None else {
                    "pos": [round(float(v), 4) for v in inp["pos"]],
                    "quat": [round(float(v), 4) for v in inp["quat"]],
                    "grip": inp["grip"],
                }
            return {
                "hands": list(self.hands),
                "engaged": dict(self._engaged),
                "weight": round(self._weight, 3),
                "solved_deg": {h: v for h, v in self._solved_deg.items()},
                "ik_err_mm": {h: round(v * 1000, 1) for h, v in self._ik_err.items()},
                "controller": controller,
                "connected": self._client is not None,
                "sent": self.sent,
                "error": self.error,
            }
