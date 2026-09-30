# Bilateral arm teleop from the Quest 3

VR controllers drive both of the R1's arms by position, plus wrist roll, so
you can reach for a small object while looking through the robot's camera.
This does **not** grasp anything -- there is no dexterous hand on this unit --
and it does not touch the legs; walking is `WALKING.md`'s separate `--walk`.

```bash
./vr.sh --teleop
```

Plain `./vr.sh` is still video and canned gestures only, exactly as before.
Starting the video feed should never be able to move an arm.

## The clutch model

Grip a controller and that hand's arm starts following it: move the
controller 10 cm forward, the arm's target moves 10 cm forward from wherever
it already was. Let go and the arm **freezes** exactly where it is -- it does
not relax, drop, or snap back to some rest pose. Grip again anywhere and a
new reference point is set from there.

This is a relative offset, not an absolute mapping from room position to
robot position -- like picking a computer mouse up and putting it down
somewhere else on the desk. It means the controller's physical location in
your room never has to line up with the arm's real position, and it means
you can reposition your hand mid-task by releasing and re-gripping without
the arm moving while you do it.

Wrist roll follows the controller's own roll the same way, relative to
wherever it was at the moment you gripped.

## Controls

| Input | Action |
|---|---|
| Grip (either hand) | engage that hand's clutch |
| Release grip | freeze that hand where it is |
| Trigger | fires the canned `clamp` gesture -- a placeholder "close hand" cue, not a real grasp (there is no gripper yet) |
| Thumbstick click | start/stop recording (only wired up with `--record`) |
| Thumbsticks | still drive walking if `--walk` is also passed; teleop does not use them |

`--teleop-hands right` or `--teleop-hands left` limits control to one arm,
recommended for the first real attempt. `--teleop-scale 0.5` halves how far
the arm moves per centimetre of controller motion, useful if the mapped
motion feels too large or too twitchy.

## What stops the arms

Same layered approach `locomotion.py` uses for walking, applied per hand:

| | Stops when | Latency |
|---|---|---|
| The clutch | you release the grip | next cycle (~15-20 ms) |
| Watchdog in `arm_teleop.py` | that hand's headset input goes stale | 0.3 s |
| Slew clamp | always active | caps joint speed at `MAX_SLEW_DEG_PER_S` (40 deg/s), regardless of cause |

The slew clamp is defined per **second** and multiplied by the measured cycle
time, not per cycle. The loop asks for 100 Hz but does not get it on Windows
-- timer granularity lands it nearer 45-65 Hz -- and a per-cycle limit would
have quietly halved the arm's speed there while running at full speed
elsewhere. A safety limit that varies with the host's clock is not a limit.

**A real incident, on hardware: none of the above stopped it.** In the first
live VR session, the arm kept moving for close to a minute after the headset
was taken off, with no software path available to stop it -- the operator
had to physically unplug the robot's ethernet. Root cause: taking the
headset off does not necessarily end the WebXR session, and it does not fire
`document.hidden`/`blur`/`pagehide` at all -- those are 2D-tab events, and
nothing was listening for the XR session's *own* visibility state. `pollTeleop`
kept running, and whatever the last grip/pose reading was (a controller left
resting against something, still tracked, still reporting pressed) kept
going out. Every layer in the table above assumes the browser keeps talking
to the server; this was the case where it didn't stop talking, it just
stopped meaning anything.

Two fixes, at different layers on purpose -- the same "several independent
reasons to stop, because they fail differently" logic `locomotion.py` already
uses for walking:

- **Client**: `xrSession.addEventListener('visibilitychange', ...)` and a
  same check inside `pollTeleop` release the moment `session.visibilityState`
  leaves `'visible'` -- the signal that actually reflects the headset, unlike
  the page-level events. `releaseTeleop()` is now also wired into every
  event `releaseDrive()` already was (`visibilitychange`, `pagehide`,
  `blur`, XR session `'end'`), which it had never been.
- **Robot**: `MAX_CONTINUOUS_ENGAGE_S` (20 s) in `arm_teleop.py` bounds one
  continuous grip regardless of what the browser does or fails to do. Once
  tripped, the hand stays refused -- not just for one cycle -- until the raw
  input genuinely reports released at least once; an early version checked
  only `clutch[hand] is None`, which a single forced-release cycle also
  satisfies, so a client still sending grip=True every cycle walked straight
  back in as a "fresh" engagement on the very next cycle. Caught by testing
  a synthetic unbroken hold before trusting it, not assumed correct.

The client fix is the one that actually matters for this exact bug. The
robot-side cap is there so a *future* bug shaped like this one -- in this
code or anything else that ever talks to `rt/lowcmd` -- has a backstop that
doesn't depend on the browser behaving.

Releasing both hands for longer than `RELEASE_GRACE` (3 s) ramps the shared
blend weight back to 0, handing the arms back to the robot's own controller.
Weight ramps up over `WEIGHT_RAMP_S` (2 s) the same way `r1_arm.py`'s
`--engage` already does, so the first movement after connecting is never a
step change.

## Known risks -- not resolved here, carried forward loudly

**`rt/arm_sdk` does not move this robot -- use `rt/lowcmd`.** Measured on
hardware: a commanded 20 deg sweep produced **0.00 deg** of motion via
arm_sdk, with frames accepted (correct `mode_machine`, valid CRC, no errors)
and silently discarded. The cause is in the R1 user manual's joint table --
`r1_arm.py` writes the arm_sdk blend weight to **slot 29**, which is the G1's
spare slot but is **HEAD_PITCH** on the R1. There is no weight slot at all
here: indices 0-30 are entirely real joints or EMPTY (14, 20, 21, 27, 28), so
arm_sdk's weighted blending is a G1 feature this robot does not have. Every
frame we sent was quietly commanding the head instead of setting a weight.

`arm_teleop.py` therefore drives `rt/lowcmd` (via `r1_lowcmd_arm.py`), which
does move the arms. Since lowcmd has no blend weight, authority is taken
gradually by **ramping stiffness** (`kp_scale` 0->1) instead.

**Releasing the controlling mode is still required, and is only safe with the
robot supported.** `ai` mode owns the joints and `lowcmd` was only ever
verified with `ai` released (`r1_mode.py --release`). Releasing makes every
joint limp, so the robot must be lying down or otherwise supported first --
learned the hard way: it was released while suspended and a hoist failure
dropped it.

**Expect the arm to under-track.** Position control against gravity leaves a
steady-state error of roughly (required torque / stiffness). Measured with
both arms lifting from a supine pose:

| KP | commanded | achieved |
|---|---|---|
| 30 | 15 deg | 3.7 deg |
| 60 | 15 deg | 7.9 deg |
| 80 | 45 deg | 33.3 deg |

Tracking improves as the arm approaches vertical, because gravity torque is
greatest at horizontal. Gravity feedforward would fix this properly; raising
stiffness indefinitely would not.

**The arm may not physically reach the floor.** The joint map has waist yaw
and roll but no waist pitch (see `r1_arm.py`), so a standing R1 bending only
at the shoulder may not be able to fold down far enough for a floor-level
object. Check with `arm_kinematics.selftest()` against a real floor-relative
target for your robot's actual mounting height before assuming it works. If
it does not reach, raise the object -- a stool or table edge, the same
choice Arnie's own G1 demo made for its box.

**One blend weight covers both arms.** `WEIGHT_SLOT` in `r1_arm.py`'s
`LowCmd_` is shared across the whole `rt/arm_sdk` frame -- engaging either
hand's clutch puts *both* arms under external authority at once. The
un-clutched arm only stays put because `arm_teleop.py` commands its current
position every single cycle regardless of which hand is active. A change
that stops doing that for the idle arm is a change that lets it drift, not
one that leaves it alone -- keep that invariant if you touch `_loop()`.

**The kinematics now come from the URDF, but the hand-motion mapping does
not.** `arm_kinematics.py` parses `unitree_r1/R1.urdf` for link offsets,
joint axes, mounting tilts and joint limits, so those are the robot's own
numbers rather than guesses. An earlier version assumed a spherical shoulder
and a 210 + 210 mm arm; the URDF shows the shoulder axes are offset (64 mm,
then 114 mm, with a -15 deg mount tilt) and the real segments are 88.5 mm and
116.7 mm -- about half. Anything written against that old model is wrong.

What is still unverified is `AXIS_MAP` in `arm_teleop.py`: which way a push
of the operator's hand maps onto the robot's frame. That needs the same
"move it and see which way it goes" check `r1_arm.py` prescribes. And the
workspace is small -- from the rest pose the hand can only move about 5 cm
before the residual grows, so expect to use the clutch often rather than
sweeping your arm around.

## Staged rollout

Do these in order. The first two need no robot at all.

0. **Robot lying flat on a mat**, then `python r1_mode.py --release`. Every
   joint goes limp, so it must already be supported. Never on a hoist.
1. **`python -m r1vision.arm_kinematics`** -- IK/FK self-check. Confirms the
   math is internally consistent; proves nothing about the real robot.
2. **`python arm_teleop_cli.py --script`** -- scripted engage/move/release
   smoke test against a simulated arm. No robot, no terminal raw-mode, runs
   anywhere. Checks the whole command -> clutch -> IK -> step-clamp -> send
   pipeline end to end.
3. **`python r1_lowcmd_arm.py --hold`** then `--lift --degrees 15` -- proves
   the actuation path on the real robot, independent of any teleop code. The
   hold stage cannot move anything; the lift stage is the first real motion.
4. **`python arm_teleop_cli.py --dry-run`** -- interactive keyboard version,
   still simulated, so you can get a feel for the clutch (`g` to engage/
   release, `wasdrf` to move, `h` to switch hands) before it means anything.
   Works on Windows and macOS/Linux alike.
5. **`python arm_teleop_cli.py`** -- keyboard, for real, watching the robot.
   This is the `r1_walk.py`-style rule: see the arm before a headset takes
   that away.
5. **`./vr.sh --teleop --teleop-hands right`** -- VR, one arm.
6. **`./vr.sh --teleop`** -- both arms.
7. **`./vr.sh --teleop --record`** -- add episode recording once control
   itself is trusted.

Clear the space around both arms, know how you will physically intervene,
and keep the robot's own remote/app to hand as a real e-stop -- nothing here
is a substitute for one.

## Recording

`--record` only opens the `/record` endpoint; nothing is written until you
start an episode (thumbstick click, or `curl '.../record?cmd=start'`).
Stopping (`cmd=stop`, or the thumbstick again) writes:

```
episodes/<timestamp>/
  video.mp4        head camera, one frame per sample
  episode.jsonl     one line per sample: t, plus ArmTeleopDriver.status()
                    (per-hand controller pose = the action, per-arm
                    commanded/measured angles = the state)
  episode.json      manifest: duration, sample count, link lengths used
```

Sample rate defaults to 30 Hz, independent of the 100 Hz control loop. A
recorder failure (disk full, a bad frame) is caught inside its own loop and
surfaces in `/record?cmd=status`'s `error` field -- it cannot affect arm
control, which never calls into the recorder at all.

## Narrowing down a failure

| Symptom | Likely cause |
|---|---|
| Grip does nothing in VR | `--teleop` was not passed, or the browser session predates the server restart -- reload the page |
| Arm moves the wrong direction | `AXIS_MAP` or `MIRROR` guessed wrong for that axis -- fix it there, not in the headset |
| Arm snaps or jumps | should not happen -- `MAX_STEP_DEG_PER_CYCLE` caps this; if it still does, the step clamp itself needs lowering, not removing |
| `python arm_teleop_cli.py --dry-run` also fails | a bug in the driver, not the network or the robot -- start there |
| `python r1_arm.py --joints` also fails | the DDS link, not teleop -- see `README.md`'s DDS/network sections |

## The reachable directions are not what you expect

Measured from the rest pose with the teleop solver's own budget (12 iterations,
no restarts), how far the hand can actually travel before the residual passes
5 mm:

| direction | travel | elbow at the end |
|---|---|---|
| forward only | 8 cm | 60.3 deg |
| forward + up | 29 cm | 65.9 deg |
| **forward + up, steep** | **39 cm** | **41.0 deg** |
| up | 17 cm | -55.9 deg |
| back | 32 cm | 60.5 deg |
| out (away from body) | 12 cm | 65.7 deg |
| down | 5 cm | 65.8 deg |

**Push forward AND up, not forward.** A door-opening motion -- hand driven
straight out at constant height -- is the single worst direction this arm has,
and it is what an operator naturally tries first. It reads as "the robot bends
its elbow instead of reaching", which is exactly what it does: at 8 cm forward
every one of 3252 sampled IK solutions has the elbow between 64.8 and 91.2 deg.
There is no straighter-elbow solution to pick. The elbow bend is the geometry,
not the solver, and no tuning changes it.

The shoulder sits at z = +0.197 m and the hand hangs below it, so travelling
forward at hanging height forces the arm to fold. Rising toward shoulder level
lets it extend, which is why the steep diagonal buys 5x the travel AND a
straighter elbow at the same time.

Raising the starting posture does NOT help and was measured, not assumed: a
"ready" pose at shoulder pitch -50 deg / elbow 40 deg gives 2 cm of forward
travel against rest's 8 cm, because a pose already extended forward has less
headroom left. Every forward-leaning candidate in a 180-pose sweep scored the
same 87% local reachability as rest with half the forward travel. `rest` stays.

### Which joint stops each motion (right arm, measured from rest)

| motion | travel | what stops it |
|---|---|---|
| out to the side | 12 cm | workspace edge |
| **out AND up** | **34 cm** | workspace edge |
| straight up | 17 cm | **elbow at its -55.9 deg lower limit** |
| forward | 9 cm | workspace edge |
| **in toward the chest** | 18 cm | **shoulder roll at its +13 deg limit** |
| forward + toward chest | 12 cm | workspace edge |
| forward + up | 29 cm | workspace edge |

Two of these are hard joint limits and no tuning will move them:

**Lifting straight up runs the elbow into its lower stop.** The arm must
straighten to rise vertically, and the elbow only extends to -55.9 deg. Raise
the arm OUT AND UP instead and you get 34 cm rather than 17, because that uses
shoulder roll -- which has 142 deg of travel -- instead of demanding elbow
extension.

**Reaching across the body stops at 18 cm** because right shoulder roll only
travels to +13 deg (left, +142 by mirror). Checked against the URDF: right is
-142..+13 and left is -13..+142, correct mirrors, so this is the robot's design
rather than another copied-limits bug of the `LIMITS[16]` kind. Each arm abducts
142 deg outward and barely crosses the midline.

Everything else on this list stops at the workspace edge with no joint at a
limit -- the arm is simply short. The elbow sits at 74-78 deg at the end of
every forward or inward reach, which is what "the elbow is always bent" is: the
only motion that straightens it is lifting, and lifting is what the elbow limit
cuts short.
