# Walking the R1 from the Quest 3

Thumbsticks drive the robot while you look through its camera, the same way
Unitree's own controller does: **left stick moves, right stick turns, let go and
it stops walking and stands there.** Releasing the stick does not make it sag or
fall — it holds its stance and keeps balancing.

Locomotion is off unless you ask for it:

```bash
./vr.sh --walk
```

Plain `./vr.sh` is video and arm gestures only, exactly as before. Starting the
video feed should never be able to make a humanoid take a step.

## How the robot is told to move

The SDK exposes locomotion as `LocoClient.SetVelocity(vx, vy, omega, duration)`
on the `sport` service. That is the entire interface — no position control, no
path following. You stream velocities and the robot's controller handles balance
and footfall. `r1vision/locomotion.py` owns that stream: one thread, 10 Hz,
whatever the headset is doing.

## What stops the robot

Three mechanisms, independent, because they fail in different ways.

| | Stops when | Latency |
|---|---|---|
| **The stick self-centring** | you take your thumb off | one 10 Hz tick |
| **Watchdog in `locomotion.py`** | the headset stops sending | 0.4 s |
| **`duration=0.5` on every command** | the robot stops hearing from us at all | 0.5 s |

The stick is the deadman. It is spring centred, so "let go" and "commanded zero"
are physically the same event — which is exactly why the Unitree controller
needs no separate hold-to-run button either.

The last one is the one that matters most. It runs on the robot, so a crashed
Mac, a dropped Wi-Fi link, or a pulled cable stops it without any code of ours
running. Given the connection drops already visible in this setup — RTSP
timeouts, broken pipes when the headset sleeps — assume it will be exercised.

The page also sends an explicit zero when the tab is hidden, the page unloads,
or the VR session ends.

If you would rather also hold a grip button before the sticks do anything —
useful while you are still getting a feel for it — start with `--deadman`.

## Stop versus damp

**Stop** commands zero velocity; the robot keeps standing and balancing. This is
what releasing the stick does, and it is what you want almost always. Nothing on
a stick or a controller button can make the robot go limp.

**Damp** releases the joints. A standing humanoid will sag and probably go over.
It is the right thing when someone is holding the robot and the wrong thing
otherwise, so it is not on any controller button — only on the 2D page behind a
confirmation, and on the keyboard CLI behind a double-press. Do not map it to a
button you can hit by flinching.

## Speed

Defaults in `locomotion.py`, deliberately slow:

| | |
|---|---|
| forward | 0.30 m/s |
| backward | 0.15 m/s — half, because you are reversing blind |
| sideways | 0.20 m/s |
| yaw | 0.40 rad/s (~23°/s) |

`--walk-speed 0.5` halves all of them. Full stick is not full speed: the
response curve is squared, so most of the stick's travel lives in the slow end
where you actually steer.

## Do the keyboard version first

```bash
python r1_walk.py --dry-run    # no robot; proves the plumbing
python r1_walk.py              # standing next to the robot, watching it
```

Same control path, same interlocks, but you can see the room. Establish that the
robot moves the way you expect and that stopping works *before* you put on a
headset that blinds you to it.

Holding a key repeats it, which refreshes the command; letting go stops the
repeats and the watchdog zeroes it. There is no key that leaves it walking.

    w / s   forward / back        a / d   strafe        q / e   turn
    space   stop                  1  start    2  stand up    0  damp    x  quit

## In the headset

1. `./vr.sh --walk`
2. Open the URL, and **before entering VR** press **Stand & balance**. The robot
   has to be in the walking state machine or velocity commands do nothing.
3. Enter VR.
4. Left stick moves, right stick turns. The HUD shows the velocity the robot was
   actually given, not what you asked for.
5. Let go of the sticks to stop. The robot stays standing; step 2's state is
   what keeps it balancing, and nothing here takes it out of that state.

## Before the first real walk

- Clear floor, and know where you will physically intervene from.
- Someone watching the robot, not the headset.
- The robot's own remote/app to hand as a real e-stop. Nothing here is a
  substitute for one.
- Check `r1_mode.py` first if commands are accepted but nothing moves — a
  built-in mode holding the joints is the usual reason.

## Why the sport backend does not work on this R1

Measured against the robot on 2026-07-30, all read-only except one zero-velocity
call:

| Probe | Result |
|---|---|
| `loco` service | not reachable |
| `sport` service, `GetFsmId` (7001) | **ok**, FSM `811` |
| `sport` getters 7002–7005 | **"API not implemented"** — 1 of 5 calls exists |
| `sport` `SetVelocity(0,0,0)` (7105) | **code 127** — refused, and FSM unchanged |
| `motion_switcher` `CheckMode` | `{'form': '0', 'name': 'ai'}` |
| `motion_switcher` `SelectMode('normal')` | **error 7004**, mode unchanged |
| arm actions, same session | **7406**, having worked earlier the same day |
| `rt/lowstate` | live at ~10 Hz, `mode_pr=0 mode_machine=1` |
| `rt/wirelesscontroller` | silent; `lowstate.wireless_remote` has no valid header |

Two conclusions. `SetVelocity` returning 127 rather than 3203 means the call
**exists and is being refused**, so this is not a wrong-service problem — the
earlier guess that `loco` was the right name was wrong, and `loco_probe.py`
disproved it. And arm actions being refused in the same session that they worked
means the robot changed, not the code: both services refuse external motion
while `ai` mode owns the joints.

There is no way in from that direction. `SelectMode` rejects `normal`, there is
no list-modes API, and `ReleaseMode` would drop a standing humanoid.

## The wireless backend

`ai` mode is Unitree's own walking controller, and it is what is standing the
robot up and balancing it right now. The robot **subscribes to
`rt/wirelesscontroller`**, whose message is a plain gamepad — `lx, ly, rx, ry`
and a `keys` bitmask. That is the channel the handheld remote drives, and `ai`
mode is what listens to it.

So rather than asking a refusing API to move the robot, publish to the channel
its own remote uses:

```bash
python r1_walk.py --backend wireless      # keyboard, standing next to it
./vr.sh --walk --backend wireless         # from the headset
```

**Read this before using it.** `WirelessController_` has no `duration` field, so
the 0.5 s robot-side command expiry that made the sport backend safe **does not
exist on this path**. Of the three stop mechanisms, only two remain:

| | Stops when | Latency |
|---|---|---|
| Stick self-centring | you let go | one 20 ms tick |
| Watchdog in `locomotion.py` | the headset stops sending | 0.4 s |
| ~~`duration=0.5`~~ | **not available on this backend** | — |

The send loop therefore never goes silent — it publishes zeros continuously
rather than stopping — and `close()` zeroes before exiting. But if the process is
killed outright, the robot holds the last stick value until its own controller
times out, and whether it does is unverified. Full stick sends only 0.35
deflection (`STICK_LIMIT`), which is the only speed limit available here, since
raw axes have no velocity units to clamp.

Given the ethernet adapter has been flapping, test this at the keyboard first,
with the Unitree remote in your hand.

## Narrowing down a failure

| Symptom | Where the problem is |
|---|---|
| No `drive ...` lines when you push a stick | the headset — sticks are not reaching the Mac |
| `drive` lines, `refused` climbing | robot hears you and rejects it — the 127/7406 case above |
| `drive` lines, `refused 0`, robot still | wireless backend: axis signs or `ai` mode not listening |
| `channel factory init error` | `en6` dropped again; check `ifconfig en6` |
| `python arm_actions.py --list` also fails | the DDS link, not locomotion |

The server prints one `drive` line a second while a stick is pushed, with the
velocity, the backend in use, and the robot's last reply code.

## Known-unverified

The locomotion path has been tested end to end against `--dry-run` only:
clamping, deadzone, watchdog timing, send rate, HTTP endpoints, shutdown. None
of it has been exercised against the robot from here, so which service answers
and which FSM id walks are still open questions — that is what `loco_probe.py`
is for, and its output should replace the guesses in this file.

The FSM ids in `locomotion.py` come from the SDK's own named helpers
(`Start()` → 500, `Damp()` → 1). Whether the R1 numbers them the same way is
unconfirmed; `loco_probe.py` prints what the robot actually reports, and if it
shows an id not in that table while the robot is standing, that id is the one
that walks.
