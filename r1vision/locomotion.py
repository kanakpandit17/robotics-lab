"""Velocity control for the R1, with the safety interlocks walking needs.

The SDK exposes locomotion as `LocoClient.SetVelocity(vx, vy, omega, duration)`
on the `sport` service. That call is the whole interface: there is no position
control, no path following. You stream velocities and the robot's own controller
handles balance and footfall.

Streaming velocity to a walking humanoid is not like driving a wheeled base. If
the commands stop arriving mid-stride the robot keeps whatever it last heard, so
every layer here is built around making a *lost* command fail to zero rather
than fail to "keep going".

Three independent things stop the robot
---------------------------------------
1. **The duration argument.** Every command carries `duration=0.5`, so the robot
   stops half a second after the last packet it received, with no help from this
   process. If the Mac crashes, the Wi-Fi drops, or the cable is pulled, the
   robot stops on its own. This is the backstop that actually matters.
2. **The watchdog thread here.** If no fresh command has arrived from the
   headset within `watchdog` seconds, this sends explicit zeros. This catches a
   headset that stopped sending -- battery, browser tab, someone taking it off.
3. **The deadman in the headset.** A grip button must be held. Releasing it is
   the normal way to stop.

Any one of the three is enough. They exist separately because they fail in
different ways.

On stopping versus damping
--------------------------
`stop()` commands zero velocity and the robot keeps standing and balancing.
`damp()` releases the joints, and a standing humanoid will sag and fall. Damp is
the right thing when someone is supporting the robot and the wrong thing when it
is standing on its own, so it is deliberately not reachable from a controller
button -- only from the CLI and the 2D page, where hitting it is a decision
rather than a twitch.
"""

from __future__ import annotations

import threading
import time

# Slower than you think you want. A humanoid at 0.3 m/s is already walking at a
# brisk-ish pace relative to how quickly you can react to it going wrong, and
# nothing about this rig is fast to abort.
MAX_VX = 0.3      # m/s forward (backward is capped at half this)
MAX_VY = 0.2      # m/s sideways
MAX_W = 0.4       # rad/s yaw, about 23 deg/s

SEND_RATE = 10.0     # Hz, for the sport backend
WIRELESS_RATE = 50.0  # Hz, closer to what a real remote publishes at
CMD_DURATION = 0.5   # s of validity per command; the robot-side deadman
WATCHDOG = 0.4       # s without a fresh command before we force zeros

# Full stick deflection sent to the robot when our stick is fully pushed. The
# wireless backend has no speed units to clamp -- it sends raw joystick axes and
# the robot's own controller decides what that means in m/s -- so this is the
# only speed limit available on that path. 0.35 is a walk.
STICK_LIMIT = 0.35

WIRELESS_TOPIC = "rt/wirelesscontroller"

# How walking commands actually reach the robot.
#
# "sport"    LocoClient.SetVelocity on the `sport` service. The documented SDK
#            path, and every command carries a 0.5 s expiry so a lost link stops
#            the robot without our help. On this R1 it answers GetFsmId but
#            returns 127 for SetVelocity, and four of its five getters report
#            "API not implemented" -- a partial implementation that refuses to
#            drive while `ai` mode owns the joints.
#
# "wireless" Publish a gamepad message to rt/wirelesscontroller, which the robot
#            subscribes to. This is the channel the handheld remote drives, and
#            `ai` mode -- the controller that is currently standing the robot up
#            and balancing it -- is what listens to it. Driving the robot the way
#            its own remote does, rather than asking a refusing API nicely.
#
#            The trade is real: WirelessController_ has no duration field, so the
#            robot-side 0.5 s expiry does not exist here. If this process is
#            killed outright the robot keeps the last stick value until its own
#            controller times out, which is unverified. The send loop therefore
#            never goes silent -- it publishes zeros continuously rather than
#            stopping -- and close() zeroes before exiting.
#
# "wireless" is the default as of the 2026-08-05 measurements, which closed the
# question. rt/wirelesscontroller was captured live while the handheld remote
# drove the robot: 639 messages, ly +0.82..+0.88 walking forward, rx +-0.89
# turning, both returning to exactly 0.000 on release. The axis mapping in
# _send() already matched. Meanwhile `sport` failed a third independent check --
# rt/sportmodestate publishes a constant fsm_id=811 with mode/task/task_time
# pinned at zero through two minutes of remote-driven walking, which is a stub,
# not a controller. Defaulting to "sport" meant connect() succeeded against that
# stub and the wireless path was never reached: the robot stood still while
# every SetVelocity returned 127.
BACKENDS = ("sport", "wireless")

FSM = {
    "zero_torque": 0,
    "damp": 1,
    "sit": 3,
    "stand_up": 4,
    "start": 500,
    "lie_to_stand": 702,
    "squat_to_stand": 706,
}

CODES = {
    0: "ok",
    3102: "service not reachable",
    3104: "timeout",
    3202: "internal server error",
    3203: "API not implemented",
    3204: "parameter error",
    3205: "rejected",
}


def describe(code: int | None) -> str:
    if code is None:
        return "no reply"
    return CODES.get(code, f"error {code}")


def shape(value: float, deadzone: float = 0.15) -> float:
    """Deadzone plus a squared response curve, in -1..1.

    Thumbsticks rest slightly off centre, and a humanoid answering a resting
    stick is how you find out the robot has been creeping across the room.
    Squaring what is left puts fine control where you spend most of the time --
    near zero -- and keeps full deflection reachable.
    """
    # A real axis never leaves -1..1, so a value outside it means the reading is
    # wrong, not that the operator wants to go faster. Read it as zero: the
    # alternative is a corrupted number arriving as full deflection.
    if not -1.001 <= value <= 1.001:
        return 0.0
    if abs(value) < deadzone:
        return 0.0
    scaled = min(1.0, (abs(value) - deadzone) / (1.0 - deadzone))
    return (scaled * scaled) * (1.0 if value > 0 else -1.0)


# Which DDS service actually implements walking is not the same on every
# Unitree humanoid, and the SDK disagrees with itself: `g1/loco` is hardcoded to
# "sport" while `h1/loco` uses "loco". This R1 advertises command topics for
# both (`rt/api/loco/request` and `rt/api/sport/request`, plus an `rr/locoReply`
# the robot actively publishes), so the name cannot be inferred from the topic
# list alone -- it has to be asked. `loco` is tried first because that is what
# the humanoid SDKs use and what this robot replies on.
SERVICE_NAMES = ("loco", "sport")


def _client_for(service: str):
    """A LocoClient bound to a chosen service name.

    The API ids, payloads and helper methods are identical between the two; only
    the topic prefix differs, so subclassing to override the name reuses the
    whole client rather than reimplementing it.
    """
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.rpc.client import Client

    class _Named(LocoClient):
        def __init__(self):
            Client.__init__(self, service, False)
            self.first_shake_hand_stage_ = -1

    return _Named()


def probe_services(iface: str = "en6", names: tuple[str, ...] = SERVICE_NAMES,
                   timeout: float = 2.0, settle: float = 2.0) -> list[dict]:
    """Ask each candidate service for the current FSM id. Moves nothing.

    GetFsmId (api 7001) is a read. Whichever service answers it is the one that
    is listening, which is the only way to tell from outside.
    """
    from r1vision.unitree_dds import (default_peers, init_channel,
                                      quiet_dds_stderr)

    results = []
    with quiet_dds_stderr():
        init_channel(iface, peers=default_peers())
        clients = []
        for name in names:
            client = _client_for(name)
            client.SetTimeout(timeout)
            client.Init()
            clients.append((name, client))
        time.sleep(settle)      # discovery must match before anything replies

        for name, client in clients:
            try:
                code, data = client.GetFsmId()
                results.append({"service": name, "code": code,
                                "fsm": data, "detail": describe(code)})
            except Exception as exc:
                results.append({"service": name, "code": None, "fsm": None,
                                "detail": f"{type(exc).__name__}: {exc}"})
    return results


class _DryClient:
    """Stands in for LocoClient so the whole path can be exercised at a desk.

    Everything above this -- clamping, the watchdog, the send loop, the headset
    plumbing -- is worth testing without a humanoid in the room.
    """

    def SetVelocity(self, vx, vy, w, duration):  # noqa: N802
        return 0

    def SetFsmId(self, fsm_id):  # noqa: N802
        return 0


class LocoDriver:
    """Owns the connection to the robot's locomotion service and one send loop.

    Commands arrive from HTTP handler threads via `command()`, which only
    records an intent. A single thread does all the talking, so the robot sees
    one coherent stream at a steady rate no matter how the headset behaves.
    """

    def __init__(self, iface: str = "en6", scale: float = 1.0,
                 rate: float | None = None, watchdog: float = WATCHDOG,
                 dry_run: bool = False, service: str = "auto",
                 backend: str = "wireless"):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.iface = iface
        self.dry_run = dry_run
        self.service = service
        self.backend = backend
        self.max_vx = MAX_VX * scale
        self.max_vy = MAX_VY * scale
        self.max_w = MAX_W * scale
        self.stick_limit = min(1.0, STICK_LIMIT * scale)
        if rate is None:
            rate = WIRELESS_RATE if backend == "wireless" else SEND_RATE
        self.period = 1.0 / rate
        self.watchdog = watchdog

        self._client = None
        self._pub = None
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)        # m/s and rad/s, for the sport backend
        self._stick = (0.0, 0.0, 0.0)      # -1..1 axes, for the wireless backend
        self._cmd_at = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.error: str | None = None
        self.sent = 0
        self.last_code: int | None = None
        self.stale = True
        self.fsm_id: int | None = None
        # Counts commands the robot actively rejected, as opposed to ones that
        # were sent into the void. A steady climb here means the service is
        # reachable but refusing -- usually the wrong FSM, not the wrong name.
        self.refused = 0

    # ---------------------------------------------------------------- connect
    def connect(self):
        """Lazily open the sport service. Video must keep working without it."""
        if self._client is not None:
            return self._client
        if self.dry_run:
            self._client = _DryClient()
            self.service = "dry-run"
            return self._client
        from r1vision.unitree_dds import (default_peers, init_channel,
                                          quiet_dds_stderr)

        with quiet_dds_stderr():
            # Unicast peers, not multicast-only. Multicast discovery does not
            # reach this robot: with peers=None the publisher is created
            # happily, Write() succeeds, and nothing arrives -- which looks
            # exactly like the robot ignoring a valid command.
            init_channel(self.iface, peers=default_peers())

            if self.backend == "wireless":
                from unitree_sdk2py.core.channel import ChannelPublisher
                from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
                    WirelessController_,
                )

                self._pub = ChannelPublisher(WIRELESS_TOPIC, WirelessController_)
                self._pub.Init()
                self._msg_type = WirelessController_
                self.service = WIRELESS_TOPIC
                time.sleep(1.0)     # let the robot's reader match before we send
                self._client = self._pub
                return self._client

            if self.service == "auto":
                # Ask before commanding. Picking the wrong service is silent --
                # the calls simply go nowhere and the robot stands there, which
                # is exactly the failure that is hard to diagnose from outside.
                candidates = []
                for name in SERVICE_NAMES:
                    probe = _client_for(name)
                    probe.SetTimeout(2.0)
                    probe.Init()
                    candidates.append((name, probe))
                time.sleep(2.0)

                chosen = None
                for name, probe in candidates:
                    try:
                        code, fsm = probe.GetFsmId()
                    except Exception:
                        continue
                    if code == 0:
                        chosen = (name, probe, fsm)
                        break
                if chosen is None:
                    tried = ", ".join(SERVICE_NAMES)
                    raise RuntimeError(
                        f"no locomotion service answered (tried {tried}). "
                        f"Run: python loco_probe.py --iface {self.iface}")
                self.service, client, self.fsm_id = chosen
                print(f"  locomotion: service '{self.service}', "
                      f"robot FSM {self.fsm_id}")
            else:
                client = _client_for(self.service)
                client.Init()
                time.sleep(2.0)

            # Short, because this timeout sits inside the send loop. A stalled
            # RPC must not hold up the next velocity command.
            client.SetTimeout(0.5)
        self._client = client
        return client

    # ------------------------------------------------------------------ drive
    def command(self, vx: float, vy: float, w: float) -> tuple[float, float, float]:
        """Record a desired velocity in normalised -1..1 units.

        Returns what was actually accepted, in m/s and rad/s, so the caller can
        show the operator the clamped figure rather than what they asked for.
        """
        vx = max(-1.0, min(1.0, vx))
        vy = max(-1.0, min(1.0, vy))
        w = max(-1.0, min(1.0, w))

        # Walking backwards blind, through a camera facing the other way, with
        # no rear sensing at all. Half speed is still generous.
        limit_x = self.max_vx if vx >= 0 else self.max_vx * 0.5
        out = (vx * limit_x, vy * self.max_vy, w * self.max_w)

        # The wireless path has no velocity units to clamp, so the same
        # reverse-is-slower rule is applied to the raw axis instead.
        stick_x = vx * (1.0 if vx >= 0 else 0.5) * self.stick_limit
        stick = (stick_x, vy * self.stick_limit, w * self.stick_limit)

        with self._lock:
            self._cmd = out
            self._stick = stick
            self._cmd_at = time.monotonic()
        return out

    def stop(self) -> None:
        with self._lock:
            self._cmd = (0.0, 0.0, 0.0)
            self._stick = (0.0, 0.0, 0.0)
            self._cmd_at = time.monotonic()

    def fsm(self, name: str) -> tuple[bool, str]:
        """Change the robot's state machine: start, damp, sit, stand_up ..."""
        fsm_id = FSM.get(name)
        if fsm_id is None:
            return False, f"unknown state {name!r}; know {sorted(FSM)}"
        if name in ("damp", "zero_torque", "sit"):
            self.stop()

        # State changes ride the `sport` service, which this robot stubs out.
        # The wireless backend is a publisher with no SetFsmId at all, and
        # letting that surface as an AttributeError is dangerous for `damp`
        # specifically: the operator reaches for it to make the robot safe and
        # is told something that does not read as failure. Say it plainly, and
        # name the control that does work.
        if self.backend == "wireless":
            return False, (
                f"cannot set {name!r}: the wireless backend drives "
                "rt/wirelesscontroller and has no state-machine channel. "
                "Velocity has been zeroed. Use the handheld remote for state "
                "changes -- L2+B held 5 s is the safe stop (damping); do NOT "
                "use L2+Y on a standing robot, it is zero-torque and it will "
                "collapse.")
        try:
            with self._quiet():
                code = self.connect().SetFsmId(fsm_id)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return code == 0, describe(code)

    def _quiet(self):
        from r1vision.unitree_dds import quiet_dds_stderr
        return quiet_dds_stderr()

    # ------------------------------------------------------------- send loop
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Zero the robot before going away, then let the loop exit."""
        self.stop()
        time.sleep(2 * self.period)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _send(self, client, vel: tuple[float, float, float],
              stick: tuple[float, float, float]) -> int:
        """One command, by whichever route this driver was built for."""
        if self.backend == "sport":
            return client.SetVelocity(*vel, CMD_DURATION)

        if self.dry_run:
            return 0

        # The remote's own axis layout: left stick walks (ly ahead, lx sideways),
        # right stick turns (rx). Signs are the robot's to interpret -- if it
        # walks backwards when you push forward, flip them here rather than in
        # the headset, so both entry points agree.
        vx, vy, w = stick
        msg = self._msg_type(lx=-vy, ly=vx, rx=-w, ry=0.0, keys=0)
        client.Write(msg)
        return 0

    def _loop(self) -> None:
        from r1vision.unitree_dds import quiet_dds_stderr

        try:
            client = self.connect()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"  locomotion unavailable: {self.error}")
            return
        print(f"  locomotion: {self.backend} backend -> '{self.service}' at "
              f"{1.0 / self.period:.0f} Hz")
        if self.backend == "wireless":
            print(f"  full stick = {self.stick_limit:.2f} deflection; "
                  "no robot-side command expiry on this path")

        with quiet_dds_stderr():
            while not self._stop.is_set():
                cycle = time.monotonic()

                with self._lock:
                    vx, vy, w = self._cmd
                    lx, ly, lw = self._stick
                    age = cycle - self._cmd_at

                if age > self.watchdog and (vx or vy or w):
                    # Nothing fresh from the headset. Do not coast.
                    print(f"  watchdog: no command for {age:.1f}s, stopping")
                    with self._lock:
                        self._cmd = (0.0, 0.0, 0.0)
                        self._stick = (0.0, 0.0, 0.0)
                    vx = vy = w = 0.0
                    lx = ly = lw = 0.0
                self.stale = age > self.watchdog

                try:
                    code = self._send(client, (vx, vy, w), (lx, ly, lw))
                    self.sent += 1
                    if code != 0 and (vx or vy or w):
                        self.refused += 1
                        # Once per run, not ten times a second. A rejected
                        # velocity almost always means the robot is not in a
                        # state that walks, so say what to do about it.
                        if self.refused == 1:
                            print(f"  robot refused a move command: "
                                  f"{describe(code)}")
                            print("  it is probably not in a walking state -- "
                                  "press 'Stand & balance', or r1_walk.py key 1")
                    if code != self.last_code:
                        self.last_code = code
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.last_code = None

                # Measure from the top of the cycle, so a slow RPC eats into
                # the sleep instead of pushing the whole schedule later.
                remaining = self.period - (time.monotonic() - cycle)
                if remaining > 0:
                    self._stop.wait(remaining)

    # ------------------------------------------------------------------ state
    def status(self) -> dict:
        with self._lock:
            vx, vy, w = self._cmd
            # Computed here rather than read from the send loop, which updates
            # it only once a cycle and would report a command as stale for up
            # to 100 ms after it arrived.
            stale = (time.monotonic() - self._cmd_at) > self.watchdog
        return {
            "vx": round(vx, 3), "vy": round(vy, 3), "w": round(w, 3),
            "limits": {"vx": self.max_vx, "vy": self.max_vy, "w": self.max_w},
            "connected": self._client is not None,
            "backend": self.backend,
            "service": self.service,
            "fsm": self.fsm_id,
            "stale": stale,
            "sent": self.sent,
            "refused": self.refused,
            "code": self.last_code,
            "detail": describe(self.last_code),
            "error": self.error,
        }
