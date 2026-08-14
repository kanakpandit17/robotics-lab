#!/usr/bin/env python3
"""Serve the R1's camera to a Meta Quest 3 (or any headset/phone browser).

    python quest_stream.py

Then open the printed URL in the Quest's browser and hit Fullscreen.

How it works
------------
The Quest browser cannot play RTSP. This Mac pulls H.264 from the robot over
ethernet, re-encodes each frame as JPEG, and serves it as multipart MJPEG over
HTTP -- which every browser plays natively, with no app to install and no
sideloading.

MJPEG rather than WebRTC or HLS: HLS buffers several seconds, which is unusable
for looking around through a robot. WebRTC would be lower latency still but
needs a signalling server and TURN setup. MJPEG lands around 150-300 ms on a
local network, which is fine for watching, and it is about 40 lines instead of
a thousand.

Networking
----------
The Quest has to reach this Mac. The robot's own Wi-Fi AP is the easiest way,
because the Mac can be on both networks at once:

    Mac  -- ethernet --> robot (192.168.123.x)   pulls the video
    Mac  -- wifi ------> robot's AP (192.168.12.x)
    Quest - wifi ------> robot's AP              opens the page

Join `R1_03302_c33840b2` on both the Quest and the Mac, keep the ethernet
plugged in, and use the 192.168.12.x URL this prints.

Campus Wi-Fi such as eduroam usually will not work: client isolation blocks
devices from reaching each other. A phone hotspot is a fine alternative.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from r1vision.sources import build_source  # noqa: E402

DEFAULT_SOURCE = "rtsp://192.168.123.161:8551/video"

_latest: dict[str, object] = {"jpeg": None, "seq": 0, "fps": 0.0}
_lock = threading.Condition()
_stop = threading.Event()

# Arm control shares this process. The video comes over RTSP, not DDS, so
# there is no domain clash with the SDK client created here.
_arm = {"client": None, "err": None, "last": 0.0}
_arm_lock = threading.Lock()
ARM_COOLDOWN = 1.5   # seconds; the robot ignores actions queued on top of each other


_loco = {"driver": None}     # LocoDriver, or None when --walk was not passed
_drive_log = {"at": 0.0}     # throttles the drive log to one line a second

_teleop = {"driver": None}   # ArmTeleopDriver, or None when --teleop was not passed
_recorder = {"instance": None}   # EpisodeRecorder, or None when --record was not passed
_estop = {"switcher": None}   # MotionSwitcherClient, warmed up at startup -- see /estop


def _latest_frame() -> tuple[bytes | None, int]:
    """What the recorder pulls: the same JPEG buffer the video stream reads."""
    with _lock:
        return _latest["jpeg"], int(_latest["seq"])


def arm_client(iface: str = "en6"):
    """Connect lazily so video still works if control is unavailable."""
    if _arm["client"] is not None:
        return _arm["client"]
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
    from r1vision.unitree_dds import init_channel, quiet_dds_stderr

    # Unicast peer discovery sweeps ~94 ports, most with nothing behind them,
    # and CycloneDDS logs a failure line for each below its own verbosity
    # control. Without this the terminal is unreadable.
    with quiet_dds_stderr():
        init_channel(iface, peers=None)
        c = G1ArmActionClient()
        c.SetTimeout(10.0)
        c.Init()
    time.sleep(2.0)          # let discovery match before the first call
    _arm["client"] = c
    return c


def trigger_action(name: str, iface: str = "en6") -> tuple[bool, str]:
    from arm_actions import ACTIONS

    aid = ACTIONS.get(name)
    if aid is None and name.isdigit():
        aid = int(name)
    if aid is None:
        return False, f"unknown action {name!r}"

    with _arm_lock:
        now = time.monotonic()
        if now - float(_arm["last"]) < ARM_COOLDOWN:
            return False, "too soon"
        _arm["last"] = now
        try:
            client = arm_client(iface)
            code = client.ExecuteAction(aid)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
    return (code == 0), ("ok" if code == 0 else f"robot returned {code}")


from r1vision.xr_page import render_page  # noqa: E402

# Plain full-screen version for phones or anything without WebXR.
FLAT_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R1 Vision</title>
<style>html,body{height:100%;margin:0;background:#000}
img{width:100%;height:100%;object-fit:contain}</style></head>
<body><img src="/stream.mjpg" alt="robot camera"></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Silent by default -- an MJPEG stream would bury the console. But with
    # everything on the server side verifiably correct and still no controller
    # data arriving, "which requests actually reach us" stops being noise and
    # becomes the only evidence that distinguishes "the headset never loaded
    # the page" from "it loaded it but never entered an XR session" from "it
    # is polling and the poses are bad". /stream is still dropped, and the
    # high-rate /teleop poll is summarised once a second rather than 30x.
    _seen: dict = {}

    def log_message(self, fmt, *a):
        try:
            path = self.path.split("?")[0]
        except Exception:
            return
        if path == "/stream":
            return
        now = time.monotonic()
        if path == "/teleop":
            n, at = self._seen.get(path, (0, 0.0))
            n += 1
            if now - at < 1.0:
                self._seen[path] = (n, at)
                return
            self._seen[path] = (0, now)
            print(f"  [http] {path} x{n} in the last second "
                  f"from {self.client_address[0]}", flush=True)
            return
        print(f"  [http] {self.command} {path} from {self.client_address[0]}",
              flush=True)

    def _json(self, payload: dict) -> None:
        import json as _json

        body = _json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            teleop_driver = _teleop["driver"]
            body = render_page(_loco["driver"] is not None,
                               self.server.deadman,
                               teleop_driver is not None,
                               _recorder["instance"] is not None,
                               teleop_hands=(teleop_driver.hands
                                            if teleop_driver else ())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # The gesture table, so anything else driving this server does not have
        # to hardcode its own copy.
        if self.path == "/actions":
            from arm_actions import ACTIONS

            self._json({"ok": True, "actions": ACTIONS})
            return

        # Velocity, ten times a second while a grip is held. Sticks arrive raw
        # in -1..1; the deadzone, curve and speed caps are applied in
        # locomotion.py so there is one place to audit them.
        if self.path.startswith("/drive"):
            from urllib.parse import parse_qs, urlparse
            from r1vision.locomotion import describe, shape

            driver = _loco["driver"]
            if driver is None:
                self._json({"ok": False, "detail": "walking disabled; restart with --walk"})
                return
            q = parse_qs(urlparse(self.path).query)

            def axis(name: str) -> float:
                try:
                    return shape(float(q.get(name, ["0"])[0]))
                except ValueError:
                    return 0.0

            vx, vy, w = driver.command(axis("vx"), axis("vy"), axis("w"))
            # One line a second while driving, so "the sticks do nothing" can be
            # told apart from "the sticks reach the Mac but the robot ignores
            # them" without instrumenting anything further.
            now = time.monotonic()
            if (vx or vy or w) and now - _drive_log["at"] > 1.0:
                _drive_log["at"] = now
                print(f"  drive vx {vx:+.2f} vy {vy:+.2f} yaw {w:+.2f}  "
                      f"[{driver.service}] sent {driver.sent} "
                      f"refused {driver.refused} {describe(driver.last_code)}")
            self._json({"ok": True, "vx": vx, "vy": vy, "w": w, "detail": "ok"})
            return

        if self.path.startswith("/loco"):
            from urllib.parse import parse_qs, urlparse

            driver = _loco["driver"]
            if driver is None:
                self._json({"ok": False, "detail": "walking disabled; restart with --walk"})
                return
            cmd = parse_qs(urlparse(self.path).query).get("cmd", [""])[0]
            if cmd == "stop":
                driver.stop()
                ok, detail = True, "stopped"
            elif cmd == "status":
                self._json({"ok": True, "detail": "ok", **driver.status()})
                return
            else:
                ok, detail = driver.fsm(cmd)
            self._json({"ok": ok, "detail": detail, "cmd": cmd})
            print(f"  loco {cmd}: {detail}")
            return

        # The one route meant to be reachable without thinking: no aim, no
        # menu, one press. Does two things a released grip and exiting VR
        # never did on their own -- drops teleop's authority over the arm
        # THIS instant (not the graceful ramp normal release uses), and
        # hands the joints back to the robot's own 'ai' controller. Built
        # after a live session where the arm moved with no input and made a
        # bad sound: 'ai' mode had never actually been released, so the
        # robot's own controller had been contesting the arm the whole
        # time, independent of anything teleop did. Releasing weight alone
        # is not enough on this robot -- something has to actually be in
        # charge, or nothing is holding the arm at all.
        if self.path.startswith("/estop"):
            import json as _json

            parts = []
            driver = _teleop["driver"]
            if driver is not None:
                driver.panic_stop()
                parts.append("teleop authority released")
            else:
                parts.append("teleop was not enabled")

            switcher = _estop["switcher"]
            if switcher is not None:
                try:
                    code, _ = switcher.SelectMode("ai")
                    parts.append("ai mode: " +
                                ("selected" if code == 0 else f"error {code}"))
                except Exception as exc:
                    parts.append(f"ai mode select failed: {exc}")
            else:
                parts.append("ai mode select unavailable (dry-run, or "
                             "teleop was never enabled this session)")

            detail = "; ".join(parts)
            body = _json.dumps({"ok": True, "detail": detail}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f"  ESTOP: {detail}")
            return

        if self.path.startswith("/action"):
            from urllib.parse import parse_qs, urlparse
            import json as _json
            q = parse_qs(urlparse(self.path).query)
            name = q.get("name", [""])[0]

            # /action fires a canned action through ExecuteAction -- the
            # robot's own high-level motion system, a completely different
            # control channel from rt/lowcmd, which teleop streams to the
            # SAME arm joints continuously while a hand is engaged. Confirmed
            # on hardware: firing one while the other held authority got the
            # action rejected (code 7404) and made the motors make a bad
            # sound -- two controllers actually fighting over the same
            # joints, not a hypothetical. The request carries no hand, so
            # there is no way to know here whether the specific arm this
            # action would move is the one under teleop authority; refusing
            # whenever ANY hand is engaged is the conservative call, same
            # reasoning as the rest of this file's "when in doubt, don't
            # send it" defaults.
            driver = _teleop["driver"]
            if driver is not None:
                st = driver.status()
                if st.get("weight", 0.0) > 1e-3:
                    detail = ("refused: teleop has an arm engaged (weight="
                              f"{st['weight']:.2f}) -- release grip first, "
                              "then retry. Firing a canned action while "
                              "teleop holds a joint puts two controllers on "
                              "it at once")
                    body = _json.dumps({"ok": False, "detail": detail,
                                        "action": name}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    print(f"  action {name}: {detail}")
                    return

            ok, detail = trigger_action(name, self.server.arm_iface)
            body = _json.dumps({"ok": ok, "detail": detail, "action": name}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f"  action {name}: {detail}")
            return

        # One controller pose per call, at whatever rate the headset polls
        # (throttled client-side). Every call answers with what the driver
        # actually did with it -- engaged/frozen, current angles -- same
        # "tell the operator what was committed" rule as /drive.
        if self.path.startswith("/teleop"):
            from urllib.parse import parse_qs, urlparse

            driver = _teleop["driver"]
            if driver is None:
                self._json({"ok": False,
                           "detail": "teleop disabled; restart with --teleop"})
                return
            q = parse_qs(urlparse(self.path).query)
            # Read-only status: /teleop?status=1 reports without commanding
            # anything. Every other path through here calls driver.command(),
            # so polling this endpoint to "just check" was itself injecting a
            # pose -- harmless when nothing else was talking, but during a
            # live headset session it fights the headset's own updates and
            # the watchdog sees a stream that never goes stale.
            if q.get("status", ["0"])[0] == "1":
                self._json({"ok": True, **driver.status()})
                return

            hand = q.get("hand", [""])[0]
            if hand not in ("left", "right"):
                self._json({"ok": False, "detail": f"bad hand {hand!r}"})
                return

            def axis(name: str, default: float = 0.0) -> float:
                try:
                    return float(q.get(name, [str(default)])[0])
                except ValueError:
                    return default

            pos = (axis("px"), axis("py"), axis("pz"))
            quat = (axis("qw", 1.0), axis("qx"), axis("qy"), axis("qz"))
            grip = q.get("grip", ["0"])[0] == "1"
            driver.command(hand, pos, quat, grip)
            self._json({"ok": True, **driver.status()})
            return

        # Recording is armed by --record but only actually writing anything
        # between an explicit start and stop, same opt-in-per-session model
        # as viewer.py's own 'r' key.
        if self.path.startswith("/record"):
            from urllib.parse import parse_qs, urlparse

            recorder = _recorder["instance"]
            if recorder is None:
                self._json({"ok": False,
                           "detail": "recording disabled; restart with --record"})
                return
            cmd = parse_qs(urlparse(self.path).query).get("cmd", ["status"])[0]
            if cmd == "status":
                self._json({"ok": True, **recorder.status()})
                return
            elif cmd == "start":
                ok, detail = recorder.start()
            elif cmd == "stop":
                ok, detail = recorder.stop()
            else:
                ok, detail = False, f"unknown cmd {cmd!r}"
            self._json({"ok": ok, "detail": detail})
            print(f"  record {cmd}: {detail}")
            return

        if self.path == "/flat":
            body = FLAT_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            last = -1
            try:
                while not _stop.is_set():
                    with _lock:
                        # Wait for a genuinely new frame rather than resending
                        # the same one, which would waste headset bandwidth.
                        _lock.wait_for(
                            lambda: _latest["seq"] != last or _stop.is_set(), 5.0)
                        jpeg, last = _latest["jpeg"], _latest["seq"]
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_error(404)


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer that does not shout when a headset walks away.

    The Quest opens and drops keep-alive connections constantly -- switching
    apps, sleeping, reloading -- and each one ends as a BrokenPipe or a reset
    inside readline(). The default handler dumps a full traceback per event,
    which buries the messages that matter.
    """

    daemon_threads = True

    # Set by main() when TLS is on. The handshake is done per-connection in
    # process_request_thread below, NOT by wrapping the listening socket --
    # see r1vision/tls.context() for why that difference wedged the server.
    ssl_ctx = None
    handshake_timeout = 10.0

    def process_request_thread(self, request, client_address):
        if self.ssl_ctx is not None:
            try:
                # Bound the handshake: a client that connects and then says
                # nothing costs one worker thread for this long, instead of
                # costing every future connection as it did in the accept loop.
                request.settimeout(self.handshake_timeout)
                request = self.ssl_ctx.wrap_socket(request, server_side=True)
                request.settimeout(None)
            except Exception:
                # A failed handshake is routine here (speculative connections,
                # certificate rejections, a headset going to sleep mid-connect)
                # and must never take the server down with it.
                self.shutdown_request(request)
                return
        super().process_request_thread(request, client_address)

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError,
                            ConnectionAbortedError, TimeoutError)):
            return
        import ssl
        if isinstance(exc, ssl.SSLError):
            return
        super().handle_error(request, client_address)


def capture(source_spec: str, rotate: int, quality: int, scale: float) -> None:
    """Pull from the robot and keep the newest JPEG ready to serve."""
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    while not _stop.is_set():
        try:
            src = build_source(source_spec)
            src.open()
            print(f"  connected: {src.name}")
        except Exception as exc:
            print(f"  connect failed: {exc}; retrying in 3s")
            time.sleep(3)
            continue

        misses = 0
        stamps: list[float] = []
        try:
            while not _stop.is_set():
                frame = src.read()
                if frame is None:
                    misses += 1
                    if misses > 60:
                        print("  stream dropped; reconnecting")
                        break
                    continue
                if frame.size == 0:
                    continue
                misses = 0

                if rotate:
                    frame = cv2.rotate(frame, rotations[rotate])
                if scale != 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)

                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ok:
                    continue

                stamps.append(time.monotonic())
                if len(stamps) > 30:
                    stamps.pop(0)

                with _lock:
                    _latest["jpeg"] = buf.tobytes()
                    _latest["seq"] = int(_latest["seq"]) + 1
                    if len(stamps) > 1:
                        span = stamps[-1] - stamps[0]
                        _latest["fps"] = (len(stamps) - 1) / span if span else 0.0
                    _lock.notify_all()
        finally:
            src.close()


def local_addresses() -> list[tuple[str, str]]:
    """(interface, ipv4) for anything a headset might reach us on."""
    # Delegates to r1vision.unitree_dds.interfaces(), which uses psutil rather
    # than shelling out to ifconfig -- that binary does not exist on Windows,
    # and duplicating a second ifconfig-based implementation here is exactly
    # how this class of bug reappears after being fixed once already.
    from r1vision.unitree_dds import interfaces

    return interfaces()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--source", default=DEFAULT_SOURCE)
    ap.add_argument("-p", "--port", type=int, default=8080)
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="the raw stream is already upright; use only if that changes")
    ap.add_argument("--quality", type=int, default=75, help="JPEG quality 1-100")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--arm-iface", default="auto",
                    help="interface for arm control (ethernet to the robot)")
    ap.add_argument("--no-control", action="store_true",
                    help="video only; disable the /action endpoint")
    ap.add_argument("--walk", action="store_true",
                    help="enable driving from the thumbsticks. Off by default: "
                         "starting the video feed should never be able to make "
                         "a humanoid take a step")
    ap.add_argument("--walk-speed", type=float, default=1.0,
                    help="scale on the speed caps in locomotion.py (0.5 = half)")
    ap.add_argument("--backend", default="sport",
                    choices=("sport", "wireless"),
                    help="how walking commands reach the robot: sport = "
                         "LocoClient.SetVelocity (returns 127 on this R1); "
                         "wireless = publish to rt/wirelesscontroller, the "
                         "channel the handheld remote drives")
    ap.add_argument("--loco-service", default="auto",
                    help="sport backend only: loco, sport, or auto to ask the "
                         "robot which one answers")
    ap.add_argument("--deadman", action="store_true",
                    help="also require a grip button held while driving. Off by "
                         "default so the sticks behave like Unitree's controller; "
                         "the spring-centred stick is already the deadman")
    ap.add_argument("--teleop", action="store_true",
                    help="enable VR controller -> arm IK teleop. Off by default: "
                         "starting the video feed should never be able to move "
                         "an arm")
    ap.add_argument("--teleop-hands", default="both",
                    choices=("left", "right", "both"),
                    help="which hand(s) the controllers drive (default both)")
    ap.add_argument("--teleop-scale", type=float, default=1.0,
                    help="controller-motion-to-arm-motion gain (0.5 = half size)")
    ap.add_argument("--teleop-dry-run", action="store_true",
                    help="run the whole VR teleop path -- headset, poses, IK, "
                         "HUD, haptics -- but never send anything to a robot. "
                         "The missing rung between the keyboard test and live "
                         "VR: it is the only way to check that moving your hand "
                         "a given direction moves the arm that direction "
                         "(AXIS_MAP) without a robot being able to move at all")
    ap.add_argument("--record", action="store_true",
                    help="enable the /record endpoint (video + pose + joint "
                         "angle logging). Recording itself still needs an "
                         "explicit start, from the headset or /record?cmd=start")
    ap.add_argument("--http", action="store_true",
                    help="plain HTTP (no VR: WebXR needs a secure context)")
    args = ap.parse_args()

    print(f"pulling {args.source}")
    threading.Thread(
        target=capture,
        args=(args.source, args.rotate, args.quality, args.scale),
        daemon=True,
    ).start()

    if args.walk:
        from r1vision.locomotion import LocoDriver

        driver = LocoDriver(iface=args.arm_iface, scale=args.walk_speed,
                            service=args.loco_service, backend=args.backend)
        driver.start()
        _loco["driver"] = driver
        print(f"  walking ENABLED: max {driver.max_vx:.2f} m/s forward, "
              f"{driver.max_w:.2f} rad/s yaw")
        if args.deadman:
            print("  hold a grip button AND push a stick to drive")
        else:
            print("  push a stick to walk, release to stop (it keeps standing)")

    if args.teleop:
        from r1vision.arm_teleop import HANDS as TELEOP_HANDS, ArmTeleopDriver

        hands = TELEOP_HANDS if args.teleop_hands == "both" else (args.teleop_hands,)
        driver = ArmTeleopDriver(iface=args.arm_iface, hands=hands,
                                 scale=args.teleop_scale,
                                 dry_run=args.teleop_dry_run)

        if not args.teleop_dry_run:
            # Set up the STOP button's mode-select BEFORE driver.start(),
            # not after. Doing it after failed on hardware with
            # `KeyError: 'cyclonedds'` from deep inside importlib -- a
            # classic CPython concurrent-import bug: driver.start() launches
            # a thread whose first action is importing unitree_sdk2py /
            # cyclonedds, and this code was importing the same modules for
            # the first time in the main thread at nearly the same instant.
            # Two threads racing to first-import one module corrupts
            # sys.modules. Reproduced standalone before believing this was
            # the actual cause, not guessed at. Doing the import here, before
            # any other thread exists to race it, is the actual fix -- an
            # explicit init_channel() call alone (tried first) did not
            # change anything, because it was never a channel-ordering
            # problem in the first place.
            try:
                from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                    MotionSwitcherClient,
                )

                from r1vision.unitree_dds import init_channel

                init_channel(args.arm_iface, peers=None)
                switcher = MotionSwitcherClient()
                switcher.SetTimeout(5.0)
                switcher.Init()
                # MotionSwitcherClient needs a moment after Init() before a
                # request actually succeeds (DDS publisher/subscriber
                # matching takes a beat), and an emergency stop is exactly
                # the wrong place to be waiting on that for the first time.
                time.sleep(2.0)
                _estop["switcher"] = switcher
                # Matches the button xr_page.py actually binds STOP to for
                # this hand set -- left grip when left is idle, the A/X
                # button once both hands are claimed by their own clutch.
                # Printing "left grip" unconditionally here was wrong the
                # moment both-hands teleop existed: telling the operator the
                # wrong stop button is worse than telling them none at all.
                stop_button = ("A/X button" if "left" in hands and "right" in hands
                              else "left grip")
                print(f"  STOP ready ({stop_button} in VR, or GET /estop): "
                      "exits VR and hands the arm back to the robot's own "
                      "'ai' mode")
            except Exception as exc:
                print(f"  STOP button's mode-select is unavailable ({exc}) -- "
                      "/estop will still release teleop authority, just not "
                      "re-select ai mode")

        driver.start()
        _teleop["driver"] = driver
        if args.teleop_dry_run:
            print(f"  arm teleop DRY RUN: hands {hands}, "
                  f"scale {args.teleop_scale:.2f}")
            print("  the full path runs, but NOTHING is sent to a robot")
        else:
            print(f"  arm teleop ENABLED: hands {hands}, "
                  f"scale {args.teleop_scale:.2f}")
        print("  grip engages that hand's clutch; release freezes it in place")

    if args.record:
        from r1vision.recorder import EpisodeRecorder

        status_fn = _teleop["driver"].status if _teleop["driver"] else (lambda: {})
        _recorder["instance"] = EpisodeRecorder(_latest_frame, status_fn)
        print("  recording ENABLED: start/stop from the headset "
             "(thumbstick click) or /record?cmd=start|stop")

    server = Server(("0.0.0.0", args.port), Handler)
    server.arm_iface = args.arm_iface
    server.deadman = args.deadman
    if args.no_control:
        globals()["trigger_action"] = lambda *a, **k: (False, "control disabled")

    addrs = [ip for _, ip in local_addresses()]
    scheme = "http"
    if not args.http:
        # navigator.xr is undefined on insecure origins, so the Quest reports
        # "WebXR not available" over plain http no matter what it supports.
        try:
            from r1vision.tls import context
            server.ssl_ctx = context(addrs)
            scheme = "https"
            print("  TLS enabled (self-signed) so WebXR is available")
        except Exception as exc:
            print(f"  TLS setup failed ({exc}); falling back to http -- VR will not work")

    print(f"\nserving on port {args.port}. Open ONE of these in the Quest browser:\n")
    for iface, ip in local_addresses():
        note = ""
        if ip.startswith("192.168.12."):
            note = "   <-- robot's Wi-Fi AP: use this if the Quest joined it"
        elif ip.startswith("192.168.123."):
            note = "   (robot ethernet segment - the Quest cannot reach this)"
        print(f"  {scheme}://{ip}:{args.port}{note}    [{iface}]")

    if scheme == "https":
        print("\n  The Quest will warn about the certificate: tap Advanced ->")
        print("  Proceed. That is expected for a self-signed cert on a LAN.")
    print("\n  In the Quest: browser -> that URL -> Enter VR.")
    print("  Ctrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        # Flush any in-progress episode before the sources it reads from go
        # away, then zero the arms, then zero locomotion -- each of those
        # close() calls already blocks until its own state is safe.
        if _recorder["instance"] is not None and _recorder["instance"].recording:
            print("  stopping recorder ...")
            _recorder["instance"].close()
        if _teleop["driver"] is not None:
            print("  releasing arms ...")
            _teleop["driver"].close()
        if _loco["driver"] is not None:
            print("  stopping robot ...")
            _loco["driver"].close()
        _stop.set()
        with _lock:
            _lock.notify_all()
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
