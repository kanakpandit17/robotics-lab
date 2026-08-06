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

    def log_message(self, *a):
        pass

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
            body = render_page(_loco["driver"] is not None,
                               self.server.deadman).encode()
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

        if self.path.startswith("/action"):
            from urllib.parse import parse_qs, urlparse
            import json as _json
            q = parse_qs(urlparse(self.path).query)
            name = q.get("name", [""])[0]
            ok, detail = trigger_action(name, self.server.arm_iface)
            body = _json.dumps({"ok": ok, "detail": detail, "action": name}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f"  action {name}: {detail}")
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
    import subprocess

    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    found, iface = [], ""
    for line in out.splitlines():
        if line and not line[0].isspace():
            iface = line.split(":")[0]
        elif line.strip().startswith("inet ") and "127.0.0.1" not in line:
            ip = line.split()[1]
            found.append((iface, ip))
    return found


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
            from r1vision.tls import wrap
            server.socket = wrap(server.socket, addrs)
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
        # Zero the robot before anything else goes away.
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
