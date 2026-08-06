#!/usr/bin/env python3
"""Meta Quest 3 first-person-view server for the Unitree R1.

Runs a Vuer WebXR server on THIS PC. The Quest 3 opens the printed URL in its browser,
enters immersive VR, and sees a head-locked video feed -> "through the robot's eyes".

Frame sources (--source):
  test     : animated test pattern (no robot needed). Use this FIRST to prove the Quest sees a live feed.
  simfile  : latest JPEG written by the Isaac Sim R1 head-camera (default path below).
             -> point Isaac Sim's camera dump at the same file and you see the SIM robot's view in VR.
  (real R1 camera = Phase 2: an image_server on the robot's onboard PC writes to this same file over the net.)

Open on the Quest 3 (same WiFi as this PC):
  1) First visit  https://<THIS_PC_IP>:8012   and ACCEPT the self-signed certificate warning (one time).
  2) Then open the client URL this script prints, and press "Enter VR".

Run (Windows, env_isaaclab):
  C:\\Users\\aimat\\miniconda3\\envs\\env_isaaclab\\python.exe C:\\Users\\aimat\\Desktop\\humanoid\\r1_quest_fpv.py --source test
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np

HUMANOID = r"C:\Users\aimat\Desktop\humanoid"
CERT = os.path.join(HUMANOID, "certs", "cert.pem")
KEY = os.path.join(HUMANOID, "certs", "key.pem")
DEFAULT_FRAME = os.path.join(HUMANOID, "r1_fpv_frame.jpg")

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["test", "simfile", "net", "screen"], default="test")
parser.add_argument("--frame", default=DEFAULT_FRAME, help="simfile: path to the JPEG the camera writes")
parser.add_argument("--robot-ip", default="", help="net: IP of the R1 running r1_image_server.py")
parser.add_argument("--robot-port", type=int, default=5555, help="net: port r1_image_server.py listens on")
parser.add_argument("--monitor", type=int, default=1, help="screen: which monitor to capture (1=primary)")
parser.add_argument("--region", default="", help="screen: capture region 'left,top,width,height' (overrides --monitor)")
parser.add_argument("--port", type=int, default=8012)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args = parser.parse_args()

from vuer import Vuer                       # noqa: E402
from vuer.schemas import ImageBackground    # noqa: E402

# ---- frame sources -------------------------------------------------------
def test_frame(w: int, h: int, t: float) -> np.ndarray:
    """A clearly-animated pattern so you can confirm the feed is LIVE in the headset."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # pulsing color background
    img[:, :, 0] = int(60 + 60 * (0.5 + 0.5 * np.sin(t * 1.3)))          # R
    img[:, :, 2] = int(60 + 60 * (0.5 + 0.5 * np.sin(t * 0.7 + 2.0)))    # B
    # a bright vertical bar sweeping left<->right (obvious motion)
    x = int((0.5 + 0.5 * np.sin(t * 1.1)) * (w - 80))
    img[:, x:x + 80, :] = 255
    # a horizon line + moving marker so orientation is readable in VR
    img[h // 2 - 3:h // 2 + 3, :, :] = 200
    y = int((0.5 + 0.5 * np.cos(t * 0.9)) * (h - 60)) + 30
    img[y - 15:y + 15, w // 2 - 15:w // 2 + 15, 1] = 255                 # green box, vertical bob
    return img


def _wait_frame(w: int, h: int, t: float, msg: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = 40
    b = int(120 + 120 * (0.5 + 0.5 * np.sin(t * 3)))                     # blinking strip = "alive, waiting"
    img[h // 2 - 20:h // 2 + 20, :, 1] = b
    return img


_last = {"img": None, "mtime": 0.0}
def simfile_frame(w: int, h: int, t: float, path: str) -> np.ndarray:
    try:
        m = os.path.getmtime(path)
        if m != _last["mtime"]:
            import cv2
            bgr = cv2.imread(path)
            if bgr is not None:
                _last["img"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                _last["mtime"] = m
    except Exception:
        pass
    if _last["img"] is None:
        return _wait_frame(w, h, t, "waiting for sim camera")
    return _last["img"]


# ---- network source: receive JPEG frames from the R1's onboard image server -------------
class NetFrameReceiver:
    """Connects to r1_image_server.py on the robot and keeps the latest frame (RGB uint8).
    Wire protocol: 4-byte big-endian length prefix + JPEG bytes, repeated."""

    def __init__(self, ip: str, port: int):
        import threading
        self.ip, self.port = ip, port
        self.latest = None
        self.connected = False
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _recv_all(self, sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _run(self):
        import socket
        import struct

        import cv2
        while not self._stop:
            try:
                s = socket.create_connection((self.ip, self.port), timeout=5)
                s.settimeout(5)
                self.connected = True
                print(f"[R1-FPV] connected to robot image server {self.ip}:{self.port}", flush=True)
                while not self._stop:
                    hdr = self._recv_all(s, 4)
                    if hdr is None:
                        break
                    (n,) = struct.unpack(">I", hdr)
                    data = self._recv_all(s, n)
                    if data is None:
                        break
                    arr = np.frombuffer(data, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        self.latest = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.connected = False
                print(f"[R1-FPV] robot conn lost/retrying: {str(e)[:60]}", flush=True)
                time.sleep(1.5)


# ---- screen source: capture a mirrored phone (scrcpy) region and stream it -------------
_sct = None
def screen_frame(t: float) -> np.ndarray:
    global _sct
    import mss
    if _sct is None:
        _sct = mss.mss()
    if args.region:
        l, tp, w, h = (int(x) for x in args.region.split(","))
        mon = {"left": l, "top": tp, "width": w, "height": h}
    else:
        mon = _sct.monitors[args.monitor]
    img = _sct.grab(mon)
    arr = np.asarray(img)                    # BGRA
    return np.ascontiguousarray(arr[:, :, [2, 1, 0]])   # -> RGB


_net = None
def get_frame(t: float) -> np.ndarray:
    global _net
    if args.source == "screen":
        return screen_frame(t)
    if args.source == "net":
        if _net is None:
            if not args.robot_ip:
                raise SystemExit("--source net requires --robot-ip <R1 IP>")
            _net = NetFrameReceiver(args.robot_ip, args.robot_port)
        if _net.latest is None:
            return _wait_frame(args.width, args.height, t, "waiting for robot camera")
        return _net.latest
    if args.source == "simfile":
        return simfile_frame(args.width, args.height, t, args.frame)
    return test_frame(args.width, args.height, t)


# ---- Vuer server ---------------------------------------------------------
app = Vuer(host="0.0.0.0", port=args.port, cert=CERT, key=KEY)


@app.spawn(start=True)
async def main(session):
    print("[R1-FPV] Quest 3 client connected. Streaming frames ...", flush=True)
    t0 = time.time()
    dt = 1.0 / args.fps
    while True:
        t = time.time() - t0
        frame = get_frame(t)
        session.upsert(
            ImageBackground(
                frame,
                format="jpeg",
                quality=80,
                key="r1-fpv",
                interpolate=True,
                aspect=args.width / args.height,
                distanceToCamera=2,
                position=[0, 1.3, -2],
                rotation=[0, 0, 0],
            ),
            to="bgChildren",
        )
        await asyncio.sleep(dt)
