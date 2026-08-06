#!/usr/bin/env python3
"""Find the R1's RTSP stream path and pull a first frame.

Run while joined to the R1's Wi-Fi. No internet needed.

    python rtsp_probe.py

The deep probe found a GStreamer RTSP Server on 192.168.12.1:8551. A GStreamer
RTSP server answers DESCRIBE with 200 + SDP on a mount point that exists and
404 on one that does not, so walking a path list identifies the real mount
exactly. On a hit this grabs one frame to captures/first-frame.jpg so there is
proof it works, plus the SDP showing codec and resolution.
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

REPORT = Path("rtsp-report.txt")
_lines: list[str] = []

PATHS = [
    "/", "/test", "/stream", "/live", "/video", "/cam", "/camera",
    "/front", "/head", "/h264", "/h265", "/main", "/color", "/rgb", "/image",
    "/media", "/ch0", "/ch1", "/0", "/1", "/left", "/right", "/stereo",
    "/binocular", "/front_camera", "/head_camera", "/videostream", "/live.sdp",
    "/stream1", "/stream0", "/track0", "/unicast", "/av0", "/preview", "/sub",
    "/realtime", "/robot", "/vision", "/eye", "/eyes", "/depth", "/rgbd",
    "/video0", "/video1", "/cam0", "/cam1", "/webcam", "/mjpeg", "/raw",
    "/unitree", "/r1", "/go2", "/default", "/session", "/play", "/source",
]

PORTS = [8551, 554, 8554, 8550, 8552, 8553, 1935, 8555]


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def rtsp_request(host: str, port: int, verb: str, path: str,
                 timeout: float = 3.0) -> tuple[int, str]:
    """One RTSP request. Returns (status_code, full_response)."""
    url = f"rtsp://{host}:{port}{path}"
    req = (
        f"{verb} {url} RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "User-Agent: r1vision\r\n"
        + ("Accept: application/sdp\r\n" if verb == "DESCRIBE" else "")
        + "\r\n"
    )
    try:
        with socket.create_connection((host, port), timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(req.encode())
            chunks = []
            while len(b"".join(chunks)) < 4096:
                try:
                    part = sock.recv(2048)
                except socket.timeout:
                    break
                if not part:
                    break
                chunks.append(part)
                if b"\r\n\r\n" in b"".join(chunks) and verb != "DESCRIBE":
                    break
            data = b"".join(chunks).decode("latin-1", "replace")
    except OSError as exc:
        return 0, str(exc)
    match = re.match(r"RTSP/1\.0 (\d+)", data)
    return (int(match.group(1)) if match else 0), data


def grab_frame(url: str, out: Path) -> str:
    """Pull a single frame with ffmpeg. Returns '' on success, else the error."""
    out.parent.mkdir(exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-rtsp_transport", "tcp", "-i", url,
        "-frames:v", "1", "-q:v", "2", str(out),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    except subprocess.TimeoutExpired:
        return "timed out after 25s"
    if out.exists() and out.stat().st_size > 0:
        return ""
    return (res.stderr or "no output").strip()[:300]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.12.1")
    ap.add_argument("--ports", default="", help="comma-separated override")
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",")] if args.ports else PORTS

    log(f"r1vision RTSP probe - {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"  host: {args.host}")

    # ------------------------------------------------------------------ 1
    log()
    log("=" * 68)
    log("1. Which ports speak RTSP")
    log("=" * 68)
    live_ports = []
    for port in ports:
        code, data = rtsp_request(args.host, port, "OPTIONS", "/")
        if code:
            methods = ""
            for line in data.splitlines():
                if line.lower().startswith("public:"):
                    methods = line.split(":", 1)[1].strip()
            server = ""
            for line in data.splitlines():
                if line.lower().startswith("server:"):
                    server = line.split(":", 1)[1].strip()
            log(f"  :{port:<6} RTSP {code}  server={server or '?'}")
            if methods:
                log(f"          methods: {methods}")
            live_ports.append(port)
        elif data and "refused" not in data.lower():
            log(f"  :{port:<6} no RTSP ({data[:60]})")
    if not live_ports:
        log("  Nothing answered RTSP. Is the robot on and the Wi-Fi joined?")
        REPORT.write_text("\n".join(_lines) + "\n")
        return 1

    # ------------------------------------------------------------------ 2
    log()
    log("=" * 68)
    log("2. Walking mount points")
    log("=" * 68)
    hits: list[tuple[int, str, str]] = []
    for port in live_ports:
        log(f"\n  -- port {port} --")
        def describe(path, _p=port):
            return path, rtsp_request(args.host, _p, "DESCRIBE", path)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for path, (code, data) in pool.map(describe, PATHS):
                if code == 200:
                    log(f"  200 OK   {path}")
                    hits.append((port, path, data))
                elif code == 401:
                    log(f"  401 auth {path}  (exists, needs credentials)")
                    hits.append((port, path, data))
                elif code and code not in (404, 400):
                    log(f"  {code}      {path}")
        if not hits:
            log("  every path returned 404")

    # ------------------------------------------------------------------ 3
    log()
    log("=" * 68)
    log("3. Stream details")
    log("=" * 68)
    if not hits:
        log("  No mount point found. The server is running but exposes nothing,")
        log("  which usually means the video pipeline is started on demand by")
        log("  the phone app. Try: open the Unitree app, start the camera view,")
        log("  leave it running, then re-run this script.")
        REPORT.write_text("\n".join(_lines) + "\n")
        return 1

    for port, path, sdp in hits:
        url = f"rtsp://{args.host}:{port}{path}"
        log(f"\n  {url}")
        for line in sdp.splitlines():
            if line.startswith(("m=", "a=rtpmap", "a=framesize", "a=fmtp", "a=control")):
                log(f"      {line.strip()}")

    # ------------------------------------------------------------------ 4
    log()
    log("=" * 68)
    log("4. Grabbing a frame")
    log("=" * 68)
    for port, path, _ in hits:
        url = f"rtsp://{args.host}:{port}{path}"
        out = Path("captures") / f"first-frame{path.replace('/', '_') or '_root'}.jpg"
        log(f"  {url} -> {out.name}")
        err = grab_frame(url, out)
        if not err:
            size = out.stat().st_size
            log(f"    SUCCESS: {size} bytes written to {out}")
            log()
            log(f"  Now run:  python -m r1vision.view --source {url}")
        else:
            log(f"    failed: {err}")

    REPORT.write_text("\n".join(_lines) + "\n")
    print(f"\n--- written to {REPORT.resolve()} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
