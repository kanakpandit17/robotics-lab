#!/usr/bin/env python3
"""R1 ONBOARD image server. Runs ON THE ROBOT'S onboard computer (Linux/Jetson).

Grabs the head camera and streams JPEG frames to the PC over TCP, where r1_quest_fpv.py
(--source net) shows them in the Meta Quest 3.

Camera auto-detect order:  Intel RealSense (pyrealsense2)  ->  USB/built-in (OpenCV /dev/videoN).
Override with --backend realsense|cv2  and  --cam-index N.

Wire protocol: 4-byte big-endian length prefix + JPEG bytes, repeated.  (matches NetFrameReceiver)

Deploy (from your PC):
  scp r1_image_server.py <user>@<ROBOT_IP>:~/
  ssh <user>@<ROBOT_IP>
  python3 ~/r1_image_server.py            # then on the PC: r1_quest_fpv.py --source net --robot-ip <ROBOT_IP>

If OpenCV is missing on the robot:  pip3 install opencv-python  (or: sudo apt install python3-opencv)
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5555)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--backend", choices=["auto", "realsense", "cv2"], default="auto")
ap.add_argument("--cam-index", type=int, default=-1, help="cv2: force /dev/videoN (default: probe 0..5)")
ap.add_argument("--jpeg-quality", type=int, default=80)
args = ap.parse_args()


class RealSenseCam:
    def __init__(self, w, h, fps):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.pipe.start(cfg)

    def read(self):
        frames = self.pipe.wait_for_frames()
        c = frames.get_color_frame()
        if not c:
            return None
        return np.asanyarray(c.get_data())

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


class CV2Cam:
    def __init__(self, w, h, fps, index):
        import cv2
        self.cv2 = cv2
        indices = [index] if index >= 0 else list(range(0, 6))
        self.cap = None
        for i in indices:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)
                ok, _ = cap.read()
                if ok:
                    self.cap = cap
                    self.index = i
                    print(f"[IMG-SRV] opened /dev/video{i}", flush=True)
                    break
                cap.release()
        if self.cap is None:
            raise RuntimeError("no working /dev/videoN found (tried %s)" % indices)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


def open_camera():
    if args.backend in ("auto", "realsense"):
        try:
            cam = RealSenseCam(args.width, args.height, args.fps)
            print("[IMG-SRV] using Intel RealSense color stream", flush=True)
            return cam
        except Exception as e:
            if args.backend == "realsense":
                raise
            print(f"[IMG-SRV] no RealSense ({str(e)[:60]}); falling back to OpenCV", flush=True)
    cam = CV2Cam(args.width, args.height, args.fps, args.cam_index)
    return cam


def serve():
    import cv2
    cam = open_camera()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"[IMG-SRV] listening on 0.0.0.0:{args.port} -- start the PC viewer now. Ctrl-C to stop.", flush=True)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[IMG-SRV] PC connected from {addr}", flush=True)
            n_sent = 0
            last = time.time()
            try:
                while True:
                    frame = cam.read()
                    if frame is None:
                        time.sleep(0.005)
                        continue
                    ok, jpg = cv2.imencode(".jpg", frame, enc)
                    if not ok:
                        continue
                    data = jpg.tobytes()
                    conn.sendall(struct.pack(">I", len(data)) + data)
                    n_sent += 1
                    if time.time() - last > 2.0:
                        print(f"[IMG-SRV] streamed {n_sent} frames", flush=True)
                        last = time.time()
            except (BrokenPipeError, ConnectionResetError):
                print("[IMG-SRV] PC disconnected; waiting for a new viewer ...", flush=True)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[IMG-SRV] stopping.", flush=True)
    finally:
        cam.close()
        srv.close()


if __name__ == "__main__":
    sys.exit(serve())
