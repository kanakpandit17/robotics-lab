"""Live robot state from rosbridge, for robots whose SDK cannot be reached.

This exists because of a measured fact about this cell: MAiRA's CORBA/omniORB
port is closed (2809 and 2800-2830 all refuse), so every `neurapy` read --
joint angles, TCP pose, mode -- fails with TRANSIENT_ConnectFailed. But the
controller runs rosbridge on :9090 and publishes `/joint_states` at full rate,
which WSL reaches without trouble.

So telemetry comes from ROS and commands go through the SDK. They are different
transports with different failure modes, and treating them separately is what
lets the dashboard show live motion while motion commands are still blocked.

The ROS boundary stops here
---------------------------
This module is the only place in the project that knows ROS exists, and it runs
on the workstation. The laptop still receives ordinary fleet telemetry over
MQTT and has no idea where the numbers came from. That was the point of the
architecture, and it survives contact with the real robot.
"""

from __future__ import annotations

import json
import logging
import threading
import time

log = logging.getLogger(__name__)

# Below this, joint velocity is sensor noise rather than motion. The arm at
# rest reads ~1e-5 rad/s; hand-guiding it is orders of magnitude above that.
MOVING_RAD_S = 0.01

DEFAULT_TOPICS = ("/joint_states", "/error_msg", "/common_status")


class RosBridgeTelemetry:
    """Keeps the newest message from a few topics, and reconnects on its own.

    One background thread owns the websocket. `snapshot()` is called from the
    agent's state publisher and never blocks on the network -- it reads whatever
    arrived last, which is the right trade for a status feed.
    """

    def __init__(self, host: str, port: int = 9090,
                 topics: tuple[str, ...] = DEFAULT_TOPICS,
                 throttle_ms: int = 100):
        self.host = host
        self.port = port
        self.topics = tuple(topics)
        self.throttle_ms = throttle_ms

        self._lock = threading.Lock()
        self._latest: dict[str, dict] = {}
        self._connected = False
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- control
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="rosbridge")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------- loop
    def _run(self) -> None:
        try:
            import websocket
        except ImportError:
            self._error = "pip install websocket-client"
            log.error("rosbridge telemetry disabled: %s", self._error)
            return

        backoff = 1.0
        while not self._stop.is_set():
            try:
                ws = websocket.create_connection(
                    f"ws://{self.host}:{self.port}", timeout=8)
                ws.settimeout(5)
                for topic in self.topics:
                    ws.send(json.dumps({"op": "subscribe", "topic": topic,
                                        "throttle_rate": self.throttle_ms}))
                self._connected = True
                self._error = None
                backoff = 1.0
                log.info("rosbridge connected to %s:%s, %d topics",
                         self.host, self.port, len(self.topics))

                while not self._stop.is_set():
                    try:
                        frame = json.loads(ws.recv())
                    except Exception as exc:
                        # A read timeout is normal on a quiet topic; anything
                        # else means the link went away and should reconnect.
                        if "timed out" in str(exc).lower():
                            continue
                        raise
                    topic = frame.get("topic")
                    if topic and "msg" in frame:
                        with self._lock:
                            self._latest[topic] = frame["msg"]
                ws.close()
            except Exception as exc:
                self._connected = False
                self._error = f"{type(exc).__name__}: {exc}"[:160]
                log.warning("rosbridge %s:%s unavailable (%s); retry in %.0fs",
                            self.host, self.port, self._error, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    # ------------------------------------------------------------- read side
    def snapshot(self) -> dict:
        """Everything worth publishing, shaped for the fleet's telemetry dict."""
        with self._lock:
            latest = dict(self._latest)

        out: dict = {"ros_connected": self._connected}
        if self._error:
            out["ros_error"] = self._error

        joints = latest.get("/joint_states")
        if joints:
            positions = list(joints.get("position") or [])
            velocities = list(joints.get("velocity") or [])
            out["joints"] = [round(v, 5) for v in positions]
            out["joint_names"] = list(joints.get("name") or [])
            out["joint_velocities"] = [round(v, 5) for v in velocities]
            # A motion indicator the SDK cannot currently give us: the arm is
            # moving if any joint is turning faster than sensor noise.
            peak = max((abs(v) for v in velocities), default=0.0)
            out["moving"] = peak > MOVING_RAD_S
            out["peak_joint_speed"] = round(peak, 5)
            stamp = (joints.get("header") or {}).get("stamp") or {}
            if stamp:
                out["joints_age_s"] = round(
                    max(0.0, time.time() - stamp.get("secs", 0)), 1)

        for topic, key in (("/error_msg", "ros_error_msg"),
                           ("/common_status", "ros_status")):
            msg = latest.get(topic)
            if msg:
                out[key] = msg.get("data", msg) if isinstance(msg, dict) else msg
        return out
