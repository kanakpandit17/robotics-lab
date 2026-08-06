"""Dump the R1's joint layout. Runs standalone so DDS starts clean.

    python -m r1vision.lowstate_dump en0 192.168.123.161:7400,...

Must be its own process: creating a cyclonedds DomainParticipant for topic
enumeration and then letting the Unitree SDK build its own Domain(0, config)
in the same process collides, and the subscriber silently receives nothing.
"""

from __future__ import annotations

import sys
import time


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: lowstate_dump.py <iface> [peers]", file=sys.stderr)
        return 2

    iface = argv[0]
    peers = [p for p in argv[1].split(",") if p] if len(argv) > 1 else []
    seconds = float(argv[2]) if len(argv) > 2 else 8.0

    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from r1vision.unitree_dds import init_channel

    init_channel(iface, peers=peers)

    latest = {}

    def on_state(msg):
        latest["msg"] = msg

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and "msg" not in latest:
        time.sleep(0.1)

    if "msg" not in latest:
        print("NO_DATA")
        return 1

    msg = latest["msg"]
    motors = msg.motor_state
    print(f"MOTORS {len(motors)}")

    imu = getattr(msg, "imu_state", None)
    if imu is not None and hasattr(imu, "rpy"):
        r, p, y = imu.rpy
        print(f"IMU_RPY {r:.4f} {p:.4f} {y:.4f}")

    print("IDX      q_rad     q_deg      dq     tau_est   temp")
    for i, m in enumerate(motors):
        q = getattr(m, "q", 0.0)
        dq = getattr(m, "dq", 0.0)
        tau = getattr(m, "tau_est", 0.0)
        temp = getattr(m, "temperature", 0)
        if isinstance(temp, (list, tuple)):
            temp = temp[0] if temp else 0
        print(f"{i:>3}  {q:>9.4f}  {q * 57.2958:>8.2f}  {dq:>7.3f}  "
              f"{tau:>8.3f}  {temp}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
