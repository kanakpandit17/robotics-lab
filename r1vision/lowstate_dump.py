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
    # float(argv[2]) directly turns a mistyped argument into "could not convert string to
    # float: '-m'", which names neither the argument nor this script. Seen in practice.
    seconds = 8.0
    if len(argv) > 2:
        try:
            seconds = float(argv[2])
        except ValueError:
            print(f"lowstate_dump: third argument must be a duration in seconds, "
                  f"got {argv[2]!r}", file=sys.stderr)
            print("usage: lowstate_dump.py <iface> [peers] [seconds]", file=sys.stderr)
            return 2

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

    # tick is the only clock the robot itself stamps onto the message. There is no
    # wall-clock field, so anything correlating lowstate against another source has to
    # pair tick with a local receive time -- the same rule the MQTT bridge follows.
    print(f"TICK {getattr(msg, 'tick', 'ABSENT')}")
    print(f"MODE_MACHINE {getattr(msg, 'mode_machine', 'ABSENT')} "
          f"MODE_PR {getattr(msg, 'mode_pr', 'ABSENT')}")

    imu = getattr(msg, "imu_state", None)
    if imu is not None and hasattr(imu, "rpy"):
        r, p, y = imu.rpy
        print(f"IMU_RPY {r:.4f} {p:.4f} {y:.4f}")
        print("IMU_QUAT " + " ".join(f"{v:.4f}" for v in imu.quaternion))
        print("IMU_GYRO " + " ".join(f"{v:.4f}" for v in imu.gyroscope))
        print("IMU_ACCEL " + " ".join(f"{v:.4f}" for v in imu.accelerometer))
        print(f"IMU_TEMP {getattr(imu, 'temperature', 'ABSENT')}")

    # ddq/vol/sensor are printed because the IDL carries them but nothing here had ever
    # observed them populated. A column of exact zeros across all 35 motors means the
    # field is transmitted but not filled by the firmware -- which is a different
    # finding from the field being absent, and only a live capture distinguishes them.
    print("IDX      q_rad     q_deg      dq       ddq    tau_est   temp   vol      sensor")
    for i, m in enumerate(motors):
        q = getattr(m, "q", 0.0)
        dq = getattr(m, "dq", 0.0)
        ddq = getattr(m, "ddq", 0.0)
        tau = getattr(m, "tau_est", 0.0)
        vol = getattr(m, "vol", 0.0)
        temp = getattr(m, "temperature", 0)
        if isinstance(temp, (list, tuple)):
            temp = temp[0] if temp else 0
        sensor = getattr(m, "sensor", None)
        sensor = ",".join(str(s) for s in sensor) if sensor is not None else "-"
        print(f"{i:>3}  {q:>9.4f}  {q * 57.2958:>8.2f}  {dq:>7.3f}  {ddq:>8.3f}  "
              f"{tau:>8.3f}  {temp:>5}  {vol:>6.2f}  {sensor}")

    # One line per field, so a capture answers "is this usable?" without reading 35 rows.
    def _live(name, vals):
        nz = sum(1 for v in vals if v != 0)
        print(f"POPULATED {name}: {nz}/{len(vals)} non-zero")

    _live("ddq", [getattr(m, "ddq", 0.0) for m in motors])
    _live("vol", [getattr(m, "vol", 0.0) for m in motors])
    _live("tau_est", [getattr(m, "tau_est", 0.0) for m in motors])
    _live("sensor", [s for m in motors for s in (getattr(m, "sensor", None) or [])])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
