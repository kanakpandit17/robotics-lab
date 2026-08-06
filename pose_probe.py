"""Find out what the R1 can tell us about where it is.

The SLAM/navigation service documented by Unitree lives on the Jetson NX at
192.168.123.164 and needs the Mid-360 lidar at 192.168.123.120. On this robot
neither host answers ARP, so that service cannot be running. Before designing
anything around it, this asks the robot directly: which topics are alive, and
does any of them carry a position?

Run it while the robot is powered and balancing:

    .venv/bin/python pose_probe.py            # 8 s listen
    .venv/bin/python pose_probe.py 20         # longer

Everything is subscribed from one process. Creating a second DomainParticipant
after the Unitree SDK has built its own silently kills the subscriptions, which
is why this is not several small scripts.
"""

from __future__ import annotations

import sys
import time

# (topic, module path, type name, why we care)
CANDIDATES = [
    # --- what we know works: joints and IMU -----------------------------
    ("rt/lowstate", "unitree_hg", "LowState_",
     "joints + IMU (no absolute position)"),
    ("rt/lf/lowstate", "unitree_hg", "LowState_",
     "low-frequency mirror of the above"),

    # --- the interesting ones: fused body position ----------------------
    ("rt/sportmodestate", "unitree_go", "SportModeState_",
     "LEG ODOMETRY -- position[3] by dead reckoning"),
    ("rt/lf/sportmodestate", "unitree_go", "SportModeState_",
     "low-frequency mirror"),
    ("rt/odommodestate", "unitree_go", "SportModeState_",
     "odometry variant on some firmware"),

    # --- lidar-derived pose, if a lidar exists --------------------------
    ("rt/utlidar/robot_pose", "geometry_msgs", "PoseStamped_",
     "built-in lidar localisation"),

    # --- the SLAM service (expected dead: no NX, no Mid-360) ------------
    ("rt/unitree/slam_relocation/odom", "nav_msgs", "Odometry_",
     "SLAM pose -- needs NX .164 + lidar .120"),
    ("rt/unitree/slam_mapping/odom", "nav_msgs", "Odometry_",
     "SLAM mapping pose"),
    ("rt/slam_info", "std_msgs", "String_",
     "SLAM status JSON"),
]


def _load(module: str, name: str):
    """Import a DDS type by module family, or None if this SDK lacks it."""
    if module in ("unitree_go", "unitree_hg"):
        path = f"unitree_sdk2py.idl.{module}.msg.dds_"
    else:
        path = f"unitree_sdk2py.idl.{module}.msg.dds_"
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name)
    except (ImportError, AttributeError):
        return None


def summarise(topic: str, msg) -> str:
    """One line describing what this message says about position."""
    try:
        # SportModeState: the one that would carry dead-reckoned position.
        pos = getattr(msg, "position", None)
        if pos is not None and len(pos) >= 3:
            rpy = getattr(getattr(msg, "imu_state", None), "rpy", None)
            yaw = f" yaw={rpy[2]:+.3f}" if rpy else ""
            mode = getattr(msg, "mode", "?")
            return (f"position=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
                    f"{yaw} mode={mode}")

        # nav_msgs/Odometry and geometry_msgs/PoseStamped.
        pose = getattr(msg, "pose", None)
        if pose is not None:
            inner = getattr(pose, "pose", pose)
            p = getattr(inner, "position", None)
            if p is not None:
                return f"position=({p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f})"

        # std_msgs/String -- the SLAM status JSON.
        data = getattr(msg, "data", None)
        if isinstance(data, str):
            return f"data={data[:110]}"

        # LowState: no absolute position, but prove the IMU is alive.
        imu = getattr(msg, "imu_state", None)
        if imu is not None and getattr(imu, "rpy", None):
            r, p_, y = imu.rpy
            return f"imu rpy=({r:+.3f}, {p_:+.3f}, {y:+.3f})  [no position]"
    except Exception as exc:  # a live topic we cannot read is still a finding
        return f"<unreadable: {type(exc).__name__}: {exc}>"
    return "<alive, no position field>"


def main(argv: list[str]) -> int:
    seconds = float(argv[0]) if argv else 8.0

    from unitree_sdk2py.core.channel import ChannelSubscriber

    from r1vision.unitree_dds import default_peers, init_channel, robot_iface

    iface = robot_iface("auto")
    peers = default_peers()
    print(f"interface {iface}, peers {', '.join(peers) or '(multicast only)'}")
    init_channel(iface, peers=peers)

    seen: dict[str, dict] = {}
    subs = []

    def make_handler(topic: str):
        def handler(msg):
            rec = seen.setdefault(topic, {"n": 0, "last": None})
            rec["n"] += 1
            rec["last"] = msg
        return handler

    skipped = []
    for topic, module, name, _why in CANDIDATES:
        dtype = _load(module, name)
        if dtype is None:
            skipped.append(f"{topic} ({module}/{name} not in this SDK)")
            continue
        try:
            sub = ChannelSubscriber(topic, dtype)
            sub.Init(make_handler(topic), 10)
            subs.append(sub)
        except Exception as exc:
            skipped.append(f"{topic} ({type(exc).__name__}: {exc})")

    print(f"listening on {len(subs)} topics for {seconds:.0f}s ...\n")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.2)

    live, dead = [], []
    for topic, _m, _n, why in CANDIDATES:
        rec = seen.get(topic)
        if rec and rec["n"]:
            live.append((topic, rec, why))
        else:
            dead.append((topic, why))

    print("=" * 74)
    if live:
        print(f"LIVE ({len(live)})")
        for topic, rec, why in live:
            hz = rec["n"] / seconds
            print(f"  {topic}")
            print(f"      {rec['n']} msgs (~{hz:.1f} Hz)   -- {why}")
            print(f"      {summarise(topic, rec['last'])}")
    else:
        print("LIVE (0)  -- nothing received at all.")
        print("  Robot powered on and past R2+A? Cable in the R1 switch?")

    print(f"\nSILENT ({len(dead)})")
    for topic, why in dead:
        print(f"  {topic:<38} {why}")

    if skipped:
        print(f"\nNOT SUBSCRIBABLE ({len(skipped)})")
        for s in skipped:
            print(f"  {s}")

    has_position = any(
        "position=" in summarise(t, r["last"]) for t, r, _ in live)
    print("\n" + "=" * 74)
    print("VERDICT: " + (
        "a position source exists -- teach-and-repeat is possible."
        if has_position else
        "no absolute position on the wire. See notes printed above."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
