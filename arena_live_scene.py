"""One Isaac Sim scene holding all three ARENA2036 shopfloor robots at their real positions.

    LARA   -- live, mirrors the real robot via lara_bridge.json (lara_real_reader.py in WSL)
    MAiRA  -- live, mirrors the real robot via MQTT (its agent runs on the workstation PC)
    KUKA   -- static geometry only, no articulation, no motion (deliberate, per request)

Run:
    python arena_live_scene.py                    # GUI, both cobots live
    python arena_live_scene.py --headless         # no window, same behaviour
    python arena_live_scene.py --fence            # draw the KUKA cell fence outline

Why the two cobots use completely different transports
------------------------------------------------------
LARA's ethernet is plugged into THIS laptop, so lara_real_reader.py can talk to it directly
through neurapy (in WSL) and drop joint angles in a shared file. MAiRA's ethernet is plugged
into a different machine, so its joints arrive over the fleet MQTT bus instead -- published by
the edge agent already running on that PC as the `telemetry.joints` field of the retained
`fleet/v1/<robot_id>/state` message (see fleet/agent/edge_agent.py:publish_state and
fleet/fleetcore/envelope.py:StateMessage). Nothing new has to run on the workstation; this
script is purely another subscriber on a bus that already exists.

Both paths are READ-ONLY. This script never commands either real robot.

Positions
---------
Every coordinate here is copied from r1_lara_maira_loop.py, where it was measured in the GUI
against the real scan rather than guessed -- see that file's own comments. They are not
re-derived here, so that the two scenes cannot drift apart.
"""
import argparse
import json
import threading
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--fence", action="store_true", default=False,
                    help="draw the visual-only KUKA cell fence (no colliders), same outline "
                         "r1_lara_maira_loop.py builds")
parser.add_argument("--lara-bridge", type=str,
                    default=r"C:\Users\aimat\Desktop\humanoid\lara_bridge.json",
                    help="file lara_real_reader.py writes; set to '' to leave LARA static")
parser.add_argument("--mqtt-host", type=str, default="127.0.0.1",
                    help="fleet broker. Use the LAN address if a second mosquitto is squatting "
                         "on loopback (see MAIRA_MQTT_SETUP notes in the repo)")
parser.add_argument("--mqtt-port", type=int, default=1883)
parser.add_argument("--maira-id", type=str, default="maira-01",
                    help="robot_id in fleet/v1/<id>/state -- matches fleet/config/maira-agent.json")
parser.add_argument("--no-maira", action="store_true", default=False,
                    help="skip the MQTT subscription entirely, leave MAiRA static")
parser.add_argument("--max-age", type=float, default=2.0,
                    help="seconds -- hold the last pose rather than follow a reading older than "
                         "this. MQTT state is retained, so without this a robot that went away "
                         "hours ago would still look live.")
args_cli, _ = parser.parse_known_args()

# h5py stub: Isaac Sim's bundled hdf5.dll shadows h5py -> DLL error loading isaaclab_tasks.
# Same workaround as maira_sim_bridge.py / lara_sim_bridge.py.
import importlib.machinery as _mach  # noqa: E402
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_h5 = _types.ModuleType("h5py")
_h5.__spec__ = _mach.ModuleSpec("h5py", loader=None)
_sys.modules.setdefault("h5py", _h5)

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

# ------------------------------------------------------------------------------ assets & layout
BASE = r"C:\Users\aimat\Desktop\humanoid\imported_cad"
SHOP_USD = BASE + r"\arena_no_walls.usd"
KUKA_USD = BASE + r"\kuka_assembly.usd"
LARA_USD = BASE + r"\lara8.usd"      # the real Neura URDF, imported by import_lara8.py
MAIRA_USD = BASE + r"\maira7M.usd"

# All from r1_lara_maira_loop.py -- see that file for how each was measured.
Z_LIFT = 12.1
KUKA_CELL_XMIN, KUKA_CELL_XMAX = -13.1, -5.2
KUKA_CELL_YMIN, KUKA_CELL_YMAX = -18.5, -11.2
LARA_XY = (-11.46287, -14.66537)   # inside the KUKA cell
MAIRA_XY = (-15.2, -15.33837)      # ~2.1 m west of the cell's west wall, outside it
DOOR_Y = -15.0
DOOR_WIDTH = 1.2

# r1_lara_maira_loop.py offsets everything by its RL env origin; a plain SimulationContext scene
# has exactly one origin and it is (0, 0), so OX/OY collapse away. Kept named for comparability.
OX, OY = 0.0, 0.0


def log(msg: str) -> None:
    print(f"[ARENA] {msg}", flush=True)


# ------------------------------------------------------------------------------------ MQTT input
class MairaMqttSource:
    """Subscribes to MAiRA's retained fleet state and keeps only the newest joint vector.

    Deliberately tolerant: the broker being down, the agent being offline, or telemetry coming
    back with joints=None (which happens when the SDK read fails and the adapter falls back)
    are all normal conditions here, not errors worth stopping the simulation over. The scene
    simply holds MAiRA's last known pose, which is what `--max-age` enforces.
    """

    def __init__(self, host: str, port: int, robot_id: str):
        self._lock = threading.Lock()
        self._joints: list[float] | None = None
        self._at: float | None = None
        self._msgs = 0
        self.connected = False
        self.error: str | None = None
        self.topic = f"fleet/v1/{robot_id}/state"

        import paho.mqtt.client as mqtt
        # paho 2.x requires an explicit callback API version; 1.x has no such argument.
        try:
            self._c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"isaac-arena-{int(time.time())}")
        except AttributeError:
            self._c = mqtt.Client(client_id=f"isaac-arena-{int(time.time())}")
        self._c.on_connect = self._on_connect
        self._c.on_disconnect = self._on_disconnect
        self._c.on_message = self._on_message
        # connect_async + loop_start so a broker that is not up yet delays nothing at startup
        # and reconnects on its own once it is.
        self._c.connect_async(host, port, keepalive=20)
        self._c.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        self.connected = True
        self.error = None
        client.subscribe(self.topic, qos=1)
        log(f"MQTT connected, subscribed to {self.topic}")

    def _on_disconnect(self, client, userdata, *a, **kw):
        self.connected = False
        log("MQTT disconnected (will retry in the background)")

    def _on_message(self, client, userdata, msg):
        try:
            d = json.loads(msg.payload.decode())
        except Exception as exc:
            self.error = f"undecodable state message: {exc}"
            return
        joints = (d.get("telemetry") or {}).get("joints")
        if not joints:
            # The agent is alive and publishing, but its own read of the robot came back
            # empty -- worth surfacing, distinct from "no message at all".
            self.error = "state message carried no telemetry.joints"
            return
        with self._lock:
            self._joints = [float(x) for x in joints]
            # Local arrival time, NOT the message's own `ts`: that clock belongs to the
            # workstation and any skew between the two machines would silently corrupt the
            # staleness check that --max-age depends on.
            self._at = time.time()
            self._msgs += 1
            self.error = None

    def latest(self):
        with self._lock:
            if self._joints is None or self._at is None:
                return None, None
            return list(self._joints), time.time() - self._at

    @property
    def count(self) -> int:
        with self._lock:
            return self._msgs

    def close(self):
        try:
            self._c.loop_stop()
            self._c.disconnect()
        except Exception:
            pass


class LaraFileSource:
    """Newest LARA joints from the shared bridge file, with the last good reading cached.

    The cache is the whole point. Reading this file fails routinely -- it is written from WSL
    and read from Windows, and either side can catch the other mid-write (lara_real_reader.py's
    write_bridge() documents the same race from the other direction). Treating one failed read
    as "the robot is gone" makes the liveness flag flap on and off many times a second even
    though the writer is perfectly healthy.

    So a failed read means "no news", not "stale": age is measured from the timestamp of the
    last reading actually obtained. A writer that genuinely stops therefore still ages out past
    --max-age within a second, while a momentary lock changes nothing.
    """

    def __init__(self, path: str):
        self.path = path
        self._joints: list[float] | None = None
        self._at: float | None = None

    def latest(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
            rj, at = d.get("real_joints"), d.get("at")
            if rj and at:
                self._joints = [float(x) for x in rj]
                self._at = float(at)
        except Exception:
            pass
        if self._joints is None or self._at is None:
            return None, None
        # The writer's own clock, not arrival time: both sides of this bridge are the same
        # physical machine, so there is no skew to worry about (unlike the MQTT path).
        return list(self._joints), time.time() - self._at


# --------------------------------------------------------------------------------- build a scene
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
stage = sim.stage

sim_utils.DomeLightCfg(intensity=1500.0).func(
    "/World/light", sim_utils.DomeLightCfg(intensity=1500.0))


def _add_ref(path, asset, translate):
    """Reference an asset at an EXACT transform, no bounding-box auto-alignment.

    The shopfloor and the KUKA cell MUST both use this, and must get the IDENTICAL translate:
    compose_scene.py established that they were authored in one shared frame and are only
    correctly registered to each other when placed identically. Auto-aligning each to its own
    bounding box would independently shift them and pull the cell off the floor plan.
    """
    p = UsdGeom.Xform.Define(stage, path)
    p.AddTranslateOp().Set(Gf.Vec3d(*translate))
    stage.DefinePrim(path + "/asset").GetReferences().AddReference(asset)
    for _ in range(6):
        simulation_app.update()
    rng = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    ).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return rng.GetMin(), rng.GetMax()


mn, mx = _add_ref("/World/Shopfloor", SHOP_USD, (OX, OY, Z_LIFT))
log(f"shopfloor bbox z=[{mn[2]:.2f},{mx[2]:.2f}] (floor should sit at ~0)")
mn, mx = _add_ref("/World/Kuka", KUKA_USD, (OX, OY, Z_LIFT))
log(f"KUKA cell (static) bbox x=[{mn[0]:.1f},{mx[0]:.1f}] y=[{mn[1]:.1f},{mx[1]:.1f}] "
    f"-- LARA at {LARA_XY} should be inside this, MAiRA at {MAIRA_XY} outside")


def build_cell_fence():
    """Visual-only outline of the cell, with a gap where the door is. No colliders: this is a
    sight-line aid for reading the scene, and giving it physics could push a robot."""
    def _wall(name, x0, x1, y0, y1, color=(0.95, 0.75, 0.05)):
        w = UsdGeom.Cube.Define(stage, f"/World/CellFence/{name}")
        w.CreateSizeAttr(1.0)
        wx = UsdGeom.Xformable(w)
        wx.AddTranslateOp().Set(Gf.Vec3d(OX + (x0 + x1) / 2.0, OY + (y0 + y1) / 2.0, 1.0))
        wx.AddScaleOp().Set(Gf.Vec3f(max(abs(x1 - x0), 0.06), max(abs(y1 - y0), 0.06), 2.0))
        w.CreateDisplayColorAttr([color])

    _wall("east", KUKA_CELL_XMAX, KUKA_CELL_XMAX, KUKA_CELL_YMIN, KUKA_CELL_YMAX)
    _wall("north", KUKA_CELL_XMIN, KUKA_CELL_XMAX, KUKA_CELL_YMAX, KUKA_CELL_YMAX)
    _wall("south", KUKA_CELL_XMIN, KUKA_CELL_XMAX, KUKA_CELL_YMIN, KUKA_CELL_YMIN)
    y0, y1 = DOOR_Y - DOOR_WIDTH / 2.0, DOOR_Y + DOOR_WIDTH / 2.0
    _wall("west_south", KUKA_CELL_XMIN, KUKA_CELL_XMIN, KUKA_CELL_YMIN, y0)
    _wall("west_north", KUKA_CELL_XMIN, KUKA_CELL_XMIN, y1, KUKA_CELL_YMAX)
    _wall("door_post_south", KUKA_CELL_XMIN - 0.05, KUKA_CELL_XMIN + 0.05,
          y0 - 0.06, y0, (0.85, 0.15, 0.1))
    _wall("door_post_north", KUKA_CELL_XMIN - 0.05, KUKA_CELL_XMIN + 0.05,
          y1, y1 + 0.06, (0.85, 0.15, 0.1))


if args_cli.fence:
    build_cell_fence()
    log(f"cell fence drawn, door gap at y=[{DOOR_Y - DOOR_WIDTH / 2:.1f},"
        f"{DOOR_Y + DOOR_WIDTH / 2:.1f}] on the west wall")

lara = Articulation(ArticulationCfg(
    prim_path="/World/Lara",
    spawn=sim_utils.UsdFileCfg(usd_path=LARA_USD),
    init_state=ArticulationCfg.InitialStateCfg(pos=(OX + LARA_XY[0], OY + LARA_XY[1], 0.0)),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=4000.0, damping=200.0)},
))
maira = Articulation(ArticulationCfg(
    prim_path="/World/Maira",
    spawn=sim_utils.UsdFileCfg(usd_path=MAIRA_USD),
    init_state=ArticulationCfg.InitialStateCfg(pos=(OX + MAIRA_XY[0], OY + MAIRA_XY[1], 0.0)),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=4000.0, damping=200.0)},
))

# Look at the two cobots, not the world origin: they sit ~15 m away near the cell, so a default
# camera would frame empty floor. Same reasoning as r1_lara_maira_loop.py's camera placement.
mid_x = OX + (LARA_XY[0] + MAIRA_XY[0]) / 2.0
mid_y = OY + (LARA_XY[1] + MAIRA_XY[1]) / 2.0
sim.set_camera_view(eye=(mid_x - 4.0, mid_y - 5.0, 3.5), target=(mid_x, mid_y, 0.8))

sim.reset()

LARA_JN, MAIRA_JN = lara.joint_names, maira.joint_names
log(f"LARA articulation: {len(LARA_JN)} joints -> {LARA_JN}")
log(f"MAiRA articulation: {len(MAIRA_JN)} joints -> {MAIRA_JN}")
if LARA_JN != ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]:
    log("WARNING: LARA joint order differs from what was verified against the real robot -- "
        "real_joints[i] may not map to joint i.")

lara_default = lara.data.default_joint_pos.clone()
maira_default = maira.data.default_joint_pos.clone()


def drive(art, default_q, njoints, angles):
    q = default_q.clone()
    for i in range(min(len(angles), njoints)):
        q[0, i] = float(angles[i])
    art.set_joint_position_target(q)


lara_src = LaraFileSource(args_cli.lara_bridge) if args_cli.lara_bridge else None

maira_src = None
if not args_cli.no_maira:
    try:
        maira_src = MairaMqttSource(args_cli.mqtt_host, args_cli.mqtt_port, args_cli.maira_id)
    except Exception as exc:
        log(f"MQTT source unavailable ({type(exc).__name__}: {exc}) -- MAiRA stays static")

log(f"LARA source: {args_cli.lara_bridge or '(none, static)'}")
log(f"MAiRA source: {'mqtt %s:%d %s' % (args_cli.mqtt_host, args_cli.mqtt_port, args_cli.maira_id) if maira_src else '(none, static)'}")
log("running -- move either real robot and watch its twin here. Ctrl-C to stop.")

# One "state changed" line per robot rather than a message per frame: at 120 Hz, logging a
# stale bridge every tick would bury everything else.
last_status = 0.0
lara_ok = maira_ok = None
mismatch_warned = False
try:
    while simulation_app.is_running():
        if lara_src is not None:
            rj, age = lara_src.latest()
            fresh = rj is not None and age is not None and age <= args_cli.max_age
            if fresh:
                drive(lara, lara_default, len(LARA_JN), rj)
            if fresh != lara_ok:
                lara_ok = fresh
                log("LARA live" if fresh else
                    "LARA not live -- holding pose (is lara_real_reader.py running in WSL?)")

        if maira_src is not None:
            rj, age = maira_src.latest()
            fresh = rj is not None and age is not None and age <= args_cli.max_age
            if fresh:
                if len(rj) != len(MAIRA_JN) and not mismatch_warned:
                    mismatch_warned = True
                    log(f"WARNING: MAiRA telemetry has {len(rj)} joints but the articulation has "
                        f"{len(MAIRA_JN)} -- mapping the first {min(len(rj), len(MAIRA_JN))} in order")
                drive(maira, maira_default, len(MAIRA_JN), rj)
            if fresh != maira_ok:
                maira_ok = fresh
                if fresh:
                    log("MAiRA live")
                else:
                    why = maira_src.error or ("no state message received yet"
                                              if maira_src.count == 0 else "state went stale")
                    log(f"MAiRA not live -- holding pose ({why})")

        lara.write_data_to_sim()
        maira.write_data_to_sim()
        sim.step()
        lara.update(1.0 / 120.0)
        maira.update(1.0 / 120.0)

        if time.time() - last_status > 5.0:
            last_status = time.time()
            lq = [round(float(lara.data.joint_pos[0, i]), 3) for i in range(len(LARA_JN))]
            mq = [round(float(maira.data.joint_pos[0, i]), 3) for i in range(len(MAIRA_JN))]
            log(f"LARA{'*' if lara_ok else ' '}={lq}")
            log(f"MAiRA{'*' if maira_ok else ' '}={mq}"
                + (f"  (mqtt msgs={maira_src.count})" if maira_src else ""))
except KeyboardInterrupt:
    log("stopping")
finally:
    if maira_src is not None:
        maira_src.close()
    simulation_app.close()
