"""Open the R1 + staircase 3 MuJoCo scene, holding the standing pose with the joint PD.

    mjpython stairs-r1/view_staircase3.py          # interactive viewer (macOS needs mjpython)
    python   stairs-r1/view_staircase3.py --check  # headless: stand 3 s, report, save a PNG

There is no locomotion policy yet -- the robot only holds its standing pose, so this checks the
model and the scene: does the R1 stand on its own feet, at the right height, in front of the flight.

STAND ASSIST: with r1_cfg.py's ankle gains (kp 20) a humanoid with no balance controller topples
in ~1 s -- in Isaac the walking policy does the balancing. So for this pose-hold only, the ankle
PD is stiffened (kp x10, kd x3). Training uses the unmodified Isaac gains from r1.xml.
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

import build_r1_mjcf
import staircase3

HERE = Path(__file__).resolve().parent


def load():
    if not (HERE / "r1.xml").exists():
        build_r1_mjcf.main()
    scene = HERE / "scene_staircase3.xml"
    scene.write_text(staircase3.scene_xml())
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.qpos[0:2] = staircase3.START
    mujoco.mj_forward(model, data)
    return model, data


def stand_assist(model, kp_scale=10.0, kd_scale=3.0):
    for i in range(model.nu):
        if "ankle" in model.actuator(i).name:
            model.actuator_gainprm[i, 0] *= kp_scale
            model.actuator_biasprm[i, 1] *= kp_scale
            model.actuator_biasprm[i, 2] *= kd_scale


def check(model, data, seconds=3.0):
    ctrl = data.ctrl.copy()
    feet = [model.geom(f"{s}_foot").id for s in ("left", "right")]
    for _ in range(int(seconds / model.opt.timestep)):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
    up = data.xmat[model.body("pelvis_link").id].reshape(3, 3)[2, 2]
    touching = {model.geom(c.geom1).name if c.geom1 in feet else model.geom(c.geom2).name
                for c in data.contact[:data.ncon] if c.geom1 in feet or c.geom2 in feet}
    print(f"after {seconds:.0f} s: pelvis z={data.qpos[2]:.3f} m (start {build_r1_mjcf.STAND_Z}), "
          f"upright cos={up:.3f}, x={data.qpos[0]:+.3f} y={data.qpos[1]:+.3f}, feet in contact={sorted(touching)}")
    ok = data.qpos[2] > 0.6 and up > 0.95 and len(touching) == 2
    print("STAND OK" if ok else "STAND FAILED")

    r = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [1.6, 0.0, 1.2]
    cam.distance, cam.azimuth, cam.elevation = 6.0, -35, -18
    r.update_scene(data, cam)
    png = HERE / "staircase3_check.png"
    import PIL.Image
    PIL.Image.fromarray(r.render()).save(png)
    print(f"saved {png}")
    return ok


def view(model, data):
    import mujoco.viewer
    ctrl = data.ctrl.copy()
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [1.6, 0.0, 1.2]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 6.0, -35, -18
        while v.is_running():
            t0 = time.time()
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    m, d = load()
    stand_assist(m)
    if args.check:
        raise SystemExit(0 if check(m, d) else 1)
    view(m, d)
