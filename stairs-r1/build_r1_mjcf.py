"""Unitree R1 URDF -> MuJoCo MJCF (r1.xml) for stair locomotion.

    python stairs-r1/build_r1_mjcf.py

What the URDF can't carry, added here:
  * a free joint on pelvis_link (the URDF's floating_base_joint is commented out)
  * position actuators with the SAME stiffness / damping / armature as r1_cfg.py (Isaac), so a
    policy sees the same joint behaviour in both simulators
  * box feet: the STL sole is 19 x 7.2 cm; a box gives clean, stable contact on step edges
  * collision only on feet, shins, pelvis and torso, and no self-collision. Everything else is
    visual. Robot geoms are contype=2/conaffinity=1, world geoms 1/1: robot-world collide,
    robot-robot never do.
The ankle linkage (A/B rods, constraint links) is fixed in the URDF, so the R1 is a plain serial
chain here too: 12 leg + 2 waist + 10 arm + 2 head = 26 actuated joints, matching Isaac.
"""
import re
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
URDF = HERE.parent / "unitree_r1" / "R1.urdf"
OUT = HERE / "r1.xml"

# r1_cfg.py ImplicitActuatorCfg, joint-name regex -> (kp, kd)
GAINS = [
    (r".*_hip_yaw_joint|.*_hip_roll_joint", 150.0, 5.0),
    (r".*_hip_pitch_joint|.*_knee_joint|waist_.*", 200.0, 5.0),
    (r".*_ankle_pitch_joint|.*_ankle_roll_joint", 20.0, 2.0),
    (r".*_shoulder_.*|.*_elbow_joint|.*_wrist_roll_joint|head_.*", 40.0, 10.0),
]
ARMATURE = 0.01
COLLIDE = re.compile(r"pelvis_link|waist_yaw_link|.*_knee_link")   # shins catch stair nosings

# r1_cfg.py InitialStateCfg
STAND = {"_hip_pitch_joint": -0.20, "_knee_joint": 0.42, "_ankle_pitch_joint": -0.23,
         "_shoulder_pitch_joint": 0.20, "_elbow_joint": 0.60}
STAND_Z = 0.72    # r1_cfg.py; the box soles start 7 mm into the floor, which MuJoCo's soft contact absorbs

# sole of left/right_ankle_roll_link.STL: x -0.055..0.135, y +-0.036, z -0.053..-0.004
FOOT_POS, FOOT_HALF = (0.040, 0.0, -0.0385), (0.095, 0.036, 0.0145)


def main():
    urdf = URDF.read_text()
    # the URDF's own <mujoco><compiler meshdir="meshes"> plus filename="meshes/x" doubles the path
    urdf = urdf.replace('meshdir="meshes"', f'meshdir="{URDF.parent}"')
    spec = mujoco.MjSpec.from_string(urdf)
    spec.modelname = "unitree_r1"
    spec.compiler.meshdir = str(URDF.parent)
    spec.option.timestep = 0.005

    pelvis = spec.body("pelvis_link")
    pelvis.pos = [0, 0, STAND_Z]
    pelvis.add_freejoint(name="floating_base")

    for g in spec.geoms:
        body = g.parent.name
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and COLLIDE.fullmatch(body):
            g.contype, g.conaffinity, g.group = 2, 1, 3
        else:
            g.contype, g.conaffinity, g.group = 0, 0, 1
    for side in ("left", "right"):
        spec.body(f"{side}_ankle_roll_link").add_geom(
            name=f"{side}_foot", type=mujoco.mjtGeom.mjGEOM_BOX, pos=FOOT_POS, size=FOOT_HALF,
            contype=2, conaffinity=1, group=3, friction=[1.0, 0.02, 0.01], rgba=[0.9, 0.4, 0.1, 0.4])
        spec.body(f"{side}_ankle_roll_link").add_site(name=f"{side}_foot_site", pos=[0.04, 0, -0.053])
    spec.body("pelvis_link").add_site(name="imu", pos=[0, 0, 0])

    for j in spec.joints:
        if j.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        kp, kd = next((kp, kd) for rx, kp, kd in GAINS if re.fullmatch(rx, j.name))
        j.armature = ARMATURE
        j.damping = [0.0, 0.0, 0.0]   # damping lives in the actuator (kd), as in Isaac
        a = spec.add_actuator(name=j.name.replace("_joint", ""), target=j.name,
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = kp
        a.biasprm[:3] = [0, -kp, -kd]
        a.ctrllimited = True
        a.ctrlrange = j.range
        if j.actfrcrange[1] > 0:            # URDF <limit effort=...>
            a.forcelimited = True
            a.forcerange = j.actfrcrange

    key = spec.add_key(name="stand")
    model = spec.compile()
    qpos = model.qpos0.copy()
    qpos[2] = STAND_Z
    ctrl = []
    for i in range(model.nu):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        q = next((v for suf, v in STAND.items() if jn.endswith(suf)), 0.0)
        qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] = q
        ctrl.append(q)
    key.qpos, key.ctrl = qpos, ctrl
    model = spec.compile()

    OUT.write_text(spec.to_xml())
    print(f"wrote {OUT}  nq={model.nq} nv={model.nv} nu={model.nu} mass={model.body_mass.sum():.2f} kg")


if __name__ == "__main__":
    main()
