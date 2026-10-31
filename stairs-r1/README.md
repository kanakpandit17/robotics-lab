# stairs-r1 — R1 on ARENA2036 staircase 3

Train an R1 stair policy in Isaac Lab, after arXiv:2603.07928 (dense unsafe-stepping penalty +
elevation map). Then test it on staircase 3 of the shopfloor in MuJoCo, the way FastStair
(arXiv:2601.10365) checks its policies sim-to-sim.

| File | Runs on | What |
|---|---|---|
| `r1_stairs_train.py` | workstation, Isaac Lab | train / play / export the stair policy |
| `eval_mujoco.py` | Mac or anywhere with MuJoCo | run the exported policy on staircase 3 |
| `staircase3.py` | — | staircase 3 geometry from the CAD → `scene_staircase3.xml` |
| `build_r1_mjcf.py` | — | `unitree_r1/R1.urdf` → `r1.xml` (free base, Isaac PD gains, box feet) |
| `view_staircase3.py` | Mac | look at the scene; `--check` = headless stand test + PNG |

## Staircase 3 (from `imported_cad/arena_full.usd`)
- 16 steps, 17.05 cm rise, 22.35 cm going, 1.05 m wide. Last riser up to the landing is 23.7 cm, kept as in the CAD.
- Landing at 2.98 m above the floor, leading to a 7.6 × 3 m platform.
- In the CAD the treads are 8.8 cm deep plates. Here each step is a solid block spanning the full going.
- Local frame: +x up the stairs, origin at the first step's front edge, floor at z = 0.
  Shopfloor ← local: `X = -31.722 - x`, `Y = -8.237 - y`, `Z = z + 0.133`.

## Steps
```powershell
# 1. workstation (Isaac Lab python)
python stairs-r1\r1_stairs_train.py                                   # ~12k iterations
python stairs-r1\r1_stairs_train.py --play   --checkpoint <model.pt>
python stairs-r1\r1_stairs_train.py --export --checkpoint <model.pt>  # -> exported\policy.pt + r1_stairs_meta.json
```
```bash
# 2. copy policy.pt + r1_stairs_meta.json into stairs-r1/policy/, then on the Mac
pip install mujoco torch
mjpython stairs-r1/eval_mujoco.py --policy stairs-r1/policy           # viewer
python   stairs-r1/eval_mujoco.py --policy stairs-r1/policy --headless --vx 0.7
```
`r1_stairs_meta.json` fixes the observation order, both joint orders, the default pose, the action
scale and the exact height-map ray layout. Because of that, the MuJoCo side rebuilds Isaac's
observation exactly rather than guessing it.

## Things that differ from the paper (see the train script's docstring)
- Height map from ground truth (ray cast) with ±2 cm noise. The paper's LiDAR + EGAU pipeline is the real-robot step.
- MLP actor and critic. The paper uses MLP + CNN.
- Extra 0.22 m treads in the curriculum, to cover staircase 3.
- Edge term implemented as the unsupported-sole fraction.
- Torque penalty on normalised torques.
- Effort limits from the URDF (60 / 50 / 33 N·m), not the 300 N·m in `r1_cfg.py`.

## Known
- With no policy, the R1 can't stand on Isaac's ankle gains (kp 20). In Isaac the policy does the
  balancing. `view_staircase3.py` stiffens the ankles for its pose-hold only.
