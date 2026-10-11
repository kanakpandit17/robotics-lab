# robotics-lab — Unitree R1 / MAiRA / KUKA in Isaac Lab

Isaac Sim / Isaac Lab work: R1 humanoid RL (walking + navigation), MAiRA sim↔real bridge,
NVIDIA GR00T data pipeline, `.e57` point-cloud import, and Meta Quest 3 FPV.

> **This repo is code + configs + trained policies + robot assets.** It does **not** contain
> Isaac Sim, the Isaac Lab framework, the Python env, or large datasets — those are installed
> or regenerated on the target machine (see setup). Git LFS holds the binary assets (USD, meshes, `*.pt`).

## What's here
| Path | What |
|------|------|
| `r1_walk_train.py`, `r1_*_nav*_train.py`, `r1_cfg.py` | RL training (rsl_rl PPO) — walking + hierarchical nav |
| `r1_cell*.py`, `r1_maira_scene.py`, `drive_r1.py` | Deploy/demo scenes (R1 + MAiRA + KUKA cell) |
| `models/` | **Trained policies** (walking `r1_flat_v2_model_3999.pt`, nav `r1_nav_v3_model_3798.pt`, exported `policy.pt`, …) |
| `imported_cad/` | Converted USDs (R1, KUKA cell) |
| `unitree_r1/` | Source robot URDF/xacro/meshes |
| `neura/` | Everything Neura (MAiRA 7M + LARA 8): descriptions, USDs, sim↔real bridges — see `neura/README.md` |
| `r1_record_ik.py`, `r1_groot_config.py`, `finetune_r1_groot.sh` | GR00T dataset + fine-tune config |
| `e57_to_usd.py`, `e57_view.py`, `e57_render.py`, `e57_locate.py` | `.e57` laser-scan → Isaac point cloud + registration |
| `r1_quest_fpv.py`, `r1_image_server.py` | Meta Quest 3 FPV (Vuer) + robot camera server |
| `r1_probe*.ps1`, `dds_sniff.py`, `r1_read_state.py` | Unitree R1 SDK / WiFi probes |
| `R1_*_Report.pdf` | Project reports |

## Setup on a new machine
1. **GPU + OS:** NVIDIA RTX GPU (developed on RTX 5070 Ti / Blackwell), Windows 11 or Linux.
2. **Isaac Sim 5.1** — install from NVIDIA (standalone or pip). 
3. **Isaac Lab** — clone and install separately (NOT in this repo):
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab && ./isaaclab.sh --install   # (Linux)  or  isaaclab.bat --install  (Windows)
   ```
4. **Python env** — the scripts run with the Isaac Lab Python (conda env, Python 3.11). Key extra deps:
   `rsl_rl`, and for tooling `pye57 usd-core matplotlib mss vuer` (the last few only for e57/Quest tools;
   note `pye57` must live in a **separate venv** — it crashes inside the Isaac/Kit process).
5. **Git LFS** — `git lfs install` before cloning so the USD/mesh/policy binaries download.

## Paths — IMPORTANT
Scripts currently use **absolute Windows paths** (`C:\Users\aimat\Desktop\humanoid\...`) and expect
policies under `IsaacLab\logs\rsl_rl\...`. On a new machine, update those paths (or place the repo at
the same location and point `--checkpoint` at `models/`). Trained policies are in `models/` here.

## Run examples (Windows; use your env's python.exe)
```powershell
# walking + KUKA/MAiRA cell demo
python r1_cell_maira.py --checkpoint models\r1_flat_v2_model_3999.pt

# import an .e57 scan (convert in the separate venv, then view in Isaac)
e57venv\Scripts\python.exe e57_to_usd.py --file scan.e57 --out scan.usd
python e57_view.py --usd scan.usd

# Quest 3 FPV test pattern
python r1_quest_fpv.py --source test
```

## Not included (regenerate / obtain separately)
- `IsaacLab/` framework, the Python env, `e57venv/` — install per above.
- LeRobot datasets (`lerobot_*/`) — regenerate with `r1_record_ik.py`.
- `.e57` scans and large scan USDs — keep externally; convert with `e57_to_usd.py`.
- `certs/` (TLS keys) — regenerate locally; **never committed**.
