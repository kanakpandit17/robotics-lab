"""Load-test the recorded LeRobot dataset inside the WSL GR00T env. Defensive: each check is
independent and reports pass/fail without aborting the rest."""
import json
import os

ROOT = "/mnt/c/Users/aimat/Desktop/humanoid/lerobot_r1_reach_img"


def sec(t):
    print(f"\n=== {t} ===", flush=True)


sec("META")
try:
    info = json.load(open(f"{ROOT}/meta/info.json"))
    print("robot_type:", info["robot_type"], "| episodes:", info["total_episodes"],
          "| frames:", info["total_frames"], "| fps:", info["fps"])
    print("features:", list(info["features"].keys()))
    eps = [json.loads(l) for l in open(f"{ROOT}/meta/episodes.jsonl")]
    tasks = [json.loads(l) for l in open(f"{ROOT}/meta/tasks.jsonl")]
    print("episodes.jsonl entries:", len(eps), "| first:", eps[0])
    print("tasks:", tasks)
except Exception as e:
    print("META FAILED:", repr(e)[:200])

sec("PARQUET (state / action)")
try:
    import numpy as np
    import pandas as pd
    df = pd.read_parquet(f"{ROOT}/data/chunk-000/episode_000000.parquet")
    print("columns:", list(df.columns))
    print("rows:", len(df), "| state dim:", len(df["observation.state"].iloc[0]),
          "| action dim:", len(df["action"].iloc[0]))
    print("sample state[:5]:", np.array(df["observation.state"].iloc[0])[:5])
    print("sample action[:5]:", np.array(df["action"].iloc[0])[:5])
except Exception as e:
    print("PARQUET FAILED:", repr(e)[:200])

sec("IMAGE LOAD (PNG frames — no ffmpeg/torchcodec needed)")
try:
    import glob
    ep0 = f"{ROOT}/images/observation.images.front/episode_000000"
    pngs = sorted(glob.glob(f"{ep0}/*.png"))
    print("png frames in ep0:", len(pngs))
    from PIL import Image
    im = np.array(Image.open(pngs[0]))
    print("PIL load OK (NO ffmpeg), first frame shape:", im.shape, "dtype:", im.dtype)
    print("info image dtype:", info["features"]["observation.images.front"]["dtype"])
except Exception as e:
    print("IMAGE LOAD FAILED:", repr(e)[:200])
sec("GR00T META (modality + stats)")
print("modality.json present:", os.path.exists(f"{ROOT}/meta/modality.json"))
print("stats.json present:", os.path.exists(f"{ROOT}/meta/stats.json"))

sec("GR00T / LEROBOT LOADERS")
try:
    import lerobot
    print("lerobot importable:", getattr(lerobot, "__version__", "?"))
except Exception as e:
    print("lerobot NOT importable:", repr(e)[:120])
try:
    import gr00t  # noqa: F401
    from gr00t.data.dataset import LeRobotSingleDataset  # noqa: F401
    print("gr00t + LeRobotSingleDataset import OK")
    # GR00T needs a meta/modality.json describing state/action/video/annotation modalities
    print("modality.json present:", os.path.exists(f"{ROOT}/meta/modality.json"))
except Exception as e:
    print("gr00t loader:", repr(e)[:200])

print("\nLOADTEST_DONE", flush=True)
