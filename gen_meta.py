"""Add the two GR00T-required meta files to the recorded dataset:
  meta/modality.json  — maps the flat 26-dim state/action + video + task into named modalities
  meta/stats.json     — per-feature normalization statistics (state, action, image, scalars)
Pure data processing (pandas/numpy/imageio) — no Isaac Sim needed.
"""
import glob
import json
import os

import imageio.v2 as iio
import numpy as np
import pandas as pd

ROOT = r"C:\Users\aimat\Desktop\humanoid\lerobot_r1_reach"


def stat(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": x.mean(0).tolist(), "std": (x.std(0) + 1e-8).tolist(),
            "max": x.max(0).tolist(), "min": x.min(0).tolist()}


# ---- state / action / scalar stats from parquet ----
data_files = sorted(glob.glob(os.path.join(ROOT, "data", "chunk-000", "*.parquet")))
states, actions = [], []
scal = {k: [] for k in ["timestamp", "frame_index", "episode_index", "index", "task_index"]}
for f in data_files:
    df = pd.read_parquet(f)
    states.append(np.array(df["observation.state"].tolist(), dtype=np.float32))
    actions.append(np.array(df["action"].tolist(), dtype=np.float32))
    for k in scal:
        scal[k].append(np.array(df[k].tolist(), dtype=np.float64))

stats = {"observation.state": stat(np.concatenate(states)), "action": stat(np.concatenate(actions))}
for k in scal:
    stats[k] = stat(np.concatenate(scal[k]).reshape(-1, 1))

# ---- image stats (sample every 10th frame across all episodes) ----
vids = sorted(glob.glob(os.path.join(ROOT, "videos", "chunk-000", "observation.images.front", "*.mp4")))
sum_c = np.zeros(3); sumsq_c = np.zeros(3); cnt = 0
mx = np.zeros(3); mn = np.ones(3)
for v in vids:
    rd = iio.get_reader(v)
    for i, frame in enumerate(rd):
        if i % 10:
            continue
        fr = frame[..., :3].astype(np.float64) / 255.0
        sum_c += fr.sum((0, 1)); sumsq_c += (fr ** 2).sum((0, 1)); cnt += fr.shape[0] * fr.shape[1]
        mx = np.maximum(mx, fr.max((0, 1))); mn = np.minimum(mn, fr.min((0, 1)))
    rd.close()
mean = sum_c / cnt
std = np.sqrt(np.maximum(sumsq_c / cnt - mean ** 2, 0)) + 1e-8


def img3(a):  # LeRobot per-channel image stat shape [3,1,1]
    return [[[float(x)]] for x in a]


stats["observation.images.front"] = {"mean": img3(mean), "std": img3(std), "max": img3(mx), "min": img3(mn)}

with open(os.path.join(ROOT, "meta", "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

# ---- modality.json (GR00T) ----
modality = {
    "state": {"joints": {"start": 0, "end": 26}},
    "action": {"joints": {"start": 0, "end": 26}},
    "video": {"front": {"original_key": "observation.images.front"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}
with open(os.path.join(ROOT, "meta", "modality.json"), "w") as f:
    json.dump(modality, f, indent=2)

print(f"OK wrote modality.json + stats.json | image mean={mean.round(3).tolist()} std={std.round(3).tolist()} | frames sampled cnt_px={cnt}")
