"""Confirm GR00T's own loader ingests the recorded image dataset end-to-end."""
import numpy as np

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ROOT = "/mnt/c/Users/aimat/Desktop/humanoid/lerobot_r1_reach_img"

modality_configs = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["joints"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)), modality_keys=["joints"],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

print("=== building LeRobotEpisodeLoader ===", flush=True)
loader = LeRobotEpisodeLoader(ROOT, modality_configs)
print("OK. episodes:", len(loader.episode_lengths), "| first lengths:", list(loader.episode_lengths[:3]), flush=True)

print("\n=== public methods ===", flush=True)
print([m for m in dir(loader) if not m.startswith("_")], flush=True)

print("\n=== pulling one sample (loader[0]) ===", flush=True)
import traceback
sample = None
try:
    sample = loader[0]
    print("loader[0] OK", flush=True)
except Exception:
    traceback.print_exc()

if sample is not None:
    print("\n=== sample keys ===", flush=True)
    print(list(sample.keys()), flush=True)
    print("\n=== per-key type/shape ===", flush=True)
    for k in sample.keys():
        v = sample[k]
        try:
            arr = np.asarray(v)
            print(f"  {k}: type={type(v).__name__} np.shape={arr.shape} dtype={arr.dtype}")
        except Exception as e:
            print(f"  {k}: type={type(v).__name__}  ({repr(e)[:60]})")
print("\nGROOT_INGEST_DONE", flush=True)
