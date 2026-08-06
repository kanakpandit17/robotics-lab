"""Pre-flight: confirm the R1 GR00T modality config registers and matches the dataset."""
import runpy

runpy.run_path("/mnt/c/Users/aimat/Desktop/humanoid/r1_groot_config.py")

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag

mc = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
print("REGISTERED OK. modalities:", list(mc.keys()))
for k, v in mc.items():
    print(f"  {k}: keys={v.modality_keys} delta={len(v.delta_indices)}")

# cross-check against the dataset loader
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

loader = LeRobotEpisodeLoader("/mnt/c/Users/aimat/Desktop/humanoid/lerobot_r1_ikreach", mc)
print("LOADER OK. episodes:", len(loader.episode_lengths))
print("R1_FT_VALIDATE_DONE")
