"""GR00T modality configuration for the Unitree R1 goal-directed reach dataset.

Passed to launch_finetune.py via --modality-config-path. Registers the R1's modalities under
EmbodimentTag.NEW_EMBODIMENT so GR00T can fine-tune on a custom (non-shipped) robot.

Keys must match the dataset's meta/modality.json:
  state.joints (26)  |  action.joints (26)  |  video.front  |  annotation.human.task_description
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

r1_config = {
    # single egocentric camera; current frame only
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front"],
    ),
    # 26 joint positions (proprioception), current reading
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joints"],
    ),
    # 26-dim joint action, 16-step prediction horizon
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["joints"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,   # joint targets -> min/max normalized
                type=ActionType.NON_EEF,             # joint space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # task instruction
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# custom robots MUST register under NEW_EMBODIMENT
register_modality_config(r1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
