# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T modality config for the local Piper/pi05-style dataset.

This config intentionally uses only the three cameras that the current real
robot client sends to the policy server:

- cam_high
- cam_left_wrist
- cam_right_wrist

The dataset also has cam_vertical, but the deploy client does not provide that
view, so training on it would make deployment fail strict observation checks.
"""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


PIPER_PI05_ACTION_HORIZON = int(os.environ.get("PIPER_PI05_ACTION_HORIZON", "40"))


piper_pi05_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_joint_position",
            "right_arm_joint_position",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(PIPER_PI05_ACTION_HORIZON)),
        modality_keys=[
            "left_arm_joint_position",
            "right_arm_joint_position",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm_joint_position",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm_joint_position",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.language.action_text"],
    ),
}


register_modality_config(piper_pi05_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
