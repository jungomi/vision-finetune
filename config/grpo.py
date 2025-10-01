import typing
from dataclasses import dataclass
from typing import Literal

from simple_parsing import choice

type GrpoScale = Literal["std", "max-len", "none"]

type GrpoImportanceWeight = Literal["token", "sequence"]


@dataclass
class GrpoConfig:
    """
    Configuration related to the Group Relative Preference Optimisation (GRPO)
    """

    # Number of generations per sample for GRPO.
    num_generations: int = 8
    # How to scale the rewards, e.g. by the standard deviation or max length. If max-len
    # is used, there must be a max_new_tokens defined, otherwise there will be no
    # scaling.
    scale_rewards: GrpoScale = choice(
        *typing.get_args(GrpoScale.__value__), default="std"
    )
    # Clip range of the advantage term. A value of 0.2 means that it will clipped in the
    # range [0.8, 1.2].
    clip_advantage: float = 0.2
    # Weight for the KL-Divergence loss term. DeepSeek-R1 used 0.04, but that seems to
    # be too high, hence the default is 0.01.
    kl_weight: float = 0.01
    # Temperature to use for the generation of the group samples for GRPO.
    temperature: float = 0.6
    # Top-P to use for the generation of the group samples for GRPO.
    top_p: float = 0.92

    # How to weight the importance of a response. "token" (default) is the original GRPO
    # weighting where each token contributes individually to the weight, whereas
    # "sequence" is GSPO, which creates one importance weight per sequence.
    importance_weight: GrpoImportanceWeight = choice(
        *typing.get_args(GrpoImportanceWeight.__value__), default="token"
    )
