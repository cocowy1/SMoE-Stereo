"""
vpt Configuration

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


@dataclass
class VPTConfig():
    """
    vpt Configuration
    """
    prompt_tokens: Optional[Union[List[int], int]] = field(default=32, metadata={"help": "The Lora rank for the attention dimension."})
    prompt_project: int = field(default=-1, metadata={"help": "projection mlp hidden dim."})
    prompt_deep: bool = field(default=False, metadata={"help": "whether do deep prompt or not, only for prepend location."})
    prompt_drop: float = field(default=0., metadata={"help":"The dropout probability for prompt."})
    prompt_initiation: str = field(
        default="random", metadata={"help": "Whether to initialize the weights of the vpt prompts."})
