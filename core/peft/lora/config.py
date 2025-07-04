"""
LoRA Configuration

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


@dataclass
class LoraConfig():
    """
    LoRA Configuration
    """
    lora_rank: Optional[Union[List[int], int]] = field(default=128, metadata={"help": "The Lora rank for the attention dimension."})
    lora_alpha: int = field(default=8, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout probability for Lora layers."})
    init_lora_weights: bool = field(
        default=True, metadata={"help": "Whether to initialize the weights of the adapter layers."})
    
    top_k: int = field(default=1, metadata={
        "help": "The k in top-k gating if the expert threshold is None."})
    threshold: float = field(default=None, metadata={
        "help": "The threshold for selecting experts if the top-k is None. "
                "The maximum threshold should be 1 / number of experts"})

