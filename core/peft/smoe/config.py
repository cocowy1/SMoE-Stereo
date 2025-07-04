"""
MoLE Configuration
"""
from dataclasses import dataclass, field
from core.peft.lora.config import LoraConfig
from typing import List, Literal, Optional, Union

@dataclass
class SMoEConfig(LoraConfig):
    """
    MoLE Configuration
    """
    lora_rank: Optional[Union[List[int], int]] = field(default_factory=lambda: [4,8,16,32], metadata={"help": "The Lora rank for the attention dimension."})
    kernel_size: Optional[Union[List[int], int]] = field(default_factory=lambda: [3,5,7,9], metadata={"help": "The Kenerl Size for Adapter."})
    lora_alpha: int = field(default=8, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout probability for Lora layers."})
    
    top_k: int = field(default=1, metadata={
        "help": "The k in top-k gating if the expert threshold is None."})
    threshold: float = field(default=None, metadata={
        "help": "The threshold for selecting experts if the top-k is None. "
                "The maximum threshold should be 1 / number of experts"})

