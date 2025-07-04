"""
AdaMoLE Configuration
"""
from dataclasses import dataclass, field

from core.peft.lora.config import LoraConfig


@dataclass
class AdaMoleConfig(LoraConfig):
    """
    AdaMoLE Configuration
    """
    num_experts: int = field(default=4, metadata={"help": "The number of experts in MoE."})
    max_threshold: float = field(default=0.25, metadata={
        "help": "The maximum threshold for selecting experts in the threshold function. "
                "The default value will be 1 / number of experts"})
