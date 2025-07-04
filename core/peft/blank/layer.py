"""
LoRA Layer

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
import math
from abc import ABC
from typing import Optional
from typing import List, Literal, Optional, Union
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

    
    
class BlankLayer(nn.Module):
    """
    FrozenLayer Implementation in a vit Layer
    """
    def __init__(
        self,
        base_layer: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__() 
        self.base_layer = base_layer
    
    def forward(self, inputs: torch.Tensor, patch_h: int, patch_w: int, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        result = self.base_layer(inputs, *args, **kwargs)
        return result
