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

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LoraLayer(nn.Module):
    """
    LoRA Layer
    """
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.lora_rank = {}
        self.lora_alpha = {}
        self.scaling = {}

        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.kwargs = kwargs

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Module):
            first_layer = next(base_layer.children())
            in_features, out_features = first_layer.in_features, first_layer.in_features # type: ignore
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, lora_rank: int, lora_alpha: int, lora_dropout: float, lora_type: str, init_lora_weights: bool,
    ) -> None:
        """
        Update the layer
        """
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout = lora_dropout_layer
        
        if lora_type == 'conv2d':
            self.lora_A = nn.Linear(self.in_features, lora_rank, bias=True)
            self.lora_B = nn.Linear(lora_rank, self.out_features, bias=True)
            self.lora_adapter = nn.Conv2d(lora_rank, lora_rank, 3, 1, 1)
            self.act = QuickGELU()
            
        elif lora_type == 'linear':
            self.lora_A = nn.Linear(self.in_features, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, self.out_features, bias=False)
        
        self.scaling = lora_alpha / lora_rank

        self.reset_parameters(init_lora_weights, lora_type)

    def reset_parameters(self, init_lora_weights: bool, lora_type: str) -> None:
        """
        Reset the parameters
        """
        if init_lora_weights is False:
            return
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            if lora_type == 'conv2d':
                nn.init.zeros_(self.lora_A.bias)
                nn.init.zeros_(self.lora_B.bias)
                nn.init.zeros_(self.lora_adapter.weight)

                self.lora_adapter.weight.data[:, :, 1, 1] += torch.eye(self.lora_rank, dtype=torch.float)
                nn.init.zeros_(self.lora_adapter.bias)


class LinearLoraLayer(LoraLayer):
    """
    LoRA Implementation in a Linear Layer
    """
    def __init__(
        self,
        base_layer: Union[nn.Linear, nn.Conv2d],
        lora_type: str = "linear",
        lora_rank: int = 8,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super(LoraLayer, self).__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.update_layer(lora_rank, lora_alpha, lora_dropout, lora_type, init_lora_weights)
        self.lora_type = lora_type
        self.activation = QuickGELU()
                
    def forward(self, inputs: torch.Tensor, patch_h: int, patch_w: int, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        previous_dtype = inputs.dtype
        result = self.base_layer(inputs, *args, **kwargs)

        if len(inputs.shape) == 4:
            b, _, _, dim = inputs.shape
            inputs = inputs.reshape(b, -1, dim)
            
        lora_A = self.lora_A
        lora_B = self.lora_B
        dropout = self.lora_dropout
        scaling = self.scaling

        if self.lora_type == 'conv2d':
            b , N, c = inputs.shape
            x_down = self.lora_A(inputs)
            x_down = self.activation(x_down)
            
            if N % 2 == 1:
                x_cls = x_down[:, :1].reshape(b, 1, 1, self.lora_rank).permute(0, 3, 1, 2).contiguous()
                x_cls = self.lora_adapter(x_cls)
                x_cls = x_cls.permute(0, 2, 3, 1).reshape(b, -1, self.lora_rank)
                x_down = x_down[:, 1:,...].permute(0, 2, 1).reshape(b , -1, patch_h, patch_w).contiguous()
            else:
                x_down = x_down[:].permute(0, 2, 1).reshape(b , -1, patch_h, patch_w).contiguous()
                
            x_down = self.lora_adapter(x_down)
            x_down = x_down.permute(0, 2, 3, 1).reshape(b, -1, self.lora_rank)
            
            if N % 2 == 1:
                x_down = torch.cat([x_cls, x_down], dim=1)
                
            x_down = self.activation(x_down)
            x_down = self.lora_dropout(x_down)
            output = self.lora_B(x_down) * self.scaling
            
        else:
            inputs = inputs.to(lora_A.weight.dtype)
            output = lora_B(lora_A(dropout(inputs))) * scaling
        
        if len(result.shape) == 4 and len(output.shape) != result.shape:
            b, h, w, d = result.shape
            output = output.reshape(b, h, w, d)
        
        final_output = output + result
        final_output = final_output.to(previous_dtype)
        return final_output
