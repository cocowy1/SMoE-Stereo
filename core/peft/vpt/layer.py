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

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
    
class VPTLayer(nn.Module):
    """
    VPT Implementation in a vit Layer
    """
    def __init__(
        self,
        base_layer: nn.Module,
        prompt_tokens: int = 128,
        prompt_drop: int = 8,
        prompt_project: int = -1,
        lora_dropout: float = 0.0,
        prompt_deep: bool = True,
        prompt_initiation: str = "random",
        **kwargs,
    ) -> None:
        super().__init__() 
        self.base_layer = base_layer
        
        self.prompt_tokens = prompt_tokens
        self.prompt_dropout = prompt_drop  
        self.prompt_dropout = Dropout(lora_dropout)
        self.prompt_initiation = prompt_initiation
        self.prompt_project = prompt_project

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


        if self.prompt_project > -1:
            # only for prepend / add
            prompt_dim = self.prompt_project
            self.prompt_proj = nn.Linear(
                prompt_dim, self.in_features)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = self.in_features
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_initiation == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, [14, 14], 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.prompt_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                

    def forward(self, inputs: torch.Tensor, patch_h: int, patch_w: int, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        if inputs.dim() == 4:
            # for conv2d
            B, _, _, C = inputs.shape
            inputs = inputs.reshape(B, -1, C).contiguous()
        
        B, N, C = inputs.shape
        
        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
            self.prompt_embeddings).expand(B, -1, -1))

        if (N % 2) == 1:
            hidden_states = torch.cat((
                    inputs[:, :1, :],
                    deep_prompt_emb,
                    inputs[:, (1+self.prompt_tokens):, :]
                ), dim=1)
        else:
            hidden_states = torch.cat((
                    deep_prompt_emb,
                    inputs[:, (self.prompt_tokens):, :]
                ), dim=1)
            hidden_states = hidden_states.reshape(B, patch_h, patch_w, -1).contiguous()     
             
        result = self.base_layer(hidden_states, *args, **kwargs)  
        return result
