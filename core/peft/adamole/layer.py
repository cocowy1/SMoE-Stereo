"""
AdaMoLE Layer
"""
import math
from abc import ABC
from typing import Optional
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.peft.smoe.layer import Expert


class AdaMoeLayer(nn.Module):
    """
    Adaptive Mixture of Experts (MoE) Layer
    """
    def __init__(self, experts: nn.ModuleList, gate: nn.Module, threshold_fn: nn.Module, max_threshold: float):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.threshold_fn = threshold_fn
        self.max_threshold = max_threshold

    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.sum(selected_experts, dim=0)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss.unsqueeze(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        gate_logits = F.softmax(self.gate(flattened_inputs), dim=-1)
        thresholds = torch.sigmoid(self.threshold_fn(flattened_inputs)) * self.max_threshold
        adapted_gate_logits = gate_logits - thresholds
        selected_experts = torch.ge(adapted_gate_logits, 0).to(torch.float)
        weights = adapted_gate_logits * selected_experts
        weight_sums = torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
        weight_sums = torch.where(weight_sums == 0, torch.ones_like(weight_sums), weight_sums)
        weights = weights / weight_sums
        results = torch.zeros_like(self.experts[0](flattened_inputs))

        for i, expert in enumerate(self.experts):
            batch_idx = torch.where(selected_experts[:, i])[0]
            if len(batch_idx) > 0:
                results[batch_idx] += weights[batch_idx, i, None] * expert(flattened_inputs[batch_idx])

        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        layer_loss = 0.
        if inputs.requires_grad:
            layer_loss = self.get_layer_loss(gate_logits=adapted_gate_logits, selected_experts=selected_experts)
        return results, layer_loss


class AdaMoleLayer(nn.Module):
    """
    AdaMoLE Layer
    """
    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.lora_rank = {}
        self.lora_alpha = {}
        self.scaling = {}

        self.lora_threshold = nn.ModuleDict({})
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_gating = nn.ModuleDict({})
        self.moe_layer = nn.ModuleDict({})
        self.kwargs = kwargs

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Module):
            in_features, out_features = base_layer.fc1.in_features, base_layer.fc1.in_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, lora_rank: int, lora_alpha: int, lora_dropout: float, lora_type: str,
                     init_lora_weights: bool, num_experts: int, max_threshold: float,
    ) -> None:
        """
        Update the layer
        """
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        if max_threshold is None:
            max_threshold = 1 / num_experts

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha


        if lora_dropout > 0.0:
            lora_dropout_layer = nn.ModuleList([nn.Dropout(p=lora_dropout) for _ in range(num_experts)])
        else:
            lora_dropout_layer = nn.ModuleList([nn.Identity(p=lora_dropout) for _ in range(num_experts)])

        self.lora_dropout = lora_dropout_layer

        if lora_type == 'conv2d':
            self.lora_A = nn.ModuleList(
                nn.Conv2d(in_channels=self.in_features, lout_channels=lora_rank, kernel_size=3, stride=1, padding=1, bias=False) for _ in range(num_experts))
            self.lora_B = nn.ModuleList(
                nn.Conv2d(in_channels=lora_rank, out_channels=self.out_features, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(num_experts))
        
        elif lora_type == 'linear':
            self.lora_A = nn.ModuleList(
                nn.Linear(self.in_features, lora_rank, bias=False) for _ in range(num_experts))
            self.lora_B = nn.ModuleList(
                nn.Linear(lora_rank, self.out_features, bias=False) for _ in range(num_experts))
            
        self.scaling = lora_alpha / lora_rank
        self.lora_gating = nn.Linear(self.in_features, num_experts, bias=False)
        self.lora_threshold = nn.Linear(self.in_features, 1)

        experts = nn.ModuleList(Expert(
            self.lora_A[i],
            self.lora_B[i],
            self.lora_dropout[i],
            self.scaling,
        ) for i in range(num_experts))

        self.moe_layer = AdaMoeLayer(
            experts=experts, gate=self.lora_gating,
            threshold_fn=self.lora_threshold, max_threshold=max_threshold)

        self.reset_parameters(init_lora_weights)

    def reset_parameters(self, init_lora_weights: bool) -> None:
        """
        Reset the parameters
        """
        if init_lora_weights is False:
            return
        else:
            for i in range(len(self.lora_A)):
                nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[i].weight)


class LinearAdaMoleLayer(AdaMoleLayer):
    """
    AdaMoLE Implementation in a Linear Layer
    """
    def __init__(
        self,
        base_layer: Union[nn.Linear, nn.Conv2d],
        lora_type: str = "linear",
        lora_rank: int = 8,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 4,
        max_threshold: float = None,
        **kwargs,
    ) -> None:
        super(AdaMoleLayer, self).__init__()
        AdaMoleLayer.__init__(self, base_layer=base_layer, **kwargs)
        self.update_layer(lora_rank, lora_alpha, lora_dropout, lora_type, init_lora_weights, num_experts, max_threshold)
        self.lora_type = lora_type
        self.layer_loss = None
    def forward(self, x: torch.Tensor,  patch_h: int, patch_w: int, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        
        moe_layer = self.moe_layer

        if self.lora_type == 'conv2d':
            assert x.dim() == 3
            b, _, c = x.size()
            class_tokens = torch.zeros_like(x[:, :1, ...])
            x = x[:, 1:,...].permute(0, 2, 1).reshape(x.shape[0] , -1, patch_h, patch_w).contiguous()
            x = x.to(moe_layer.experts[0].lora_A.weight.dtype)
            x, layer_loss = moe_layer(x)
            x = x.reshape(b, c, -1).permute(0, 2, 1).contiguous()
            x = torch.cat([class_tokens, x], dim=1)
            result += x

        else:
            x = x.to(moe_layer.experts[0].lora_A.weight.dtype)
            x, layer_loss = moe_layer(x)
            result += x

        self.layer_loss = layer_loss
        result = result.to(previous_dtype)
        return result
