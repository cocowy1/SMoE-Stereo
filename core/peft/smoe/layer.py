"""
MoLE Layer
"""
import math
from abc import ABC
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from core.peft.lora.layer import LoraLayer
from torch.autograd import Variable
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class TopKMoeLayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer with the Top-k

    Adapted from https://github.com/mistralai/mistral-src
    """

    def __init__(self, experts: nn.ModuleList, clean_gate: nn.Module, noise_gate: nn.Module, top_k: int, lora_type: str):
        super().__init__()
        self.experts = experts
        self.clean_gate = clean_gate
        self.noise_gate = noise_gate
        self.top_k = top_k
        self.lora_type =lora_type
        self.balance_loss = None
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-6

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def get_balance_loss(self, gates: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        return loss

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten().cuda()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def noisy_top_k_gating(self, x, noise_epsilon=1e-3):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        
        num_experts = len(self.experts)
        clean_logits = x @ self.clean_gate
        if self.training:
            raw_noise_stddev = x @ self.noise_gate
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            
        top_logits, top_indices = logits.topk(min(self.top_k + 1, num_experts), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        
        # 将源张量转换为半精度
        zeros_fp16 = zeros.to(torch.float16)
        top_k_gates_fp16 = top_k_gates.to(torch.float16) # 如果索引也需要转换
        # 进行半精度计算
        gates_fp16 = zeros_fp16.scatter(1, top_k_indices, top_k_gates_fp16)

        # 将结果转换回单精度
        gates = gates_fp16.to(torch.float32)

        # gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
            
        return gates, load, top_k_logits, top_k_indices
    
    def forward(self, inputs: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Forward propagation
        """
        outputs = {}
        if len(inputs.shape) == 4:
            b, _, _, dim = inputs.shape
            inputs = inputs.reshape(b, -1, dim)
            
        b, _, dim = inputs.shape
        if self.lora_type == 'conv2d':
            inputs_global = torch.mean(inputs, dim=1, keepdim=False)
            flattened_inputs = inputs_global.view((-1, inputs_global.shape[-1]))
            gates, load, top_k_logits, top_k_indices = self.noisy_top_k_gating(flattened_inputs)
            weights = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True, dtype=inputs.dtype)
            weights = weights.unsqueeze(1)
            
        elif self.lora_type == 'linear':
            flattened_inputs = inputs.view((-1, inputs.shape[-1]))
            gates, load, top_k_logits, top_k_indices = self.noisy_top_k_gating(flattened_inputs)
            weights = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True, dtype=inputs.dtype)
            
        if self.lora_type == 'conv2d':
            results = torch.zeros_like(self.experts[0](inputs, patch_h, patch_w))
        elif self.lora_type == 'linear':
            results = torch.zeros_like(self.experts[0](flattened_inputs, patch_h, patch_w))

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(top_k_indices == i)
            
            if len(batch_idx) > 0:
                if self.lora_type == 'conv2d':
                    results[batch_idx] += expert(inputs[batch_idx], patch_h, patch_w)
                elif self.lora_type == 'linear':
                    results[batch_idx] += expert(flattened_inputs[batch_idx], patch_h, patch_w)

        results = results.view((*inputs.shape[:-1], results.shape[-1]))


#####################################################################################################################
        # if self.lora_type == 'conv2d':
        #     results = torch.zeros_like(self.experts[0](inputs, patch_h, patch_w))
        # elif self.lora_type == 'linear':
        #     flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        #     results = torch.zeros_like(self.experts[0](flattened_inputs, patch_h, patch_w))
            
        # for i, expert in enumerate(self.experts):
        #     if self.lora_type == 'conv2d':
        #         results += expert(inputs, patch_h, patch_w)
        #     elif self.lora_type == 'linear':
        #         flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        #         results += expert(flattened_inputs, patch_h, patch_w)
            
        # results = results.view((*inputs.shape[:-1], results.shape[-1]))
        
        if self.training:
            self.balance_loss = self.get_balance_loss(gates=gates, load=load)
        
        outputs.update({'moe_results': results})
        outputs.update({'moe_experts': top_k_indices})
        outputs.update({'moe_type': self.lora_type})
        
        return outputs


class ThresholdMoeLayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer with the Threshold
    """
    def __init__(self, experts: nn.ModuleList, gate: nn.Module, threshold: float, lora_type: str):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.threshold = threshold
        self.lora_type = lora_type
        self.balance_loss = None

    def get_balance_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.sum(selected_experts, dim=0)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        balance_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return balance_loss

    def forward(self, inputs: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Forward propagation
        """
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        gate_logits = F.softmax(self.gate(flattened_inputs), dim=-1)
        selected_experts = torch.ge(gate_logits, self.threshold).to(torch.float)
        weights = gate_logits * selected_experts
        weight_sums = torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
        weight_sums = torch.where(weight_sums == 0, torch.ones_like(weight_sums), weight_sums)
        weights = weights / weight_sums
        if self.lora_type == 'conv2d':
            results = torch.zeros_like(self.experts[0](inputs, patch_h, patch_w))
        elif self.lora_type == 'linear':
            results = torch.zeros_like(self.experts[0](flattened_inputs, patch_h, patch_w))

        for i, expert in enumerate(self.experts):
            batch_idx = torch.where(selected_experts[:, i])[0]
            if len(batch_idx) > 0:
                if self.lora_type == 'conv2d':
                    results[batch_idx] += weights[batch_idx, i, None] * expert(inputs, patch_h, patch_w)[batch_idx]
                elif self.lora_type == 'linear':
                     results[batch_idx] += weights[batch_idx, i, None] * expert(flattened_inputs[batch_idx], patch_h, patch_w)

                # results[batch_idx] += weights[batch_idx, i, None] * expert(flattened_inputs[batch_idx], patch_h, patch_w)

        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        if inputs.requires_grad:
            self.balance_loss = self.get_balance_loss(gate_logits=gate_logits, selected_experts=selected_experts)
        return results


class Expert(nn.Module):
    """
    LoRA Expert
    """
    def __init__(self, lora_A: nn.Module, lora_B: nn.Module, lora_dropout: nn.Module, lora_rank: int, scaling: float, lora_type: str, lora_adapter: nn.Module=None):
        super().__init__()
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_dropout = lora_dropout
        self.scaling = scaling
        self.lora_type = lora_type
        self.lora_adapter = lora_adapter
        self.lora_rank = lora_rank
        self.activation = QuickGELU()
        
    def forward(self, inputs: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Forward propagation
        """
        if self.lora_type == "linear":
            outputs = self.lora_B(self.lora_A(self.lora_dropout(inputs))) * self.scaling
        
        elif self.lora_type == 'conv2d':
            b, N , c = inputs.shape
            x_down = self.lora_A(inputs)
            
            # check cls token
            if N % 2 == 1:
                x_cls = x_down[:, :1].reshape(b, 1, 1, self.lora_rank).permute(0, 3, 1, 2).contiguous()
                x_cls = self.lora_adapter(x_cls)
                x_cls = x_cls.permute(0, 2, 3, 1).reshape(b, -1, self.lora_rank).contiguous()
                x_down = x_down[:, 1:,...].permute(0, 2, 1).reshape(b , -1, patch_h, patch_w).contiguous()
            else:
                x_down = x_down[:].permute(0, 2, 1).reshape(b , -1, patch_h, patch_w).contiguous()
                

            x_down = self.lora_adapter(x_down)
            x_down = x_down.permute(0, 2, 3, 1).reshape(b, -1, self.lora_rank).contiguous()

            if N % 2 == 1:
                x_down = torch.cat([x_cls, x_down], dim=1)
                
            x_down = self.activation(x_down)
            outputs = self.lora_B(x_down) * self.scaling
            # outputs = outputs.reshape(-1, c)
            
        return outputs


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SMoELayer(nn.Module):
    """
    MoLE Layer
    """
    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.lora_rank = {}
        self.lora_alpha = {}
        self.scaling = {}

        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.moe_layer = nn.ModuleDict({})
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
        self, lora_rank: List[int], lora_alpha: int, lora_dropout: float, lora_type: str,
        init_lora_weights: bool, top_k: int, threshold: float, kernel_size: List[int],
    ) -> None:
        """
        Update the layer
        """
        for rank in lora_rank:
            if rank <= 0:
                raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        assert (top_k is not None) ^ (threshold is not None)
        # if (top_k is not None) and (threshold is not None):
        #     raise ValueError(f"Only one of the top-k {top_k} and the threshold {threshold} can be used.")
        # elif (top_k is None) and (threshold is None):
        #     raise ValueError(f"At least one of the top-k {top_k} and the threshold {threshold} should be used.")

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.kernel_size = kernel_size
    
        self.scaling = 1

        if lora_type == 'conv2d':
            num_experts = len(self.kernel_size)
            self.lora_w_gate = nn.Parameter(torch.randn(self.in_features,  num_experts), requires_grad=True)
            self.lora_w_noise = nn.Parameter(torch.randn(self.in_features,  num_experts), requires_grad=True)
            
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.ModuleList(nn.Dropout(p=lora_dropout) for _ in range(num_experts))
            else:
                lora_dropout_layer = nn.ModuleList(nn.Identity(p=lora_dropout) for _ in range(num_experts))
            
            self.lora_dropout = lora_dropout_layer
            
            self.lora_A = nn.ModuleList(
               nn.Linear(self.in_features, lora_rank[3], bias=True) for i in range(num_experts))
            self.lora_B = nn.ModuleList(
                nn.Linear(lora_rank[3], self.out_features, bias=True) for i in range(num_experts))
            
            if kernel_size is not None:
                self.lora_adapter = nn.ModuleList(
                nn.Conv2d(lora_rank[3], lora_rank[3], kernel_size[i], stride=1, padding=kernel_size[i]//2) for i in range(num_experts))

            self.act = QuickGELU()

            experts = nn.ModuleList(Expert(
            self.lora_A[i],
            self.lora_B[i],
            self.lora_dropout[i],
            self.lora_rank[3],
            self.scaling, lora_type,
            self.lora_adapter[i],
        ) for i in range(num_experts))
            
            
        elif lora_type == 'linear':
            num_experts = len(self.lora_rank)
            self.lora_w_gate = nn.Parameter(torch.randn(self.in_features, num_experts), requires_grad=True)
            self.lora_w_noise = nn.Parameter(torch.randn(self.in_features, num_experts), requires_grad=True)
        
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.ModuleList(nn.Dropout(p=lora_dropout) for _ in range(num_experts))
            else:
                lora_dropout_layer = nn.ModuleList(nn.Identity(p=lora_dropout) for _ in range(num_experts))
            
            self.lora_dropout = lora_dropout_layer
 
            self.lora_A = nn.ModuleList(
                nn.Linear(self.in_features, lora_rank[i], bias=False) for i in range(num_experts))
            self.lora_B = nn.ModuleList(
                nn.Linear(lora_rank[i], self.out_features, bias=False) for i in range(num_experts))
            
            experts = nn.ModuleList(Expert(
            self.lora_A[i],
            self.lora_B[i],
            self.lora_dropout[i],
            self.lora_rank[i],
            self.scaling, lora_type, None,
        ) for i in range(num_experts))

        if top_k is not None:
            self.moe_layer = TopKMoeLayer(
                experts=experts, clean_gate=self.lora_w_gate, noise_gate=self.lora_w_noise, top_k=top_k, lora_type=lora_type)
        if threshold is not None:
            self.moe_layer = ThresholdMoeLayer(
                experts=experts, clean_gate=self.lora_w_gate, noise_gate=self.lora_w_noise, threshold=threshold, lora_type=lora_type)

        self.reset_parameters(init_lora_weights, lora_type)
    
    def reset_parameters(self, init_lora_weights: bool, lora_type: str) -> None:
        """
        Reset the parameters
        """
        if init_lora_weights is False:
            return
        else:
            for i in range(len(self.lora_A)):
                nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[i].weight)
  
            if lora_type == 'conv2d':
                nn.init.zeros_(self.lora_A[i].bias)
                nn.init.zeros_(self.lora_B[i].bias)
                nn.init.zeros_(self.lora_adapter[i].weight)
                self.lora_adapter[i].weight.data[:, :, 1, 1] += torch.eye(self.lora_rank[3], dtype=torch.float)
                nn.init.zeros_(self.lora_adapter[i].bias)


class BlockLayerSelect(nn.Module):
    def __init__(self, dim_in, num_sub_layer=2, tau=5, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Linear(dim_in, num_sub_layer, bias=bias)
        # self.norm = LayerNorm(dim_in)
        self.tau = tau
        self.threshold = threshold
        self.add_noise = True
        self.random_policy = False
        self.random_layer = False
        self.random_layer_ratio = 1.

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:                    
        #             nn.init.constant_(m.bias, 1)    
    
    def get_random_policy(self, policy, ratio):
        random_p = torch.empty_like(policy).fill_(ratio).bernoulli() + policy * 0.0  # add policy * 0.0 into the loop of loss calculation to avoid the DDP issue
        return random_p

    def set_tau(self, tau):
        self.tau = tau

    def _gumbel_sigmoid(self, logits, tau=1, hard=False, eps=1e-10, training = True, threshold=0.8):
        if self.training:
            # ~Gumbel(0,1)`
            gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            ) 
            gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
            y_soft = gumbels.sigmoid()
        else:
            y_soft = logits.sigmoid()

        if hard:
            # Straight through.
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).masked_fill(y_soft > threshold, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft

        return ret

    def forward(self, x):
        if len(x.shape) == 4:
            b, _, _, d = x.shape
            x = x.reshape(b, -1, d).sum(1)
            logits = self.mlp_head(x[:])
        else:
            logits = self.mlp_head(x[:, 0])
            
        sample = self._gumbel_sigmoid(logits=logits, tau=self.tau, hard=True, threshold=self.threshold)
        if self.random_policy or self.random_layer:
            sample = self.get_random_policy(sample, self.random_layer_ratio)
        sample = sample #(b,2)

        return sample, logits
    

class LinearSMoELayer(SMoELayer):
    """
    MoLE Implementation in a Linear Layer
    """
    def __init__(
        self,
        base_layer: Union[nn.Linear, nn.Conv2d],
        lora_type: str = "linear",
        lora_rank: List[int] = [4,8,16,32,64],
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = False,
        top_k: int = 1,
        threshold: float = None,
        kernel_size: List[int] = [3,5,7,9],
        layer_selection: bool = False,
        count_experts: bool = True,
        **kwargs,
    ) -> None:
        super(SMoELayer, self).__init__()
        SMoELayer.__init__(self, base_layer=base_layer, **kwargs)
        self.update_layer(lora_rank, lora_alpha, lora_dropout, lora_type, init_lora_weights, top_k, threshold, kernel_size)
        self.lora_type = lora_type
        self.layer_selection = layer_selection
        self.count_experts = count_experts

        if self.count_experts:
            self.experts = torch.FloatTensor([0])

        self.layer_logits = None
        self.balance_loss = None
        self.sub_select_layer = None
        
        if isinstance(base_layer, nn.Linear):
            self.dim = base_layer.in_features
        elif isinstance(base_layer, nn.Conv2d):
            self.dim = base_layer.in_channels
        elif isinstance(base_layer, nn.Module):
            first_layer = next(base_layer.children())
            self.dim = first_layer.in_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        # self.lora_norm_layer = nn.LayerNorm(self.dim)
        if self.layer_selection:
            self.lora_layer_select = BlockLayerSelect(self.dim, 2)
            self.lora_layer_loss = None

    def count_activated_experts(self, moe_experts, moe_type, activated_layers=None):

        _, num_expert1 = torch.where(moe_experts == 0)
        _, num_expert2 = torch.where(moe_experts == 1)
        _, num_expert3 = torch.where(moe_experts == 2)
        _, num_expert4 = torch.where(moe_experts == 3)
        
        num_expert1 = num_expert1.shape[-1]
        num_expert2 = num_expert2.shape[-1]
        num_expert3 = num_expert3.shape[-1]
        num_expert4 = num_expert4.shape[-1]
        
        if activated_layers != None:
            num_expert1 = activated_layers.mean() * num_expert1/moe_experts.shape[0]
            num_expert2 = activated_layers.mean() * num_expert2/moe_experts.shape[0]
            num_expert3 = activated_layers.mean() * num_expert3/moe_experts.shape[0]
            num_expert4 = activated_layers.mean() * num_expert4/moe_experts.shape[0]
        else:
            num_expert1 = torch.FloatTensor([num_expert1/moe_experts.shape[0]])
            num_expert2 = torch.FloatTensor([num_expert2/moe_experts.shape[0]])
            num_expert3 = torch.FloatTensor([num_expert3/moe_experts.shape[0]])
            num_expert4 = torch.FloatTensor([num_expert4/moe_experts.shape[0]])
        
        self.experts = torch.stack([num_expert1.unsqueeze(0), num_expert2.unsqueeze(0), 
                        num_expert3.unsqueeze(0), num_expert4.unsqueeze(0)], dim=1)

            
        # print(f"moe_type:{moe_type}, number_expert1:{num_expert1:.2f}, number_expert2:{num_expert2:.2f}, \
        #       number_expert3:{num_expert3:.2f}, number_expert4:{num_expert4:.2f}".format(moe_type, num_expert1, num_expert2, num_expert3, num_expert4))
    
    def forward(self, x: torch.Tensor, patch_h: int, patch_w: int, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        moe_layer = self.moe_layer
        x = x.to(moe_layer.experts[0].lora_A.weight.dtype)
            
        if self.layer_selection == True:
            sub_select_layer, layer_logits = self.lora_layer_select(x)
            self.sub_select_layer = sub_select_layer
            self.layer_logits = layer_logits
            
            moe_output = moe_layer(x, patch_h, patch_w)
            moe_result = moe_output["moe_results"]
            moe_result = sub_select_layer[:, 0][:, None, None] * moe_result
            moe_result = moe_result.reshape(*result.shape).contiguous()
            self.count_activated_experts(moe_output["moe_experts"], moe_output["moe_type"], sub_select_layer[:, 0])
        else:
            moe_output = moe_layer(x, patch_h, patch_w)
            moe_result = moe_output["moe_results"].reshape(*result.shape).contiguous()
               
            self.count_activated_experts(moe_layer(x, patch_h, patch_w)["moe_experts"], moe_layer(x, patch_h, patch_w)["moe_type"])
        
        if self.training:
            self.balance_loss = moe_layer.balance_loss   
            
        result += moe_result
        result = result.to(previous_dtype)

        return result
