# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from core.segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from typing import List, Literal, Optional, Union

from core.peft.lora.config import LoraConfig
from core.peft.lora.layer import LinearLoraLayer

from core.peft.smoe.config import SMoEConfig
from core.peft.smoe.layer import LinearSMoELayer

from core.peft.adamole.config import AdaMoleConfig
from core.peft.adamole.layer import LinearAdaMoleLayer

from core.peft.vpt.config import VPTConfig
from core.peft.vpt.layer import VPTLayer

from core.peft.blank.config import BlankConfig
from core.peft.blank.layer import BlankLayer  


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h

def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vith": build_sam_vit_h,
    "vitl": build_sam_vit_l,
    "vitb": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            print('loading model from:{}'.format(checkpoint))
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=True)
    return sam


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



##################### peft #########################33

class sam_SMoE(nn.Module):
    prefix: dict = {"lora": "lora_", "adapter": "lora_","smoe": "lora_", "vpt":"prompt_", "tuning": "tuning_"}
    def __init__(self, vfm_size: str, peft_type: str, tunable_layers: Optional[Union[List[int], int]], layer_selection: bool,  *args, **kwargs):
        super(sam_SMoE, self).__init__()

        self.vfm_size = vfm_size
        self.peft_type = peft_type
        self.layer_selection = layer_selection
        self.encoder_global_attn_indexes = [2,5,8,11]

        if self.peft_type == 'smoe':
            self.config = SMoEConfig()
        elif self.peft_type == 'lora':
            self.config = LoraConfig()
        elif self.peft_type == 'adapter':
            self.config = LoraConfig()
        elif self.peft_type == 'vpt':
            self.config = VPTConfig()
        elif self.peft_type == 'tuning' or self.peft_type == 'ff':
            self.config = BlankConfig()
        else:
            raise NotImplementedError

        # self.tunable_layers = self.config.target_modules
        self.tunable_layers = tunable_layers
        self.layer_target_ratio = 0.6
        self.layer_entropy_weight = 0.1
        
        vfm_path = '/data1/ywang/my_projects/SMoEStereo/checkpoints/vfm/sam/{}.pth'.format(self.vfm_size)
        self.vfm_model = sam_model_registry[self.vfm_size](vfm_path)

        # from core.utils.utils import torch_init_model
        # torch_init_model(self.vfm_model, state_dict, key='none')

        # self.vfm_model.eval()
        
        self._create_and_replace(peft_config=self.config, peft_type=self.peft_type)
        for name, param in self.vfm_model.named_parameters():
            param.requires_grad = True
            
        if self.peft_type != 'ff':
            for name, param in self.vfm_model.named_parameters():
                if self.prefix[self.peft_type] not in name:
                    param.requires_grad = False

        num_params = sum(p.numel() for p in self.vfm_model.parameters())
        print("VFM model Parameter Count: %d, VFM's learnable parameter : %d" %(num_params, 
                                            sum(p.numel() for p in self.vfm_model.parameters() if p.requires_grad)))


    def _create_and_replace(self, peft_config: Union[LoraConfig, SMoEConfig, VPTConfig], peft_type:str, **kwargs) -> None:
        """
        Inplace replacement of the target module with the adapter layer
        """
        if peft_type == 'smoe' or peft_type == 'lora' or peft_type == 'adapter':
            lora_kwargs = {
            "lora_rank": peft_config.lora_rank,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
            "init_lora_weights": peft_config.init_lora_weights,
            "top_k": peft_config.top_k,
            "threshold": peft_config.threshold,
            }
            
        elif peft_type == 'vpt':
            vpt_kwargs = {
            "prompt_tokens": peft_config.prompt_tokens,
            "prompt_project": peft_config.prompt_project,
            "prompt_dropout": peft_config.prompt_drop,
            "prompt_initiation": peft_config.prompt_initiation,
            }
        elif peft_type == 'tuning' or peft_type == 'ff':
            pass
        
        
        for i, block in enumerate(self.vfm_model.image_encoder.blocks):
            if i not in self.tunable_layers:
                continue
    
            if isinstance(block, nn.Module):
            
                base_layer = block.attn.qkv
                if self.peft_type == 'lora':
                    base_layer = LinearLoraLayer(base_layer=base_layer, lora_type='linear', **lora_kwargs)
                elif self.peft_type == 'vpt':
                    base_layer = VPTLayer(base_layer=base_layer, **vpt_kwargs)
                elif self.peft_type == 'smoe':
                    base_layer = LinearSMoELayer(base_layer=base_layer, lora_type='linear', layer_selection=self.layer_selection, **lora_kwargs)
                elif self.peft_type == 'tuning' or self.peft_type == 'ff':
                    base_layer = BlankLayer(base_layer=base_layer, **kwargs)
                block.attn.qkv = base_layer
                
            ## print(base_layer)

                base_layer = block.mlp
                if self.peft_type == 'adapter':
                    base_layer = LinearLoraLayer(base_layer=base_layer, lora_type='conv2d', **lora_kwargs)
                elif self.peft_type == 'vpt':
                    base_layer = VPTLayer(base_layer=base_layer, **vpt_kwargs)
                elif self.peft_type == 'smoe':
                    base_layer = LinearSMoELayer(base_layer=base_layer, lora_type='conv2d', layer_selection=self.layer_selection, **lora_kwargs)
                elif self.peft_type == 'tuning' or self.peft_type == 'ff':
                    base_layer = BlankLayer(base_layer=base_layer, **kwargs)
                block.mlp = base_layer

                ## print(base_layer)
   
        # for name, module in self.vfm_model.depth_head.scratch.named_modules():
        #     if '_rn' in name:
        #         if isinstance(module, nn.Conv2d):
        #             base_layer = module
        #             base_layer = LinearLoraLayer(base_layer=base_layer, lora_type='conv2d', **kwargs)
        #             setattr(self.vfm_model.depth_head.scratch, name, base_layer)
        #             print(base_layer)


    def get_aux_loss(self) -> torch.Tensor:
        """
        Get the load balancing loss for the whole model
        """
        moe_balance_loss = torch.tensor([0], dtype=torch.float).cuda()

        for name, module in self.vfm_model.named_modules():
            if isinstance(module, LinearSMoELayer):
                if  module.balance_loss is not None:
                    balance_loss = module.balance_loss
                    moe_balance_loss += balance_loss

        return moe_balance_loss

    def _convert_list_to_tensor(self, list_convert, encoder_global_attn_indexes=None):
        if len(list_convert):
            if encoder_global_attn_indexes is None:
                result = torch.stack(list_convert, dim=1)
            else:
                list_convert = [list_convert[i] for i in range(len(list_convert)) if i not in encoder_global_attn_indexes]
                result = torch.stack(list_convert, dim=1)
        else :
            result = None
        return result 

    def binaray_entropy(self, prob, eps=1e-6) -> torch.Tensor:
        neg_entro = prob * prob.clamp(min=eps).log() + (1-prob) * (1-prob).clamp(min=eps).log()
        return - neg_entro
    
    def count_activated_experts(self)-> torch.Tensor:
        target_selected_lora_experts = []
        target_selected_adapter_experts = []
        
        for name, module in self.vfm_model.named_modules():
            if isinstance(module, LinearSMoELayer):
                if 'attn' in name: 
                    target_selected_lora_experts.append(module.experts)
                elif 'mlp' in name:
                    target_selected_adapter_experts.append(module.experts)
                else:
                    pass
        
        target_selected_lora_experts = torch.cat(target_selected_lora_experts, 0)   
        target_selected_adapter_experts = torch.cat(target_selected_adapter_experts, 0)   
              
        return [target_selected_lora_experts.mean(0).unsqueeze(0), target_selected_adapter_experts.mean(0).unsqueeze(0)]
    

    def count_activated_layers(self)-> torch.Tensor:
        target_selected_lora_layer = []
        target_layer_lora_logits = []
        
        target_selected_adapter_layer = []
        target_layer_adapter_logits = []
        
        for name, module in self.vfm_model.named_modules():
            if isinstance(module, LinearSMoELayer):
                if 'attn' in name: 
                    target_selected_lora_layer.append(module.sub_select_layer)
                    target_layer_lora_logits.append(module.layer_logits)
                elif 'mlp' in name:
                    target_selected_adapter_layer.append(module.sub_select_layer)
                    target_layer_adapter_logits.append(module.layer_logits)
                else:
                    pass
        
        activated_lora_layers = self._convert_list_to_tensor(target_selected_lora_layer, self.encoder_global_attn_indexes)
        activated_lora_logits = self._convert_list_to_tensor(target_layer_lora_logits, self.encoder_global_attn_indexes)
        
        activated_adapter_layers = self._convert_list_to_tensor(target_selected_adapter_layer, self.encoder_global_attn_indexes)
        activated_adapter_logits = self._convert_list_to_tensor(target_layer_adapter_logits, self.encoder_global_attn_indexes)
        
        return [activated_lora_layers, activated_adapter_layers], [activated_lora_logits, activated_adapter_logits]
    
    
    def _get_layer_loss(self, lora_layer_weight=1, adapter_layer_weight=1) -> torch.Tensor:
        layer_loss = torch.tensor([0], dtype=torch.float).cuda()
        activated_layers, activated_logits = self.count_activated_layers()
        
        layer_lora_mean = activated_layers[0][:,:, 0].mean()
        layer_adapter_mean = activated_layers[1][:, :, 0].mean()
        
        layer_flops_loss = lora_layer_weight * (activated_layers[0][:,:, 0].mean() -0.7).abs().mean() + \
                            adapter_layer_weight * (activated_layers[1][:,:, 0].mean() - 0.5).abs().mean()

        if self.layer_entropy_weight > 0:
            layer_entropy = (self.binaray_entropy(activated_logits[0].sigmoid()).mean() +  \
                            self.binaray_entropy(activated_logits[1].sigmoid()).mean()) /2
        else:
            layer_entropy = 0

        layer_loss = layer_flops_loss - self.layer_entropy_weight * layer_entropy
        return layer_loss.unsqueeze(0), layer_lora_mean.unsqueeze(0), layer_adapter_mean.unsqueeze(0)
    
    
    
    def forward(self, x):
        outputs = {}
        feature_outputs = self.vfm_model.forward_features(x, multimask_output=False, tunable_layers=self.tunable_layers)
        outputs.update({'feature_outputs': feature_outputs})
        
        if self.peft_type == 'smoe' and self.layer_selection == True:

            layer_loss, layer_lora_ratio, layer_adapter_ratio = self._get_layer_loss()
            outputs.update({'layer_lora_ratio': layer_lora_ratio})
            outputs.update({'layer_adapter_ratio': layer_adapter_ratio})
            
            lora_experts, adapter_experts = self.count_activated_experts()
            outputs.update({'lora_experts': lora_experts})
            outputs.update({'adapter_experts': adapter_experts})
            
            
        if self.training:
            moe_balance_loss = self.get_aux_loss()
            outputs.update({'moe_balance_loss': moe_balance_loss})
            
            if self.peft_type == 'smoe' and self.layer_selection == True:
                outputs.update({'layer_loss': layer_loss})

        return outputs



class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Feature(SubModule):
    def __init__(self, vfm_size='vitb', peft_type='smoe', tunable_layers=[2,5,8,11], layer_selection=False):
        super(Feature, self).__init__()
        self.model = sam_SMoE(vfm_size, peft_type, tunable_layers, layer_selection)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            x = torch.cat(x, dim=0)

        outputs = self.model.forward(x)
        return outputs