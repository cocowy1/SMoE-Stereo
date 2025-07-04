import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import math
from typing import List, Literal, Optional, Union

from core.depthanything_v2.dinov2 import DINOv2
from core.depthanything_v2.util.blocks import FeatureFusionBlock, _make_scratch
from core.depthanything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
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


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        # path_3 = self.scratch.refinenet3(F.interpolate(path_4, scale_factor=2, mode="bilinear", align_corners=True), 
        #                                 layer_3_rn, size=layer_2_rn.shape[2:])
        # path_2 = self.scratch.refinenet2(F.interpolate(path_3, scale_factor=2, mode="bilinear", align_corners=True),
        #                                 layer_2_rn, size=layer_1_rn.shape[2:])
        # path_1 = self.scratch.refinenet1(F.interpolate(path_2, scale_factor=2, mode="bilinear", align_corners=True), layer_1_rn)
        
        # out = self.scratch.output_conv1(path_1)
        # out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # out = self.scratch.output_conv2(out)
        
        # return out
        # return [path_4, path_3, path_2, path_1]
        return [layer_4_rn, layer_3_rn, layer_2_rn, layer_1_rn]

class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        tunable_layers=None,
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl', 'vitg']
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.tunable_layers = tunable_layers
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], 
                                                           patch_h=patch_h, patch_w=patch_w, tunable_layers=self.tunable_layers, return_class_token=True)
        outputs = self.depth_head(features, patch_h, patch_w)
        # depth = F.relu(depth)
        
        return outputs
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)


class damv2_small(DepthAnythingV2):
    def __init__(self, **kwargs):
        super(damv2_small, self).__init__(
           encoder='vits', features=64, out_channels=[48, 96, 192, 384], use_bn=False, use_clstoken=False,)

class damv2_base(DepthAnythingV2):
    def __init__(self, **kwargs):
        super(damv2_base, self).__init__(
           encoder='vitb', features=128, out_channels=[96, 192, 384, 768], use_bn=False, use_clstoken=False,)

class damv2_large(DepthAnythingV2):
    def __init__(self, **kwargs):
        super(damv2_large, self).__init__(
           encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False,)



##################### peft #########################33
intermediate_layer_idx = {
    'vits': [2, 5, 8, 11],
    'vitb': [2, 5, 8, 11], 
    'vitl': [4, 11, 17, 23], 
    'vitg': [9, 19, 29, 39]
}

output_channels = {
    'vits': [48, 96, 192, 384],
    'vitb': [96, 192, 384, 768],
    'vitl': [256, 512, 1024, 1024],
    }
features_dimension = {
    'vits': 64,
    'vitb': 128, 
    'vitl': 256, 
    'vitg': 256
}


class damv2_SMoE(nn.Module):
    prefix: dict = {"lora": "lora_", "adapter": "lora_","smoe": "lora_", "vpt":"prompt_", "tuning": "tuning_"}
    def __init__(self, vfm_size: str, peft_type: str, tunable_layers: Optional[Union[List[int], int]], layer_selection: bool,  *args, **kwargs):
        super(damv2_SMoE, self).__init__()

        self.vfm_size = vfm_size
        self.peft_type = peft_type
        self.layer_selection = layer_selection
        
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
        
        self.vfm_model = DepthAnythingV2(encoder=self.vfm_size, features=features_dimension[self.vfm_size], 
                        out_channels=output_channels[self.vfm_size], tunable_layers=self.tunable_layers, use_bn=False, use_clstoken=False)

        vfm_path = '/data1/ywang/my_projects/SMoEStereo/checkpoints/vfm/damv2/{}.pth'.format(self.vfm_size)
        print('load model from:{}'.format(vfm_path))
        
        state_dict = torch.load(vfm_path, map_location='cuda')
        from core.utils.utils import torch_init_model
        torch_init_model(self.vfm_model, state_dict, key='none')

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

    def _create_and_replace(self, peft_config: Union[LoraConfig, SMoEConfig, AdaMoleConfig, VPTConfig, BlankConfig], peft_type:str, **kwargs) -> None:
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
        
        for i, block in enumerate(self.vfm_model.pretrained.blocks):
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
                
                # print(base_layer)
   
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

    def _convert_list_to_tensor(self, list_convert):
        if len(list_convert) :
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
        
        activated_lora_layers = self._convert_list_to_tensor(target_selected_lora_layer)
        activated_lora_logits = self._convert_list_to_tensor(target_layer_lora_logits)
        
        activated_adapter_layers = self._convert_list_to_tensor(target_selected_adapter_layer)
        activated_adapter_logits = self._convert_list_to_tensor(target_layer_adapter_logits)
        
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
        feature_outputs = self.vfm_model.forward(x)
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
        self.model = damv2_SMoE(vfm_size, peft_type, tunable_layers, layer_selection)
    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            x = torch.cat(x, dim=0)

        outputs = {}
        x_scaled = F.interpolate(x, scale_factor=7/8, mode="bilinear", align_corners=False)
        # with torch.no_grad():
        
        outputs = self.model.forward(x_scaled)
        return outputs