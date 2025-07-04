import torch
import torch.nn as nn
import torch.nn.functional as F
from core.depthanything_v2.dpt import Feature

from core.depthanything_v2.util.blocks import FeatureFusionBlock
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


class Adapter_Tuning(nn.Module):
    def __init__(
        self,
        in_channels=[128, 128, 128, 128], # Added in_channels parameter
        out_channels=[128, 128, 128, 128],
        use_bn=False,
    ):
        super(Adapter_Tuning, self).__init__()

        if len(in_channels) != len(out_channels):
            raise ValueError("Length of in_channels and out_channels must be equal.")

        self.scratch = self._make_scratch(in_channels, out_channels, use_bn)
        self.scratch.stem_transpose = None

    def _make_scratch(self, in_channels, out_channels, use_bn):
        scratch = nn.Module()
        
        if len(in_channels) != len(out_channels):
            raise ValueError("Length of in_channels and out_channels must match.")

        for i, (in_c, out_c) in enumerate(zip(in_channels, out_channels)):
            layer = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
            setattr(scratch, f"layer{i+1}_rn", layer)
            setattr(scratch, f"refinenet{i+1}", _make_fusion_block(out_c, use_bn)) # Add refinenet block after each conv layer

        return scratch

    
    def forward(self, x):

        layer_4, layer_3, layer_2, layer_1 = x[0], x[1], x[2], x[3]
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(F.interpolate(path_4, scale_factor=2, mode="bilinear", align_corners=True), 
                                        layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(F.interpolate(path_3, scale_factor=2, mode="bilinear", align_corners=True),
                                        layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(F.interpolate(path_2, scale_factor=2, mode="bilinear", align_corners=True), layer_1_rn)

        return [path_4, path_3, path_2, path_1]



class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x, dual_inp=False):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x

class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)



class MultiVFMDecoder(nn.Module):
    def __init__(self, input_dim=[128], output_dim=[128], norm_fn='batch', dropout=0.0):
        super(MultiVFMDecoder, self).__init__()
        self.norm_fn = norm_fn

        output_list = []
        
        self.vfm_08 =None
        if input_dim[2] != output_dim[0][2]:
            self.vfm_08 = nn.Sequential(nn.Conv2d(input_dim[2], output_dim[0][2], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][2]),
                                        nn.ReLU(inplace=True)
                                        )
            
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(dim[2], dim[2], self.norm_fn, stride=1),
                nn.Conv2d(dim[2], dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        
        self.vfm_16 = None
        if input_dim[1] != output_dim[0][1]:
            self.vfm_16 = nn.Sequential(nn.Conv2d(input_dim[1], output_dim[0][1], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][1]),
                                        nn.ReLU(inplace=True)
                                        )
            
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(dim[1], dim[1], self.norm_fn, stride=1),
                nn.Conv2d(dim[1], dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        
        self.vfm_32 = None
        if input_dim[0] != output_dim[0][0]:
            self.vfm_32 = nn.Sequential(nn.Conv2d(input_dim[0], output_dim[0][0], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][0]),
                                        nn.ReLU(inplace=True)
                                        )
            
        for dim in output_dim:
            conv_out = nn.Conv2d(dim[0], dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, vfm_output, num_layers=3):
        vfm_outputs08 = vfm_output[-1]
        if self.vfm_08:
            vfm_outputs08 = self.vfm_08(vfm_outputs08)
        outputs08 = [f(vfm_outputs08) for f in self.outputs08]
        if num_layers == 1:
            return outputs08

        vfm_outputs16 = vfm_output[-2]
        if self.vfm_16:
            vfm_outputs16 = self.vfm_08(vfm_outputs16)
        outputs16 = [f(vfm_outputs16) for f in self.outputs16]

        if num_layers == 2:
            return outputs08, outputs16

        vfm_outputs32 = vfm_output[-3]
        if self.vfm_32:
            vfm_outputs32 = self.vfm_32(vfm_outputs32)
        outputs32 = [f(vfm_outputs32) for f in self.outputs32]

        return outputs08, outputs16, outputs32


class BasicConv_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU(inplace=True)(x)  
        return x


class MatchingHead(nn.Module):
    def __init__(self, model_size='base', input_dim=128, output_dim=[128], norm_fn='instance', dropout=0.0, downsample=3):
        super(MatchingHead, self).__init__()
        self.model_size = model_size
        self.norm_fn = norm_fn
        self.downsample = downsample

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 64, kernel_size=7, stride=2, padding=3),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU()
            )

        self.final_output = nn.Sequential(
            nn.Conv2d(128 + input_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, vfm_output, x, dual_inp=False):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        stem_2 = self.stem_2(x)
        stem_4 = self.stem_4(stem_2)

        vfm_output = torch.cat([vfm_output, stem_4], dim=1)
        output = self.final_output(vfm_output)
        
        if is_list:
            output = output.split(split_size=batch_dim, dim=0)

        return output


class MatchingDecoder(nn.Module):
    def __init__(self, model_size='base', input_dim=128, output_dim=[128], norm_fn='instance', dropout=0.0, downsample=3):
        super(MatchingDecoder, self).__init__()
        self.model_size = model_size
        self.norm_fn = norm_fn
        self.downsample = downsample

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        
        self.layer_1 = self._make_layer(64, stride=1)
        self.layer_2 = self._make_layer(128, stride=2)

        self.final_output = nn.Sequential(
            nn.Conv2d(128 + input_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, vfm_output, x, dual_inp=False):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        
        vfm_output = torch.cat([vfm_output, x], dim=1)
        output = self.final_output(vfm_output)
        
        if is_list:
            output = output.split(split_size=batch_dim, dim=0)

        return output
