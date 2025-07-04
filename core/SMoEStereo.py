import torch
import torch.nn as nn
import torch.nn.functional as F
from core.peft import lora
from core.update import BasicMultiUpdateBlock
from core.extractor import MatchingDecoder, MultiVFMDecoder, ResidualBlock, MatchingHead, Adapter_Tuning
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8, optimal_transport
from core.decoder import Decoder

import math

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


class Regression(SubModule):
    def __init__(self,
                 max_disparity=192,
                 top_k=3):
        super(Regression, self).__init__()
        self.max_disp = max_disparity
        self.top_k = top_k

    def topkpool(self, cost, k):
        if k == 1:
            _, ind = cost.sort(1, True)
            pool_ind_ = ind[:, :k, ...]
            b, _, _, h, w = pool_ind_.shape
            pool_ind = pool_ind_.new_zeros((b, 1, 3, h, w))
            pool_ind[:, :, 1:2] = pool_ind_
            pool_ind[:, :, 0:1] = torch.max(
                pool_ind_-1, pool_ind_.new_zeros(pool_ind_.shape))
            pool_ind[:, :, 2:] = torch.min(
                pool_ind_+1, self.D*pool_ind_.new_ones(pool_ind_.shape))
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        else:
            _, ind = cost.sort(1, True)
            pool_ind = ind[:, :k, ...]
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        return cv, disp
    
    def forward(self, cost):
        corr, disp = self.topkpool(cost, self.top_k)
        corr = F.softmax(corr, 1)

        init_disp = torch.sum(corr*disp, 1, keepdim=True)
        return init_disp


class SMoEStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
        context_dims = args.hidden_dims
        # self.regression = Regression(self.args.max_disp//4, 3)

        if self.args.vfm_type == 'dam':
            from core.depthanything.dpt import Feature
        elif self.args.vfm_type == 'damv2':
            from core.depthanything_v2.dpt import Feature
        elif self.args.vfm_type == 'sam':
            from core.segment_anything.build_sam import Feature
        else:
            raise ValueError(f"Unsupported model type {(self.args.vfm_type)}")

        self.backbone = Feature(vfm_size=self.args.vfm_size, peft_type=self.args.peft_type, 
                                 tunable_layers=self.args.tunable_layers, layer_selection=self.args.use_layer_selection)

        if self.args.peft_type == 'tuning':
            input_dim = self.args.VFM_dims[0]
            self.adapter_tuning = Adapter_Tuning(in_channels=[input_dim, input_dim, input_dim, input_dim], 
                                          out_channels=[input_dim, input_dim, input_dim, input_dim])
            
        self.cnet = MultiVFMDecoder(input_dim=args.VFM_dims, output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=self.args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], self.args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if self.args.recon:
            self.img_decode = Decoder(256, 64, 2, 64//2)
            
        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = MatchingDecoder(input_dim=args.VFM_dims[0], output_dim=256, norm_fn='instance')
            
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask, iters, test_mode=False):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)

        if not test_mode:
            temp = 1 + 1/ (torch.exp((torch.FloatTensor([iters])-1).cuda()))
            mask = torch.softmax(mask/temp, dim=2)
        else:
            mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)

    def forward(self, image1, image2, iters=16, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        b, _, h, w = image1.shape
        
        outputs = {}
        flow_predictions = []

        # run the context network
        with autocast(enabled=self.args.mixed_precision):

            if self.args.peft_type == 'tuning':
                with torch.no_grad():
                    vfm_output = self.backbone([image1, image2])
                    vfm_features = vfm_output["feature_outputs"]
                vfm_features = self.adapter_tuning(vfm_features)
            
            else:
                vfm_output = self.backbone([image1, image2])
                vfm_features = vfm_output["feature_outputs"]

            if self.args.peft_type == "smoe" and self.args.use_layer_selection == True:
                outputs.update({'layer_adapter_ratio': vfm_output["layer_adapter_ratio"]})
                outputs.update({'layer_lora_ratio': vfm_output["layer_lora_ratio"]})
                outputs.update({'lora_experts': vfm_output["lora_experts"]})
                outputs.update({'adapter_experts': vfm_output["adapter_experts"]})
             
            if self.training:
                outputs.update({'moe_balance_loss': vfm_output["moe_balance_loss"]})

                if self.args.peft_type == "smoe" and self.args.use_layer_selection == True:
                    outputs.update({'layer_loss': vfm_output["layer_loss"]})
                                                
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                left_vfm_feature_output = [left_output.split(split_size=b, dim=0)[0] for left_output in vfm_features]
                cnet_list = self.cnet(left_vfm_feature_output, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet(vfm_features[-1], [image1, image2])
                
                # fmap1, fmap2 = vfm_features[-1].split(split_size=b, dim=0)

            if self.training and self.args.recon:
                recon_img = self.img_decode(fmap1)
                outputs.update({'recon_img': recon_img})
                
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if self.args.feature_maps:
            outputs.update({'fmap1': fmap1})
            outputs.update({'fmap2': fmap2})

        if flow_init is not None:
            coords1 = coords1 + flow_init


        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru: # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask, itr, test_mode=test_mode)
            flow_up = flow_up[:, :1]

            flow_predictions.append(-flow_up)

        outputs.update({'disp_preds': flow_predictions})
        
        return outputs
