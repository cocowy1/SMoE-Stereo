import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8
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



class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
        context_dims = args.hidden_dims
        self.regression = Regression(args.max_disp//8, 3)

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def init_disp(self, left, right):
        disparity_range =  self.args.max_disp//8
        b, _, h, w = left.shape
        cost_volume = left.new_zeros(b, disparity_range, h, w)

        for i in range(disparity_range):
            if i > 0:
                cost_volume[:, i, :, i:] = (left[:, :, :, i:] *
                                                right[:, :, :, :-i]).mean(dim=1)
            else:
                cost_volume[:, i, :, :] = (left * right).mean(dim=1)

        cost_volume = cost_volume.contiguous()

        disp_pred = self.regression(cost_volume)

        return disp_pred
   
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

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        outputs = {}
        flow_predictions = []

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])
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

        if self.args.proxy:
            outputs.update({'fmap1': fmap1})
            outputs.update({'fmap2': fmap2})

        if flow_init is not None:
            coords1 = coords1 + flow_init

        # disp_init = self.init_disp(fmap1, fmap2)
        # flow_init = torch.cat((-disp_init, torch.zeros_like(disp_init)), dim=1)
        # coords1 = coords1 + flow_init

        # flow_init_up = F.interpolate(flow_init, scale_factor=4, mode='bilinear')
        # flow_predictions.append(-flow_init_up[:,:1])

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

        if test_mode:
            return outputs
        
        return outputs
