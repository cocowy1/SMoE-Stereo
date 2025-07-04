from __future__ import print_function
import sys
from typing import Optional, Union, List

from core import SMoEStereo
import argparse
from core.SMoEStereo import SMoEStereo
from thop import profile
import torch.nn.parallel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with torch.cuda.device(0):
    parser = argparse.ArgumentParser(description='SMoE-Stereo')
    parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')
    parser.add_argument('--model_name', default='smoe', choices=["smoe", "raft"], type=str, nargs='+')
    # hyper-parameters
    parser.add_argument('--max_disp', default=192, type=int,
                        help='exclude very large disparity in the loss function')
    parser.add_argument('--img_height', default=320, type=int)
    parser.add_argument('--img_width', default=896, type=int)
    parser.add_argument('--padding_factor', default=32, type=int)
    
    # mode parameters
    parser.add_argument('--peft_type', default='smoe', choices=["lora", "smoe", "adapter", "tuning", "ff"], type=str)
    parser.add_argument('--vfm_type', default='damv2', choices=["sam","dam","damv2","dinov2"], type=str)
    parser.add_argument('--vfm_size', default='vitb', choices=["vits","vitb","vitl"], type=str)
    
    parser.add_argument('--tunable_layers', default=[0,1,2,3,4,5,6,7,8,9,10,11], type=List)
    parser.add_argument('--use_layer_selection', default=True, type=bool)
    parser.add_argument('--recon',default=False, type=bool, help="recon the img")
    parser.add_argument('--feature_maps', default=True, action='store_false', help='use mixed precision')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--VFM_dims', nargs='+', type=int, default=[128]*3, help="[SAM, VFM_dims = 768], [other VFMs, VFM_dims = 128]")


    # model: parameter-free
    parser.add_argument('--validate_iters', default=24, type=int,
                        help='number of additional local regression refinement')
    parser.add_argument('--proxy',default=False, type=bool, help="hidden state and context dimensions")

    # evaluation
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')
    parser.add_argument('--save_vis_disp', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--middlebury_resolution', default='H', choices=['Q', 'H', 'F'])

    args = parser.parse_args()

    model = SMoEStereo(args)
    model.cuda() 
    model.eval()
    left = torch.randn(1, 3, 256, 512).cuda()
    right = torch.randn(1, 3, 256, 512).cuda()
    macs, params = profile(model, inputs=(left, right), verbose=True)
    print('{:<30}  {:<10}'.format('Computational complexity: ', macs/1000000000))
    print('{:<30}  {:<10}'.format('Number of parameters: ', params/1000000))

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
