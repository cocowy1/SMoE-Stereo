from __future__ import print_function, division
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from core.utils import misc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import torch.nn.functional as F

from core.utils.dist_utils import get_dist_info, init_dist, setup_for_distributed

from torch.utils.tensorboard import SummaryWriter
from dataloader.datasets import build_dataset
from evaluate_stereo import validate_robust
import core.stereo_datasets as datasets
from core.SMoEStereo import SMoEStereo

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
            pct_start=0.02, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs/robust/')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs/')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs/')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    misc.check_path(args.save_model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    
    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))
        
        setup_for_distributed(args.local_rank == 0)

    model = nn.DataParallel(SMoEStereo(args))
    num_params = sum(p.numel() for p in model.parameters())
    print("Parameter Count: %d, learnable parameter : %d" %(num_params, sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model.to(device),
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank)
    #     model_without_ddp = model.module
    # else:
    #     if torch.cuda.device_count() > 1:
    #         print('Use %d GPUs' % torch.cuda.device_count())
    #         model = torch.nn.DataParallel(model)

    #         model_without_ddp = model.module
    #     else:
    #         model_without_ddp = model

    train_data = build_dataset(args)
    print('=> {} training samples found in the training set'.format(len(train_data)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank
        )
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=train_sampler is None,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              sampler=train_sampler,
                              )

    
    optimizer, scheduler = fetch_optimizer(args, model)
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        print('found the pre-trained model %s'%(args.restore_ckpt))
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=False)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    scaler = GradScaler(enabled=args.mixed_precision)

    global_batch_num = 0
    total_epe = 0.
    total_steps = 0
    
    while total_steps < args.num_steps:
        model.train()

        for i_batch, sample in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            left = sample['left'].to(device).to(torch.float32)  # [B, 3, H, W]
            right = sample['right'].to(device).to(torch.float32)
            gt_disp = sample['disp'].to(device).to(torch.float32) # [B, H, W]
            gt_disp = gt_disp.unsqueeze(1)

            assert model.training
            pred_results = model(left, right, iters=args.train_iters)
            assert model.training

            mask = ((gt_disp > 0) & (gt_disp < args.max_disp))

            disp_loss = 0
            total_loss = 0
            pred_disps = pred_results["disp_preds"]
        
            # loss weights
            loss_weights = [0.9 ** (len(pred_disps) - 1 - power) for power in
                            range(len(pred_disps))]

            for k in range(len(pred_disps)):
                pred_disp = pred_disps[k].to(torch.float32)
                weight = loss_weights[k]
                curr_loss = F.l1_loss(pred_disp[mask], gt_disp[mask],
                                             reduction='mean')
                disp_loss += weight * curr_loss
            
            total_loss += disp_loss
            
            if args.peft_type != 'ff' or args.peft_type != 'frozen':
                moe_balance_loss = pred_results["moe_balance_loss"].sum()
                total_loss +=  0.1 * moe_balance_loss
                
                # layer_loss = torch.FloatTensor([0.])
                # if args.use_layer_selection:
                #     layer_loss = pred_results["layer_loss"].sum()
                #     scale = disp_loss.item() / layer_loss.item()
                #     total_loss += 0.1 * scale * layer_loss

            # if "recon_img" in pred_results:
            #     recon_loss = torch.mean((left - pred_results["recon_img"]) ** 2) 
            #     total_loss += recon_loss
            #     logger.writer.add_scalar("recon_loss", recon_loss.item(), global_batch_num)
            
            # if "layer_lora_ratio" in pred_results:
            #     logger.writer.add_scalar("layer_lora_ratio", 
            #                              pred_results["layer_lora_ratio"].mean().item(), global_batch_num)
            # if "layer_adapter_ratio" in pred_results:
            #     logger.writer.add_scalar("layer_adapter_ratio", 
            #                              pred_results["layer_adapter_ratio"].mean().item(), global_batch_num)
            
            # if "lora_experts" in pred_results:
            #     logger.writer.add_scalar("lora_experts_0", 
            #                              pred_results["lora_experts"].mean(0)[0].item(), global_batch_num)
            #     logger.writer.add_scalar("lora_experts_1", 
            #                              pred_results["lora_experts"].mean(0)[1].item(), global_batch_num)
            #     logger.writer.add_scalar("lora_experts_2", 
            #                              pred_results["lora_experts"].mean(0)[2].item(), global_batch_num)
            #     logger.writer.add_scalar("lora_experts_3", 
            #                              pred_results["lora_experts"].mean(0)[3].item(), global_batch_num)
                 
            # if "adapter_experts" in pred_results:
            #     logger.writer.add_scalar("adapter_experts_0", 
            #                              pred_results["adapter_experts"].mean(0)[0].item(), global_batch_num)
            #     logger.writer.add_scalar("adapter_experts_1", 
            #                              pred_results["adapter_experts"].mean(0)[1].item(), global_batch_num)
            #     logger.writer.add_scalar("adapter_experts_2", 
            #                              pred_results["adapter_experts"].mean(0)[2].item(), global_batch_num)
            #     logger.writer.add_scalar("adapter_experts_3", 
            #                              pred_results["adapter_experts"].mean(0)[3].item(), global_batch_num)
                 
                
            epe = F.l1_loss(gt_disp[mask], pred_disps[-1][mask], reduction='mean')
            total_epe += epe.item()
            logger.writer.add_scalar("disp_loss", disp_loss.item(), global_batch_num)
            logger.writer.add_scalar("total_epe", total_epe/(1+global_batch_num), global_batch_num)
            if args.peft_type != "ff" or args.peft_type != "frozen":
                logger.writer.add_scalar("moe_balance_loss", moe_balance_loss.item(), global_batch_num)

            # if args.peft_type == 'smoe' and args.use_layer_selection:
            #     logger.writer.add_scalar("layer_loss", layer_loss.item(), global_batch_num)

            logger.writer.add_scalar("epe", epe.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1

            ### trick for gradient value is nan
            # parameters = model.parameters()
            # gradients = [param.grad for param in parameters if param.grad is not None]

            # # 检查并替换 NaN 值
            # for grad in gradients:
            #     if torch.isnan(grad).any():
            #         grad[torch.isnan(grad)] = 0  # 将 NaN 值替换为 0 或其他合适的数值
            #         print("detect the unbounded graident, revise it to zero !")

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.99)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if total_steps > 8889 and (total_steps +1) % args.summary_freq == 0:
                print('step: %06d \t mean_epe: %.3f \t lr: %.7f' % (total_steps, total_epe/global_batch_num, scheduler.get_last_lr()[0]))

            if total_steps > 8889 and (total_steps + 1) % args.eval_freq == 0:
                if total_steps > 8889:
                    save_path = Path(args.save_model_path +'/%d_%s.pth' % (total_steps + 1, args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    save_dict = {
                        'model': model.state_dict(),
                    }
                    torch.save(save_dict, save_path)

                val_results = {}
                if 'robust' in args.val_dataset:
                    model.eval()
                    if args.bf16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                            results_dict = validate_robust(model.module, padding_factor=args.padding_factor,
                                            inference_size=args.inference_size,
                                            iterations=args.validate_iters,
                                        )
                    else:
                        results_dict = validate_robust(model.module, padding_factor=args.padding_factor,
                                    inference_size=args.inference_size,
                                    iterations=args.validate_iters,
                                    )

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                model.train()
                model.module.freeze_bn()
            total_steps += 1

    print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='SMoE-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default="/data1/ywang/my_projects/SMoE-Stereo/ckpt/sceneflow/damv2_sceneflow.pth", help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')
    parser.add_argument('--save_model_path', default='/data1/ywang/my_projects/SMoE-Stereo/ckpt/robust/', type=str)
    parser.add_argument('--seed', default=326, type=int)

    # distributed training
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_dataset', nargs='+', default='robust', type=str, help="training datasets.")
    parser.add_argument('--val_dataset', default='robust', type=str, nargs='+')
    parser.add_argument('--bf16', default=False, type=bool)
    parser.add_argument('--num_workers', default=8, type=int)
    
    
    parser.add_argument('--peft_type', default='smoe', choices=["smoe", "lora", "vpt","tuning","adapter","frozen"], type=str)
    parser.add_argument('--vfm_type', default='damv2', choices=["sam","dam","damv2","dinov2"], type=str)
    parser.add_argument('--vfm_size', default='vitb', type=str)
    parser.add_argument('--tunable_layers', default=[0,1,2,3,4,5,6,7,8,9,10,11], type=List)
    parser.add_argument('--use_layer_selection', default=False, type=bool)
    parser.add_argument('--recon',default=False, type=bool, help="recon the img")

    parser.add_argument('--lr', type=float, default=2e-4, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=20000, help="length of training schedule.")
    parser.add_argument('--summary_freq', type=int, default=50, help="length of training schedule.")
    parser.add_argument('--eval_freq', type=int, default=50, help="length of training schedule.")

    parser.add_argument('--img_height', default=320, type=int)
    parser.add_argument('--img_width', default=832, type=int)
    parser.add_argument('--padding_factor', default=32, type=int)
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=2e-5, help="Weight decay in optimizer.")
    parser.add_argument('--feature_maps',default=False, type=bool, help="obtain the feature maps for visualization")

    # Validation parameters
    parser.add_argument('--validate_iters', type=int, default=24, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--max_disp', type=int, default=256, help='number of flow-field updates during validation forward pass')

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
 
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)