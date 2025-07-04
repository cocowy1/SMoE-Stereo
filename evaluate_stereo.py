from pickle import FALSE, TRUE
import time
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from ast import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
from tqdm import tqdm 
from imageio import imread, imsave
import cv2
from PIL import Image
from glob import glob

from core.stereo_metric import d1_metric, thres_metric
from dataloader.datasets import (Driving, RobustDrivingStereo, DrivingStereo, MS2, KITTI12, KITTI15, ETH3DStereo, MiddleburyEval3)
from dataloader import transforms
from dataloader.utils import InputPadder
from dataloader.utils import write_pfm
from core.utils.visualization import vis_disparity

from core.raft_stereo import RAFTStereo
from core.SMoEStereo import SMoEStereo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def create_kitti_submission(model,
                            output_path='output/disp_0/',
                            padding_factor=32,
                            inference_size=None,
                            iterations=24,
                            ):
    """ create submission for the KITTI leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]


    val_transform = transforms.Compose(val_transform_list)

    test_dataset = KITTI15(mode='testing', transform=val_transform)

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, sample in tqdm(enumerate(test_dataset)):
        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        pred_disp = model(image1=left, image2=right,
                          iters=iterations,
                          test_mode=True,
                          )['disp_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        save_name = os.path.join(output_path, left_name)

        imsave(save_name, (pred_disp.squeeze().cpu().numpy() * 256.).astype(np.uint16))


@torch.no_grad()
def create_eth3d_submission(model,
                            output_path='output/low_res_two_view/',
                            padding_factor=32,
                            inference_size=None,
                            iterations=24,
                            save_vis_disp=False,
                            ):
    """ create submission for the eth3d stereo leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    test_dataset = ETH3DStereo(mode='test',
                               transform=val_transform,
                               save_filename=True)

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, sample in tqdm(enumerate(test_dataset)):
        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        # warpup to measure inference time
        if i == 0:
            for _ in range(5):
                model(image1=left, image2=right,
                      iters=iterations,
                      test_mode=True,
                      )

        torch.cuda.synchronize()
        time_start = time.perf_counter()

        pred_disp = model(image1=left, image2=right,
                          iters=iterations,
                          test_mode=True,
                          )['disp_preds'][-1]  # [1, H, W]

        torch.cuda.synchronize()
        inference_time = time.perf_counter() - time_start

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        filename = os.path.basename(os.path.dirname(left_name))

        if save_vis_disp:
            save_name = os.path.join(output_path, filename + '.png')
            disp = vis_disparity(pred_disp.squeeze().cpu().numpy())
            cv2.imwrite(save_name, disp)
        else:
            save_disp_name = os.path.join(output_path, filename + '.pfm')
            # save disp
            write_pfm(save_disp_name, pred_disp.squeeze().cpu().numpy())
            # save runtime
            save_runtime_name = os.path.join(output_path, filename + '.txt')
            with open(save_runtime_name, 'w') as f:
                f.write('runtime ' + str(inference_time))



@torch.no_grad()
def create_middlebury_submission(model,
                                output_path='output/testH/',
                                padding_factor=32,
                                inference_size=None,
                                iterations=32,
                                submission_type='test',
                                save_vis_disp=True,
                                ):
    """ create submission for the Middlebury leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    test_dataset = MiddleburyEval3(mode=submission_type,
                                   resolution='H',
                                   transform=val_transform,
                                   save_filename=True,
                                   half_resolution=False
                                   )

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, sample in tqdm(enumerate(test_dataset)):
        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        disp = sample['disp'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        # warpup to measure inference time
        if i == 0:
            for _ in range(5):
                model(image1=left, image2=right,
                      iters=iterations,
                      test_mode=True,
                      )

        torch.cuda.synchronize()
        time_start = time.perf_counter()

        pred_disp = model(image1=left, image2=right,
                          iters=iterations,
                          test_mode=True,
                          )['disp_preds'][-1] # [1, H, W]

        torch.cuda.synchronize()
        inference_time = time.perf_counter() - time_start

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        filename = os.path.basename(os.path.dirname(left_name))  # works for both windows and linux

        if save_vis_disp:
            img_path = 'output/mid_test/'
            save_name = os.path.join(img_path, filename + '.png')
            pred_disp = vis_disparity(pred_disp.squeeze().cpu().numpy())
            color_disp = cv2.applyColorMap(pred_disp, cv2.COLORMAP_JET)
            cv2.imwrite(save_name, color_disp)
        else:
            save_disp_dir = os.path.join(output_path, filename)
            os.makedirs(save_disp_dir, exist_ok=True)

            save_disp_name = os.path.join(save_disp_dir, 'disp0SMoEStereo.pfm')
            # save disp
            write_pfm(save_disp_name, pred_disp.squeeze().cpu().numpy())
            # save runtime
            save_runtime_name = os.path.join(save_disp_dir, 'timeSMoEStereo.txt')
            with open(save_runtime_name, 'w') as f:
                f.write(str(inference_time-0.3))       
                
@torch.no_grad()
def validate_robust(model,
                     padding_factor=32,
                     inference_size=None,
                     iterations=32,
                     count_time=True,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    # val_transform_list = None
    
    val_transform = transforms.Compose(val_transform_list)

    val_dataset_15 = KITTI15(transform=val_transform,)

    val_dataset_12 = KITTI12(transform=val_transform,)
    
    val_dataset_mid = MiddleburyEval3(transform=val_transform,)

    val_dataset_eth3d = ETH3DStereo(transform=val_transform,)

    val_datasets = {"kit12": val_dataset_12, "kit15": val_dataset_15, "mid": val_dataset_mid, "eth3d": val_dataset_eth3d,}

    # val_datasets = { "mid": val_dataset_mid, "eth3d": val_dataset_eth3d,}


    for name, val_dataset in val_datasets.items():
        num_samples = len(val_dataset)
        print('=> %d samples found in the validation %s dataset' %(num_samples, name))

        val_epe = 0
        val_d1 = 0
        val_thres3 = 0
        val_thres2 = 0
        val_thres1 = 0

        selection_lora_ratio = 0.
        selection_adapter_ratio = 0.
        
        valid_samples = 0
        elapse_counter = 0
        elapse_time = 0
        avgtime = 0
        
        
        lora_experts_0 = 0
        lora_experts_1 = 0
        lora_experts_2 = 0
        lora_experts_3 = 0
        
        adapter_experts_0 = 0
        adapter_experts_1 = 0
        adapter_experts_2 = 0
        adapter_experts_3 = 0
            
        for i, sample in tqdm(enumerate(val_dataset)):

            left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            gt_disp = sample['disp'].to(device).to(torch.float32).unsqueeze(0)  # [H, W]

            if inference_size is None:
                padder = InputPadder(left.shape, padding_factor=padding_factor)
                left, right = padder.pad(left, right)
            else:
                ori_size = left.shape[-2:]
                left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
                right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

            valid_samples += 1

            if i > 10:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                elapse_counter += 1
                
            with torch.no_grad():
                outputs = model(image1=left, image2=right,
                              iters=iterations, 
                              test_mode=True
                              ) # [1, H, W]
            
            if i > 10:
                elapse_time += time.perf_counter() - start_time
          
            pred_disp = outputs['disp_preds'][-1] 
            
            if 'layer_lora_ratio' in outputs:
                layer_lora_ratio = outputs["layer_lora_ratio"].unsqueeze(0).mean().item()
                selection_lora_ratio += layer_lora_ratio
            
            if 'layer_adapter_ratio' in outputs:
                layer_adapter_ratio = outputs["layer_adapter_ratio"].unsqueeze(0).mean().item()
                selection_adapter_ratio += layer_adapter_ratio
                
            if "lora_experts" in outputs:
                lora_experts_0 += outputs["lora_experts"].mean(0)[0].item()
                lora_experts_1 += outputs["lora_experts"].mean(0)[1].item()
                lora_experts_2 += outputs["lora_experts"].mean(0)[2].item()
                lora_experts_3 += outputs["lora_experts"].mean(0)[3].item()
                 
            if "adapter_experts" in outputs:
                adapter_experts_0 += outputs["adapter_experts"].mean(0)[0].item()
                adapter_experts_1 += outputs["adapter_experts"].mean(0)[1].item()
                adapter_experts_2 += outputs["adapter_experts"].mean(0)[2].item()
                adapter_experts_3 += outputs["adapter_experts"].mean(0)[3].item()
                     
            # remove padding
            if inference_size is None:
                pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
            else:
                # resize back
                pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
                pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

            pred_disp = pred_disp.squeeze()
            gt_disp = gt_disp.squeeze()
            

            mask = (gt_disp > 0.5) & (gt_disp < 192) 
            if not mask.any():
                continue
                
            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            d1 = d1_metric(pred_disp, gt_disp, mask)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
            thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

            val_epe += epe.item()
            val_d1 += d1.item()
            val_thres3 += thres3.item()
            val_thres2 += thres2.item()
            val_thres1 += thres1.item()
        
        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples * 100
        mean_thres3 = val_thres3 / valid_samples * 100
        mean_thres2 = val_thres2 / valid_samples * 100
        mean_thres1 = val_thres1 / valid_samples * 100
        avg_lora_ratio = selection_lora_ratio / valid_samples * 100
        avg_adapter_ratio = selection_adapter_ratio / valid_samples * 100
        avgtime = elapse_time / elapse_counter
        
        avg_lora_experts_0 = lora_experts_0 / valid_samples * 100
        avg_lora_experts_1 = lora_experts_1 / valid_samples * 100
        avg_lora_experts_2 = lora_experts_2 / valid_samples * 100
        avg_lora_experts_3 = lora_experts_3 / valid_samples * 100
    
        avg_adapter_experts_0 = adapter_experts_0 / valid_samples * 100
        avg_adapter_experts_1 = adapter_experts_1 / valid_samples * 100
        avg_adapter_experts_2 = adapter_experts_2 / valid_samples * 100
        avg_adapter_experts_3 = adapter_experts_3 / valid_samples * 100
        
        print('Validation %s dataset EPE: %.3f, D1: %.4f, 1px: %.4f,  2px: %.4f, 3px: %.4f, layer_lora_ratio: %.3f, layer_adapter_ratio: %.3f, \
            avg_lora_experts_0:%.2f, avg_lora_experts_1:%.2f, avg_lora_experts_2:%.2f, avg_lora_experts_3:%.2f, \
            avg_adapter_experts_0:%.2f, avg_adapter_experts_1:%.2f, avg_adapter_experts_2:%.2f, avg_adapter_experts_3:%.2f, avgtime: %.4f' % (
            name, mean_epe, mean_d1, mean_thres1, mean_thres2, mean_thres3, avg_lora_ratio, avg_adapter_ratio, 
            avg_lora_experts_0, avg_lora_experts_1, avg_lora_experts_2, avg_lora_experts_3, avg_adapter_experts_0,
            avg_adapter_experts_1, avg_adapter_experts_2, avg_adapter_experts_3, avgtime))

        results[str(name) + '_epe'] = mean_epe
        results[str(name) + '_d1'] = mean_d1
        results[str(name) + '_3px'] = mean_thres3
        results[str(name) + '_2px'] = mean_thres2
        results[str(name) + '_1px'] = mean_thres1
        results[str(name) + '_avg_lora_ratio'] = avg_lora_ratio
        results[str(name) + '_avg_adapter_ratio'] = avg_adapter_ratio
        results[str(name) + '_avgtime'] = avgtime
    return results


@torch.no_grad()
def validate_kitti12(model,
                     padding_factor=16,
                     inference_size=None,
                     iterations=24,
                     count_time=True,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = KITTI12(transform=val_transform,
                          )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0
    val_thres1 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device).to(torch.float32).unsqueeze(0)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              iters=iterations,
                              )['disp_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        pred_disp = pred_disp.squeeze()
        gt_disp = gt_disp.squeeze()
        mask = gt_disp > 0
        
        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples * 100
    mean_thres3 = val_thres3 / valid_samples * 100
    mean_thres1 = val_thres1 / valid_samples * 100

    print('Validation KITTI12 EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres1, mean_thres3))

    results['kitti12_epe'] = mean_epe
    results['kitti12_d1'] = mean_d1
    results['kitti12_3px'] = mean_thres3
    results['kitti12_1px'] = mean_thres1

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def validate_kitti15(model,
                     padding_factor=16,
                     inference_size=None,
                     iterations=24,
                     count_time=True,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = KITTI15(transform=val_transform,
                          )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0
    val_thres1 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device).to(torch.float32).unsqueeze(0)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              iters=iterations,
                              )['disp_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        pred_disp = pred_disp.squeeze()
        gt_disp = gt_disp.squeeze()
        mask = gt_disp > 0
        
        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples * 100
    mean_thres3 = val_thres3 / valid_samples * 100
    mean_thres1 = val_thres1 / valid_samples * 100

    print('Validation KITTI15 EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres1, mean_thres3))

    results['kitti15_epe'] = mean_epe
    results['kitti15_d1'] = mean_d1
    results['kitti15_3px'] = mean_thres3
    results['kitti15_1px'] = mean_thres1

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def validate_eth3d(model,
                   padding_factor=16,
                   inference_size=None,
                   iterations=24,
                   ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = ETH3DStereo(transform=val_transform,
                              )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres1 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              iters=iterations,
                              )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_thres1 = val_thres1 / valid_samples

    print('Validation ETH3D EPE: %.3f, 1px: %.4f' % (
        mean_epe, mean_thres1))

    results['eth3d_epe'] = mean_epe
    results['eth3d_1px'] = mean_thres1

    return results


@torch.no_grad()
def validate_middlebury(model,
                        padding_factor=16,
                        inference_size=None,
                        iterations=24,
                        resolution='H',
                        ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = MiddleburyEval3(transform=val_transform,
                                  resolution=resolution,
                                  )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres2 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]

            left = F.interpolate(left, size=inference_size,
                                 mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size,
                                  mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              iters=iterations,
                              )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres2 += thres2.item()

    mean_epe = val_epe / valid_samples
    mean_thres2 = val_thres2 / valid_samples

    print('Validation Middlebury EPE: %.3f, 2px: %.4f' % (
        mean_epe, mean_thres2))

    results['middlebury_epe'] = mean_epe
    results['middlebury_2px'] = mean_thres2

    return results


@torch.no_grad()
def validate_drivingstereo(model,
                     max_disp=128,
                     padding_factor=64,
                     num_reg_refine=1,
                     data_path=None,
                     inference_size=None,
                     attn_type=None,
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = DrivingStereo(transform=val_transform,
                                mode="testing",)

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0
    val_thres1 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if debug and i > 10:
            break

        if i % 5000 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0
        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              max_disp=max_disp,
                              num_reg_refine=num_reg_refine,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              )['disp_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp.unsqueeze(0).unsqueeze(0))[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        pred_disp = pred_disp.squeeze()
        gt_disp = gt_disp.squeeze()
        mask = gt_disp > 0
        
        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples * 100
    mean_thres3 = val_thres3 / valid_samples * 100
    mean_thres1 = val_thres1 / valid_samples * 100

    print('Validation drivingstereo EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres1, mean_thres3))

    results['drivingstereo_epe'] = mean_epe
    results['drivingstereo_d1'] = mean_d1
    results['drivingstereo_3px'] = mean_thres3
    results['drivingstereo_1px'] = mean_thres1

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def validate_robust_weather(model,
                     padding_factor=32,
                     inference_size=None,
                     iterations=32,
                     count_time=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    data_path = '/data1/ywang/dataset/DrivingStereo/DrivingStereo/adverse_weather/'
    weathers = ["clear", "cloudy", "foggy", "rainy"]

    for weather in weathers:
        val_dataset = RobustDrivingStereo(data_dir=data_path+weather,
                                transform=val_transform, mode="testing")
    
        num_samples = len(val_dataset)
        print('=> %d samples found in the validation %s weather set' % (num_samples, weather))

        val_epe = 0
        val_d1 = 0
        val_thres3 = 0
        val_thres1 = 0

        if count_time:
            total_time = 0
            num_runs = 100

        valid_samples = 0

        for i, sample in tqdm(enumerate(val_dataset)):

            if i % 100 == 0:
                print('=> Validating %d/%d' % (i, num_samples))

            left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

            if inference_size is None:
                padder = InputPadder(left.shape, padding_factor=padding_factor)
                left, right = padder.pad(left, right)
            else:
                ori_size = left.shape[-2:]
                left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
                right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

            mask = (gt_disp > 0) & (gt_disp < 192)
            # if not mask.any():
            #     continue

            valid_samples += 1

            if count_time and i >= 5:
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            with torch.no_grad():
                pred_disp = model(image1=left, image2=right,
                              iters=iterations, 
                              test_mode=True
                              )['disp_preds'][-1]  # [1, H, W]

            if count_time and i >= 5:
                torch.cuda.synchronize()
                total_time += time.perf_counter() - time_start

            # remove padding
            if inference_size is None:
                pred_disp = padder.unpad(pred_disp.unsqueeze(0).unsqueeze(0))[0]  # [H, W]
            else:
                # resize back
                pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
                pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

            pred_disp = pred_disp.squeeze()
            gt_disp = gt_disp.squeeze()
            mask = (gt_disp > 0)

            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            d1 = d1_metric(pred_disp, gt_disp, mask)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

            val_epe += epe.item()
            val_d1 += d1.item()
            val_thres3 += thres3.item()
            val_thres1 += thres1.item()

        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples * 100
        mean_thres3 = val_thres3 / valid_samples * 100
        mean_thres1 = val_thres1 / valid_samples * 100

        print('Validation robust drivingstereo %s weather subset EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        weather, mean_epe, mean_d1, mean_thres1, mean_thres3))

        results.update({str(weather)+'_drivingstereo_epe': mean_epe})
        results.update({str(weather)+'_drivingstereo_d1': mean_d1})
        results.update({str(weather)+'_drivingstereo_3px': mean_thres3})
        results.update({str(weather)+'_drivingstereo_1px': mean_thres1})

    # if count_time:
    #     print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def validate_MS2(model,
                     max_disp=128,
                     padding_factor=32,
                     num_reg_refine=1,
                     inference_size=None,
                     attn_type=None,
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = MS2(transform=val_transform,
                                mode="testing",)

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0
    val_thres1 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if debug and i > 10:
            break

        if i % 5000 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0
        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              max_disp=max_disp,
                              num_reg_refine=num_reg_refine,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              )['disp_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        pred_disp = pred_disp.squeeze()
        gt_disp = gt_disp.squeeze()
        mask = gt_disp > 0
        
        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples * 100
    mean_thres3 = val_thres3 / valid_samples * 100
    mean_thres1 = val_thres1 / valid_samples * 100

    print('Validation MS2 EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres1, mean_thres3))

    results['MS2_epe'] = mean_epe
    results['MS2_d1'] = mean_d1
    results['MS2_3px'] = mean_thres3
    results['MS2_1px'] = mean_thres1

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        vgrid = grid.cuda()
    else:
        vgrid = grid
    
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
        
    return output



@torch.no_grad()
def LR_Check_v1(dispL, dispR, thres=2.1):
    # # left to right
    disp_L2R = warp(dispL, -dispR)
    dispR_thres = (disp_L2R - dispR).abs()
    mask_R = dispR_thres > thres
    dispR[mask_R] = 0.

    # right to left
    disp_R2L = warp(dispR, dispL)
    dispL_thres = (disp_R2L - dispL).abs()
    mask_L = dispL_thres > thres

    return (~mask_L).detach()
    


@torch.no_grad()
def validate_eth3d(model,
                   padding_factor=16,
                   iterations=24,
                   inference_size=None,
                   ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = ETH3DStereo(transform=val_transform,
                              )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres1 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              iters=iterations,
                              )['disp_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_thres1 = val_thres1 / valid_samples

    print('Validation ETH3D EPE: %.3f, 1px: %.4f' % (
        mean_epe, mean_thres1))

    results['eth3d_epe'] = mean_epe
    results['eth3d_1px'] = mean_thres1

    return results


@torch.no_grad()
def validate_middlebury(model,
                        padding_factor=16,
                        iterations=24,
                        inference_size=None,
                        resolution='H',
                        ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = MiddleburyEval3(transform=val_transform,
                                  resolution=resolution,
                                  )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres2 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]

            left = F.interpolate(left, size=inference_size,
                                 mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size,
                                  mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                                iters=iterations,
                              )['disp_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres2 += thres2.item()

    mean_epe = val_epe / valid_samples
    mean_thres2 = val_thres2 / valid_samples

    print('Validation Middlebury EPE: %.3f, 2px: %.4f' % (
        mean_epe, mean_thres2))

    results['middlebury_epe'] = mean_epe
    results['middlebury_2px'] = mean_thres2

    return results


@torch.no_grad()
def inference_stereo(model,
                     inference_dir=None,
                     inference_dir_left=None,
                     inference_dir_right=None,
                     output_path='output',
                     padding_factor=32,
                     inference_size=None,
                     iterations=32,
                     save_pfm_disp=True,
                     count_time=True,
                     ):
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    assert inference_dir or (inference_dir_left and inference_dir_right)

    if inference_dir is not None:
        filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))

        left_filenames = filenames[::2]
        right_filenames = filenames[1::2]

    else:
        left_filenames = sorted(glob(inference_dir_left + '/*.png') + glob(inference_dir_left + '/*.jpg'))
        right_filenames = sorted(glob(inference_dir_right + '/*.png') + glob(inference_dir_right + '/*.jpg'))

    assert len(left_filenames) == len(right_filenames)

    num_samples = len(left_filenames)
    print('%d test samples found' % num_samples)

    for i in range(num_samples):

        if (i + 1) % 50 == 0:
            print('predicting %d/%d' % (i + 1, num_samples))

        left_name = left_filenames[i]
        right_name = right_filenames[i]

        left = np.array(Image.open(left_name).convert('RGB')).astype(np.float32)
        right = np.array(Image.open(right_name).convert('RGB')).astype(np.float32)
        sample = {'left': left, 'right': right}

        sample = val_transform(sample)

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]

        # resize to nearest size or specified size
        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        with torch.no_grad():

            pred_disp = model(image1=left, image2=right,
                          iters=iterations,
                          test_mode=True,
                          )['disp_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        save_name = os.path.join(output_path, os.path.basename(left_name)[:-4] + '_disp.png')
        disp = pred_disp[0].squeeze().cpu().numpy()

        if save_pfm_disp:
            save_name_pfm = save_name[:-4] + '.pfm'
            write_pfm(save_name_pfm, disp)

        disp = vis_disparity(disp)
        cv2.imwrite(save_name, disp)

    print('Done!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset & submussion
    parser.add_argument('--eval_dataset', default='robust', type=str, nargs='+')
    parser.add_argument('--submission', default='none', type=str, nargs='+')
    parser.add_argument('--model_name', default='smoe', choices=["smoe", "raft"], type=str, nargs='+')
    # hyper-parameters
    parser.add_argument('--max_disp', default=192, type=int,
                        help='exclude very large disparity in the loss function')
    parser.add_argument('--img_height', default=320, type=int)
    parser.add_argument('--img_width', default=896, type=int)
    parser.add_argument('--padding_factor', default=32, type=int)
    
    # mode parameters
    parser.add_argument('--peft_type', default='smoe', choices=["lora", "smoe", "adapter", "tuning", "ff", "vpt"], type=str)
    parser.add_argument('--vfm_type', default='damv2', choices=["sam","dam","damv2","dinov2"], type=str)
    parser.add_argument('--vfm_size', default='vitb', choices=["vits","vitb","vitl"], type=str)
    
    parser.add_argument('--tunable_layers', default=[0,1,2,3,4,5,6,7,8,9,10,11], type=List)
    parser.add_argument('--use_layer_selection', default=True, type=bool)
    parser.add_argument('--recon',default=True, type=bool, help="recon the img")
    
    # resume pretrained model or resume training
    parser.add_argument('--resume', default='/data1/ywang/my_projects/SMoE-Stereo/ckpt/sceneflow/damv2_sceneflow.pth', type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_false',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--mixed_precision', action='store_false', help='use mixed precision')

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
    parser.add_argument('--validate_iters', default=32, type=int,
                        help='number of additional local regression refinement')
    parser.add_argument('--feature_maps',default=False, type=bool, help="obtain the feature maps for visualization")

    # evaluation
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')
    parser.add_argument('--save_vis_disp', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--middlebury_resolution', default='H', choices=['Q', 'H', 'F'])

    args = parser.parse_args()
    
    if args.model_name == 'raft':
         model = RAFTStereo(args).to(device)
    elif args.model_name == 'smoe':
        model = SMoEStereo(args).to(device)
    else:
        raise ValueError(f"Unsupported model type {type(args.model_name)}")
    
    model = torch.nn.DataParallel(model)
    if args.resume:
        print("=> Load checkpoint: %s" % args.resume)
        loc = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)

        model.load_state_dict(checkpoint["model"], strict=True)

    model.cuda()
    model.eval()

    print(f"The model has {format(sum(p.numel() for p in model.parameters())/1e6, '.2f')}M parameters.")

    if args.eval_dataset == 'eth3d':
        validate_eth3d(model, )

    elif args.eval_dataset == 'robust':
        validate_robust(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)
        
    elif args.eval_dataset == 'kitti15':
        validate_kitti15(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)

    elif args.eval_dataset == 'kitti12':
        validate_kitti12(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)

    elif args.eval_dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)
        
    elif args.eval_dataset == 'drivingstereo':
        validate_drivingstereo(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)

    elif args.eval_dataset == 'robust_weather':
        validate_robust_weather(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)
    else:
        pass

    if args.submission == 'kit15':
        create_kitti_submission(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)
        
    elif args.submission == 'eth3d':
        create_eth3d_submission(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)

    elif args.submission == 'mid':
        create_middlebury_submission(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iterations=args.validate_iters)
    else:
        pass