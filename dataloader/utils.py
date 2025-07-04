from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from PIL import Image
import sys
import cv2
import json
import os

import torch
import torch.nn.functional as F
import numpy as np


def read_img(filename):
    # convert to RGB for scene flow finalpass data
    img = np.array(Image.open(filename).convert('RGB')).astype(np.float32)
    return img


def read_disp(filename, subset=False, vkitti2=False, sintel=False,
              drivingstereo=False, MS2=False, booster=False,
              tartanair=False, instereo2k=False, crestereo=False,
              fallingthings=False,
              argoverse=False,
              raw_disp_png=False,
              ):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp
    # VKITTI2 dataset
    elif vkitti2:
        disp = _read_vkitti2_disp(filename)
    # Sintel
    elif sintel:
        disp = _read_sintel_disparity(filename)
    elif tartanair:
        disp = _read_tartanair_disp(filename)
    elif instereo2k:
        disp = _read_instereo2k_disp(filename)
    elif crestereo:
        disp = _read_crestereo_disp(filename)
    elif fallingthings:
        disp = _read_fallingthings_disp(filename)
    elif argoverse:
        disp = _read_argoverse_disp(filename)
    elif booster:
        disp = _read_booster_disp(filename)
    elif raw_disp_png:
        disp = np.array(Image.open(filename)).astype(np.float32)
    elif drivingstereo:
        disp = _read_drivingstereo_disp(filename)
    elif MS2:
        disp = _read_MS2_disp(filename) 

    # KITTI
    elif filename.endswith('png'):
        disp = _read_kitti_disp(filename)
    elif filename.endswith('npy'):
        disp = np.load(filename)
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]


def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)


def _read_drivingstereo_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth

def _read_MS2_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth


def _read_booster_disp(filename):
    disp = np.load(filename, encoding='bytes', allow_pickle=True)
    valid = disp > 0
    return disp


def _read_kitti_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth


def _read_vkitti2_disp(filename):
    # read depth
    depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # in cm
    depth = (depth / 100).astype(np.float32)  # depth clipped to 655.35m for sky

    valid = (depth > 0) & (depth < 655)  # depth clipped to 655.35m for sky

    # convert to disparity
    focal_length = 725.0087  # in pixels
    baseline = 0.532725  # meter

    disp = baseline * focal_length / depth

    disp[~valid] = 0.000001  # invalid as very small value

    return disp


def _read_sintel_disparity(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))

    d_r = f_in[:, :, 0].astype('float32')
    d_g = f_in[:, :, 1].astype('float32')
    d_b = f_in[:, :, 2].astype('float32')

    depth = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    return depth


def _read_tartanair_disp(filename):
    # the infinite distant object such as the sky has a large depth value (e.g. 10000)
    depth = np.load(filename)

    # change to disparity image
    disparity = 80.0 / depth

    return disparity


def _read_instereo2k_disp(filename):
    disp = np.array(Image.open(filename))
    disp = disp.astype(np.float32) / 100.
    return disp


def _read_crestereo_disp(filename):
    disp = np.array(Image.open(filename))
    return disp.astype(np.float32) / 32.


def _read_fallingthings_disp(filename):
    depth = np.array(Image.open(filename))
    camera_file = os.path.join(os.path.dirname(filename), '_camera_settings.json')
    with open(camera_file, 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / depth.astype(np.float32)

    return disp


def _read_argoverse_disp(filename):
    disparity_map = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return np.float32(disparity_map) / 256.


def extract_video(video_name):
    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened(), f'Failed to load video file {video_name}'
    # get video info
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('video size (hxw): %dx%d' % (size[1], size[0]))
    print('fps: %d' % fps)

    imgs = []
    while cap.isOpened():
        # get frames
        flag, img = cap.read()
        if not flag:
            break
        # to rgb format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    return imgs, fps



class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def bilinear_sampler(img, coords, mode='bilinear', mask=False, padding_mode='zeros'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    if coords.size(-1) != 2:  # [B, 2, H, W] -> [B, H, W, 2]
        coords = coords.permute(0, 2, 3, 1)

    H, W = img.shape[-2:]
    # H = height if height is not None else img.shape[-2]
    # W = width if width is not None else img.shape[-1]

    xgrid, ygrid = coords.split([1, 1], dim=-1)

    # To handle H or W equals to 1 by explicitly defining height and width
    if H == 1:
        assert ygrid.abs().max() < 1e-8
        H = 10
    if W == 1:
        assert xgrid.abs().max() < 1e-8
        W = 10

    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.squeeze(-1).float()

    return img


def coords_grid(batch, ht, wd, normalize=False):
    if normalize:  # [-1, 1]
        coords = torch.meshgrid(2 * torch.arange(ht) / (ht - 1) - 1,
                                2 * torch.arange(wd) / (wd - 1) - 1)
    else:
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)  # [B, 2, H, W]


def coords_grid_np(h, w):  # used for accumulating high speed sintel flow testdata
    coords = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing='ij')
    coords = np.stack(coords[::-1], axis=-1)  # [H, W, 2]

    return coords


def compute_out_of_boundary_mask(flow, downsample_factor=None):
    # flow: [B, 2, H, W]
    assert flow.dim() == 4 and flow.size(1) == 2
    b, _, h, w = flow.shape
    init_coords = coords_grid(b, h, w).to(flow.device)
    corres = init_coords + flow  # [B, 2, H, W]

    if downsample_factor is not None:
        assert w % downsample_factor == 0 and h % downsample_factor == 0
        # the actual max disp can predict is in the downsampled feature resolution, then upsample
        max_w = (w // downsample_factor - 1) * downsample_factor
        max_h = (h // downsample_factor - 1) * downsample_factor
        # print('max_w: %d, max_h: %d' % (max_w, max_h))
    else:
        max_w = w - 1
        max_h = h - 1

    valid_mask = (corres[:, 0] >= 0) & (corres[:, 0] <= max_w) & (corres[:, 1] >= 0) & (corres[:, 1] <= max_h)

    # in case very large flow
    flow_mask = (flow[:, 0].abs() <= max_w) & (flow[:, 1].abs() <= max_h)

    valid_mask = valid_mask & flow_mask

    return valid_mask  # [B, H, W]


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    # grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sampler(feature, grid, mask=mask, padding_mode=padding_mode)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def bilinear_upflow(flow, scale_factor=8):
    assert flow.size(1) == 2
    flow = F.interpolate(flow, scale_factor=scale_factor,
                         mode='bilinear', align_corners=True) * scale_factor

    return flow


def upsample_flow(flow, img):
    if flow.size(-1) != img.size(-1):
        scale_factor = img.size(-1) / flow.size(-1)
        flow = F.interpolate(flow, size=img.size()[-2:],
                             mode='bilinear', align_corners=True) * scale_factor
    return flow


def count_parameters(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
