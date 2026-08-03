#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp

from .general_utils import compute_plane_smoothness


def l1(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _plane_regulation(grid):
    total = 0
    # model.grids is 6 x [1, rank * F_dim, reso, reso]
    for grids in grid:
        if len(grids) == 3:
            time_grids = []
        else:
            time_grids = [0, 1, 3]
        for grid_id in time_grids:
            total += compute_plane_smoothness(grids[grid_id])
    return total


def _time_regulation(grid):
    total = 0
    # model.grids is 6 x [1, rank * F_dim, reso, reso]
    for grids in grid:
        if len(grids) == 3:
            time_grids = []
        else:
            time_grids = [2, 4, 5]
        for grid_id in time_grids:
            total += compute_plane_smoothness(grids[grid_id])
    return total


def _l1_regulation(grid):
    # model.grids is 6 x [1, rank * F_dim, reso, reso]
    total = 0.0
    for grids in grid:
        if len(grids) == 3:
            continue
        else:
            # These are the spatiotemporal grids
            spatiotemporal_grids = [2, 4, 5]
        for grid_id in spatiotemporal_grids:
            total += torch.abs(1 - grids[grid_id]).mean()
    return total


def tv(grid, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
    return (
        plane_tv_weight * _plane_regulation(grid)
        + time_smoothness_weight * _time_regulation(grid)
        + l1_time_planes_weight * _l1_regulation(grid)
    )

def eloss(pred_events, gt_events):
    # STE
    quantized_events = torch.where(pred_events >= 0, torch.floor(pred_events), torch.ceil(pred_events))
    pred_events = pred_events + (quantized_events - pred_events).detach()

    return torch.mean(torch.abs(pred_events - gt_events) / torch.clamp(torch.abs(gt_events), min=1))