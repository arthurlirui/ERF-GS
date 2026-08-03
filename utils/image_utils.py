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
import numpy as np

import os

from PIL import Image, ImageDraw, ImageFont

from .event_utils import event_from_image
from .general_utils import rgb_to_luma


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is not None:
        img1 = img1.flatten(2)
        img2 = img2.flatten(2)

        mask = mask.flatten()
        mask = torch.where(mask != 0, True, False)
        img1 = img1[:, :, mask]
        img2 = img2[:, :, mask]

        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

    else:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
    if mask is not None:
        if torch.isinf(psnr).any():
            print(mse.mean(), psnr.mean())
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
            psnr = psnr[~torch.isinf(psnr)]

    return psnr


def event_overlay(img, events, max=10):
    if max is None:
        max = torch.abs(events).max().item()

    event_img = torch.zeros_like(events).unsqueeze(0).repeat(3, 1, 1)
    mag = torch.clip(torch.abs(events), max=max)

    event_img[0, :, :] = (events > 0).float()
    event_img[2, :, :] = (events < 0).float()

    pos_add = (events > 0).float() * (1 - mag / max)
    event_img[1, :, :] += pos_add
    event_img[2, :, :] += pos_add

    neg_add = (events < 0).float() * (1 - mag / max)
    event_img[0, :, :] += neg_add
    event_img[1, :, :] += neg_add

    img = torch.where(events.abs() < 1, img.cpu(), event_img)
    return img


@torch.no_grad()
def render_training_image(
    out_dir,
    gaussians,
    viewpoints,
    render_func,
    background,
    stage,
    iteration,
    time_now,
    args,
):
    def render(viewpoint, path, events=None):
        render_pkg = render_func(
            viewpoint,
            gaussians,
            background,
            stage,
            args,
        )
        label1 = f"stage:{stage},iter:{iteration}"
        times = time_now / 60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        gt_image = viewpoint.image

        if gt_image is None:
            gt_image = torch.ones_like(image)

        if events is not None:
            next_image = render_func(
                viewpoint, gaussians, background, stage, args, event_idx=0
            )["render"]
            image_event = event_from_image(
                image, next_image
            ).cpu()
            image1 = event_overlay(image, image_event)  # motion_event)
            image2 = image
            gt_image = event_overlay(gt_image, events)
        else:
            image1 = image
            image2 = image
            image2 = rgb_to_luma(image2).unsqueeze(0).repeat(3, 1, 1)

        gt_np = gt_image.permute(1, 2, 0).cpu().numpy()
        image_np = image1.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        depth_np = image2.permute(1, 2, 0).cpu().numpy()
        image_np = np.concatenate((gt_np, image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray(
            (np.clip(image_np, 0, 1) * 255).astype("uint8")
        )
        draw1 = ImageDraw.Draw(image_with_labels)
        font = ImageFont.truetype("./utils/TIMES.TTF", size=40)
        text_color = (255, 0, 0)
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        image_with_labels.save(path)

    render_base_path = os.path.join(out_dir, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path, "pointclouds")
    image_path = os.path.join(render_base_path, "images")
    os.makedirs(render_base_path, exist_ok=True)
    os.makedirs(point_cloud_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(viewpoints)):
            image_save_path = os.path.join(image_path, f"{iteration}_{idx}.jpg")
            if "train" in stage and viewpoints[idx].event_images is not None:
                events = (
                    viewpoints[idx].event_images[0, 0, :, :]
                    - viewpoints[idx].event_images[1, 0, :, :]
                )
            else:
                events = None
            render(viewpoints[idx], image_save_path, events=events)
        pc_mask = gaussians.get_opacity
        pc_mask = pc_mask > 0.1


def event2vis(events, max=10):
    if max is None:
        max = torch.abs(events).max().item()

    event_img = torch.zeros_like(events).unsqueeze(0).repeat(3, 1, 1)
    mag = torch.clip(torch.abs(events), max=max)

    event_img[0, :, :] = (events > 0).float()
    event_img[2, :, :] = (events < 0).float()

    pos_add = (events > 0).float() * (1 - mag / max)
    event_img[1, :, :] += pos_add
    event_img[2, :, :] += pos_add

    neg_add = (events < 0).float() * (1 - mag / max)
    event_img[0, :, :] += neg_add
    event_img[1, :, :] += neg_add

    img = torch.where(events == 0, 1, event_img)
    img = (img.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
    img = Image.fromarray(img)
    return img


def img2vis(img):
    img = (img.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
    img = Image.fromarray(img)
    return img
