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
import stat
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as tf

import concurrent.futures
import imageio
import json
import os

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from pytorch_msssim import ms_ssim
from time import time
from tqdm import tqdm

from data import get_data
from gaussian_renderer import render
from lpipsPyTorch import lpips
from models import get_model
from utils.general_utils import load_args
from utils.image_utils import psnr
from utils.loss_utils import ssim

to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)


def evaluate(scene_dir, tag):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("Scene: {}, tag: {}".format(scene_dir, tag))

    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / tag

    for method in os.listdir(test_dir):
        print("Method:", method)

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        dynamic_mask_dir = method_dir / "dynamic_mask"

        ssims = []
        psnrs = []
        lpipss = []
        lpipsa = []
        ms_ssims = []
        Dssims = []

        dyn_psnrs, dyn_ssims, dyn_lpips = [], [], []
        stat_psnrs, stat_ssims, stat_lpips = [], [], []

        image_names = []
        for fname in tqdm(os.listdir(renders_dir), desc="Metric evaluation progress"):
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)

            render_img = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
            gt_img = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

            if tag == "test":
                dynamic_mask = Image.open(dynamic_mask_dir / fname)
                dyn_mask_img = tf.to_tensor(dynamic_mask)[:1, :, :].unsqueeze(0).cuda()

            image_names.append(fname)

            ssims.append(ssim(render_img, gt_img).item())
            psnrs.append(psnr(render_img, gt_img).mean().item())
            lpipss.append(lpips(render_img, gt_img, net_type="vgg").item())
            ms_ssims.append(
                ms_ssim(render_img, gt_img, data_range=1, size_average=True).item()
            )
            lpipsa.append(lpips(render_img, gt_img, net_type="alex").item())
            Dssims.append((1 - ms_ssims[-1]) / 2)

            if tag == "test" and dyn_mask_img.sum() != 0:
                dyn_ssims.append(
                    ssim(render_img * dyn_mask_img, gt_img * dyn_mask_img).item()
                )
                dyn_psnrs.append(
                    psnr(render_img, gt_img, mask=dyn_mask_img).mean().item()
                )
                dyn_lpips.append(
                    lpips(
                        render_img * dyn_mask_img, gt_img * dyn_mask_img, net_type="vgg"
                    ).item()
                )

                stat_ssims.append(
                    ssim(
                        render_img * (1 - dyn_mask_img), gt_img * (1 - dyn_mask_img)
                    ).item()
                )
                stat_psnrs.append(
                    psnr(render_img, gt_img, mask=(1 - dyn_mask_img)).mean().item()
                )
                stat_lpips.append(
                    lpips(
                        render_img * (1 - dyn_mask_img),
                        gt_img * (1 - dyn_mask_img),
                        net_type="vgg",
                    ).item()
                )

        print(
            "Scene: ",
            scene_dir,
            "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"),
        )
        print(
            "Scene: ",
            scene_dir,
            "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"),
        )
        print(
            "Scene: ",
            scene_dir,
            "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"),
        )
        print(
            "Scene: ",
            scene_dir,
            "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"),
        )
        print(
            "Scene: ",
            scene_dir,
            "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"),
        )
        print(
            "Scene: ",
            scene_dir,
            "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"),
        )
        if tag == "test":
            print(
                "Scene: ",
                scene_dir,
                "Dynamic-SSIM : {:>12.7f}".format(torch.tensor(dyn_ssims).mean(), ".5"),
            )
            print(
                "Scene: ",
                scene_dir,
                "Dynamic-PSNR : {:>12.7f}".format(torch.tensor(dyn_psnrs).mean(), ".5"),
            )
            print(
                "Scene: ",
                scene_dir,
                "Dynamic-LPIPS : {:>12.7f}".format(
                    torch.tensor(dyn_lpips).mean(), ".5"
                ),
            )
            print(
                "Scene: ",
                scene_dir,
                "Static-SSIM : {:>12.7f}".format(torch.tensor(stat_ssims).mean(), ".5"),
            )
            print(
                "Scene: ",
                scene_dir,
                "Stati-PSNR : {:>12.7f}".format(torch.tensor(stat_psnrs).mean(), ".5"),
            )
            print(
                "Scene: ",
                scene_dir,
                "Static-LPIPS : {:>12.7f}".format(
                    torch.tensor(stat_lpips).mean(), ".5"
                ),
            )

        if tag == "test":
            full_dict[scene_dir][method].update(
                {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                    "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                    "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                    "D-SSIM": torch.tensor(Dssims).mean().item(),
                    "Dynamic-SSIM": torch.tensor(dyn_ssims).mean().item(),
                    "Dynamic-PSNR": torch.tensor(dyn_psnrs).mean().item(),
                    "Dynamic-LPIPS": torch.tensor(dyn_lpips).mean().item(),
                    "Static-SSIM": torch.tensor(stat_ssims).mean().item(),
                    "Static-PSNR": torch.tensor(stat_psnrs).mean().item(),
                    "Static-LPIPS": torch.tensor(stat_lpips).mean().item(),
                },
            )
        else:
            full_dict[scene_dir][method].update(
                {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                    "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                    "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                    "D-SSIM": torch.tensor(Dssims).mean().item(),
                },
            )

        if tag == "test":
            per_view_dict[scene_dir][method].update(
                {
                    "SSIM": {
                        name: ssim
                        for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                    },
                    "PSNR": {
                        name: psnr
                        for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                    },
                    "LPIPS-vgg": {
                        name: lp
                        for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                    },
                    "LPIPS-alex": {
                        name: lp
                        for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)
                    },
                    "MS-SSIM": {
                        name: lp
                        for lp, name in zip(
                            torch.tensor(ms_ssims).tolist(), image_names
                        )
                    },
                    "D-SSIM": {
                        name: lp
                        for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)
                    },
                    "Dynamic-SSIM": {
                        name: lp
                        for lp, name in zip(
                            torch.tensor(dyn_ssims).tolist(), image_names
                        )
                    },
                    "Dynamic-PSNR": {
                        name: lp
                        for lp, name in zip(
                            torch.tensor(dyn_psnrs).tolist(), image_names
                        )
                    },
                    "Dynamic-LPIPS": {
                        name: lp
                        for lp, name in zip(
                            torch.tensor(dyn_lpips).tolist(), image_names
                        )
                    },
                    "Static-SSIM": {
                        name: lp
                        for lp, name in zip(
                            torch.tensor(stat_ssims).tolist(), image_names
                        )
                    },
                    "Static-PSNR": {
                        name: lp
                        for lp, name in zip(
                            torch.tensor(stat_psnrs).tolist(), image_names
                        )
                    },
                    "Static-LPIPS": {
                        name: lp
                        for lp, name in zip(
                            torch.tensor(stat_lpips).tolist(), image_names
                        )
                    },
                }
            )
        else:
            per_view_dict[scene_dir][method].update(
                {
                    "SSIM": {
                        name: ssim
                        for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                    },
                    "PSNR": {
                        name: psnr
                        for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                    },
                    "LPIPS-vgg": {
                        name: lp
                        for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                    },
                    "LPIPS-alex": {
                        name: lp
                        for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)
                    },
                    "MS-SSIM": {
                        name: lp
                        for lp, name in zip(
                            torch.tensor(ms_ssims).tolist(), image_names
                        )
                    },
                    "D-SSIM": {
                        name: lp
                        for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)
                    },
                }
            )

    with open(scene_dir + "/results.json", "w") as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", "w") as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def multithread_write(image_list, path, start_count=0):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, os.path.join(path, "{0:05d}".format(count) + ".png")
            )
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index + start_count, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index + start_count, path)


def render_set(output_dir, tag, iteration, dataset, gaussians, background, stage):
    render_path = os.path.join(output_dir, tag, str(iteration), "renders")
    gts_path = os.path.join(output_dir, tag, str(iteration), "gt")
    dyn_mask_path = os.path.join(output_dir, tag, str(iteration), "dynamic_mask")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(dyn_mask_path, exist_ok=True)

    batch_size = 200
    render_images = []
    gt_list = []
    render_list = []
    dynamic_mask_list = []
    write_time = 0
    print("point nums:", gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(dataset, desc="Rendering progress")):
        time1 = time()
        rendering = render(view, gaussians, background, stage, args)["render"]
        write_time += time() - time1
        render_images.append(to8b(rendering).transpose(1, 2, 0))
        render_list.append(rendering.detach().cpu())
        gt = view.image[0:3, :, :].cpu()
        gt_list.append(gt)
        if tag == "test":
            dynamic_mask_list.append(view.dynamic_mask)

        if idx % batch_size == batch_size - 1:
            print(200 / write_time)
            multithread_write(
                gt_list, gts_path, start_count=idx // batch_size * batch_size
            )
            multithread_write(
                render_list, render_path, start_count=idx // batch_size * batch_size
            )
            if tag == "test":
                multithread_write(
                    dynamic_mask_list,
                    dyn_mask_path,
                    start_count=idx // batch_size * batch_size,
                )
            gt_list = []
            render_list = []
            dynamic_mask_list = []

    time2 = time()
    print("FPS:", (len(dataset) - 1) / (time2 - time1 - write_time))

    multithread_write(gt_list, gts_path, start_count=idx // batch_size * batch_size)
    multithread_write(
        render_list, render_path, start_count=idx // batch_size * batch_size
    )

    if tag == "test":
        multithread_write(
            dynamic_mask_list,
            dyn_mask_path,
            start_count=idx // batch_size * batch_size,
        )

    video_path = os.path.join(output_dir, tag, str(iteration), "video_rgb.mp4")
    print(video_path)
    print(len(render_images))
    imageio.mimwrite(video_path, render_images, fps=30, format="MP4")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--test_stage", type=str, default="fine")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--config_file", type=str, default="")
    args = parser.parse_known_args()[0]
    args = load_args(args.config_file, args)

    output_dir = os.path.join("./output/", args.expname)
    scene_info, train_dataset, test_dataset, video_dataset = get_data(
        args.data_name, args.data
    )
    gaussians = get_model(scene_info.pcd, scene_info.radius, args.model, args.optimize)
    bg_color = [1, 1, 1] if args.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ckpt_path = (
        output_dir
        + "/chkpnt"
        + f"_{args.test_stage}_"
        + str(args.test_iteration)
        + ".pth"
    )
    gaussians.load(ckpt_path, args.optimize)

    if not args.skip_train:
        render_set(
            output_dir,
            "train",
            args.test_iteration,
            train_dataset,
            gaussians,
            background,
            args.test_stage,
        )
    if not args.skip_test:
        render_set(
            output_dir,
            "test",
            args.test_iteration,
            test_dataset,
            gaussians,
            background,
            args.test_stage,
        )
    if not args.skip_video:
        render_set(
            output_dir,
            "video",
            args.test_iteration,
            video_dataset,
            gaussians,
            background,
            args.test_stage,
        )

    if not args.skip_train:
        evaluate(output_dir, "train")
    if not args.skip_test:
        evaluate(output_dir, "test")
