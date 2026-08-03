from calendar import c
import os
import cv2
import glob
import argparse
from pathlib import Path
from natsort import natsorted

import torchvision.transforms as tf

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image

import numpy as np

import subprocess

from torch import native_batch_norm
from loguru import logger as guru
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_root",
        type=str,
        default="/mnt/ssd2/zhenyang/datasets/nvidia_ds/",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default="Balloon1",
    )
    parser.add_argument(
        "--affix",
        type=str,
        default="png",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    root_dir = args.scene_root
    for scene_name in os.listdir(root_dir):
        if scene_name != args.scene_name:
            continue
        guru.info(f"process: {scene_name=}")
        scene_dir = os.path.join(root_dir, scene_name)

        target_dir = os.path.join(scene_dir, 'ts{}-new'.format(args.subsample))
        os.makedirs(target_dir, exist_ok=True)

        cams_dir = Path(scene_dir)
        cam_list = natsorted(seq=cams_dir.glob("cam??/"))
        for cam in tqdm(cam_list):
            cam_dir = str(cam)
            cam_name = cam_dir.split('/')[-1]

            target_cam_dir = os.path.join(target_dir, cam_name)
            os.makedirs(target_cam_dir, exist_ok=True)

            num_imgs = max(
                [
                    int(img.split(".")[0])
                    for img in os.listdir(cam_dir)
                    if img.endswith((".png", ".jpg", ".jpeg"))
                ]
            ) + 1

            image_stack = []
            for i in tqdm(range(num_imgs)):
                image = Image.open(os.path.join(cam_dir, '{}.{}'.format(i, args.affix)))
                image_tensor = tf.ToTensor()(image)

                image_stack.append(image_tensor)

                if i % args.subsample == args.subsample - 1:
                    target_image_tensor = sum(image_stack[:3]) / len(image_stack[:3])
                    target_image = Image.fromarray((target_image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    target_image.save(os.path.join(target_cam_dir, '{}.{}'.format(i // args.subsample * args.subsample, args.affix)))
                    image_stack = []
            if len(image_stack) != 0:
                target_image_tensor = sum(image_stack[:3]) / len(image_stack[:3])
                target_image = Image.fromarray((target_image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                target_image.save(os.path.join(target_cam_dir, '{}.{}'.format(i // args.subsample * args.subsample, args.affix)))