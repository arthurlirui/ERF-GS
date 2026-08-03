import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import h5py
import os
import random

from loguru import logger as guru
from natsort import index_natsorted, order_by_index
from pathlib import Path
from plyfile import PlyData, PlyElement
from PIL import Image
from tqdm import tqdm

from .data_structures import CameraInfo, SceneInfo, PointCloud
from utils.colmap import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    qvec2rotmat,
    read_points3D_binary,
    read_points3D_text,
)
from utils.event_utils import events_to_voxel
from utils.graphics_utils import (
    focal2fov,
    getWorld2View2,
    getProjectionMatrix,
    get_spiral,
)

dataset_paths = {
    "nvidia-long": "YOUR_PATH_TO_DATASET/Nvidia",
    "neu3d": "YOUR_PATH_TO_DATASET/Neu3D",
}


def store_ply(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "f4"),
        ("green", "f4"),
        ("blue", "f4"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def fetch_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    return PointCloud(points=positions, colors=colors, normals=None)


def read_cameras(
    cam_extrinsics,
    cam_intrinsics,
    cam_folder,
    event_folder,
    split,
    max_time,
    dataset,
    llffhold,
    time_subsample,
    motion_blur,
    with_dynamic_mask,
):
    scene_name = cam_folder.split("/")[-1]

    if split == "train":
        cam_extrinsics = [
            cam_extrinsics[key] for key in cam_extrinsics if (key - 1) % llffhold != 0
        ]
    else:
        cam_extrinsics = [
            cam_extrinsics[key] for key in cam_extrinsics if (key - 1) % llffhold == 0
        ]

    if dataset == "nvidia-long":
        cam_list = [
            "cam" + os.path.basename(extr.name)[5:-4].zfill(2)
            for extr in cam_extrinsics
        ]
    else:
        existing_list = [itm.split('.mp4')[0] for itm in os.listdir(cam_folder) if '.mp4' in itm]
        cam_index = index_natsorted(existing_list)
        existing_list = order_by_index(existing_list, cam_index)
        cam_list = [existing_list[int(os.path.basename(extr.name)[3:5])] for extr in cam_extrinsics]

    cam_length = len(cam_list)
    image_length = min(
        [len(os.listdir(os.path.join(cam_folder, cam))) for cam in cam_list]
    )
    image_length = image_length if image_length <= max_time else max_time

    full_image_length = image_length

    event_dir = Path(event_folder)

    images = []
    image_paths = []
    image_poses = []
    image_times = []
    if with_dynamic_mask:
        dynamic_masks = []
    else:
        dynamic_masks = None
    for idx, extr in enumerate(cam_extrinsics):
        guru.info(f"processing cam: {extr.name}")
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if time_subsample != 1 and motion_blur:
            images_folder = os.path.join(cam_folder, 'ts{}'.format(time_subsample), cam_list[idx])
        else:
            images_folder = os.path.join(cam_folder, cam_list[idx])
        dynamic_mask_folder = os.path.join(cam_folder, "dynamic_masks", cam_list[idx])

        image_range = list(range(image_length))

        if split == "train":
            image_range = image_range[
                :-1
            ]  # last frame does not have corresponding events

        for i in tqdm(image_range):
            # temporal subsample
            if not (i % time_subsample == 0):
                continue
            num = i

            image_path = os.path.join(images_folder, str(num) + ".png")
            images.append(image_path)
            image_paths.append(image_path)
            image_poses.append((R, T))
            image_times.append(float(i / image_length))
            if with_dynamic_mask:
                dynamic_masks.append(
                    os.path.join(dynamic_mask_folder, str(num) + ".png")
                )

    image_paths_index = index_natsorted(image_paths)
    images = order_by_index(images, image_paths_index)
    image_paths = order_by_index(image_paths, image_paths_index)
    image_poses = order_by_index(image_poses, image_paths_index)
    image_times = order_by_index(image_times, image_paths_index)
    if with_dynamic_mask:
        dynamic_masks = order_by_index(dynamic_masks, image_paths_index)

    if split == "train":
        if time_subsample == 1:
            image_length -= 1
        else:
            image_length = len(images) // len(cam_extrinsics)  # last frame does not have corresponding events

    image_sample = Image.open(image_paths[0])
    img_w, img_h = image_sample.size

    cam_list = order_by_index(
        cam_list, index_natsorted(cam_list)
    )

    if split == "train":
        total_event = 0
        event_images = []
        if time_subsample != 1:
            event_root = f"preprocess_event_images/{scene_name}/ts{time_subsample}"
        else:
            event_root = f"preprocess_event_images/{scene_name}"
        os.makedirs(event_root, exist_ok=True)

        if (
            os.path.exists(
                os.path.join(event_root, f"{cam_length * image_length - 1}.bin")
            )
            and len([itm for itm in os.listdir(event_root) if itm.endswith(".bin")])
            == cam_length * image_length
        ):
            guru.info("event_images all already saved!!")
            for cam_idx in tqdm(range(cam_length)):
                guru.info(f"recoding event_cam: {cam_idx=}")
                for j in tqdm(range(image_length), desc="record event paths"):
                    event_images.append(os.path.join(event_root, f"{total_event}.bin"))
                    total_event += 1
        else:
            for cam_idx in tqdm(range(cam_length)):
                for j in tqdm(range(image_length), desc="process frames"):

                    if j == image_length - 1:
                        num_events = full_image_length - 1 - j * time_subsample
                    else:
                        num_events = time_subsample

                    event_images = []
                    for k in range(num_events):
                        event_file = os.path.join(event_dir, f"{cam_list[cam_idx]}_{j * time_subsample + k}.h5")
                        assert os.path.exists(event_file)
                        events_h5 = h5py.File(event_file, mode="r")
                        events = events_h5["events"]
                        events = torch.tensor(
                            events.astype(np.float32), dtype=torch.float32
                        )
                        ts, xs, ys, ps = torch.chunk(events, 4, dim=1)

                        if 'Dynamicface' in cam_folder:
                            ys = torch.round(ys * 1005 / 1011)
                        elif 'Playground' in cam_folder:
                            ys = torch.round(ys * 923 / 1011)

                        ps = 2 * (ps - 0.5)  # convert {0, 1} to {-1, 1}
                        event_image = events_to_voxel(
                            xs, ys, ts, ps, False, img_h, img_w
                        )  # [2, 1, h, w]
                        event_images.append(event_image.unsqueeze(1))
                        events_h5.close()
                    event_image = torch.cat(event_images, dim=1)

                    # for neu3d only
                    if dataset == 'neu3d':
                        h, w = event_image.size(-2), event_image.size(-1)
                        if event_image.size(1) % 2 == 1:
                            event_image = event_image[:,:-1,:,:]
                        event_image = event_image.view(2, -1, 2, h, w).sum(dim=2)
                        event_image = F.interpolate(event_image, scale_factor=(0.5, 0.5))

                    with open(os.path.join(event_root, f"{total_event}.bin"), "w") as f:
                        event_image.numpy().tofile(f)
                    event_images.append(os.path.join(event_root, f"{total_event}.bin"))
                    total_event += 1

        assert len(image_paths) == len(event_images)
    else:
        event_images = None

    return (
        images,
        image_paths,
        image_poses,
        image_times,
        event_images,
        full_image_length,
        len(cam_list),
        dynamic_masks,
    )


class MultiviewEventDataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        event_folder,
        max_time,
        split,
        dataset,
        llffhold,
        time_subsample,
        disjoint_views,
        motion_blur,
        with_dynamic_mask,
    ):
        self.dataset = dataset

        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height = cam_intrinsics[1].height
        width = cam_intrinsics[1].width

        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)

        self.transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
            ]
        )

        (
            self.images,
            self.image_paths,
            self.image_poses,
            self.image_times,
            self.event_images,
            image_length,
            cam_length,
            self.dynamic_masks,
        ) = read_cameras(
            cam_extrinsics,
            cam_intrinsics,
            cam_folder,
            event_folder,
            split,
            max_time,
            dataset,
            llffhold,
            time_subsample,
            motion_blur,
            with_dynamic_mask,
        )
        if dataset == 'neu3d':
            self.frame_time = 2.0 / image_length
        else:
            self.frame_time = 1.0 / image_length
        self.image_per_view = len(self.images) // cam_length
        if self.event_images is not None:
            self.event_per_view = len(self.event_images) // cam_length
        self.cam_length = cam_length

        self.times = list(set(self.image_times))
        self.times.sort()

        centers = []
        for R, T in self.image_poses:
            w2c = torch.tensor(getWorld2View2(R, T))
            c2w = np.linalg.inv(w2c)
            centers.append(c2w[:3, 3:4])
        centers = np.hstack(centers)
        avg_center = np.mean(centers, axis=1, keepdims=True)
        dist = np.linalg.norm(centers - avg_center, axis=0, keepdims=True)
        diag = np.max(dist)
        self.radius = diag * 1.1

        self.time_subsample = time_subsample

        self.disjoint_idx = random.sample(list(range(cam_length)), disjoint_views)
        
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        w, h = img.size[0], img.size[1]
        if self.dataset == 'neu3d':
            w, h = w // 2, h // 2
        img = self.transform(img)
        if self.event_images is not None:
            event_img = np.fromfile(self.event_images[index], dtype=np.float32).reshape(
                    2, -1, h, w
                )
            event_img = torch.from_numpy(event_img)
            event_time = [self.image_times[index] + self.frame_time * k for k in range(event_img.shape[1] + 1)]
            if event_img.size(-1) != img.size(-1) or event_img.size(-2) != img.size(-2):
                event_img = F.interpolate(event_img, size=(img.size(-2), img.size(-1)))
            event_time = torch.tensor(event_time).float()
        else:
            event_img = None
            event_time = None

        if self.dynamic_masks is not None:
            dynamic_mask = Image.open(self.dynamic_masks[index])
            dyn_mask = self.transform_mask(dynamic_mask)
        else:
            dyn_mask = None

        R, T = self.image_poses[index]
        time = self.image_times[index]

        world_view_transform = torch.from_numpy(getWorld2View2(R, T)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=self.FovX, fovY=self.FovY
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        image_height = img.shape[1]
        image_width = img.shape[2]

        if len(self.disjoint_idx) != 0 and (index // self.image_per_view) in self.disjoint_idx:
            img = None
        if len(self.disjoint_idx) != 0 and not (index // self.image_per_view) in self.disjoint_idx:
            event_img = None

        return CameraInfo(
            idx=index,
            R=R,
            T=T,
            fovy=self.FovY,
            fovx=self.FovX,
            image=img,
            image_height=image_height,
            image_width=image_width,
            image_name=f"{index}",
            world_view_transform=world_view_transform,
            projection_matrix=projection_matrix,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            time=time,  
            event_images=event_img,
            event_times=event_time,
            dynamic_mask=dyn_mask,
        )

    def __len__(self):
        return len(self.images)

    def group_by_timestamp(self, timestamp):
        target_time = timestamp
        indices = [
            idx
            for idx in range(len(self.image_times))
            if self.image_times[idx] == target_time and (len(self.disjoint_idx) == 0 or (idx // self.image_per_view) in self.disjoint_idx)
        ]
        for idx in indices:
            yield self[idx]

    def get_event_size(self, timestamp):
        for idx in range(len(self.image_times)):
            if self.image_times[idx] == timestamp and (len(self.disjoint_idx) == 0 or (idx // self.image_per_view) in self.disjoint_idx):
                event_img = self[idx].event_images
                return event_img.size(1)

def get_video_cam_infos(test_cams, datadir):
    poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    near_fars = poses_arr[:, -2:]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    N_views = 300
    val_poses = get_spiral(poses, near_fars, N_views=N_views)

    cameras = []
    len_poses = len(val_poses)
    times = [i / len_poses for i in range(len_poses)]
    image = Image.open(test_cams.image_paths[0])
    image = test_cams.transform(image)

    for idx, p in enumerate(val_poses):
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3, :] = p[:3, :]
        R = pose[:3, :3]
        R = -R
        R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)
        FovX = test_cams.FovX
        FovY = test_cams.FovY

        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        cameras.append(
            CameraInfo(
                idx=idx,
                R=R,
                T=T,
                fovy=FovY,
                fovx=FovX,
                image=image,
                image_height=image.shape[1],
                image_width=image.shape[2],
                image_name=image_name,
                world_view_transform=world_view_transform,
                projection_matrix=projection_matrix,
                full_proj_transform=full_proj_transform,
                camera_center=camera_center,
                time=time,
                event_images=None,
                event_times=None,
                dynamic_mask=None,
            )
        )
    return cameras


def load_data(dataset, task, max_time, llffhold, time_subsample, disjoint_views, motion_blur, with_dynamic_mask, **ignore_kwargs):
    datadir = os.path.join(dataset_paths[dataset], task)

    if dataset == "nvidia-long":
        cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
        cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    elif dataset == "neu3d":
        cameras_extrinsic_file = os.path.join(datadir, "colmap/sparse/0/images.bin")
        cameras_intrinsic_file = os.path.join(datadir, "colmap/sparse/0/cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    event_dir = os.path.join(datadir, "events")

    train_cams = MultiviewEventDataset(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        cam_folder=datadir,
        event_folder=event_dir,
        max_time=int(max_time),
        split="train",
        dataset=dataset,
        llffhold=llffhold,
        time_subsample=time_subsample,
        disjoint_views=disjoint_views,
        motion_blur=motion_blur,
        with_dynamic_mask=False,
    )

    test_cams = MultiviewEventDataset(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        cam_folder=datadir,
        event_folder=event_dir,
        max_time=int(max_time),
        split="test",
        dataset=dataset,
        llffhold=llffhold,
        time_subsample=1,
        disjoint_views=0,
        motion_blur=False,
        with_dynamic_mask=with_dynamic_mask,
    )

    video_cams = get_video_cam_infos(test_cams, datadir)

    radius = train_cams.radius

    ply_path = os.path.join(datadir, "points3D_multipleview.ply")
    bin_path = os.path.join(datadir, "points3D_multipleview.bin")
    txt_path = os.path.join(datadir, "points3D_multipleview.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        store_ply(ply_path, xyz, rgb)
    pcd = fetch_ply(ply_path)

    scene_info = SceneInfo(
        pcd,
        train_cams,
        test_cams,
        video_cams,
        radius,
    )
    return scene_info
