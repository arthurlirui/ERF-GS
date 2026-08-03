import numpy as np
import torch

from typing import NamedTuple


class CameraInfo(NamedTuple):
    idx: int
    R: np.array
    T: np.array
    fovy: np.array
    fovx: np.array
    image: torch.tensor
    image_height: int
    image_width: int
    image_name: str
    world_view_transform: torch.tensor
    projection_matrix: torch.tensor
    full_proj_transform: torch.tensor
    camera_center: torch.tensor
    time: float
    event_images: torch.tensor
    event_times: list
    dynamic_mask: torch.tensor


class PointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class SceneInfo(NamedTuple):
    pcd: PointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    radius: float
