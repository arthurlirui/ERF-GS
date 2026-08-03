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

import math

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh


def render(viewpoint_camera, gaussians, bg_color, stage, args, event_idx=None):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            gaussians.get_xyz,
            dtype=gaussians.get_xyz.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = gaussians.get_xyz
    tanfovx = math.tan(viewpoint_camera.fovx * 0.5)
    tanfovy = math.tan(viewpoint_camera.fovy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=False,
    )
    if event_idx is None:
        time = (
            torch.tensor(viewpoint_camera.time)
            .to(means3D.device)
            .repeat(means3D.shape[0], 1)
        )
    else:
        time = (
            torch.tensor(viewpoint_camera.event_times[event_idx+1])
            .to(means3D.device)
            .repeat(means3D.shape[0], 1)
        )
        time_final = (
            torch.tensor(viewpoint_camera.event_times[-1])
            .to(means3D.device)
            .repeat(means3D.shape[0], 1)
        )
        time_first = (
            torch.tensor(viewpoint_camera.time)
            .to(means3D.device)
            .repeat(means3D.shape[0], 1)
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = gaussians._opacity
    shs = gaussians.get_features

    scales = gaussians._scaling
    rotations = gaussians._rotation

    reg_loss = None

    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
            means3D,
            scales,
            rotations,
            opacity,
            shs,
        )
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
            gaussians._deformation(means3D, scales, rotations, opacity, shs, time)
        )
    else:
        raise NotImplementedError

    scales_final = gaussians.scaling_activation(scales_final)
    rotations_final = gaussians.rotation_activation(rotations_final)
    opacity = gaussians.opacity_activation(opacity_final)

    # precompute color from SH
    shs_view = shs_final.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    dir_pp = means3D_final - viewpoint_camera.camera_center.cuda().repeat(
        gaussians.get_features.shape[0], 1
    )
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    shs_final = None

    if event_idx is not None:
        with torch.no_grad():
            means3D_plus, shs_plus = (
                gaussians._deformation.forward_pos_and_color(means3D, shs, time_final)
            )
            means3D_first, shs_first = (
                gaussians._deformation.forward_pos_and_color(means3D, shs, time_first)
            )
            vec_fp = means3D_plus - means3D_first
            vec_pf = means3D_first - means3D_plus
            means3D_pred = (means3D_plus - means3D_first) / (time_final - time_first) * (time - time_first) + means3D_first
        
            #precompute color from SH
            shs_view = shs_first.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
            dir_pp = means3D_first - viewpoint_camera.camera_center.cuda().repeat(
                gaussians.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_init = torch.clamp_min(sh2rgb + 0.5, 0.0)

        reg_means = torch.abs(means3D_pred - means3D_final).mean()
        reg_color = torch.abs(colors_precomp - colors_init).mean()

        reg_loss = reg_means + reg_color

    rendered, radii, depth, rendered_alpha = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=None,
    )

    rendered_image = rendered[:3, ...]

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "alpha": rendered_alpha,
        "reg_loss": reg_loss
    }
