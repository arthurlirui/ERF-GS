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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import random

from plyfile import PlyData, PlyElement
from pytorch3d.ops import knn_points, knn_gather
from simple_knn._C import distCUDA2

from .deformation.deformation import Deformation
from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    pix2ndc,
    ndc2pix,
)
from utils.sh_utils import RGB2SH


class GaussianModel:
    def __init__(
        self, pcd, radius, sh_degree, deformation, percent_dense, **ignore_kwargs
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._deformation = Deformation(deformation).cuda()
        self.setup_functions()
        self.create_from_pcd(pcd, radius)

        self.percent_dense = percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def create_from_pcd(self, pcd, radius):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        xyz_max = pcd.points.max(axis=0)
        xyz_min = pcd.points.min(axis=0)
        self._deformation.set_aabb(xyz_max, xyz_min)

        self.radius = radius

    def build_opt(self, args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": args.position_lr_init * self.radius,
                "name": "xyz",
            },
            {
                "params": list(self._deformation.get_mlp_parameters()),
                "lr": args.deformation_lr_init * self.radius,
                "name": "deformation",
            },
            {
                "params": list(self._deformation.get_grid_parameters()),
                "lr": args.grid_lr_init * self.radius,
                "name": "grid",
            },
            {"params": [self._features_dc], "lr": args.feature_lr, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-8)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.position_lr_init * self.radius,
            lr_final=args.position_lr_final * self.radius,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.position_lr_max_steps,
        )
        self.deformation_scheduler_args = get_expon_lr_func(
            lr_init=args.deformation_lr_init * self.radius,
            lr_final=args.deformation_lr_final * self.radius,
            lr_delay_mult=args.deformation_lr_delay_mult,
            max_steps=args.position_lr_max_steps,
        )
        self.grid_scheduler_args = get_expon_lr_func(
            lr_init=args.grid_lr_init * self.radius,
            lr_final=args.grid_lr_final * self.radius,
            lr_delay_mult=args.deformation_lr_delay_mult,
            max_steps=args.position_lr_max_steps,
        )

    def load(self, checkpoint_file, args):
        model_params, first_iter = torch.load(checkpoint_file)

        (
            self.active_sh_degree,
            self._xyz,
            deform_state,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            opt_dict,
            self.radius,
        ) = model_params
        self._deformation.load_state_dict(deform_state)
        self.build_opt(args)
        self.optimizer.load_state_dict(opt_dict)

        return first_iter

    def update(self, iteration):
        if iteration % 1000 == 0:
            if self.active_sh_degree < self.max_sh_degree:
                self.active_sh_degree += 1

        # Learning rate scheduling per step
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                # return lr
            if "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group["lr"] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group["lr"] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save(self, dir, iteration, stage):
        info = (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.radius,
        )
        torch.save(
            (info, iteration), dir + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth"
        )

        if stage == "coarse":
            point_cloud_path = os.path.join(
                dir, "point_cloud/coarse_iteration_{}".format(iteration)
            )
        else:
            point_cloud_path = os.path.join(
                dir, "point_cloud/iteration_{}".format(iteration)
            )
        os.makedirs(point_cloud_path, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(os.path.join(point_cloud_path, "point_cloud.ply"))

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_grids(self):
        return self._deformation.grid.grids

    def densify(
        self,
        viewspace_point_tensor_grad,
        visibility_filter,
        radii,
        iteration,
        stage,
        densify_args,
    ):
        if iteration < densify_args.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(
                self.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            self.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

            if stage == "coarse":
                opacity_threshold = densify_args.opacity_threshold_coarse
                densify_threshold = densify_args.densify_grad_threshold_coarse
            else:
                opacity_threshold = (
                    densify_args.opacity_threshold_fine_init
                    - iteration
                    * (
                        densify_args.opacity_threshold_fine_init
                        - densify_args.opacity_threshold_fine_after
                    )
                    / (densify_args.densify_until_iter)
                )
                densify_threshold = (
                    densify_args.densify_grad_threshold_fine_init
                    - iteration
                    * (
                        densify_args.densify_grad_threshold_fine_init
                        - densify_args.densify_grad_threshold_after
                    )
                    / (densify_args.densify_until_iter)
                )

            if (
                iteration > densify_args.densify_from_iter
                and iteration % densify_args.densification_interval == 0
                and self.get_xyz.shape[0] < 200000
            ):
                if iteration > densify_args.opacity_reset_interval:
                    size_threshold = 20
                else:
                    size_threshold = None

                prev = self._xyz.size(0)

                grads = self.xyz_gradient_accum / self.denom
                grads[grads.isnan()] = 0.0

                self.densify_and_clone(grads, densify_threshold, self.radius)
                self.densify_and_split(grads, densify_threshold, self.radius)

                if prev != self._xyz.size(0):
                    print("Added: {}, total: {}".format(self._xyz.size(0) - prev, self._xyz.size(0)))

            if (
                iteration > densify_args.pruning_from_iter
                and iteration % densify_args.pruning_interval == 0
                and self.get_xyz.shape[0] > 150000
            ):
                if iteration > densify_args.opacity_reset_interval:
                    size_threshold = 20
                else:
                    size_threshold = None

                self.prune(opacity_threshold, self.radius, size_threshold)

            if iteration % densify_args.opacity_reset_interval == 0:
                print("reset opacity")
                self.reset_opacity()

    def event_densify(
        self, viewset, event_idx, time, time_next, depth, iteration, stage, ev_args
    ):
        if not ev_args.apply_ev_densify:
            return 0
        if (
            stage == "coarse"
            or iteration > ev_args.densify_until_iter
            or iteration < ev_args.densify_from_iter
        ):
            return 0
        if iteration % ev_args.densify_interval != 0:
            return 0
        if self._xyz.size(0) > 250000:
            return 0

        min_d = depth.min()
        max_d = depth.max()
        depth_range = (
            torch.linspace(min_d, max_d, ev_args.num_depths)
            .view(-1, 1, 1)
            .repeat(1, depth.size(1), depth.size(2))
            .cuda()
        )

        R = ev_args.downsample_scale
        r = 1 / R

        event_images = []
        views = []
        for view in viewset:
            gt_event = (
                view.event_images[0, event_idx, :, :]
                - view.event_images[1, event_idx, :, :]
            ).cuda()
            event_mask = (
                F.avg_pool2d(
                    torch.abs(gt_event).unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                > 0.2
            )
            gt_event = gt_event * event_mask.squeeze(0).squeeze(0)
            gt_event = (
                F.interpolate(gt_event.unsqueeze(0).unsqueeze(0), scale_factor=(r, r))
                .squeeze(0)
                .squeeze(0)
            )

            views.append(view)
            event_images.append(gt_event)

        if len(event_images) > 3:
            source_idx = random.sample(range(len(event_images)), 3)
        else:
            source_idx = [random.choice(range(len(event_images)))]

        event_samples = []
        sample_counts = []
        for idx in source_idx:
            view = views[idx]
            ev_idx = (torch.abs(event_images[idx]) >= ev_args.ev_thr).nonzero()

            u = ev_idx[:, 0]
            v = ev_idx[:, 1]

            h, w = view.image_height // R, view.image_width // R

            c2w = view.world_view_transform.T.inverse().cuda()
            proj_inv = view.projection_matrix.T.inverse().cuda()

            for didx in range(ev_args.num_depths):
                ndcu, ndcv = pix2ndc(u.float(), h).unsqueeze(1), pix2ndc(
                    v.float(), w
                ).unsqueeze(1)
                ndc_cam = torch.cat(
                    (ndcv, ndcu, torch.ones_like(ndcu), torch.ones_like(ndcv)), dim=1
                )
                local_point_uv = ndc_cam @ proj_inv.T
                dir_local = local_point_uv / local_point_uv[:, 3:]

                d = depth_range[didx, u, v].unsqueeze(1)

                rate = d / dir_local[:, 2:3]
                local_point = dir_local * rate
                local_point[:, -1] = 1

                world_point = local_point @ c2w.T
                world_point = world_point / world_point[:, 3:]

                xyz = world_point[:, :3]
                event_samples.append(xyz)
            sample_counts.append(xyz.size(0))

        event_samples = torch.cat(event_samples, dim=0)  # Nx3
        mask = torch.ones(event_samples.size(0), device=event_samples.device)

        for idx in range(len(event_images)):
            view = views[idx]
            wt = view.world_view_transform.T.to("cuda")
            pj = view.projection_matrix.T.to("cuda")
            curr_xyz = torch.cat(
                (
                    event_samples,
                    torch.ones(event_samples.size(0), 1, device=event_samples.device),
                ),
                dim=1,
            )
            proj_xyz = pj @ (wt @ curr_xyz.T)

            screen_coords_x = torch.clamp(
                ndc2pix(
                    proj_xyz[0, :] / (proj_xyz[3, :] + 1e-8), view.image_width // R
                ).int(),
                min=0,
                max=view.image_width // R - 1,
            )
            screen_coords_y = torch.clamp(
                ndc2pix(
                    proj_xyz[1, :] / (proj_xyz[3, :] + 1e-8), view.image_height // R
                ).int(),
                min=0,
                max=view.image_height // R - 1,
            )
            pos2d = torch.cat(
                (screen_coords_x.unsqueeze(0), screen_coords_y.unsqueeze(0)), dim=0
            )  # 2xN
            selected_events = event_images[idx][pos2d[1, :], pos2d[0, :]]

            mask = torch.logical_and(mask, torch.abs(selected_events) >= 1)

        total = 0
        result_samples = []
        for sc in sample_counts:
            sample_batch = event_samples[total : total + sc * ev_args.num_depths, :]
            mask_batch = mask[total : total + sc * ev_args.num_depths]
            total += sc * ev_args.num_depths
            mask_batch = mask_batch.view(ev_args.num_depths, -1)
            idx = torch.argmax(mask_batch.float(), dim=0)
            sample_idx = idx * sc + torch.arange(sc).cuda()
            final_batch = sample_batch[sample_idx, :]
            final_mask = mask_batch.view(-1)[sample_idx]
            final_batch = final_batch[final_mask, :]
            result_samples.append(final_batch)

        event_samples = torch.cat(result_samples, dim=0)

        time = torch.tensor(time).cuda().repeat(self._xyz.shape[0], 1)
        time_next = torch.tensor(time_next).cuda().repeat(self._xyz.shape[0], 1)
        xyz_prev = self._deformation.forward_pos(self._xyz, time)
        xyz_post = self._deformation.forward_pos(self._xyz, time_next)

        knn_prev = knn_points(
            event_samples.unsqueeze(0), xyz_prev.unsqueeze(0), K=ev_args.knn
        )
        knn_post = knn_points(
            event_samples.unsqueeze(0), xyz_post.unsqueeze(0), K=ev_args.knn
        )
        dist_prev = knn_prev.dists[0, :, :].mean(dim=-1)
        dist_post = knn_post.dists[0, :, :].mean(dim=-1)
        dist_two = torch.minimum(dist_prev, dist_post)

        if dist_two.size(0) > 2 * ev_args.num_candidates:
            candidate_idx = torch.topk(
                dist_two, 2 * ev_args.num_candidates, dim=0, largest=True
            ).indices
        else:
            candidate_idx = torch.arange(dist_two.size(0)).cuda()
        candidate_idx = candidate_idx[
            torch.randperm(candidate_idx.size(0))[: ev_args.num_candidates]
        ]
        xyz = event_samples[candidate_idx, :]

        if xyz.size(0) == 0:
            return 0

        post_xyz = self._deformation(
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self.get_features,
            time,
        )[0]
        dxyz = post_xyz - self._xyz

        knn = knn_points(
            xyz.unsqueeze(0), post_xyz.unsqueeze(0), K=ev_args.knn, return_nn=True
        )
        warp_xyz = (
            knn_gather(dxyz.unsqueeze(0), knn.idx).squeeze(0)
            * knn.dists[0, :, :].unsqueeze(-1)
        ).sum(dim=1) / knn.dists[0, :, :].unsqueeze(-1).sum(dim=1)

        num = xyz.size(0)
        test_xyz = self._deformation(
            xyz - warp_xyz,
            self._scaling[:num, :],
            self._rotation[:num, :],
            self._opacity[:num, :],
            self.get_features[:num, :, :],
            time[:num, :],
        )[0]

        float_mask = ((test_xyz - xyz) ** 2).sum(dim=1) <= ev_args.float_thr
        new_xyz = (xyz - warp_xyz)[float_mask, :]

        num = new_xyz.size(0)

        features_dc = self._features_dc.view(self._features_dc.size(0), -1)
        new_features_dc = (
            knn_gather(features_dc.unsqueeze(0), knn.idx)
            .squeeze(0)
            .mean(dim=1)
            .view(-1, self._features_dc.size(1), self._features_dc.size(2))[
                float_mask, :
            ]
        )
        # using KNN-sampled features_rest will cause problem
        new_features_rest = torch.zeros(num, self._features_rest.size(1), 3).cuda()

        new_opacity = torch.ones(num, 1).cuda()
        rot = torch.zeros(num, 4).cuda()
        rot[:, 0] = 1.0
        new_rotation = rot

        tmpxyz = torch.cat((new_xyz, self._xyz), dim=0)
        dist2 = torch.clamp_min(distCUDA2(tmpxyz).float(), 1e-7)
        dist2 = dist2[: new_xyz.shape[0]]
        new_scales = torch.log(torch.sqrt(dist2) / 6)[..., None].repeat(1, 3)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scales,
            new_rotation,
        )
        return num

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        grads_accum_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            grads_accum_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
