import torch
import torch.nn as nn
import torch.nn.init as init

from .grid import DenseGrid
from .hexplane import HexPlaneField


def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight, gain=1)


class Deformation(nn.Module):
    def __init__(self, args):
        super(Deformation, self).__init__()

        self.W = args.net_width
        self.D = args.defor_depth
        self.input_ch = 3 + (3 * args.posebase_pe) * 2
        self.grid_pe = args.grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        if args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])
        else:
            self.empty_voxel = None
        if args.static_mlp:
            self.static_mlp = nn.Sequential(
                nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1)
            )
        else:
            self.static_mlp = None

        if self.grid_pe != 0:
            grid_out_dim = self.grid.feat_dim + (self.grid.feat_dim) * 2
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(grid_out_dim, self.W)]
        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)
        )
        self.scales_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)
        )
        self.rotations_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4)
        )
        self.opacity_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1)
        )
        self.shs_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 16 * 3)
        )

        self.no_dx = args.no_dx
        self.no_ds = args.no_ds
        self.no_dr = args.no_dr
        self.no_do = args.no_do
        self.no_dshs = args.no_dshs

        self.register_buffer(
            "pos_poc", torch.FloatTensor([(2**i) for i in range(args.posebase_pe)])
        )
        self.register_buffer(
            "rotation_scaling_poc",
            torch.FloatTensor([(2**i) for i in range(args.scale_rotation_pe)]),
        )
        self.register_buffer(
            "opacity_poc", torch.FloatTensor([(2**i) for i in range(args.opacity_pe)])
        )
        self.apply(initialize_weights)

    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb", xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.empty_voxel is not None:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                parameter_list.append(param)
        return parameter_list

    def query_time(self, rays_pts_emb, time_emb):
        if self.no_grid:
            hidden = torch.cat([rays_pts_emb[:, :3], time_emb[:, :1]], -1)
        else:
            grid_feature = self.grid(rays_pts_emb[:, :3], time_emb[:, :1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            hidden = torch.cat([grid_feature], -1)
        hidden = self.feature_out(hidden)
        return hidden
    
    def forward_pos(self, point, times_sel):
        point_emb = poc_fre(point, self.pos_poc)
        hidden = self.query_time(point_emb, times_sel)

        dx = self.pos_deform(hidden)
        return point_emb[:, :3] + dx
    
    def forward_pos_and_color(self, point, shs, times_sel):
        point_emb = poc_fre(point, self.pos_poc)
        hidden = self.query_time(point_emb, times_sel)

        dx = self.pos_deform(hidden)
        dshs = self.shs_deform(hidden).reshape([shs.shape[0], 16, 3])
        return point_emb[:, :3] + dx, shs + dshs

    def forward(
        self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None
    ):
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        opacity_emb = opacity
        shs_emb = shs

        hidden = self.query_time(point_emb, times_sel)
        if self.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.empty_voxel:
            mask = self.empty_voxel(point_emb[:, :3])
        else:
            mask = torch.ones_like(opacity_emb[:, 0]).unsqueeze(-1)

        if self.no_dx:
            pts = point_emb[:, :3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(point_emb[:, :3])
            pts = point_emb[:, :3] * mask + dx

        if self.no_ds:
            scales = scales_emb[:, :3]
        else:
            ds = self.scales_deform(hidden)
            scales = torch.zeros_like(scales_emb[:, :3])
            scales = scales_emb[:, :3] * mask + ds

        if self.no_dr:
            rotations = rotations_emb[:, :4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = torch.zeros_like(rotations_emb[:, :4])
            rotations = rotations_emb[:, :4] + dr

        if self.no_do:
            opacity = opacity_emb[:, :1]
        else:
            do = self.opacity_deform(hidden)
            opacity = torch.zeros_like(opacity_emb[:, :1])
            opacity = opacity_emb[:, :1] * mask + do

        if self.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3])
            shs = torch.zeros_like(shs_emb)
            shs = shs_emb * mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs
