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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.gaussian_source = torch.empty(0, dtype=torch.int32, device="cuda")
        self.gaussian_birth_iter = torch.empty(0, dtype=torch.int32, device="cuda")
        self.gaussian_level = torch.empty(0, dtype=torch.int32, device="cuda")
        self.gaussian_parent = torch.empty(0, dtype=torch.int64, device="cuda")
        self.reset_densify_event_stats()
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.gaussian_source,
            self.gaussian_birth_iter,
            self.gaussian_level,
            self.gaussian_parent,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.gaussian_source,
        self.gaussian_birth_iter,
        self.gaussian_level,
        self.gaussian_parent) = model_args
        self.training_setup(training_args)
        self.reset_densify_event_stats()
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.gaussian_source = torch.zeros((self.get_xyz.shape[0],), dtype=torch.int32, device="cuda")  # 0=base, 1=clone, 2=split
        self.gaussian_birth_iter = torch.full((self.get_xyz.shape[0],), -1, dtype=torch.int32, device="cuda")
        self.gaussian_level = torch.zeros((self.get_xyz.shape[0],), dtype=torch.int32, device="cuda")
        self.gaussian_parent = torch.full((self.get_xyz.shape[0],), -1, dtype=torch.int64, device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.reset_densify_event_stats()
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))


    def create_from_random(self, cam_infos, spatial_lr_scale: float, num_points: int = 20000,
                           init_extent_factor: float = 1.0, init_scale: float = 0.01,
                           init_opacity: float = 0.05, color_mode: str = "random"):
        self.spatial_lr_scale = spatial_lr_scale
        device = "cuda"

        def _camera_center_from_info(cam):
            if hasattr(cam, "camera_center"):
                cc = cam.camera_center
                if not torch.is_tensor(cc):
                    return torch.tensor(cc, dtype=torch.float32, device=device)
                return cc.to(device=device, dtype=torch.float32)

            if hasattr(cam, "R") and hasattr(cam, "T"):
                R = torch.tensor(cam.R, dtype=torch.float32, device=device)
                T = torch.tensor(cam.T, dtype=torch.float32, device=device)
                return (-R @ T).float()

            raise AttributeError("Camera object has neither camera_center nor (R, T).")

        cam_centers = []
        for cam in cam_infos:
            cam_centers.append(_camera_center_from_info(cam))

        if len(cam_centers) == 0:
            raise ValueError("create_from_random requires at least one camera")

        cam_centers = torch.stack(cam_centers, dim=0)
        center = cam_centers.mean(dim=0)
        extent = float(spatial_lr_scale) * float(init_extent_factor)
        if extent <= 0:
            extent = 1.0

        fused_point_cloud = center[None, :] + (torch.rand((num_points, 3), device=device) * 2.0 - 1.0) * extent

        if color_mode == "gray":
            fused_rgb = torch.full((num_points, 3), 0.5, device=device)
        else:
            fused_rgb = torch.rand((num_points, 3), device=device)

        fused_color = RGB2SH(fused_rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device=device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at random initialisation : ", fused_point_cloud.shape[0])

        init_scale = max(float(init_scale), 1e-6)
        scales = torch.full((num_points, 3), float(np.log(init_scale)), dtype=torch.float32, device=device)
        rots = torch.zeros((num_points, 4), device=device)
        rots[:, 0] = 1.0
        init_opacity = min(max(float(init_opacity), 1e-4), 1.0 - 1e-4)
        opacities = self.inverse_opacity_activation(init_opacity * torch.ones((num_points, 1), dtype=torch.float32, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        self.gaussian_source = torch.zeros((self.get_xyz.shape[0],), dtype=torch.int32, device=device)
        self.gaussian_birth_iter = torch.full((self.get_xyz.shape[0],), -1, dtype=torch.int32, device=device)
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device=device)[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.gaussian_source = torch.zeros((self.get_xyz.shape[0],), dtype=torch.int32, device="cuda")
        self.gaussian_birth_iter = torch.full((self.get_xyz.shape[0],), -1, dtype=torch.int32, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if hasattr(self, "_densify_event_stats") and mask.numel() == self.gaussian_source.numel():
            pruned_total = int(mask.sum().item())
            if pruned_total > 0:
                pruned_src = self.gaussian_source[mask]
                pruned_lvl = self.gaussian_level[mask]
                self._densify_event_stats["pruned_total"] += pruned_total
                self._densify_event_stats["pruned_base"] += int((pruned_src == 0).sum().item())
                self._densify_event_stats["pruned_clone"] += int((pruned_src == 1).sum().item())
                self._densify_event_stats["pruned_split"] += int((pruned_src == 2).sum().item())
                uniq, cnt = torch.unique(pruned_lvl, return_counts=True)
                for u, c in zip(uniq.tolist(), cnt.tolist()):
                    key = int(u)
                    self._densify_event_stats["pruned_level_hist"][key] = self._densify_event_stats["pruned_level_hist"].get(key, 0) + int(c)
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
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        self.gaussian_source = self.gaussian_source[valid_points_mask]
        self.gaussian_birth_iter = self.gaussian_birth_iter[valid_points_mask]
        self.gaussian_level = self.gaussian_level[valid_points_mask]
        self.gaussian_parent = self.gaussian_parent[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def append_new_gaussians(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii=None, source_label: int = 1, birth_iter: int = -1, levels=None, parents=None):
        if new_tmp_radii is None:
            new_tmp_radii = torch.zeros((new_xyz.shape[0],), device=new_xyz.device, dtype=new_xyz.dtype)
        self.tmp_radii = torch.zeros((self.get_xyz.shape[0],), device=self.get_xyz.device, dtype=self.get_xyz.dtype)
        source_labels = torch.full((new_xyz.shape[0],), int(source_label), dtype=torch.int32, device=new_xyz.device)
        birth_iters = torch.full((new_xyz.shape[0],), int(birth_iter), dtype=torch.int32, device=new_xyz.device)
        if levels is None:
            levels = torch.zeros((new_xyz.shape[0],), dtype=torch.int32, device=new_xyz.device)
        if parents is None:
            parents = torch.full((new_xyz.shape[0],), -1, dtype=torch.int64, device=new_xyz.device)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, source_labels, birth_iters, levels, parents)
        self.tmp_radii = None

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, source_labels=None, birth_iters=None, levels=None, parents=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        if source_labels is None:
            source_labels = torch.zeros((new_xyz.shape[0],), dtype=torch.int32, device=new_xyz.device)
        if birth_iters is None:
            birth_iters = torch.full((new_xyz.shape[0],), -1, dtype=torch.int32, device=new_xyz.device)
        if levels is None:
            levels = torch.zeros((new_xyz.shape[0],), dtype=torch.int32, device=new_xyz.device)
        if parents is None:
            parents = torch.full((new_xyz.shape[0],), -1, dtype=torch.int64, device=new_xyz.device)
        self.gaussian_source = torch.cat((self.gaussian_source, source_labels))
        self.gaussian_birth_iter = torch.cat((self.gaussian_birth_iter, birth_iters))
        self.gaussian_level = torch.cat((self.gaussian_level, levels))
        self.gaussian_parent = torch.cat((self.gaussian_parent, parents))
        if hasattr(self, "_densify_event_stats"):
            n_new = int(new_xyz.shape[0])
            self._densify_event_stats["created_total"] += n_new
            self._densify_event_stats["created_clone"] += int((source_labels == 1).sum().item())
            self._densify_event_stats["created_split"] += int((source_labels == 2).sum().item())
            uniq, cnt = torch.unique(levels, return_counts=True)
            for u, c in zip(uniq.tolist(), cnt.tolist()):
                key = int(u)
                self._densify_event_stats["created_level_hist"][key] = self._densify_event_stats["created_level_hist"].get(key, 0) + int(c)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def reset_densify_event_stats(self):
        self._densify_event_stats = {
            "created_total": 0,
            "created_clone": 0,
            "created_split": 0,
            "created_level_hist": {},
            "pruned_total": 0,
            "pruned_base": 0,
            "pruned_clone": 0,
            "pruned_split": 0,
            "pruned_level_hist": {},
        }

    def get_densify_event_stats(self):
        return dict(self._densify_event_stats)

    def count_gaussians_by_source(self):
        total = int(self.get_xyz.shape[0])
        if total == 0:
            return {"total": 0, "base_alive": 0, "clone_alive": 0, "split_alive": 0}
        base_alive = int((self.gaussian_source == 0).sum().item())
        clone_alive = int((self.gaussian_source == 1).sum().item())
        split_alive = int((self.gaussian_source == 2).sum().item())
        return {"total": total, "base_alive": base_alive, "clone_alive": clone_alive, "split_alive": split_alive}

    def gaussian_level_stats(self):
        total = int(self.get_xyz.shape[0])
        if total == 0:
            return {"max_level": 0, "mean_level": 0.0, "median_level": 0.0, "per_level_counts": {}}
        levels = self.gaussian_level.detach().cpu().numpy().astype(int)
        unique, counts = np.unique(levels, return_counts=True)
        per_level_counts = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        return {
            "max_level": int(levels.max()),
            "mean_level": float(levels.mean()),
            "median_level": float(np.median(levels)),
            "per_level_counts": per_level_counts,
        }

    def lineage_summary(self):
        counts = self.count_gaussians_by_source()
        levels = self.gaussian_level_stats()
        out = {"counts": counts, "levels": levels, "base_levels": {}, "clone_levels": {}, "split_levels": {}}
        total = int(self.get_xyz.shape[0])
        if total == 0:
            return out
        src = self.gaussian_source.detach().cpu().numpy().astype(int)
        lvl = self.gaussian_level.detach().cpu().numpy().astype(int)
        for src_id, key in [(0, "base_levels"), (1, "clone_levels"), (2, "split_levels")]:
            mask = src == src_id
            if mask.any():
                u, c = np.unique(lvl[mask], return_counts=True)
                out[key] = {int(k): int(v) for k, v in zip(u.tolist(), c.tolist())}
        return out

    def triangulated_age_stats(self, current_iter: int):
        if self.gaussian_source.numel() == 0:
            return {"triangulated_alive": 0, "triangulated_mean_age": 0.0, "triangulated_median_age": 0.0}
        mask = self.gaussian_source == 1
        if not bool(mask.any()):
            return {"triangulated_alive": 0, "triangulated_mean_age": 0.0, "triangulated_median_age": 0.0}
        ages = (int(current_iter) - self.gaussian_birth_iter[mask].to(torch.int64)).clamp_min(0)
        return {
            "triangulated_alive": int(mask.sum().item()),
            "triangulated_mean_age": float(ages.float().mean().item()),
            "triangulated_median_age": float(ages.float().median().item()),
        }


    def export_lineage_arrays(self):
        return {
            "xyz": self.get_xyz.detach().cpu().numpy(),
            "source": self.gaussian_source.detach().cpu().numpy(),
            "birth_iter": self.gaussian_birth_iter.detach().cpu().numpy(),
            "level": self.gaussian_level.detach().cpu().numpy(),
            "parent": self.gaussian_parent.detach().cpu().numpy(),
            "opacity": self.get_opacity.detach().cpu().numpy(),
        }

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        selected_idx = torch.where(selected_pts_mask)[0]
        new_levels = (self.gaussian_level[selected_idx].repeat(N) + 1).to(torch.int32)
        new_parents = selected_idx.repeat(N).to(torch.int64)
        new_sources = torch.full((new_xyz.shape[0],), 2, dtype=torch.int32, device=new_xyz.device)
        new_births = torch.full((new_xyz.shape[0],), -1, dtype=torch.int32, device=new_xyz.device)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_sources, new_births, new_levels, new_parents)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        selected_idx = torch.where(selected_pts_mask)[0]
        new_levels = (self.gaussian_level[selected_idx] + 1).to(torch.int32)
        new_parents = selected_idx.to(torch.int64)
        new_sources = torch.full((new_xyz.shape[0],), 1, dtype=torch.int32, device=new_xyz.device)
        new_births = torch.full((new_xyz.shape[0],), -1, dtype=torch.int32, device=new_xyz.device)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_sources, new_births, new_levels, new_parents)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1



