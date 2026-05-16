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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"

        # Detail-aware losses. Disabled by default.
        # lambda_grad_loss matches image gradients between render and GT.
        # lambda_edge_l1 upweights RGB L1 loss on GT edge/detail regions.
        self.lambda_grad_loss = 0.0
        self.lambda_edge_l1 = 0.0

        # RGB-D residual point injection. This is disabled by default.
        # It injects a small number of new Gaussians at high RGB-residual pixels
        # where metric depth is valid.
        self.use_residual_injection = False
        # If enabled, also bias injection toward regions where GT contains strong
        # high-frequency detail but the current render is missing/smoothing it out.
        self.use_detail_aware_injection = False
        # Aggressive mode relaxes detail candidate filtering so the injector can
        # fill the requested point budget. Use this as a stress test.
        self.aggressive_detail_injection = False

        # Bundle-adjustment-style per-camera pose refinement. Disabled by default.
        self.optimize_camera_poses = False
        self.pose_refine_from_iter = 3000
        self.pose_refine_until_iter = 15000
        self.pose_lr = 1e-4
        self.pose_reg_weight = 0.01
        # joint: optimize Gaussians and poses together during pose window.
        # pose_only: freeze Gaussian/exposure optimizer and densification during pose window.
        # interval_ba: every pose_ba_interval iterations, run pose_ba_steps
        # pose-only camera-refinement iterations on one camera, then resume Gaussian training.
        # window_ba: same idea, but optimize a local window of nearby cameras together.
        self.pose_opt_mode = "joint"
        self.pose_log_interval = 1000
        self.pose_ba_interval = 500
        self.pose_ba_steps = 50
        # Number of cameras in a local window for pose_opt_mode=window_ba.
        # Uses the train camera order: center +/- floor(window/2).
        self.pose_ba_window = 5
        # Smoothness prior for local-window BA. Penalizes neighboring pose
        # corrections that differ too much, improving trajectory consistency.
        self.pose_smooth_weight = 0.0
        # In interval_ba/window_ba mode, print inner-loop BA progress every N inner steps.
        self.pose_ba_inner_log_every = 10

        # Start residual injection only after the model has stabilized.
        # It still stops at densify_until_iter, same as vanilla densification.
        self.residual_inject_from_iter = 7000
        self.residual_inject_interval = 500
        self.residual_inject_points = 2000
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
