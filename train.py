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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.residual_injection_utils import RGBDResidualInjector
from utils.pose_refinement_utils import PoseRefinementManager
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def image_luminance(image):
    """Convert a CxHxW image tensor in [0,1] to 1xHxW luminance."""
    if image.shape[0] == 3:
        r, g, b = image[0:1], image[1:2], image[2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return image.mean(dim=0, keepdim=True)


def gradient_loss_and_edge_weight(image, gt_image, eps=1e-6):
    """
    Compute simple finite-difference image gradient loss and a normalized GT edge map.

    Returns:
        grad_loss: scalar L1 loss between rendered and GT image gradients
        edge_norm: 1xHxW normalized GT edge strength in [0, 1]
    """
    img_l = image_luminance(image)
    gt_l = image_luminance(gt_image)

    img_dx = img_l[:, :, 1:] - img_l[:, :, :-1]
    gt_dx = gt_l[:, :, 1:] - gt_l[:, :, :-1]
    img_dy = img_l[:, 1:, :] - img_l[:, :-1, :]
    gt_dy = gt_l[:, 1:, :] - gt_l[:, :-1, :]

    grad_loss = torch.abs(img_dx - gt_dx).mean() + torch.abs(img_dy - gt_dy).mean()

    # Edge strength from GT only. Pad back to full HxW.
    edge = torch.zeros_like(gt_l)
    edge[:, :, 1:] += torch.abs(gt_dx)
    edge[:, 1:, :] += torch.abs(gt_dy)
    edge = edge.detach()
    edge_norm = edge / (edge.amax() + eps)
    return grad_loss, edge_norm




def _pose_state_for_cameras(pose_refiner, cameras):
    idxs = pose_refiner.camera_indices(cameras)
    if not idxs:
        return idxs, None, None
    idx = torch.tensor(idxs, device="cuda", dtype=torch.long)
    return idxs, pose_refiner.rot_delta[idx].detach().clone(), pose_refiner.trans_delta[idx].detach().clone()


def _restore_pose_state_for_indices(pose_refiner, idxs, rot_state, trans_state):
    if not idxs or rot_state is None or trans_state is None:
        return
    idx = torch.tensor(idxs, device="cuda", dtype=torch.long)
    with torch.no_grad():
        pose_refiner.rot_delta[idx].copy_(rot_state)
        pose_refiner.trans_delta[idx].copy_(trans_state)


def _camera_window_names(cameras):
    return [getattr(c, "image_name", "UNKNOWN") for c in cameras]

def camera_pose_inner_ba_burst(
    viewpoint_cam,
    gaussians,
    pose_refiner,
    pose_optimizer,
    pipe,
    background,
    dataset,
    opt,
    iteration,
    num_steps,
    separate_sh,
):
    """
    SplaTAM-style short tracking/BA burst on one fixed view.

    This freezes the Gaussian map and repeatedly optimizes the pose delta for
    the same selected camera view for num_steps inner iterations. After the
    burst, Gaussian/exposure gradients are cleared so the outer training step
    remains a normal Gaussian-only step.
    """
    if pose_refiner is None or pose_optimizer is None or num_steps <= 0:
        return

    gt_image = viewpoint_cam.original_image.cuda()
    log_every = max(int(getattr(opt, "pose_ba_inner_log_every", 10)), 1)

    print(f"\n[Pose BA inner] outer iter {iteration}: optimizing camera {viewpoint_cam.image_name} "
          f"for {num_steps} inner pose steps")

    # Keep the best pose state from the inner loop. Photometric BA can easily
    # overshoot or make the selected view worse; do not accept a worse endpoint.
    best_loss = float("inf")
    best_rot = None
    best_trans = None
    ba_indices, initial_rot, initial_trans = _pose_state_for_cameras(pose_refiner, [viewpoint_cam])

    for inner_step in range(1, num_steps + 1):
        pose_optimizer.zero_grad(set_to_none=True)
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)

        render_cam = pose_refiner.apply(viewpoint_cam, enable_grad=True)
        render_pkg = render(
            render_cam,
            gaussians,
            pipe,
            background,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=separate_sh,
        )
        image = render_pkg["render"]
        if viewpoint_cam.alpha_mask is not None:
            image = image * viewpoint_cam.alpha_mask.cuda()

        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        ba_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if opt.pose_reg_weight > 0.0:
            ba_loss = ba_loss + opt.pose_reg_weight * pose_refiner.regularization(viewpoint_cam)

        loss_scalar = float(ba_loss.detach().item())
        if loss_scalar < best_loss and ba_indices:
            best_loss = loss_scalar
            _, best_rot, best_trans = _pose_state_for_cameras(pose_refiner, [viewpoint_cam])

        ba_loss.backward()

        if inner_step == 1 or inner_step == num_steps or (inner_step % log_every == 0):
            pose_refiner.print_grad_summary(prefix=f"Pose BA inner {iteration} step {inner_step}/{num_steps}")
            pose_refiner.print_camera_summary(viewpoint_cam, prefix=f"Pose BA inner {iteration} step {inner_step}/{num_steps}")
            print(f"[Pose BA inner {iteration} step {inner_step}/{num_steps}] loss={ba_loss.item():.6f}")

        pose_optimizer.step()

    # Restore the best observed pose state. If no step improved over the
    # starting state, this becomes a no-op / rollback to the initial pose.
    if best_rot is not None and best_trans is not None:
        _restore_pose_state_for_indices(pose_refiner, ba_indices, best_rot, best_trans)
    elif initial_rot is not None and initial_trans is not None:
        _restore_pose_state_for_indices(pose_refiner, ba_indices, initial_rot, initial_trans)

    # Do not let inner BA gradients leak into the normal Gaussian update.
    pose_optimizer.zero_grad(set_to_none=True)
    gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.exposure_optimizer.zero_grad(set_to_none=True)

    print(f"[Pose BA inner {iteration}] accepted best loss={best_loss:.6f}")
    pose_refiner.print_camera_summary(viewpoint_cam, prefix=f"Pose BA inner {iteration} final camera")
    pose_refiner.print_summary(prefix=f"Pose BA inner {iteration} global")



def camera_pose_window_ba_burst(
    center_cam,
    train_cameras,
    gaussians,
    pose_refiner,
    pose_optimizer,
    pipe,
    background,
    dataset,
    opt,
    iteration,
    num_steps,
    separate_sh,
):
    """
    Local-window BA burst.

    This freezes the Gaussian map and optimizes pose deltas for a contiguous
    window of nearby training cameras together. Compared with one-camera BA,
    this adds multi-view pressure and a local trajectory smoothness prior.
    """
    if pose_refiner is None or pose_optimizer is None or num_steps <= 0:
        return

    window_size = max(int(getattr(opt, "pose_ba_window", 5)), 1)
    ba_cams = pose_refiner.window_cameras(train_cameras, center_cam, window_size=window_size)
    log_every = max(int(getattr(opt, "pose_ba_inner_log_every", 10)), 1)

    print(
        f"\n[Pose BA window] outer iter {iteration}: center={center_cam.image_name}, "
        f"window={len(ba_cams)} cameras, inner steps={num_steps}"
    )
    print(f"[Pose BA window {iteration}] temporal window images={_camera_window_names(ba_cams)}")
    pose_refiner.print_window_summary(ba_cams, prefix=f"Pose BA window {iteration} initial")

    # Keep the best pose state over the inner loop. This is important because
    # some windows reduce loss at step 100 but are worse by step 200, and some
    # windows are worse than the initial state.
    best_loss = float("inf")
    best_rot = None
    best_trans = None
    ba_indices, initial_rot, initial_trans = _pose_state_for_cameras(pose_refiner, ba_cams)

    for inner_step in range(1, num_steps + 1):
        pose_optimizer.zero_grad(set_to_none=True)
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device="cuda")
        total_l1 = 0.0

        for cam in ba_cams:
            gt_image = cam.original_image.cuda()
            render_cam = pose_refiner.apply(cam, enable_grad=True)
            render_pkg = render(
                render_cam,
                gaussians,
                pipe,
                background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=separate_sh,
            )
            image = render_pkg["render"]
            if cam.alpha_mask is not None:
                image = image * cam.alpha_mask.cuda()

            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            view_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            total_loss = total_loss + view_loss
            total_l1 += float(Ll1.detach().item())

        total_loss = total_loss / max(len(ba_cams), 1)

        if opt.pose_reg_weight > 0.0:
            total_loss = total_loss + opt.pose_reg_weight * pose_refiner.regularization_for_cameras(ba_cams)

        smooth_weight = float(getattr(opt, "pose_smooth_weight", 0.0))
        if smooth_weight > 0.0:
            total_loss = total_loss + smooth_weight * pose_refiner.smoothness_regularization_for_cameras(ba_cams)

        loss_scalar = float(total_loss.detach().item())
        if loss_scalar < best_loss and ba_indices:
            best_loss = loss_scalar
            _, best_rot, best_trans = _pose_state_for_cameras(pose_refiner, ba_cams)

        total_loss.backward()

        if inner_step == 1 or inner_step == num_steps or (inner_step % log_every == 0):
            pose_refiner.print_grad_summary(prefix=f"Pose BA window {iteration} step {inner_step}/{num_steps}")
            pose_refiner.print_window_summary(ba_cams, prefix=f"Pose BA window {iteration} step {inner_step}/{num_steps}")
            print(
                f"[Pose BA window {iteration} step {inner_step}/{num_steps}] "
                f"loss={total_loss.item():.6f}, mean_L1={total_l1 / max(len(ba_cams), 1):.6f}"
            )

        pose_optimizer.step()

    # Restore best state / rollback if the BA burst made this local objective worse.
    if best_rot is not None and best_trans is not None:
        _restore_pose_state_for_indices(pose_refiner, ba_indices, best_rot, best_trans)
    elif initial_rot is not None and initial_trans is not None:
        _restore_pose_state_for_indices(pose_refiner, ba_indices, initial_rot, initial_trans)

    # Do not let inner BA gradients leak into the normal Gaussian update.
    pose_optimizer.zero_grad(set_to_none=True)
    gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.exposure_optimizer.zero_grad(set_to_none=True)

    print(f"[Pose BA window {iteration}] accepted best loss={best_loss:.6f}")
    pose_refiner.print_window_summary(ba_cams, prefix=f"Pose BA window {iteration} final")
    pose_refiner.print_summary(prefix=f"Pose BA window {iteration} global")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    residual_injector = RGBDResidualInjector(dataset.source_path) if opt.use_residual_injection else None
    residual_injected_total = 0

    pose_refiner = None
    pose_optimizer = None
    if opt.optimize_camera_poses:
        pose_refiner = PoseRefinementManager(scene.getTrainCameras(), pose_lr=opt.pose_lr)
        pose_optimizer = pose_refiner.make_optimizer(opt.pose_lr)
        print(f"[Pose refinement] Enabled for {len(scene.getTrainCameras())} training cameras")
        print(f"[Pose refinement] active from iter {opt.pose_refine_from_iter} to {opt.pose_refine_until_iter}")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    train_cameras_full = scene.getTrainCameras()
    viewpoint_stack = train_cameras_full.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Optionally render through small optimized camera-pose corrections.
        # We keep applying the refined pose after pose_refine_from_iter, but only
        # optimize pose parameters during the requested schedule.
        use_refined_pose = pose_refiner is not None and iteration >= opt.pose_refine_from_iter
        in_pose_window = use_refined_pose and iteration <= opt.pose_refine_until_iter

        pose_grad_enabled = False
        pose_only_active = False

        if in_pose_window:
            if opt.pose_opt_mode == "joint":
                # Update Gaussians and poses on the same iterations.
                pose_grad_enabled = True
                pose_only_active = False
            elif opt.pose_opt_mode == "pose_only":
                # Freeze Gaussians for the entire pose window.
                pose_grad_enabled = True
                pose_only_active = True
            elif opt.pose_opt_mode == "interval_ba":
                # Inner-loop BA on the selected camera only.
                ba_interval = max(int(opt.pose_ba_interval), 1)
                ba_steps = max(int(opt.pose_ba_steps), 0)
                phase = (iteration - opt.pose_refine_from_iter) % ba_interval
                if phase == 0 and ba_steps > 0:
                    camera_pose_inner_ba_burst(
                        viewpoint_cam=viewpoint_cam,
                        gaussians=gaussians,
                        pose_refiner=pose_refiner,
                        pose_optimizer=pose_optimizer,
                        pipe=pipe,
                        background=bg,
                        dataset=dataset,
                        opt=opt,
                        iteration=iteration,
                        num_steps=ba_steps,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )
                # The actual outer iteration remains Gaussian-only. The refined
                # pose is used but detached so only Gaussians update here.
                pose_grad_enabled = False
                pose_only_active = False
            elif opt.pose_opt_mode == "window_ba":
                # Inner-loop BA on a local window of nearby cameras.
                ba_interval = max(int(opt.pose_ba_interval), 1)
                ba_steps = max(int(opt.pose_ba_steps), 0)
                phase = (iteration - opt.pose_refine_from_iter) % ba_interval
                if phase == 0 and ba_steps > 0:
                    camera_pose_window_ba_burst(
                        center_cam=viewpoint_cam,
                        train_cameras=train_cameras_full,
                        gaussians=gaussians,
                        pose_refiner=pose_refiner,
                        pose_optimizer=pose_optimizer,
                        pipe=pipe,
                        background=bg,
                        dataset=dataset,
                        opt=opt,
                        iteration=iteration,
                        num_steps=ba_steps,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )
                # The actual outer iteration remains Gaussian-only. The refined
                # pose is used but detached so only Gaussians update here.
                pose_grad_enabled = False
                pose_only_active = False
            else:
                raise ValueError(f"Unknown pose_opt_mode: {opt.pose_opt_mode}")

        render_cam = pose_refiner.apply(viewpoint_cam, enable_grad=pose_grad_enabled) if use_refined_pose else viewpoint_cam

        render_pkg = render(render_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        residual_for_injection = None

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Detail-aware supervision. These are disabled by default.
        # lambda_grad_loss penalizes missing image gradients/edges directly.
        # lambda_edge_l1 increases RGB supervision where the GT image has high edge strength.
        if opt.lambda_grad_loss > 0.0 or opt.lambda_edge_l1 > 0.0:
            grad_detail_loss, gt_edge_norm = gradient_loss_and_edge_weight(image, gt_image)
            if viewpoint_cam.alpha_mask is not None:
                gt_edge_norm = gt_edge_norm * viewpoint_cam.alpha_mask.detach().cuda()

            if opt.lambda_grad_loss > 0.0:
                loss = loss + opt.lambda_grad_loss * grad_detail_loss

            if opt.lambda_edge_l1 > 0.0:
                per_pixel_l1 = torch.abs(image - gt_image).mean(dim=0, keepdim=True)
                edge_weight = 1.0 + gt_edge_norm
                edge_l1 = (per_pixel_l1 * edge_weight).mean()
                loss = loss + opt.lambda_edge_l1 * edge_l1

        if opt.use_residual_injection:
            residual_for_injection = torch.mean(torch.abs(image.detach() - gt_image.detach()), dim=0)
            if viewpoint_cam.alpha_mask is not None:
                residual_for_injection = residual_for_injection * viewpoint_cam.alpha_mask.detach().cuda().squeeze(0)

        # Bundle-adjustment-style pose regularization. This keeps learned pose
        # corrections small so they only absorb local PF/RGB-D pose noise.
        if pose_refiner is not None and pose_grad_enabled and opt.pose_reg_weight > 0.0:
            loss = loss + opt.pose_reg_weight * pose_refiner.regularization(viewpoint_cam)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        if (pose_refiner is not None and pose_grad_enabled
            and opt.pose_log_interval > 0
            and iteration % opt.pose_log_interval == 0):
            pose_refiner.print_grad_summary(prefix=f"Pose refinement iter {iteration}")

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if pose_refiner is not None:
                    pose_refiner.save(dataset.model_path, iteration)
                    pose_refiner.print_summary(prefix=f"Pose refinement iter {iteration}")

            # Densification. In pose_only mode, keep the Gaussian map fixed so
            # the pose optimizer cannot co-adapt with changing geometry.
            if (not pose_only_active) and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # RGB-D residual point injection. This supplements vanilla densification
            # by adding new Gaussians at high RGB-error pixels with valid metric depth.
            if (opt.use_residual_injection and residual_injector is not None
                and iteration >= opt.residual_inject_from_iter
                and iteration < opt.densify_until_iter
                and iteration % opt.residual_inject_interval == 0
                and residual_for_injection is not None):
                xyz_new, rgb_new = residual_injector.propose_points(
                    render_cam,
                    residual_for_injection,
                    gt_image.detach(),
                    opt.residual_inject_points,
                    rendered_depth=render_pkg.get("depth", None),
                    rendered_image=image.detach(),
                    use_detail_aware=opt.use_detail_aware_injection,
                    aggressive_detail=opt.aggressive_detail_injection,
                )
                n_injected = gaussians.inject_points(xyz_new, rgb_new, scene.cameras_extent)
                residual_injected_total += n_injected
                if n_injected > 0:
                    print(f"\n[ITER {iteration}] Residual injection added {n_injected} Gaussians "
                          f"(total injected: {residual_injected_total}, total count: {gaussians.get_xyz.shape[0]})")

            # Optimizer step
            if iteration < opt.iterations:
                if not pose_only_active:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    # Clear Gaussian gradients without stepping them.
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if pose_optimizer is not None:
                    if pose_grad_enabled:
                        pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
