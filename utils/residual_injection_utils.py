import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.graphics_utils import fov2focal


class RGBDResidualInjector:
    """
    Helper for RGB-D residual point injection.

    V6: depth-aware + detail-aware candidate selection.

    Depth-aware branch (existing idea):
      - observed metric depth is valid, and
      - the current rendered depth is missing or disagrees with observed depth.

    Detail-aware branch (new idea):
      - observed metric depth is valid, and
      - the GT image has strong high-frequency detail, and
      - the current render is missing/smoothing that detail.

    The two branches are combined into a single score. This lets the method add
    Gaussians both for missing geometry and for missing appearance detail on
    already-covered surfaces (e.g. hose ridges / thin stripes).
    """

    def __init__(self, source_path, depth_dir_candidates=("depth", "depths"), min_depth=0.1, max_depth=5.0):
        self.source_path = source_path
        self.depth_dir_candidates = depth_dir_candidates
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = self._read_depth_scale(source_path)

        # Conservative defaults. Kept mostly internal to avoid too many knobs.
        self.metric_depth_threshold_m = 0.10
        self.inverse_depth_threshold = 0.05
        self.residual_quantile = 0.75
        self.nms_kernel_size = 7

        # Detail-aware defaults.
        self.detail_edge_quantile = 0.70
        self.detail_missing_floor = 0.05
        self.detail_branch_weight = 1.25
        self.depth_branch_weight = 1.0

        # Aggressive detail-injection presets. These are intentionally loose:
        # they are for testing whether the bottleneck is simply that too few
        # detail candidates are being inserted.
        self.aggressive_residual_quantile = 0.05
        self.aggressive_nms_kernel_size = 3
        self.aggressive_detail_edge_quantile = 0.25
        self.aggressive_detail_branch_weight = 6.0
        self.aggressive_rgb_fallback_weight = 0.15

    def _read_depth_scale(self, source_path):
        intr_path = os.path.join(source_path, "intrinsics.json")
        if os.path.exists(intr_path):
            with open(intr_path, "r") as f:
                intr = json.load(f)
            return float(intr.get("depth_scale", 0.001))
        return 0.001

    def _depth_path_for_image(self, image_name):
        stem = os.path.splitext(image_name)[0]
        candidates = []
        for dname in self.depth_dir_candidates:
            droot = os.path.join(self.source_path, dname)
            candidates.extend([
                os.path.join(droot, image_name),
                os.path.join(droot, stem + ".png"),
                os.path.join(droot, stem + ".jpg"),
            ])
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def load_metric_depth(self, image_name, target_hw):
        depth_path = self._depth_path_for_image(image_name)
        if depth_path is None:
            return None

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            return None

        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        target_h, target_w = target_hw
        if depth_m.shape[:2] != (target_h, target_w):
            depth_m = cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(depth_m).cuda()

    @staticmethod
    def _nms_residual(residual, kernel_size=7):
        """Simple local-max suppression on a [H, W] residual map."""
        if kernel_size <= 1:
            return residual
        pad = kernel_size // 2
        x = residual[None, None]
        pooled = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)[0, 0]
        return torch.where(residual >= pooled, residual, torch.zeros_like(residual))

    @staticmethod
    def _prepare_rendered_depth(rendered_depth, target_hw):
        """Return rendered depth as a [H, W] CUDA tensor, or None."""
        if rendered_depth is None:
            return None
        rd = rendered_depth.detach().float()
        while rd.ndim > 2:
            rd = rd.squeeze(0)
        if rd.ndim != 2:
            return None
        H, W = target_hw
        if rd.shape != (H, W):
            rd = F.interpolate(rd[None, None], size=(H, W), mode="nearest")[0, 0]
        return rd

    @staticmethod
    def _to_luminance(img):
        """Convert [3,H,W] or [1,H,W] image in [0,1] to [1,H,W]."""
        if img is None:
            return None
        if img.ndim == 2:
            return img[None]
        if img.shape[0] == 3:
            r, g, b = img[0:1], img[1:2], img[2:3]
            return 0.299 * r + 0.587 * g + 0.114 * b
        return img[:1]

    @staticmethod
    def _sobel_magnitude(img_1hw):
        """Sobel edge magnitude for [1,H,W] image, returned as [H,W]."""
        x = img_1hw[None]  # [1,1,H,W]
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return mag[0, 0]

    def _detail_score(self, gt_image, rendered_image, observed_valid, aggressive=False):
        """
        Build a score for regions where GT has detail but render is smooth.
        Returns [H,W] score >= 0.
        """
        if gt_image is None or rendered_image is None:
            return None

        gt_l = self._to_luminance(gt_image.detach().float())
        rd_l = self._to_luminance(rendered_image.detach().float())
        gt_edge = self._sobel_magnitude(gt_l)
        rd_edge = self._sobel_magnitude(rd_l)

        # Normalize edge maps robustly using maxima. Safe because later we also
        # gate with a quantile threshold to focus on genuinely detailed regions.
        gt_edge_norm = gt_edge / (gt_edge.amax() + 1e-6)
        rd_edge_norm = rd_edge / (rd_edge.amax() + 1e-6)

        missing_edge = torch.relu(gt_edge_norm - rd_edge_norm)

        valid_gt_edges = gt_edge_norm[observed_valid]
        if valid_gt_edges.numel() == 0:
            return None

        if aggressive:
            # Aggressive mode: do not require render edge to be completely missing.
            # Strong GT edges plus RGB residual are enough to request detail points.
            edge_thresh = torch.quantile(valid_gt_edges, self.aggressive_detail_edge_quantile)
            detail_mask = observed_valid & (gt_edge_norm >= edge_thresh)
            detail_score = gt_edge_norm * (0.50 + missing_edge)
            detail_score[~detail_mask] = 0.0
            return detail_score

        # Conservative mode: focus on GT edges that the render is actually missing.
        edge_thresh = torch.quantile(valid_gt_edges, self.detail_edge_quantile)
        detail_mask = observed_valid & (gt_edge_norm >= edge_thresh) & (missing_edge >= self.detail_missing_floor)
        if detail_mask.sum() == 0:
            return torch.zeros_like(gt_edge_norm)

        detail_score = gt_edge_norm * missing_edge
        detail_score[~detail_mask] = 0.0
        return detail_score

    def propose_points(
        self,
        viewpoint_cam,
        residual,
        gt_image,
        num_points,
        rendered_depth=None,
        rendered_image=None,
        use_detail_aware=False,
        aggressive_detail=False,
    ):
        """
        Args:
            viewpoint_cam: scene Camera object.
            residual: [H, W] CUDA tensor, high values indicate RGB error.
            gt_image: [3, H, W] CUDA tensor in [0, 1].
            num_points: maximum number of points to propose.
            rendered_depth: optional renderer output depth [H, W] or [1, H, W].
            rendered_image: optional rendered RGB [3,H,W] used for detail-aware mode.
            use_detail_aware: if True, also inject where GT has strong detail but
                the current render is missing it.
            aggressive_detail: if True, relax filtering so more detail candidates
                are inserted. This is meant as a stress test.

        Returns:
            xyz_world: [N, 3] CUDA tensor
            rgb: [N, 3] CUDA tensor in [0, 1]
        """
        if num_points <= 0:
            return None, None

        H, W = residual.shape
        depth = self.load_metric_depth(viewpoint_cam.image_name, (H, W))
        if depth is None:
            return None, None

        observed_valid = (depth > self.min_depth) & (depth < self.max_depth) & torch.isfinite(depth)
        if observed_valid.sum() == 0:
            return None, None

        rgb_residual = residual.detach().float().clone()
        rgb_residual[~observed_valid] = 0.0

        score = torch.zeros_like(rgb_residual)

        # Depth-aware geometric branch.
        rd = self._prepare_rendered_depth(rendered_depth, (H, W))
        if rd is not None:
            rendered_valid = torch.isfinite(rd) & (rd > 1e-6)

            obs_metric = depth
            obs_inv = torch.zeros_like(depth)
            obs_inv[observed_valid] = 1.0 / torch.clamp(depth[observed_valid], min=1e-6)

            metric_mismatch = torch.abs(rd - obs_metric) / self.metric_depth_threshold_m
            inv_mismatch = torch.abs(rd - obs_inv) / self.inverse_depth_threshold
            mismatch = torch.minimum(metric_mismatch, inv_mismatch)

            depth_candidate = observed_valid & ((~rendered_valid) | (mismatch > 1.0))
            depth_weight = torch.clamp(mismatch, min=0.0, max=10.0)
            depth_score = rgb_residual * depth_candidate.float() * (1.0 + 0.25 * depth_weight)
            score = score + self.depth_branch_weight * depth_score
        else:
            # Fallback when depth render is unavailable.
            score = score + self.depth_branch_weight * rgb_residual
            score[~observed_valid] = 0.0

        # Detail-aware branch. This can add points even when rendered depth is
        # already roughly correct, as long as valid observed depth exists.
        if use_detail_aware and rendered_image is not None:
            detail_score = self._detail_score(gt_image, rendered_image, observed_valid, aggressive=aggressive_detail)
            if detail_score is not None:
                weight = self.aggressive_detail_branch_weight if aggressive_detail else self.detail_branch_weight
                score = score + weight * (rgb_residual * detail_score)

        if aggressive_detail:
            # Ensure the injector can actually reach the requested point budget.
            # This adds a weak valid-depth RGB-residual fallback, while the detail
            # branch still dominates high-edge regions.
            score = score + self.aggressive_rgb_fallback_weight * rgb_residual * observed_valid.float()

        candidate_values = score[score > 0]
        if candidate_values.numel() == 0:
            return None, None

        qval = self.aggressive_residual_quantile if aggressive_detail else self.residual_quantile
        q = torch.quantile(candidate_values, qval)
        score = torch.where(score >= q, score, torch.zeros_like(score))

        nms_kernel = self.aggressive_nms_kernel_size if aggressive_detail else self.nms_kernel_size
        score = self._nms_residual(score, kernel_size=nms_kernel)
        score[~observed_valid] = 0.0

        flat = score.reshape(-1)
        k = min(int(num_points), int((flat > 0).sum().item()))
        if k <= 0:
            return None, None

        _, top_idx = torch.topk(flat, k=k, largest=True, sorted=False)
        v = torch.div(top_idx, W, rounding_mode="floor").float()
        u = (top_idx % W).float()
        z = depth.reshape(-1)[top_idx]

        fx = fov2focal(viewpoint_cam.FoVx, W)
        fy = fov2focal(viewpoint_cam.FoVy, H)
        cx = 0.5 * W
        cy = 0.5 * H

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = torch.stack([x, y, z], dim=1)

        # Use the actual camera transform being rendered. This supports both
        # original cameras and pose-refined camera wrappers. GraphDECO stores
        # world_view_transform in row-vector convention: W2C_col^T.
        w2c_col = viewpoint_cam.world_view_transform.transpose(0, 1)
        c2w_col = torch.inverse(w2c_col)
        R_c2w = c2w_col[:3, :3]
        cam_center = c2w_col[:3, 3]
        pts_world = (R_c2w @ pts_cam.T).T + cam_center[None]

        rgb = gt_image[:, v.long(), u.long()].T.contiguous().clamp(0.0, 1.0)
        return pts_world.contiguous(), rgb
