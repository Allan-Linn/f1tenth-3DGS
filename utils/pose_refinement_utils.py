import os
import torch


class RefinedCameraView:
    """Lightweight wrapper around a Camera with refined differentiable transforms."""

    def __init__(self, base_camera, world_view_transform, full_proj_transform, camera_center):
        self._base_camera = base_camera
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center

    def __getattr__(self, name):
        return getattr(self._base_camera, name)


def _skew(v):
    """Batch skew-symmetric matrices for v [..., 3]."""
    zero = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([
        torch.stack([zero, -vz, vy], dim=-1),
        torch.stack([vz, zero, -vx], dim=-1),
        torch.stack([-vy, vx, zero], dim=-1),
    ], dim=-2)


def so3_exp_map(rotvec):
    """Rodrigues exponential map from axis-angle vector(s) to rotation matrix."""
    orig_shape = rotvec.shape[:-1]
    w = rotvec.reshape(-1, 3)
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp(min=1e-8)
    K = _skew(w)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[0], 3, 3)

    theta2 = theta * theta
    A = torch.sin(theta) / theta
    B = (1.0 - torch.cos(theta)) / theta2

    # For very small rotations, use stable Taylor approximations.
    small = theta < 1e-4
    A = torch.where(small, 1.0 - theta2 / 6.0, A)
    B = torch.where(small, 0.5 - theta2 / 24.0, B)

    R = eye + A[..., None] * K + B[..., None] * (K @ K)
    return R.reshape(*orig_shape, 3, 3)


def make_delta_row_transform(rotvec, trans):
    """
    Build row-vector convention delta transform.

    GraphDECO stores camera.world_view_transform as W2C^T because points are
    multiplied as row vectors. If column-vector W2C is left-multiplied by a
    small camera-frame correction Delta, then the row-vector matrix is right-
    multiplied by Delta^T.
    """
    R = so3_exp_map(rotvec)
    delta_col = torch.eye(4, device=rotvec.device, dtype=rotvec.dtype)
    delta_col[:3, :3] = R
    delta_col[:3, 3] = trans
    return delta_col.transpose(0, 1)


class PoseRefinementManager(torch.nn.Module):
    """
    Per-training-image SE(3) pose corrections.

    This is a lightweight bundle-adjustment-style refinement: each training
    camera has a small axis-angle rotation and translation correction optimized
    jointly with the Gaussians, regularized to stay near the input PF/RGB-D pose.
    """

    def __init__(self, train_cameras, pose_lr=1e-4):
        super().__init__()
        self.image_names = [cam.image_name for cam in train_cameras]
        self.name_to_idx = {name: i for i, name in enumerate(self.image_names)}
        n = len(self.image_names)
        self.rot_delta = torch.nn.Parameter(torch.zeros(n, 3, device="cuda"))
        self.trans_delta = torch.nn.Parameter(torch.zeros(n, 3, device="cuda"))
        self.pose_lr = pose_lr

    def has_camera(self, camera):
        return camera.image_name in self.name_to_idx

    def _idx(self, camera):
        return self.name_to_idx.get(camera.image_name, None)

    def apply(self, camera, enable_grad=True):
        idx = self._idx(camera)
        if idx is None:
            return camera

        rot = self.rot_delta[idx]
        trans = self.trans_delta[idx]
        if not enable_grad:
            rot = rot.detach()
            trans = trans.detach()

        delta_row = make_delta_row_transform(rot, trans)
        base_wv = camera.world_view_transform
        refined_wv = base_wv @ delta_row
        refined_full = (refined_wv.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)
        refined_center = torch.inverse(refined_wv)[3, :3]
        return RefinedCameraView(camera, refined_wv, refined_full, refined_center)

    def regularization(self, camera=None, rot_scale=0.01, trans_scale=0.05):
        """
        Dimensionless regularization. Defaults roughly mean:
          rot_scale=0.01 rad (~0.57 deg), trans_scale=5 cm.
        """
        if camera is not None:
            idx = self._idx(camera)
            if idx is None:
                return torch.zeros((), device="cuda")
            rot = self.rot_delta[idx]
            trans = self.trans_delta[idx]
        else:
            rot = self.rot_delta
            trans = self.trans_delta
        return ((rot / rot_scale) ** 2).mean() + ((trans / trans_scale) ** 2).mean()

    def make_optimizer(self, lr=None):
        return torch.optim.Adam([
            {"params": [self.rot_delta], "lr": lr or self.pose_lr},
            {"params": [self.trans_delta], "lr": lr or self.pose_lr},
        ])



    @staticmethod
    def _image_sort_key(camera):
        """Sort by numeric filename stem when possible, otherwise by name.

        GraphDECO's train camera stack may be shuffled, so a raw list slice is
        not a temporal/local window. For RGB-D sequences, image names like
        849.png, 850.png, ... give the correct temporal order.
        """
        name = str(getattr(camera, "image_name", ""))
        stem = os.path.splitext(os.path.basename(name))[0]
        try:
            return (0, int(stem), name)
        except Exception:
            return (1, stem, name)

    def temporal_sorted_cameras(self, train_cameras):
        return sorted(list(train_cameras), key=self._image_sort_key)

    def window_cameras(self, train_cameras, center_camera, window_size=5):
        """Return a temporal/numeric camera window around center_camera.

        Earlier versions used the current train_cameras order. In 3DGS this order
        can be shuffled/randomized, which produced windows like [834, 1032,
        1134, 1267, 1227] around center 1134. That is not a local BA window and
        makes the smoothness prior meaningless.
        """
        if self._idx(center_camera) is None:
            return [center_camera]

        ordered = self.temporal_sorted_cameras(train_cameras)
        center_name = getattr(center_camera, "image_name", None)
        center_order_idx = None
        for i, cam in enumerate(ordered):
            if getattr(cam, "image_name", None) == center_name:
                center_order_idx = i
                break
        if center_order_idx is None:
            return [center_camera]

        window_size = max(int(window_size), 1)
        radius = window_size // 2
        start = max(0, center_order_idx - radius)
        end = min(len(ordered), center_order_idx + radius + 1)
        # If near boundary, extend the other side to keep the requested size when possible.
        if end - start < window_size:
            missing = window_size - (end - start)
            start = max(0, start - missing)
            end = min(len(ordered), end + missing)
        return list(ordered[start:end])

    def regularization_for_cameras(self, cameras, rot_scale=0.01, trans_scale=0.05):
        vals = []
        for cam in cameras:
            vals.append(self.regularization(cam, rot_scale=rot_scale, trans_scale=trans_scale))
        if not vals:
            return torch.zeros((), device="cuda")
        return torch.stack(vals).mean()

    def smoothness_regularization_for_cameras(self, cameras, rot_scale=0.01, trans_scale=0.05):
        """Penalize changes in pose deltas between neighboring cameras in the provided order."""
        idxs = []
        seen = set()
        for cam in cameras:
            idx = self._idx(cam)
            if idx is not None and idx not in seen:
                idxs.append(idx)
                seen.add(idx)
        if len(idxs) < 2:
            return torch.zeros((), device="cuda")
        idx = torch.tensor(idxs, device="cuda", dtype=torch.long)
        rot = self.rot_delta[idx]
        trans = self.trans_delta[idx]
        drot = rot[1:] - rot[:-1]
        dtrans = trans[1:] - trans[:-1]
        return ((drot / rot_scale) ** 2).mean() + ((dtrans / trans_scale) ** 2).mean()

    def camera_indices(self, cameras):
        idxs = []
        seen = set()
        for cam in cameras:
            idx = self._idx(cam)
            if idx is not None and idx not in seen:
                idxs.append(idx)
                seen.add(idx)
        return idxs

    def print_window_summary(self, cameras, prefix="Pose BA window"):
        names = [getattr(c, "image_name", "UNKNOWN") for c in cameras]
        idxs = [self._idx(c) for c in cameras if self._idx(c) is not None]
        if not idxs:
            print(f"[{prefix}] no tracked cameras")
            return
        idx = torch.tensor(idxs, device="cuda", dtype=torch.long)
        with torch.no_grad():
            rot_norm = torch.linalg.norm(self.rot_delta[idx], dim=-1)
            trans_norm = torch.linalg.norm(self.trans_delta[idx], dim=-1)
            print(
                f"[{prefix}] n={len(idxs)} images={names} "
                f"rot mean/max rad={rot_norm.mean().item():.6f}/{rot_norm.max().item():.6f}, "
                f"rot mean/max deg={rot_norm.mean().item() * 180.0 / 3.141592653589793:.4f}/{rot_norm.max().item() * 180.0 / 3.141592653589793:.4f}, "
                f"trans mean/max m={trans_norm.mean().item():.6f}/{trans_norm.max().item():.6f}"
            )

    def print_grad_summary(self, prefix="Pose refinement grad"):
        def _stat(g):
            if g is None:
                return (0.0, 0.0)
            n = torch.linalg.norm(g.detach(), dim=-1)
            return (n.mean().item(), n.max().item())
        rot_mean, rot_max = _stat(self.rot_delta.grad)
        trans_mean, trans_max = _stat(self.trans_delta.grad)
        print(
            f"[{prefix}] "
            f"rot grad mean/max: {rot_mean:.6e}/{rot_max:.6e}, "
            f"trans grad mean/max: {trans_mean:.6e}/{trans_max:.6e}"
        )

    def save(self, model_path, iteration):
        out_dir = os.path.join(model_path, "pose_refinement")
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "image_names": self.image_names,
            "rot_delta": self.rot_delta.detach().cpu(),
            "trans_delta": self.trans_delta.detach().cpu(),
        }
        torch.save(payload, os.path.join(out_dir, f"iteration_{iteration}.pt"))
        torch.save(payload, os.path.join(out_dir, "latest.pt"))

    def load(self, model_path, iteration=-1):
        if iteration is None or iteration < 0:
            path = os.path.join(model_path, "pose_refinement", "latest.pt")
        else:
            path = os.path.join(model_path, "pose_refinement", f"iteration_{iteration}.pt")
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location="cuda")
        loaded_names = payload["image_names"]
        loaded_name_to_idx = {n: i for i, n in enumerate(loaded_names)}
        with torch.no_grad():
            for i, name in enumerate(self.image_names):
                if name in loaded_name_to_idx:
                    j = loaded_name_to_idx[name]
                    self.rot_delta[i].copy_(payload["rot_delta"][j].to("cuda"))
                    self.trans_delta[i].copy_(payload["trans_delta"][j].to("cuda"))
        return True


    def print_camera_summary(self, camera, prefix="Pose refinement camera"):
        idx = self._idx(camera)
        if idx is None:
            print(f"[{prefix}] camera not tracked: {getattr(camera, 'image_name', 'UNKNOWN')}")
            return
        with torch.no_grad():
            rot_norm = torch.linalg.norm(self.rot_delta[idx]).item()
            trans_norm = torch.linalg.norm(self.trans_delta[idx]).item()
            print(
                f"[{prefix}] image={camera.image_name} "
                f"rot rad={rot_norm:.6f}, rot deg={rot_norm * 180.0 / 3.141592653589793:.4f}, "
                f"trans m={trans_norm:.6f}"
            )
    def print_summary(self, prefix="Pose refinement"):
        with torch.no_grad():
            rot_norm = torch.linalg.norm(self.rot_delta, dim=-1)
            trans_norm = torch.linalg.norm(self.trans_delta, dim=-1)
            print(
                f"[{prefix}] "
                f"rot mean/max rad: {rot_norm.mean().item():.6f}/{rot_norm.max().item():.6f}, "
                f"trans mean/max m: {trans_norm.mean().item():.6f}/{trans_norm.max().item():.6f}"
            )
