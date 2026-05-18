import os
import math
import time
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class MiniCamera:
    """
    Minimal camera object needed by gaussian_renderer.render().
    """

    def __init__(self, R, T, FoVx, FoVy, width, height, device="cuda"):
        self.uid = 0
        self.colmap_id = 0
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.znear = 0.01
        self.zfar = 100.0

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T),
            dtype=torch.float32,
            device=device,
        ).transpose(0, 1)

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).to(device)

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )

        self.camera_center = self.world_view_transform.inverse()[3, :3]


def focal2fov(focal, pixels):
    return 2.0 * math.atan(pixels / (2.0 * focal))


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("Zero-norm quaternion")

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return R


def transform_stamped_to_matrix(tf_msg):
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def yaw_from_T_map_laser(T_map_laser):
    """
    Extract planar yaw from ROS-style map->laser transform.
    Assumes map x-y is ground plane and z is up.
    """
    R = T_map_laser[:3, :3]
    return math.atan2(R[1, 0], R[0, 0])


def map_laser_to_camera_extrinsics(
    T_map_laser,
    camera_forward_offset_m=0.05,
    camera_height_m=0.0,
    yaw_offset_rad=0.0,
):
    """
    Convert live map->laser TF pose into the camera extrinsic used by 3DGS.

    Convention used by our RGB-D builder:
      ROS/map x -> 3DGS/world X
      ROS/map y -> 3DGS/world Z
      3DGS/world Y is vertical

    Camera convention:
      x = right
      y = down
      z = forward
    """

    x_ros = float(T_map_laser[0, 3])
    y_ros = float(T_map_laser[1, 3])
    yaw_ros = yaw_from_T_map_laser(T_map_laser) + yaw_offset_rad

    forward_w = np.array([math.cos(yaw_ros), 0.0, math.sin(yaw_ros)], dtype=np.float64)
    left_w = np.array([-math.sin(yaw_ros), 0.0, math.cos(yaw_ros)], dtype=np.float64)
    up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    cam_right_w = -left_w
    cam_down_w = -up_w
    cam_forward_w = forward_w

    R_c2w = np.stack([cam_right_w, cam_down_w, cam_forward_w], axis=1)

    C_w = np.array([x_ros, camera_height_m, y_ros], dtype=np.float64)
    C_w = C_w + camera_forward_offset_m * forward_w

    R_w2c = R_c2w.T
    T_w2c = -R_w2c @ C_w

    R_for_3dgs = R_w2c.T.astype(np.float32)
    T_w2c = T_w2c.astype(np.float32)

    pose_info = {
        "x_ros": x_ros,
        "y_ros": y_ros,
        "yaw_rad": yaw_ros,
        "yaw_deg": math.degrees(yaw_ros),
        "camera_center_world": C_w,
    }

    return R_for_3dgs, T_w2c, pose_info


def load_intrinsics_from_cameras_txt(source_path):
    cameras_txt = Path(source_path) / "sparse" / "0" / "cameras.txt"

    if not cameras_txt.exists():
        raise FileNotFoundError(
            f"Could not find cameras.txt at {cameras_txt}. "
            f"Provide --width --height --fx --fy manually or use a valid COLMAP source path."
        )

    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue

            parts = line.split()
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx, fy = f, f
            else:
                raise ValueError(f"Unsupported camera model: {model}")

            return width, height, fx, fy

    raise RuntimeError(f"No camera entry found in {cameras_txt}")


def find_latest_ply(model_path):
    model_path = Path(model_path)
    pc_root = model_path / "point_cloud"

    if not pc_root.exists():
        raise FileNotFoundError(f"No point_cloud folder found under {model_path}")

    candidates = []
    for p in pc_root.glob("iteration_*"):
        try:
            it = int(p.name.replace("iteration_", ""))
        except ValueError:
            continue

        ply = p / "point_cloud.ply"
        if ply.exists():
            candidates.append((it, ply))

    if not candidates:
        raise FileNotFoundError(f"No point_cloud.ply found under {pc_root}")

    it, ply_path = sorted(candidates)[-1]
    print(f"[Model] using iteration {it}: {ply_path}")
    return ply_path


def save_tensor_image(tensor, out_path):
    img = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).astype(np.uint8)
    Image.fromarray(img).save(out_path)


class GSRosInferenceNode(Node):
    def __init__(self, args):
        super().__init__("gs_inference_from_tf")

        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.load_intrinsics()
        self.load_model()

        self.pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=False,
        )

        bg_color = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        self.frame_idx = 0

        self.get_logger().info("Inference node initialized.")
        self.get_logger().info(f"Listening for TF: {args.parent_frame} -> {args.child_frame}")

    def load_intrinsics(self):
        if (
            self.args.width is not None
            and self.args.height is not None
            and self.args.fx is not None
            and self.args.fy is not None
        ):
            self.width = self.args.width
            self.height = self.args.height
            self.fx = self.args.fx
            self.fy = self.args.fy
            self.get_logger().info("Using manual intrinsics.")
        else:
            self.width, self.height, self.fx, self.fy = load_intrinsics_from_cameras_txt(
                self.args.source_path
            )
            self.get_logger().info("Loaded intrinsics from cameras.txt.")

        self.FoVx = focal2fov(self.fx, self.width)
        self.FoVy = focal2fov(self.fy, self.height)

        self.get_logger().info(
            f"Intrinsics: width={self.width}, height={self.height}, "
            f"fx={self.fx:.3f}, fy={self.fy:.3f}"
        )

    def load_model(self):
        self.gaussians = GaussianModel(self.args.sh_degree)

        if self.args.iteration > 0:
            ply_path = (
                Path(self.args.model_path)
                / "point_cloud"
                / f"iteration_{self.args.iteration}"
                / "point_cloud.ply"
            )
            if not ply_path.exists():
                raise FileNotFoundError(f"PLY not found: {ply_path}")
        else:
            ply_path = find_latest_ply(self.args.model_path)

        self.gaussians.load_ply(str(ply_path))
        self.get_logger().info("Loaded Gaussian model.")

    def lookup_tf_pose(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.args.parent_frame,
                self.args.child_frame,
                Time(),
                timeout=Duration(seconds=self.args.tf_timeout_sec),
            )
            return transform_stamped_to_matrix(tf_msg), "ok"
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            return None, f"tf_fail:{type(e).__name__}"

    def render_once(self, out_path):
        T_map_laser, status = self.lookup_tf_pose()
        if T_map_laser is None:
            self.get_logger().warn(f"Could not render: {status}")
            return False

        R, T, pose_info = map_laser_to_camera_extrinsics(
            T_map_laser,
            camera_forward_offset_m=self.args.camera_forward_offset_m,
            camera_height_m=self.args.camera_height_m,
            yaw_offset_rad=math.radians(self.args.yaw_offset_deg),
        )

        camera = MiniCamera(
            R=R,
            T=T,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            width=self.width,
            height=self.height,
            device=self.device,
        )

        with torch.no_grad():
            render_pkg = render(camera, self.gaussians, self.pipe, self.background)
            image = render_pkg["render"]

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        save_tensor_image(image, out_path)

        self.get_logger().info(
            f"Rendered {out_path} | "
            f"x={pose_info['x_ros']:.3f}, y={pose_info['y_ros']:.3f}, "
            f"yaw={pose_info['yaw_deg']:.2f} deg"
        )

        return True

    def run_once(self):
        deadline = time.time() + self.args.wait_sec

        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            ok = self.render_once(self.args.out)
            if ok:
                return True

        self.get_logger().error("Failed to render before timeout.")
        return False

    def run_loop(self):
        if self.args.out_dir is None:
            raise ValueError("For continuous mode, provide --out_dir.")

        os.makedirs(self.args.out_dir, exist_ok=True)

        dt = 1.0 / self.args.rate_hz
        self.get_logger().info(f"Starting continuous rendering at {self.args.rate_hz:.2f} Hz")

        while rclpy.ok():
            start = time.time()
            rclpy.spin_once(self, timeout_sec=0.01)

            out_path = os.path.join(self.args.out_dir, f"render_{self.frame_idx:06d}.png")
            ok = self.render_once(out_path)
            if ok:
                self.frame_idx += 1

            elapsed = time.time() - start
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)

    parser.add_argument("--parent_frame", type=str, default="map")
    parser.add_argument("--child_frame", type=str, default="laser")
    parser.add_argument("--tf_timeout_sec", type=float, default=0.5)

    parser.add_argument("--out", type=str, default=None, help="Output image path for one-shot mode")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for continuous mode")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--rate_hz", type=float, default=2.0)
    parser.add_argument("--wait_sec", type=float, default=10.0)

    parser.add_argument("--camera_forward_offset_m", type=float, default=0.05)
    parser.add_argument("--camera_height_m", type=float, default=0.0)
    parser.add_argument("--yaw_offset_deg", type=float, default=0.0)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")

    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)

    args = parser.parse_args()

    if not args.continuous and args.out is None:
        raise ValueError("One-shot mode requires --out.")

    if args.continuous and args.out_dir is None:
        raise ValueError("Continuous mode requires --out_dir.")

    return args


def main():
    args = parse_args()

    rclpy.init()
    node = None

    try:
        node = GSRosInferenceNode(args)

        if args.continuous:
            node.run_loop()
        else:
            node.run_once()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()