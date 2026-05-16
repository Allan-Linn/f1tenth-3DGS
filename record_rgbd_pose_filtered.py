import os
import csv
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


WIDTH = 640
HEIGHT = 480
FPS = 15
SAVE_EVERY = 1
MAX_FRAMES = None
ALIGN_DEPTH_TO_COLOR = True
RESET_DEVICE_ON_START = True

# For your particle-filter setup, map -> laser is the stable localization pose.
TARGET_PARENT_FRAME = "map"
TARGET_CHILD_FRAME = "laser"

# Slightly longer than 0.2 sec to avoid occasional TF timing misses.
TF_LOOKUP_TIMEOUT_SEC = 0.5

# If the accepted pose suddenly moves more than this relative to the previous
# accepted pose, skip the frame. This removes PF relocalization outliers.
DEFAULT_MAX_STEP_M = 1.0


def get_intrinsics_dict(profile):
    intr = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("Zero-norm quaternion")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    return R


def transform_to_matrix(transform_stamped):
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def reset_realsense():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found")

    dev = devices[0]
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    print(f"Resetting RealSense: {name} ({serial})")
    dev.hardware_reset()
    time.sleep(4.0)


def wait_for_first_valid_frames(pipeline, timeout_ms=3000, max_tries=30):
    for i in range(max_tries):
        try:
            frames = pipeline.wait_for_frames(timeout_ms)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame and depth_frame:
                print(f"Received first valid RGB-D frames on try {i + 1}")
                return
        except RuntimeError:
            pass
        time.sleep(0.1)
    raise RuntimeError("Could not get valid RGB-D frames after startup")


class RGBDPoseRecorder(Node):
    def __init__(self, args):
        super().__init__("rgbd_pose_recorder_filtered")

        self.args = args

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pipeline = None
        self.align = None

        self.frame_count = 0          # raw camera frame count
        self.saved_count = 0          # accepted/saved frame count
        self.skipped_count = 0
        self.running = True

        self.prev_good_pos = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_dir is None:
            self.root_dir = os.path.join("rgbd_pose_capture", f"capture_{timestamp}")
        else:
            self.root_dir = args.output_dir

        self.color_dir = os.path.join(self.root_dir, "color")
        self.depth_dir = os.path.join(self.root_dir, "depth")
        self.pose_dir = os.path.join(self.root_dir, "poses")

        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.pose_dir, exist_ok=True)

        self.intrinsics_path = os.path.join(self.root_dir, "intrinsics.json")
        self.metadata_path = os.path.join(self.root_dir, "metadata.csv")
        self.skipped_metadata_path = os.path.join(self.root_dir, "skipped_metadata.csv")

        self.csvfile = open(self.metadata_path, "w", newline="")
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow([
            "saved_index",
            "raw_frame_count",
            "camera_timestamp_ms",
            "ros_wall_time_sec",
            "color_filename",
            "depth_filename",
            "pose_filename",
            "tf_status",
            "step_from_prev_good_m",
        ])

        self.skipfile = open(self.skipped_metadata_path, "w", newline="")
        self.skip_writer = csv.writer(self.skipfile)
        self.skip_writer.writerow([
            "skipped_count",
            "raw_frame_count",
            "camera_timestamp_ms",
            "ros_wall_time_sec",
            "reason",
            "step_from_prev_good_m",
        ])

        self.start_camera()

    def start_camera(self):
        if RESET_DEVICE_ON_START:
            reset_realsense()

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

        self.get_logger().info("Starting RealSense pipeline...")
        profile = self.pipeline.start(config)

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        self.get_logger().info(f"Depth scale: {depth_scale} meters per unit")

        self.align = rs.align(rs.stream.color) if ALIGN_DEPTH_TO_COLOR else None

        self.get_logger().info("Waiting for first valid frames...")
        wait_for_first_valid_frames(self.pipeline)

        color_profile = profile.get_stream(rs.stream.color)
        depth_profile = profile.get_stream(rs.stream.depth)

        intrinsics_data = {
            "color_intrinsics": get_intrinsics_dict(color_profile),
            "depth_intrinsics": get_intrinsics_dict(depth_profile),
            "depth_scale": depth_scale,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "save_every": SAVE_EVERY,
            "depth_aligned_to_color": ALIGN_DEPTH_TO_COLOR,
            "target_parent_frame": TARGET_PARENT_FRAME,
            "target_child_frame": TARGET_CHILD_FRAME,
            "tf_lookup_timeout_sec": TF_LOOKUP_TIMEOUT_SEC,
            "max_step_m": self.args.max_step_m,
            "file_naming": "zero_padded_raw_frame_count",
        }

        with open(self.intrinsics_path, "w") as f:
            json.dump(intrinsics_data, f, indent=2)

        self.get_logger().info(f"Saving to: {self.root_dir}")

    def lookup_pose(self):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                TARGET_PARENT_FRAME,
                TARGET_CHILD_FRAME,
                Time(),
                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC),
            )
            T = transform_to_matrix(tf_msg)
            return T, "ok"
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            return None, f"tf_fail:{type(e).__name__}"

    def log_skip(self, cam_ts_ms, ros_wall_time_sec, reason, step=None):
        self.skip_writer.writerow([
            self.skipped_count,
            self.frame_count,
            f"{cam_ts_ms:.3f}" if cam_ts_ms is not None else "",
            f"{ros_wall_time_sec:.9f}",
            reason,
            f"{step:.6f}" if step is not None else "",
        ])
        self.skipfile.flush()
        self.skipped_count += 1

    def save_one(self):
        try:
            frames = self.pipeline.wait_for_frames(3000)
        except RuntimeError:
            self.get_logger().warn("Frame timeout, retrying...")
            return

        if ALIGN_DEPTH_TO_COLOR:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            self.get_logger().warn("Missing color or depth frame")
            return

        if self.frame_count % SAVE_EVERY != 0:
            self.frame_count += 1
            return

        cam_ts_ms = frames.get_timestamp()
        ros_wall_time_sec = self.get_clock().now().nanoseconds * 1e-9

        # Look up pose BEFORE saving RGB/depth.
        T_map_laser, tf_status = self.lookup_pose()
        if T_map_laser is None:
            self.get_logger().warn(f"Skipping raw frame {self.frame_count}: {tf_status}")
            self.log_skip(cam_ts_ms, ros_wall_time_sec, tf_status)
            self.frame_count += 1
            return

        # Reject physically impossible pose jumps.
        current_pos = T_map_laser[:3, 3]
        if self.prev_good_pos is not None:
            step = float(np.linalg.norm(current_pos - self.prev_good_pos))
            if step > self.args.max_step_m:
                reason = f"pose_jump:{step:.3f}>{self.args.max_step_m:.3f}"
                self.get_logger().warn(f"Skipping raw frame {self.frame_count}: {reason}")
                self.log_skip(cam_ts_ms, ros_wall_time_sec, reason, step)
                self.frame_count += 1
                return
        else:
            step = 0.0

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Use raw frame count and zero padding.
        # This preserves skipped-frame gaps and fixes lexicographic sorting later.
        color_name = f"{self.frame_count:06d}.png"
        depth_name = f"{self.frame_count:06d}.png"
        pose_name = f"{self.frame_count:06d}.txt"

        color_path = os.path.join(self.color_dir, color_name)
        depth_path = os.path.join(self.depth_dir, depth_name)
        pose_path = os.path.join(self.pose_dir, pose_name)

        ok_color = cv2.imwrite(color_path, color_image)
        ok_depth = cv2.imwrite(depth_path, depth_image)

        if not (ok_color and ok_depth):
            self.get_logger().warn("Failed to save RGB-D pair")
            self.log_skip(cam_ts_ms, ros_wall_time_sec, "image_write_fail", step)
            self.frame_count += 1
            return

        np.savetxt(pose_path, T_map_laser, fmt="%.8f")
        self.prev_good_pos = current_pos.copy()

        self.writer.writerow([
            self.saved_count,
            self.frame_count,
            f"{cam_ts_ms:.3f}",
            f"{ros_wall_time_sec:.9f}",
            color_name,
            depth_name,
            pose_name,
            tf_status,
            f"{step:.6f}",
        ])
        self.csvfile.flush()

        self.get_logger().info(
            f"Saved {self.saved_count} raw={self.frame_count} step={step:.3f} m"
        )

        self.saved_count += 1
        self.frame_count += 1

        if MAX_FRAMES is not None and self.saved_count >= MAX_FRAMES:
            self.get_logger().info("Reached MAX_FRAMES, stopping.")
            self.running = False

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                self.get_logger().info("Pipeline stopped.")
            except Exception as e:
                self.get_logger().warn(f"Pipeline stop warning: {e}")

        try:
            self.csvfile.close()
        except Exception:
            pass

        try:
            self.skipfile.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.get_logger().info(
            f"Done. Saved {self.saved_count} RGB-D frames to {self.root_dir}. "
            f"Skipped {self.skipped_count} frames."
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. If omitted, saves to rgbd_pose_capture/capture_TIMESTAMP",
    )
    parser.add_argument(
        "--max_step_m",
        type=float,
        default=DEFAULT_MAX_STEP_M,
        help="Reject a frame if map->laser translation jumps more than this from the last accepted pose.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = None

    try:
        node = RGBDPoseRecorder(args)

        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.01)
            node.save_one()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping cleanly...")

    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
