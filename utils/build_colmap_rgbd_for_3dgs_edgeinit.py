import os
import re
import json
import shutil
import argparse
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def extract_index(filename):
    nums = re.findall(r"\d+", filename)
    if not nums:
        raise ValueError(f"No numeric index found in filename: {filename}")
    return int(nums[-1])


def list_indexed_files(folder, ext):
    files = {}
    for fname in os.listdir(folder):
        if fname.endswith(ext):
            files[extract_index(fname)] = fname
    return files


def load_pose(path):
    T = np.loadtxt(path)
    if T.shape != (4, 4):
        raise ValueError(f"Pose is not 4x4: {path}")
    return T


def c2w_to_w2c(T_c2w):
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = t_w2c
    return T_w2c


def rotmat_to_colmap_quat(Rm):
    qx, qy, qz, qw = R.from_matrix(Rm).as_quat()
    return qw, qx, qy, qz


def make_T_laser_camopt(camera_forward_offset_m=0.05):
    R_laser_cam = np.array([
        [0.0,  0.0,  1.0],
        [-1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_laser_cam
    T[:3, 3] = [camera_forward_offset_m, 0.0, 0.0]
    return T


def normalize01(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def compute_edge_score(color_bgr, depth_raw, depth_scale, max_depth):
    """
    Returns an RGB/depth edge score in [0,1].
    Safety: invalid depth pixels get zero depth-edge score and are excluded later.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_edge = normalize01(cv2.magnitude(gx, gy))

    depth_m = depth_raw.astype(np.float32) * depth_scale
    valid = (depth_m > 0.1) & (depth_m < max_depth)

    depth_for_grad = depth_m.copy()
    depth_for_grad[~valid] = np.median(depth_m[valid]) if np.any(valid) else 0.0
    depth_for_grad = cv2.GaussianBlur(depth_for_grad, (3, 3), 0)
    dgx = cv2.Sobel(depth_for_grad, cv2.CV_32F, 1, 0, ksize=3)
    dgy = cv2.Sobel(depth_for_grad, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge = normalize01(cv2.magnitude(dgx, dgy))
    depth_edge[~valid] = 0.0

    score = 0.65 * rgb_edge + 0.35 * depth_edge
    score[~valid] = 0.0
    return score, valid


def sample_pixels_uniform(depth_raw, depth_scale, max_depth, pixel_stride):
    h, w = depth_raw.shape
    vs = np.arange(0, h, pixel_stride)
    us = np.arange(0, w, pixel_stride)
    u_grid, v_grid = np.meshgrid(us, vs)
    z = depth_raw[v_grid, u_grid].astype(np.float32) * depth_scale
    valid = (z > 0.1) & (z < max_depth)
    return u_grid[valid].astype(np.int32), v_grid[valid].astype(np.int32)


def sample_pixels_edge_aware(color_bgr, depth_raw, depth_scale, max_depth, pixel_stride, edge_fraction, rng):
    """
    Mixture sampling:
      - keep broad uniform coverage from the normal pixel-stride grid
      - add extra valid-depth samples biased toward RGB/depth edges
    """
    h, w = depth_raw.shape

    u_uni, v_uni = sample_pixels_uniform(depth_raw, depth_scale, max_depth, pixel_stride)
    n_uniform_keep = int(round((1.0 - edge_fraction) * len(u_uni)))
    if len(u_uni) > 0 and n_uniform_keep < len(u_uni):
        choice = rng.choice(len(u_uni), size=max(n_uniform_keep, 1), replace=False)
        u_uni, v_uni = u_uni[choice], v_uni[choice]

    edge_stride = max(1, pixel_stride // 2)
    score, valid_full = compute_edge_score(color_bgr, depth_raw, depth_scale, max_depth)

    vs = np.arange(0, h, edge_stride)
    us = np.arange(0, w, edge_stride)
    u_grid, v_grid = np.meshgrid(us, vs)
    valid = valid_full[v_grid, u_grid]
    cand_u = u_grid[valid].astype(np.int32)
    cand_v = v_grid[valid].astype(np.int32)

    if len(cand_u) == 0:
        return u_uni, v_uni

    cand_score = score[cand_v, cand_u].astype(np.float64) + 1e-6
    cand_score = cand_score / cand_score.sum()

    # Approximate final mixture ratio without adding too many knobs.
    n_total_target = max(len(u_uni), 1) / max(1.0 - edge_fraction, 1e-6)
    n_edge = int(round(edge_fraction * n_total_target))
    n_edge = min(n_edge, len(cand_u))

    if n_edge > 0:
        choice = rng.choice(len(cand_u), size=n_edge, replace=False, p=cand_score)
        u = np.concatenate([u_uni, cand_u[choice]])
        v = np.concatenate([v_uni, cand_v[choice]])
    else:
        u, v = u_uni, v_uni

    if len(u) > 0:
        uv = np.unique(np.stack([u, v], axis=1), axis=0)
        u, v = uv[:, 0].astype(np.int32), uv[:, 1].astype(np.int32)
    return u, v


def backproject_pixels(depth_raw, u, v, fx, fy, cx, cy, depth_scale):
    z = depth_raw[v, u].astype(np.float32) * depth_scale
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    pix = np.stack([u.astype(np.int32), v.astype(np.int32)], axis=1)
    return pts, pix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Cleaned RGB-D dataset folder with color/, depth/, poses/, intrinsics.json")
    parser.add_argument("--out_dir", required=True,
                        help="Output COLMAP-style scene folder")
    parser.add_argument("--image_mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--pixel_stride", type=int, default=4)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=8.0)
    parser.add_argument("--camera_forward_offset_m", type=float, default=0.05)
    parser.add_argument("--max_points", type=int, default=500000)

    # Minimal new controls.
    parser.add_argument("--edge_aware_init", action="store_true",
                        help="Use RGB/depth edge-aware sampling for initialization")
    parser.add_argument("--edge_fraction", type=float, default=0.40,
                        help="Approximate fraction of raw samples biased toward edges")

    args = parser.parse_args()
    args.edge_fraction = float(np.clip(args.edge_fraction, 0.0, 0.9))

    color_dir = os.path.join(args.data_dir, "color")
    depth_dir = os.path.join(args.data_dir, "depth")
    pose_dir = os.path.join(args.data_dir, "poses")
    intr_path = os.path.join(args.data_dir, "intrinsics.json")
    out_images_dir = os.path.join(args.out_dir, "images")
    out_sparse_dir = os.path.join(args.out_dir, "sparse", "0")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_sparse_dir, exist_ok=True)

    with open(intr_path, "r") as f:
        intr = json.load(f)

    fx = intr["color_intrinsics"]["fx"]
    fy = intr["color_intrinsics"]["fy"]
    cx = intr["color_intrinsics"]["ppx"]
    cy = intr["color_intrinsics"]["ppy"]
    width = intr["width"]
    height = intr["height"]
    depth_scale = intr["depth_scale"]

    color_files = list_indexed_files(color_dir, ".png")
    depth_files = list_indexed_files(depth_dir, ".png")
    pose_files = list_indexed_files(pose_dir, ".txt")
    indices = sorted(set(color_files) & set(depth_files) & set(pose_files))
    if len(indices) == 0:
        raise RuntimeError("No matching RGB-D-pose triplets found.")

    print("\n[Input]")
    print(f"  data_dir: {args.data_dir}")
    print(f"  matched frames: {len(indices)}")
    print(f"  first index: {indices[0]}")
    print(f"  last index : {indices[-1]}")

    print("\n[Intrinsics]")
    print("  camera model: PINHOLE")
    print(f"  size: {width} x {height}")
    print(f"  fx fy: {fx:.3f}, {fy:.3f}")
    print(f"  cx cy: {cx:.3f}, {cy:.3f}")
    print(f"  depth_scale: {depth_scale}")

    print("\n[Initialization]")
    print("  mode:", "edge-aware RGB-D" if args.edge_aware_init else "uniform RGB-D")
    if args.edge_aware_init:
        print(f"  edge_fraction: {args.edge_fraction:.2f}")

    T_laser_cam = make_T_laser_camopt(args.camera_forward_offset_m)

    print("\n[Images]")
    for idx in indices:
        src = os.path.join(color_dir, color_files[idx])
        dst = os.path.join(out_images_dir, color_files[idx])
        if os.path.exists(dst):
            continue
        if args.image_mode == "copy":
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)
    print(f"  images written to: {out_images_dir}")
    print(f"  mode: {args.image_mode}")

    cameras_txt = os.path.join(out_sparse_dir, "cameras.txt")
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    images_txt = os.path.join(out_sparse_dir, "images.txt")
    T_world_cam_by_idx = {}
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(indices)}\n")

        for image_id, idx in enumerate(indices, start=1):
            T_map_laser = load_pose(os.path.join(pose_dir, pose_files[idx]))
            T_map_cam = T_map_laser @ T_laser_cam
            T_world_cam_by_idx[idx] = T_map_cam
            T_w2c = c2w_to_w2c(T_map_cam)
            qw, qx, qy, qz = rotmat_to_colmap_quat(T_w2c[:3, :3])
            t_w2c = T_w2c[:3, 3]
            image_name = color_files[idx]
            f.write(
                f"{image_id} "
                f"{qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} "
                f"1 {image_name}\n"
            )
            f.write("\n")

    print("\n[Point cloud]")
    all_points = []
    all_colors = []
    rng = np.random.default_rng(0)
    total_sampled_pixels = 0

    for idx in tqdm(indices, desc="Backproject RGB-D"):
        color_path = os.path.join(color_dir, color_files[idx])
        depth_path = os.path.join(depth_dir, depth_files[idx])
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if color is None:
            print(f"[WARN] Could not read color: {color_path}")
            continue
        if depth is None:
            print(f"[WARN] Could not read depth: {depth_path}")
            continue

        if args.edge_aware_init:
            u, v = sample_pixels_edge_aware(
                color_bgr=color,
                depth_raw=depth,
                depth_scale=depth_scale,
                max_depth=args.max_depth,
                pixel_stride=args.pixel_stride,
                edge_fraction=args.edge_fraction,
                rng=rng,
            )
        else:
            u, v = sample_pixels_uniform(depth, depth_scale, args.max_depth, args.pixel_stride)

        if len(u) == 0:
            continue

        pts_cam, pix = backproject_pixels(depth, u, v, fx, fy, cx, cy, depth_scale)
        total_sampled_pixels += len(pts_cam)

        rgb = color[pix[:, 1], pix[:, 0], ::-1].astype(np.float64) / 255.0
        T_map_cam = T_world_cam_by_idx[idx]
        pts_world = (T_map_cam[:3, :3] @ pts_cam.T).T + T_map_cam[:3, 3]
        all_points.append(pts_world)
        all_colors.append(rgb)

    if not all_points:
        raise RuntimeError("No RGB-D points were generated. Check depth images/intrinsics/max_depth.")

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    print(f"  sampled RGB-D pixels: {total_sampled_pixels}")
    print(f"  raw RGB-D points: {len(all_points)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    pcd = pcd.voxel_down_sample(args.voxel_size)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    print(f"  after voxel downsample ({args.voxel_size} m): {len(pts)}")

    if len(pts) > args.max_points:
        chosen = rng.choice(len(pts), size=args.max_points, replace=False)
        pts = pts[chosen]
        cols = cols[chosen]
        print(f"  after random subsample: {len(pts)}")

    points3d_txt = os.path.join(out_sparse_dir, "points3D.txt")
    with open(points3d_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(pts)}\n")
        for point_id, (p, c) in enumerate(zip(pts, cols), start=1):
            r, g, b = np.clip(c * 255.0, 0, 255).astype(np.uint8)
            f.write(
                f"{point_id} "
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(r)} {int(g)} {int(b)} "
                f"1.0\n"
            )

    preview_ply = os.path.join(args.out_dir, "preview_rgbd_points.ply")
    preview_pcd = o3d.geometry.PointCloud()
    preview_pcd.points = o3d.utility.Vector3dVector(pts)
    preview_pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(preview_ply, preview_pcd)

    init_meta = {
        "edge_aware_init": args.edge_aware_init,
        "edge_fraction": args.edge_fraction,
        "pixel_stride": args.pixel_stride,
        "voxel_size": args.voxel_size,
        "max_depth": args.max_depth,
        "max_points": args.max_points,
        "sampled_rgbd_pixels": int(total_sampled_pixels),
        "final_points": int(len(pts)),
    }
    with open(os.path.join(args.out_dir, "init_metadata.json"), "w") as f:
        json.dump(init_meta, f, indent=2)

    print("\n[Output]")
    print(f"  COLMAP scene root: {args.out_dir}")
    print(f"  images:            {out_images_dir}")
    print(f"  sparse model:      {out_sparse_dir}")
    print(f"  cameras.txt:       {cameras_txt}")
    print(f"  images.txt:        {images_txt}")
    print(f"  points3D.txt:      {points3d_txt}")
    print(f"  preview ply:       {preview_ply}")
    print(f"  init metadata:     {os.path.join(args.out_dir, 'init_metadata.json')}")
    print("\nDone.")
    print("Use this path as your 3DGS scene/source path:")
    print(f"  {args.out_dir}")


if __name__ == "__main__":
    main()
