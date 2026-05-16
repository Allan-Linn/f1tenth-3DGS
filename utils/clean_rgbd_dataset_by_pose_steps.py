import os
import re
import csv
import argparse
import shutil
import numpy as np


def extract_index(filename):
    nums = re.findall(r"\d+", filename)
    if not nums:
        raise ValueError(f"No numeric index found in {filename}")
    return int(nums[-1])


def load_pose(path):
    T = np.loadtxt(path)
    if T.shape != (4, 4):
        raise ValueError(f"Pose is not 4x4: {path}")
    return T


def list_indexed_files(folder, ext):
    out = {}
    for fname in os.listdir(folder):
        if fname.endswith(ext):
            idx = extract_index(fname)
            out[idx] = fname
    return out


def copy_if_exists(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True,
                        help="Original dataset folder containing color/, depth/, poses/, intrinsics.json")
    parser.add_argument("--out_dir", required=True,
                        help="Cleaned output dataset folder")

    parser.add_argument("--max_step_per_index", type=float, default=0.5,
                        help="Reject transition if distance/index_gap exceeds this many meters")

    parser.add_argument("--max_raw_step", type=float, default=4.0,
                        help="Reject transition if raw step exceeds this many meters, regardless of gap")

    parser.add_argument("--min_index_gap", type=int, default=1,
                        help="Usually leave as 1")

    args = parser.parse_args()

    color_dir = os.path.join(args.data_dir, "color")
    depth_dir = os.path.join(args.data_dir, "depth")
    pose_dir = os.path.join(args.data_dir, "poses")

    out_color_dir = os.path.join(args.out_dir, "color")
    out_depth_dir = os.path.join(args.out_dir, "depth")
    out_pose_dir = os.path.join(args.out_dir, "poses")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(out_color_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_pose_dir, exist_ok=True)

    color_files = list_indexed_files(color_dir, ".png")
    depth_files = list_indexed_files(depth_dir, ".png")
    pose_files = list_indexed_files(pose_dir, ".txt")

    # Only use indices that have RGB, depth, and pose.
    common_indices = sorted(set(color_files) & set(depth_files) & set(pose_files))

    if len(common_indices) < 2:
        raise RuntimeError("Not enough matched RGB-D-pose triplets found.")

    print(f"[Input]")
    print(f"  color files: {len(color_files)}")
    print(f"  depth files: {len(depth_files)}")
    print(f"  pose files : {len(pose_files)}")
    print(f"  matched triplets: {len(common_indices)}")
    print(f"  first index: {common_indices[0]}")
    print(f"  last index : {common_indices[-1]}")

    poses = {}
    positions = {}

    for idx in common_indices:
        T = load_pose(os.path.join(pose_dir, pose_files[idx]))
        poses[idx] = T
        positions[idx] = T[:3, 3]

    keep = [common_indices[0]]
    rejected = []

    prev_idx = common_indices[0]
    prev_pos = positions[prev_idx]

    for idx in common_indices[1:]:
        curr_pos = positions[idx]

        raw_step = float(np.linalg.norm(curr_pos - prev_pos))
        index_gap = max(idx - prev_idx, args.min_index_gap)
        step_per_index = raw_step / index_gap

        bad = False
        reasons = []

        if step_per_index > args.max_step_per_index:
            bad = True
            reasons.append(
                f"step_per_index {step_per_index:.4f} > {args.max_step_per_index:.4f}"
            )

        if raw_step > args.max_raw_step:
            bad = True
            reasons.append(
                f"raw_step {raw_step:.4f} > {args.max_raw_step:.4f}"
            )

        if bad:
            rejected.append({
                "index": idx,
                "prev_kept_index": prev_idx,
                "index_gap": index_gap,
                "raw_step_m": raw_step,
                "step_per_index_m": step_per_index,
                "reason": "; ".join(reasons),
            })
            # Important: do NOT update prev_idx / prev_pos.
            # We compare future candidates against last accepted pose.
            continue

        keep.append(idx)
        prev_idx = idx
        prev_pos = curr_pos

    print("\n[Cleaning]")
    print(f"  kept: {len(keep)}")
    print(f"  rejected: {len(rejected)}")

    # Copy intrinsics and metadata if present.
    for fname in ["intrinsics.json", "metadata.csv", "skipped_metadata.csv"]:
        src = os.path.join(args.data_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out_dir, fname))

    # Copy files.
    for idx in keep:
        copy_if_exists(
            os.path.join(color_dir, color_files[idx]),
            os.path.join(out_color_dir, color_files[idx]),
        )
        copy_if_exists(
            os.path.join(depth_dir, depth_files[idx]),
            os.path.join(out_depth_dir, depth_files[idx]),
        )
        copy_if_exists(
            os.path.join(pose_dir, pose_files[idx]),
            os.path.join(out_pose_dir, pose_files[idx]),
        )

    # Save kept/rejected reports.
    kept_csv = os.path.join(args.out_dir, "kept_indices.csv")
    with open(kept_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "color_file", "depth_file", "pose_file"])
        for idx in keep:
            writer.writerow([idx, color_files[idx], depth_files[idx], pose_files[idx]])

    rejected_csv = os.path.join(args.out_dir, "rejected_indices.csv")
    with open(rejected_csv, "w", newline="") as f:
        fieldnames = [
            "index",
            "prev_kept_index",
            "index_gap",
            "raw_step_m",
            "step_per_index_m",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rejected:
            writer.writerow(row)

    # Recompute stats on kept trajectory.
    kept_positions = np.array([positions[idx] for idx in keep])
    kept_steps = np.linalg.norm(np.diff(kept_positions, axis=0), axis=1)

    kept_gaps = np.diff(np.array(keep))
    kept_step_per_index = kept_steps / np.maximum(kept_gaps, 1)

    print("\n[Kept trajectory stats]")
    print(f"  mean raw step: {kept_steps.mean():.4f} m")
    print(f"  max raw step : {kept_steps.max():.4f} m")
    print(f"  mean step/index: {kept_step_per_index.mean():.4f} m")
    print(f"  max step/index : {kept_step_per_index.max():.4f} m")

    print("\n[Saved]")
    print(f"  cleaned dataset: {args.out_dir}")
    print(f"  kept report    : {kept_csv}")
    print(f"  rejected report: {rejected_csv}")

    total_input = len(common_indices)
    total_kept = len(keep)
    total_rejected = len(rejected)

    print("\n[Frame count summary]")
    print(f"  input matched RGB-D-pose frames: {total_input}")
    print(f"  frames left after cleaning:     {total_kept}")
    print(f"  frames removed:                 {total_rejected}")
    print(f"  kept percentage:                {100.0 * total_kept / total_input:.2f}%")


if __name__ == "__main__":
    main()