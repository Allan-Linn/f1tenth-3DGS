import os
import re
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


def extract_index(filename):
    """
    Extract numeric index from filenames like:
    0.txt, 15.txt, frame_0015.txt, pose_15.txt
    """
    nums = re.findall(r"\d+", filename)
    if not nums:
        raise ValueError(f"No numeric index found in filename: {filename}")
    return int(nums[-1])


def load_poses(pose_dir):
    pose_entries = []

    for fname in os.listdir(pose_dir):
        if not fname.endswith(".txt"):
            continue

        idx = extract_index(fname)
        path = os.path.join(pose_dir, fname)
        T = np.loadtxt(path)

        if T.shape != (4, 4):
            print(f"[WARN] Skipping invalid pose shape {T.shape}: {fname}")
            continue

        pose_entries.append((idx, fname, T))

    pose_entries.sort(key=lambda x: x[0])

    indices = np.array([e[0] for e in pose_entries], dtype=int)
    files = [e[1] for e in pose_entries]
    poses = np.array([e[2] for e in pose_entries])

    return indices, files, poses


def find_missing_ranges(indices):
    missing = []

    for a, b in zip(indices[:-1], indices[1:]):
        if b > a + 1:
            missing.append((a + 1, b - 1, b - a - 1))

    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_dir", required=True, help="Folder containing pose txt files")
    parser.add_argument("--out_dir", default="pose_analysis", help="Output directory")

    # threshold for absolute jump distance
    parser.add_argument("--max_step_m", type=float, default=1.0,
                        help="Flag transition if total step distance exceeds this many meters")

    # threshold for motion normalized by missing frame gap
    parser.add_argument("--max_step_per_index_m", type=float, default=0.5,
                        help="Flag transition if step/index_gap exceeds this many meters")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    indices, files, poses = load_poses(args.pose_dir)

    if len(poses) < 2:
        raise RuntimeError("Need at least 2 poses to analyze trajectory.")

    positions = poses[:, :3, 3]

    # Consecutive recorded pose differences
    diffs = np.diff(positions, axis=0)
    steps = np.linalg.norm(diffs, axis=1)

    # Original filename index gaps
    index_gaps = np.diff(indices)
    normalized_steps = steps / np.maximum(index_gaps, 1)

    # Flag suspicious jumps:
    # A transition is suspicious if it is too large absolutely
    # AND still too large even after accounting for skipped pose indices.
    suspicious = (steps > args.max_step_m) & (normalized_steps > args.max_step_per_index_m)

    print("\n[Pose loading]")
    print(f"  pose_dir: {args.pose_dir}")
    print(f"  loaded poses: {len(poses)}")
    print(f"  first index: {indices[0]}")
    print(f"  last index : {indices[-1]}")

    missing_ranges = find_missing_ranges(indices)
    total_missing = sum(x[2] for x in missing_ranges)

    print("\n[Missing pose indices]")
    print(f"  missing count: {total_missing}")
    print(f"  missing ranges: {len(missing_ranges)}")

    if len(missing_ranges) > 0:
        print("  first few missing ranges:")
        for start, end, count in missing_ranges[:10]:
            print(f"    {start} to {end}  ({count} missing)")

    print("\n[Step stats: raw distance between consecutive recorded poses]")
    print(f"  mean step:   {steps.mean():.4f} m")
    print(f"  median step: {np.median(steps):.4f} m")
    print(f"  max step:    {steps.max():.4f} m")
    print(f"  p95 step:    {np.percentile(steps, 95):.4f} m")
    print(f"  p99 step:    {np.percentile(steps, 99):.4f} m")

    print("\n[Step stats: normalized by filename index gap]")
    print(f"  mean step/index:   {normalized_steps.mean():.4f} m")
    print(f"  median step/index: {np.median(normalized_steps):.4f} m")
    print(f"  max step/index:    {normalized_steps.max():.4f} m")
    print(f"  p95 step/index:    {np.percentile(normalized_steps, 95):.4f} m")
    print(f"  p99 step/index:    {np.percentile(normalized_steps, 99):.4f} m")

    print("\n[Flagged suspicious transitions]")
    print(f"  count: {int(suspicious.sum())}")

    suspicious_rows = []
    for k in np.where(suspicious)[0]:
        row = {
            "transition_id": int(k),
            "from_index": int(indices[k]),
            "to_index": int(indices[k + 1]),
            "from_file": files[k],
            "to_file": files[k + 1],
            "index_gap": int(index_gaps[k]),
            "step_m": float(steps[k]),
            "step_per_index_m": float(normalized_steps[k]),
            "from_x": float(positions[k, 0]),
            "from_y": float(positions[k, 1]),
            "from_z": float(positions[k, 2]),
            "to_x": float(positions[k + 1, 0]),
            "to_y": float(positions[k + 1, 1]),
            "to_z": float(positions[k + 1, 2]),
        }
        suspicious_rows.append(row)

    suspicious_rows_sorted = sorted(
        suspicious_rows,
        key=lambda r: r["step_per_index_m"],
        reverse=True
    )

    for r in suspicious_rows_sorted[:20]:
        print(
            f"  {r['from_file']} -> {r['to_file']} | "
            f"gap={r['index_gap']} | "
            f"step={r['step_m']:.3f} m | "
            f"step/gap={r['step_per_index_m']:.3f} m"
        )

    # Save CSV for all transitions
    transition_csv = os.path.join(args.out_dir, "pose_transition_analysis.csv")
    with open(transition_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transition_id",
            "from_index",
            "to_index",
            "from_file",
            "to_file",
            "index_gap",
            "step_m",
            "step_per_index_m",
            "suspicious",
            "from_x", "from_y", "from_z",
            "to_x", "to_y", "to_z",
        ])

        for k in range(len(steps)):
            writer.writerow([
                k,
                indices[k],
                indices[k + 1],
                files[k],
                files[k + 1],
                index_gaps[k],
                steps[k],
                normalized_steps[k],
                bool(suspicious[k]),
                positions[k, 0], positions[k, 1], positions[k, 2],
                positions[k + 1, 0], positions[k + 1, 1], positions[k + 1, 2],
            ])

    # Save missing ranges
    missing_csv = os.path.join(args.out_dir, "missing_pose_ranges.csv")
    with open(missing_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["missing_start", "missing_end", "missing_count"])
        for start, end, count in missing_ranges:
            writer.writerow([start, end, count])

    print("\n[Saved CSV]")
    print(f"  {transition_csv}")
    print(f"  {missing_csv}")

    # ---------------- Plot 1: trajectory with suspicious jumps highlighted ----------------
    plt.figure(figsize=(8, 6))

    for k in range(len(positions) - 1):
        p1 = positions[k]
        p2 = positions[k + 1]

        if suspicious[k]:
            color = "red"
            linewidth = 2.5
            alpha = 0.9
        else:
            color = "blue"
            linewidth = 1.0
            alpha = 0.55

        plt.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    plt.scatter(positions[0, 0], positions[0, 1], c="green", s=80, label="start")
    plt.scatter(positions[-1, 0], positions[-1, 1], c="black", s=80, label="end")

    plt.title("Trajectory X-Y with suspicious jumps highlighted")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(args.out_dir, "trajectory_xy_jumps.png")
    plt.savefig(path, dpi=150)
    plt.show()

    # ---------------- Plot 2: raw step distance ----------------
    plt.figure(figsize=(10, 4))
    plt.plot(indices[:-1], steps, label="raw step distance")
    plt.axhline(args.max_step_m, color="red", linestyle="--", label=f"max_step_m={args.max_step_m}")
    plt.title("Raw frame-to-frame distance between recorded poses")
    plt.xlabel("From pose index")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(args.out_dir, "raw_steps.png")
    plt.savefig(path, dpi=150)
    plt.show()

    # ---------------- Plot 3: normalized step distance ----------------
    plt.figure(figsize=(10, 4))
    plt.plot(indices[:-1], normalized_steps, label="step / index_gap")
    plt.axhline(
        args.max_step_per_index_m,
        color="red",
        linestyle="--",
        label=f"max_step_per_index_m={args.max_step_per_index_m}",
    )
    plt.title("Step distance normalized by skipped pose index gap")
    plt.xlabel("From pose index")
    plt.ylabel("Distance per index (m)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(args.out_dir, "normalized_steps.png")
    plt.savefig(path, dpi=150)
    plt.show()

    # ---------------- Plot 4: index gaps ----------------
    plt.figure(figsize=(10, 4))
    plt.plot(indices[:-1], index_gaps)
    plt.title("Pose filename index gaps")
    plt.xlabel("From pose index")
    plt.ylabel("Index gap")
    plt.tight_layout()

    path = os.path.join(args.out_dir, "index_gaps.png")
    plt.savefig(path, dpi=150)
    plt.show()

    print("\n[Saved plots]")
    print(f"  {os.path.join(args.out_dir, 'trajectory_xy_jumps.png')}")
    print(f"  {os.path.join(args.out_dir, 'raw_steps.png')}")
    print(f"  {os.path.join(args.out_dir, 'normalized_steps.png')}")
    print(f"  {os.path.join(args.out_dir, 'index_gaps.png')}")


if __name__ == "__main__":
    main()