# f1tenth-3DGS

3DGS Visual Simulator for F1Tenth Vehicles from captured RGBD Sequence and 2D LiDAR poses.

Demo video/GIF: 

![](assets/moore_demo1.gif)

![](assets/levine_demo.gif)

Rendered trajectory from F1Tenth tracks.

## Overview

3D reconstruction for F1Tenth autonomous racing environments. Standard 3DGS relies on successful Structure-from-Motion (SfM) initialization, but F1Tenth sequences are difficult for SfM because the vehicle motion is planar, the baseline is limited, and many views contain blur or weak texture.

Instead, a practical RGB-D reconstruction pipeline that synthesizes RGB, depth, camera intrinsics, and vehicle poses into a 3D Gaussian Splatting training setup is proposed. Visual quality is further improved through camera-gradient bundle-adjusted pose refinement.

The recovered Gaussian scene can then be used as a vision-based simulator by rendering onboard RGB observations from vehicle poses, moving the F1Tenth stack toward camera-based simulation and policy training.

## Main Contributions

1. **RGBD to 3DGS pipeline**  
   Converted F1Tenth RGBD captures into a dataset viable for 3D Gaussian Splatting.

2. **Successful Reconstruction of Indoor Scenes**  
    Trained 3D models that can be deployed on the F1tenth vehicle, enabling novel view inference in 0.1s.

3. **Edge-aware RGBD Gaussian initialization**  
   Sampled RGBD points with emphasis on image edges and high-detail regions.

4. **Selective Gaussian Insertion**  
   Evaluated residual/detail-aware Gaussian insertion instead of naively increasing scene size.

5. **Temporal-window Camera-gradient Bundle Adjustment**  
   Implemented a constrained local-window pose refinement strategy that optimized neighboring camera poses together rather than optimizing each frame independently.


## Setup

```bash
conda env create -f environment.yml

conda activate gaussian_splatting_camgrad
```

## Data Collection

This pipeline builds on top of the [SLAM](https://f1tenth-coursekit.readthedocs.io/en/latest/lectures/ModuleC/tutorial5.html) & [Particle Filter](https://f1tenth-coursekit.readthedocs.io/en/latest/lectures/ModuleC/lecture08.html) modules established in the F1tenth stack. 

Follow the installation guide for F1tenth/RoboRacer [here](https://f1tenth-coursekit.readthedocs.io/en/latest/lectures/ModuleC/tutorial5.html): 

Once complete, perform SLAM and save the map of the environment on the f1Tenth vehicle by running simultaneously:

```bash
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
```

In terminal 1, run:

```bash
 ros2 launch f1tenth_stack sick_bringup_launch.py
```

In terminal 2, run:

```bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=path/to/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml
```

In terminal 3, run Foxglove to visualize the map:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

![](assets/moore_map.png)

After the map has been saved, load it for particle filter and run the following:

In terminal 1, run:

```bash
 ros2 launch f1tenth_stack sick_bringup_launch.py
```

In terminal 2, run:

```bash
ros2 launch particle_filter localize_launch.py
```

In terminal 3, run:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

Publish the vehicle's initial 2D position on the map with the Foxglove interface.

In terminal 4, run:

```bash
python record_rgbd_pose_filtered.py \
  --output_dir path_to_output_folder \
  --max_step_m 5.0
```

Then drive the car around the environment. Use a lower speed to reduce blurry frames.

The captured scene has the format:

```text
scene_name/
  rgb/
    0.png
    1.png
    ...
  depth/
    0.png
    1.png
    ...
  poses/
    0.txt
    1.txt
    ...
  intrinsics.json
```

## Data Preprocessing and Visualization

2D LiDAR poses can be noisy and impact reconstruction quality. Use the following to clean and visualize the data:

```bash
cd utils

python analyze_pose_trajectory_with_gaps.py   --pose_dir path_to_data_folder/poses   --out_dir output_folder
```

To eliminate large jumps, run:

```bash
python clean_rgbd_dataset_by_pose_steps.py \
  --data_dir path_to_data_folder \
  --out_dir output_folder_clean \
  --max_step_m 5.0
```

Optionally, refine poses by aligning RGBD sequences:

```bash
python refine_rgbd_poses.py \
  --data_dir data_clean \
  --out_dir data_rgbd_refined \
  --camera_forward_offset_m 0.05 \
  --max_depth 5.0 \
  --blend_alpha 0.35 \
  --max_rel_trans_correction 0.08 \
  --max_rel_rot_correction_deg 3.0 \
  --depth_diff_max 0.07 \
  --grayscale_odometry
```

It uses RGB-D odometry between neighboring frames to estimate a better relative motion from frame i to frame i+1.

## RGBD to Point Cloud Converter

The following scripts converts the cleaned data into a pointcloud for 3DGS. It back-projects pixels values to form a dense point cloud, then reduces the scene to max_points.

```bash
python build_colmap_rgbd_for_3dgs_edgeinit.py \
  --data_dir /path/to/data_rgbd_refined \
  --out_dir /path/to/data_colmap \
  --image_mode symlink \
  --pixel_stride 4 \
  --voxel_size 0.05 \
  --max_depth 5.0 \
  --camera_forward_offset_m 0.05 \
  --max_points 300000 \
  --edge_aware_init
```
This creates:

```text
data_colmap/
  images/
  sparse/0/
    cameras.txt
    images.txt
    points3D.txt
  preview_rgbd_points.ply
  init_metadata.json
```

Convert text files to binary

```bash
colmap model_converter \
  --input_path /path/to/data_colmap/sparse/0 \
  --output_path /path/to/data_colmap/sparse/0 \
  --output_type BIN
```

## Train 3DGS Model

```bash
conda activate env_name

cd /local-scratch2/allan/F1_splat/f1tenth_3DGS_camgrad

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s /path/to/data_colmap \
  -m /path/to/output_folder \
  --optimize_camera_poses \
  --pose_refine_from_iter 5000 \
  --pose_refine_until_iter 15000 \
  --pose_opt_mode window_ba \
  --pose_ba_interval 500 \
  --pose_ba_steps 100 \
  --pose_ba_window 5 \
  --pose_lr 2e-5 \
  --pose_reg_weight 0.05 \
  --pose_smooth_weight 0.05 \
  --pose_ba_inner_log_every 50 \
  --pose_log_interval 500
```

## Inference on the F1Tenth Vehicle

While planning with particle filter, run:

```bash
python inference.py \
  --model_path /path/to/model \
  --source_path /path/to/data/scene \
  --parent_frame map \
  --child_frame laser \
  --continuous \
  --rate_hz 2 \
  --out_dir /outputs/live_tf_renders
```










