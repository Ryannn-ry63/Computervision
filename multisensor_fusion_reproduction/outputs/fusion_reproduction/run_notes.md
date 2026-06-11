# Fusion Reproduction Notes

## Result

- Outputs include `camera_raw.png`, `lidar_projection_depth.png`, `bev_pointcloud.png`, `fusion_ppt_panel.png`, `bevfusion_result.png`, and `bevfusion_kitti_000008_style.png`.
- Data source: MMDetection3D built-in KITTI demo sample `000008`.
- Method: projected Velodyne LiDAR points into the CAM2 image using the sample calibration matrix; colors encode depth.

## Command

```bash
python scripts/kitti_lidar_projection.py \
  --image mmdetection3d/demo/data/kitti/000008.png \
  --pointcloud mmdetection3d/demo/data/kitti/000008.bin \
  --calib mmdetection3d/demo/data/kitti/000008.pkl \
  --out-dir outputs/fusion_reproduction
```

## Environment Observation

- `motiondetection` uses PyTorch `2.0.1+cu118`, CUDA available on an NVIDIA GeForce RTX 4090.
- Upgraded the OpenMMLab stack for the cloned MMDetection3D tree: `mmcv 2.1.0`, `mmengine 0.10.7`, `mmdet 3.2.0`, editable `mmdet3d 1.4.0`.
- Compiled BEVFusion CUDA ops with the conda environment's CUDA 11.8 `nvcc`.

## BEVFusion Demo

- Data source: MMDetection3D built-in nuScenes demo sample.
- Model: official BEVFusion LiDAR-camera checkpoint.
- Converted 21 `pts_middle_encoder` sparse convolution weights from checkpoint layout `[out, kD, kH, kW, in]` to the current sparse conv layout `[kD, kH, kW, in, out]`.
- Output: `bevfusion_result.png`, a multi-camera 3D detection visualization with projected 3D boxes.
- KITTI `000008` is also presented as a same-frame comparison figure for PPT layout consistency. This figure uses the KITTI camera image, LiDAR projection, BEV point cloud, and the available single-frame 3D predictions; it is a visualization aid rather than an official BEVFusion benchmark result.

```bash
python ../scripts/convert_bevfusion_sparse_weights.py \
  --input checkpoints/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
  --output checkpoints/bevfusion/bevfusion_lidar-cam_converted_sparse.pth

CUDA_HOME=/root/miniconda3/envs/motiondetection \
PATH=/root/miniconda3/envs/motiondetection/bin:$PATH \
/root/miniconda3/envs/motiondetection/bin/python \
  projects/BEVFusion/demo/multi_modality_demo.py \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
  demo/data/nuscenes/ \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  checkpoints/bevfusion/bevfusion_lidar-cam_converted_sparse.pth \
  --cam-type all \
  --score-thr 0.2 \
  --out-dir outputs/bevfusion_demo/bevfusion_result.png
```

## KITTI 000008 Same-frame Figure

```bash
python scripts/bevfusion_kitti_000008_style.py \
  --out-dir outputs/fusion_reproduction
```
