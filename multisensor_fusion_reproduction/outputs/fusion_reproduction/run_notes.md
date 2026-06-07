# Fusion Reproduction Notes

## Result

- Generated `camera_raw.png`, `lidar_projection_depth.png`, `bev_pointcloud.png`, `fusion_ppt_panel.png`, `bevfusion_result.png`, and `bevfusion_result_sample10.png`.
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
- Additional output: `bevfusion_result_sample10.png`, generated from a second nuScenes test sample in `/inspire/hdd/global_public/public_datas/nuScenes/raw/test`. Public data was read only; no files were modified there.

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

## Second BEVFusion Sample

```bash
python scripts/build_nuscenes_demo_ann.py \
  --dataroot /inspire/hdd/global_public/public_datas/nuScenes/raw/test \
  --version v1.0-test \
  --sample-index 10 \
  --out outputs/fusion_reproduction/nuscenes_test_sample_10.pkl

CUDA_HOME=/root/miniconda3/envs/motiondetection \
PATH=/root/miniconda3/envs/motiondetection/bin:$PATH \
/root/miniconda3/envs/motiondetection/bin/python \
  projects/BEVFusion/demo/multi_modality_demo.py \
  /inspire/hdd/global_public/public_datas/nuScenes/raw/test/samples/LIDAR_TOP/n008-2018-08-01-16-03-27-0400__LIDAR_TOP__1533153863047401.pcd.bin \
  /inspire/hdd/global_public/public_datas/nuScenes/raw/test \
  ../outputs/fusion_reproduction/nuscenes_test_sample_10.pkl \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  checkpoints/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
  --cam-type all \
  --score-thr 0.2 \
  --out-dir outputs/bevfusion_demo/bevfusion_result_sample10.png
```
