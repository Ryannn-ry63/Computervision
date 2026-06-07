# 多传感器融合可视化复现工作总结

## 目标

本次工作的目标是为课程汇报中“多传感器融合”部分准备可放入 PPT 的真实可视化结果，重点支撑以下内容：

- 相机与 LiDAR 的空间标定和坐标对齐；
- LiDAR 点云向相机图像的投影；
- BEV 鸟瞰空间表示；
- BEVFusion 将相机和 LiDAR 融合后进行 3D 目标检测的效果展示。

## 完成内容

### 1. LiDAR 与相机空间对齐可视化

基于 MMDetection3D 自带的 KITTI demo 样例 `000008`，完成了 LiDAR 点云到相机图像的投影。实现中读取相机图像、LiDAR 点云和标定信息，将三维点云投影到二维图像平面，并按深度进行颜色编码。

输出文件：

- `outputs/fusion_reproduction/camera_raw.png`
- `outputs/fusion_reproduction/lidar_projection_depth.png`
- `outputs/fusion_reproduction/bev_pointcloud.png`
- `outputs/fusion_reproduction/fusion_ppt_panel.png`

其中 `fusion_ppt_panel.png` 是适合直接放入 PPT 的三联图，包含原始相机图、LiDAR 投影图和 BEV 鸟瞰点云图。

### 2. BEVFusion 多模态 3D 检测可视化

在已 clone 的 MMDetection3D 仓库中配置并运行了 BEVFusion demo。使用 nuScenes demo 样例作为输入，加载官方 BEVFusion LiDAR-camera 预训练权重，输出多视角相机图像上的 3D 检测框可视化结果。

输出文件：

- `outputs/fusion_reproduction/bevfusion_result.png`

这张图展示了 BEVFusion 将多视角相机和 LiDAR 点云融合后，对车辆、行人等目标进行 3D 检测，并将三维检测框投影回相机视角的结果。

## 环境与工程处理

为了跑通 BEVFusion，对 `motiondetection` 环境做了以下处理：

- 安装并对齐新版 OpenMMLab 依赖：
  - `mmcv 2.1.0`
  - `mmengine 0.10.7`
  - `mmdet 3.2.0`
  - editable 安装当前 MMDetection3D 仓库，版本为 `mmdet3d 1.4.0`
- 使用环境自带 CUDA 11.8 `nvcc` 编译 BEVFusion CUDA ops；
- 下载官方 BEVFusion LiDAR-camera checkpoint；
- 处理 sparse convolution 权重布局不匹配问题，生成可用于当前环境的 checkpoint；
- 编写辅助脚本记录和复现关键步骤。

新增或使用的主要脚本：

- `scripts/kitti_lidar_projection.py`：生成 LiDAR 投影图和 BEV 点云图；
- `scripts/convert_bevfusion_sparse_weights.py`：转换 BEVFusion checkpoint 中 sparse convolution 权重布局。

## PPT 使用建议

建议在 PPT 中分两层展示：

1. **融合前提：空间标定与坐标对齐**

   使用：

   - `outputs/fusion_reproduction/fusion_ppt_panel.png`

   讲解重点：

   > 不同传感器的数据不能直接融合，必须先通过标定参数完成坐标系转换。LiDAR 点云投影到相机图像后，可以直观看到点云与道路、车辆等图像内容对齐。

2. **BEVFusion：统一到 BEV 空间的多模态融合检测**

   使用：

   - `outputs/fusion_reproduction/bevfusion_result.png`

   讲解重点：

   > BEVFusion 将相机和 LiDAR 特征统一到 BEV 鸟瞰空间，在该空间中完成 3D 目标检测。图中的彩色 3D 框表示模型检测出的交通参与物，体现了多传感器融合从“对齐”进一步发展到“联合感知”的过程。

## 最终交付

最终可用于汇报的核心图片为：

- `outputs/fusion_reproduction/fusion_ppt_panel.png`
- `outputs/fusion_reproduction/bevfusion_result.png`

详细命令和运行记录见：

- `outputs/fusion_reproduction/run_notes.md`
