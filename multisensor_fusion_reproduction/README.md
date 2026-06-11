# 多传感器融合复现附件说明

## 复现定位

本附件用于支撑报告中“复现实验与可视化验证”一节。复现目标是验证多传感器融合的核心工程流程，并生成可解释的可视化结果；不声称完整复现 PointPainting、BEVFusion 或 CLOCs 的论文 benchmark 指标。

本次覆盖三类融合层次：

- PointPainting：数据层/点级融合，将图像语义分数追加到 LiDAR 点云特征。
- BEVFusion：特征层融合，将相机与 LiDAR 特征统一到 BEV 鸟瞰空间。
- CLOCs：决策层/候选级融合，对 2D 与 3D detection candidates 做几何关联和重打分。

## 推荐查看的核心图片

- `outputs/fusion_reproduction/fusion_ppt_panel.png`：相机原图、LiDAR 投影、BEV 点云三联图。
- `outputs/fusion_reproduction/bevfusion_result.png`：BEVFusion 官方 nuScenes demo 多模态 3D 检测可视化。
- `outputs/fusion_reproduction/bevfusion_kitti_000008_style.png`：使用 KITTI `000008` 输入整理的同帧对比图，便于和 PointPainting、CLOCs 放在同一页展示。
- `outputs/pointpainting_mmseg_legacy/official_pointpainting_panel.png`：PointPainting 仓库内置 MMSegmentation DeepLabV3+ painting 流程可视化。
- `outputs/pointpainting_detector_300iter/pointpainting_detector_panel.png`：painted PointPillar 单帧 detector 链路验证。
- `outputs/clocs_official_000008/official_clocs_panel.png`：CLOCs 官方 checkpoint 单帧推理可视化。
- `outputs/clocs_reproduction/clocs_panel.png`：CLOCs 候选级 IoU 关联与融合重打分机制图。

## 目录内容

```text
multisensor_fusion_reproduction/
  README.md
  scripts/
    build_nuscenes_demo_ann.py
    bevfusion_kitti_000008_style.py
    clocs_reproduction_demo.py
    convert_bevfusion_sparse_weights.py
    kitti_lidar_projection.py
    pointpainting_demo.py
    run_pointpainting_detector.py
    visualize_clocs_official.py
    visualize_official_pointpainting.py
  summaries/
    BEVFusion工作总结.md
    CLOCs工作总结.md
    PointPainting工作总结.md
  patches/
    pointpainting_mmseg_legacy_compat.md
    clocs_legacy_compat.md
  outputs/
    fusion_reproduction/
    pointpainting_mmseg_legacy/
    pointpainting_official/
    pointpainting_detector_300iter/
    clocs_official_000008/
    clocs_reproduction/
```

## 复现限制

- BEVFusion 使用 nuScenes demo 样例和官方预训练权重进行可视化推理，未重新训练完整模型。
- PointPainting 在独立 `pointpainting_legacy` 环境中跑通了仓库内置 MMSegmentation DeepLabV3+ painting 分支；detector 部分使用 KITTI 单帧验证权重检查链路，不是官方完整评测结果。
- CLOCs 在独立 `clocs_legacy` 环境中跑通了 KITTI 单帧官方 checkpoint 流程；由于缺完整 KITTI benchmark 数据组织，且运行方式为 spconv2 兼容路径，未报告完整 KITTI AP。
- 每个输出目录中的 `run_notes.md` 记录了实际命令、输入数据和限制说明。

## 报告引用口径

建议在报告中表述为：

```text
本文完成了核心流程复现和单帧可视化验证，结果用于说明多传感器融合机制，不作为论文指标复现。
```

避免表述为：

```text
完整复现了论文结果，或复现了 KITTI / nuScenes benchmark 精度。
```
