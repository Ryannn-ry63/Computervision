# PointPainting 工作总结

## 一句话结论

我基于已经 clone 下来的 `PointPainting/` 仓库，完成了 PointPainting 的核心 painting 流程，并进一步跑通了 painted PointPillar 3D detector 的 checkpoint 加载和推理链路。后续又在独立 `pointpainting_legacy` 环境中跑通了仓库内置 MMSegmentation DeepLabV3+ painting 分支。最终有三类结果：

- 仓库内置 MMSeg DeepLabV3+ painting 结果：`outputs/pointpainting_mmseg_legacy/official_pointpainting_panel.png`
- torchvision DeepLabV3 painting 备用结果：`outputs/pointpainting_official/official_pointpainting_panel.png`
- 完整 detector 链路结果：`outputs/pointpainting_detector_300iter/pointpainting_detector_panel.png`

这些图不是割裂的关系。painting 图展示 PointPainting 如何把相机语义“涂”到 LiDAR 点上；detector 图使用 painted point cloud 作为输入，再送入 PointPillar 做 3D 检测。

完整关系是：

```text
image + lidar + calib
-> 2D semantic segmentation
-> project LiDAR points to image
-> append semantic scores to each point
-> painted point cloud
-> PointPillar detector
-> 3D boxes
```

## 已完成的工作

1. 阅读并确认 `PointPainting/README.md` 中的训练和推理方式。
2. 使用仓库自带 `painting/painting.py` 跑通 painting 流程。
3. 准备了一个 KITTI 风格 demo 样本 `000008`，包括图像、点云和标定文件。
4. 在独立 `pointpainting_legacy` 环境中跑通仓库内置 MMSegmentation DeepLabV3+ 分支，生成了 painted LiDAR 文件：

```text
outputs/pointpainting_mmseg_legacy/painted_lidar/000008.npy
```

5. 生成了 painting 可视化结果：

```text
outputs/pointpainting_mmseg_legacy/official_pointpainting_panel.png
outputs/pointpainting_mmseg_legacy/official_pointpainting_projection.png
outputs/pointpainting_mmseg_legacy/official_painted_bev.png
outputs/pointpainting_mmseg_legacy/official_painted_label_counts.png
```

6. 修通 detector 侧 PointPillar painted 模型在当前环境下的运行问题。
7. 编译并验证了 PointPillar 需要的 OpenPCDet CUDA ops。
8. 写了无 GUI 的 detector 脚本：

```text
scripts/run_pointpainting_detector.py
```

9. 用 KITTI demo 单帧 `000008` overfit 训练了一个本地 checkpoint，用于验证完整 detector 链路：

```text
outputs/pointpainting_detector_300iter/pointpillar_painted_000008_overfit.pth
```

10. 使用该 checkpoint 跑出 3D 检测输出和 PPT 可视化图：

```text
outputs/pointpainting_detector_300iter/pointpainting_detector_panel.png
outputs/pointpainting_detector_300iter/pointpainting_detector_image.png
outputs/pointpainting_detector_300iter/pointpainting_detector_bev.png
```

## Painting 结果

仓库内置 MMSeg DeepLabV3+ painting 分支输出的 painted 点云为：

```text
outputs/pointpainting_mmseg_legacy/painted_lidar/000008.npy
```

形状：

```text
(17153, 8)
```

列格式：

```text
x, y, z, intensity, s0, s1, s2, s3
```

这和 detector 配置文件中的 8 维 painted 点云输入一致：

```text
PointPainting/detector/tools/cfgs/dataset_configs/painted_kitti_dataset.yaml
```

语义统计结果：

```text
background: 11099
car: 6054
bicycle/cyclist: 0
person: 0
```

更新后的 DeepLabV3+ 分支语义统计结果：

```text
background: 10728
bicycle/cyclist: 6
car: 6389
person: 30
```

## Detector 结果

README 中 detector 推理的 checkpoint 加载方式是：

```bash
cd PointPainting/detector/tools
python demo.py \
  --cfg_file cfgs/kitti_models/pointpillar_painted.yaml \
  --ckpt ${your trained ckpt} \
  --data_path ${painted .npy file} \
  --ext .npy
```

但是仓库没有提供官方 `pointpillar_painted` detector checkpoint，只要求用户传入 `${your trained ckpt}`。因此 detector 部分没有作为官方预训练结果报告，而是用 KITTI demo 单帧 `000008` 做了一个本地 overfit checkpoint，用于验证完整流程能跑通。

训练命令：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision

/root/miniconda3/envs/motiondetection/bin/python scripts/run_pointpainting_detector.py \
  --mode overfit \
  --iters 300 \
  --lr 0.003 \
  --score-thresh 0.05 \
  --output-dir outputs/pointpainting_detector_300iter
```

结果摘要：

```text
checkpoint: outputs/pointpainting_detector_300iter/pointpillar_painted_000008_overfit.pth
num_predictions: 6
labels: Car x 6
scores: 0.6458, 0.5431, 0.5073, 0.4632, 0.4553, 0.4243
```

这个结果和 `000008.pkl` 中 6 个有效 Car 标注数量一致。

重新加载 checkpoint 推理的命令：

```bash
/root/miniconda3/envs/motiondetection/bin/python scripts/run_pointpainting_detector.py \
  --mode infer \
  --ckpt outputs/pointpainting_detector_300iter/pointpillar_painted_000008_overfit.pth \
  --score-thresh 0.05 \
  --output-dir outputs/pointpainting_detector_300iter
```

## 我做过的环境适配

当前环境和原仓库年代差异比较大，所以 detector 侧需要适配：

- 当前环境使用 PyTorch 2.x，原 OpenPCDet 分支中的 PointNet2 扩展会因为旧 CUDA/THC 接口编译失败。
- PointPillar painted 不需要 PointNet2，所以我让 detector 只加载 PointPillar 必要模块，避免无关 PointNet2 import。
- 当前环境使用 spconv 2.x，原始 voxel generator API 已变化，所以我给 `DataProcessor` 增加了 `spconv.pytorch.utils.PointToVoxel` 兼容分支。
- 原 `demo.py` 依赖 `mayavi`，当前环境没有 GUI/mayavi，因此我写了 headless 脚本输出 PNG。

这些改动的目标是让 PointPillar painted 路线可运行，而不是重写算法。

## DeepLabV3+ 分支复现情况

`PointPainting/painting/painting.py` 支持三种语义来源：

- `--seg-net 0`：torchvision DeepLabV3
- `--seg-net 1`：仓库内置旧版 mmseg DeepLabV3+
- `--seg-net 2`：HMA

原 `motiondetection` 环境为了 BEVFusion 复现已经使用 `mmcv==2.1.0`，而该仓库内置旧版 mmseg 要求 `mmcv<=1.4.0`。因此我没有破坏已有 BEVFusion 环境，而是新建了独立 `pointpainting_legacy` 环境。

在该独立环境中，`--seg-net 1` 的仓库内置 MMSeg DeepLabV3+ 分支已经跑通。由于采用 CPU 推理，需要做两点兼容处理：一是将 DeepLabV3+ 配置中的 `SyncBN` 替换为普通 `BN`；二是避免导入未使用但依赖 `mmcv-full` 自定义算子的 decode head。该处理不改变 DeepLabV3+ painting 主流程。

## PPT 推荐口径

推荐先放：

```text
outputs/pointpainting_mmseg_legacy/official_pointpainting_panel.png
```

讲：

> PointPainting 的核心是点级融合。先用图像语义分割得到像素级语义信息，再根据 KITTI 标定把 LiDAR 点投影到图像上，从对应像素采样语义分数，并追加到点云特征中，得到 8 维 painted point cloud。

再放：

```text
outputs/pointpainting_detector_300iter/pointpainting_detector_panel.png
```

讲：

> 第一阶段得到的 painted point cloud 可以作为 3D detector 的输入。我进一步跑通了 painted PointPillar 的 checkpoint 加载和推理链路，并在 KITTI demo 单帧上 overfit 了一个本地 checkpoint 来验证完整流程可以输出 3D 检测框。

必须补充限制：

> detector 部分不是官方 KITTI benchmark 结果，因为原仓库没有提供 detector 预训练权重。这里的 checkpoint 是单帧 overfit，只用于说明完整流程跑通。

## 和 BEVFusion 的对比

可以这样讲：

- PointPainting 是点级融合：把图像语义追加到每个 LiDAR 点上。
- BEVFusion 是 BEV 级融合：把相机和 LiDAR 特征统一到 BEV 空间后再融合。
- PointPainting 的流程更直观，适合解释“相机语义如何帮助点云检测”。
- BEVFusion 的融合位置更高层，通常更适合端到端多模态 BEV 感知。

## 相关文件

主要说明文档：

```text
PointPainting复现指南.md
outputs/pointpainting_official/run_notes.md
PointPainting工作总结.md
```

主要脚本：

```text
PointPainting/painting/painting.py
scripts/visualize_official_pointpainting.py
scripts/run_pointpainting_detector.py
```

主要结果：

```text
outputs/pointpainting_official/official_pointpainting_panel.png
outputs/pointpainting_detector_300iter/pointpainting_detector_panel.png
outputs/pointpainting_detector_300iter/pointpillar_painted_000008_overfit.pth
```
