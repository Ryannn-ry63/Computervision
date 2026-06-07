# CLOCs 复现指南

## 结论

本次复现优先使用已 clone 的 `CLOCs/` 仓库理解官方流程，并从 README 指向的官方 Google Drive 下载了预训练/候选文件。当前有两类结果：

官方 checkpoint 单帧运行图，PPT 推荐优先使用：

```text
outputs/clocs_official_000008/official_clocs_panel.png
```

候选级融合机制解释图，适合作为方法页补充：

```text
outputs/clocs_reproduction/clocs_panel.png
```

这张图展示 CLOCs 的核心思想：

```text
2D detector candidates + 3D detector candidates -> 2D/3D IoU 关联 -> 候选级融合重打分
```

注意：官方代码依赖旧版 SECOND/spconv v1。本次另外建立了 `clocs_legacy` 隔离环境（Python 3.7、PyTorch 1.7.1、spconv-cu118 2.3.6）重跑单帧推理，并把 spconv v1 checkpoint 的 sparse conv 权重转成 spconv2 布局。单帧推理已跑通，但由于本地没有完整 KITTI label/raw/reduced 数据，不能做完整 KITTI AP benchmark。

## 已完成输出

输出目录：

```text
outputs/clocs_official_000008/
```

关键文件：

- `official_clocs_panel.png`：PPT 推荐使用的官方 checkpoint 单帧组合图。
- `official_clocs_image.png`：相机视角，橙色为官方 Cascade-RCNN 2D candidates，青色为 CLOCs 输出。
- `official_clocs_bev.png`：BEV 中的 CLOCs 3D 输出。
- `official_clocs_summary.json`：原始导出行数、去重后预测数和分数。
- `run_notes.md`：单帧官方运行说明。

官方单帧运行使用：

- SECOND checkpoint：`CLOCs/model_dir_spconv2/voxelnet-30950.tckpt`，由官方 `second_model.zip` 转换得到。
- CLOCs fusion checkpoint：`CLOCs/CLOCs_SecCas_pretrained/fusion_layer-11136.tckpt`。
- 2D candidates：`CLOCs/d2_detection_data/data/000008.txt`，来自官方 `cascade_rcnn_sigmoid_data.zip`。
- mini KITTI 单帧：`CLOCs/mini_kitti/`。
- 运行环境：`/root/miniconda3/envs/clocs_legacy`，用于隔离 CLOCs 的旧版 SECOND 依赖。

输出里官方 CLOCs txt 为：

```text
CLOCs/CLOCs_SecCas_pretrained/eval_results/step_30950/000008.txt
```

该 txt 原始导出 100 行，在当前 spconv2 兼容运行下是重复框；可视化时按 KITTI 行内容去重，展示 1 个唯一预测框，score 为 `0.8681`。

候选机制可视化输出目录：

```text
outputs/clocs_reproduction/
```

关键文件：

- `clocs_panel.png`：PPT 推荐使用的 2x2 组合图。
- `clocs_candidates_image.png`：相机视角中的 2D 候选框、投影后的 3D 候选框和最佳匹配连线。
- `clocs_fused_bev.png`：BEV 点云中 3D 候选框及融合前后分数。
- `clocs_pair_iou_matrix.png`：2D/3D 候选对 IoU 矩阵。
- `clocs_score_fusion.png`：融合前 3D 分数与 CLOCs-style 融合分数对比。
- `fusion_summary.json`：候选数量、匹配 IoU、融合分数等数值摘要。
- `run_notes.md`：命令、结果和限制说明。

候选机制图使用：

- 3D candidates：`outputs/pointpainting_detector_300iter/predictions_000008.npz`，来自之前跑通的单帧 PointPainting detector overfit 结果。
- 2D candidates：用 KITTI demo pkl 中的 2D 标注框作为 2D detector proxy，因为本地没有官方 Cascade-RCNN 候选文件。
- 融合特征：参考 CLOCs 代码中的候选对特征，使用 `2D IoU, 3D score, 2D score, normalized distance`。

## 实际运行命令

下载到的官方文件：

```text
CLOCs/official_downloads/second_model.zip
CLOCs/official_downloads/CLOCs_SecCas_pretrained.zip
CLOCs/official_downloads/cascade_rcnn_sigmoid_data.zip
CLOCs/official_downloads/kitti_infos_val.pkl
```

官方文件来源是 CLOCs README 中的 Google Drive folder：

```text
https://drive.google.com/drive/folders/16Z9_c8VbZVsvrHczn67ZCeOHYx1x5VDj?usp=sharing
```

单帧官方 CLOCs evaluate：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision/CLOCs/second

PYTHONPATH=/inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision/CLOCs \
CLOCS_SECOND_CONFIG=./configs/car.local_000008.config \
CLOCS_SECOND_MODEL_DIR=../model_dir_spconv2 \
/root/miniconda3/envs/clocs_legacy/bin/python pytorch/train.py evaluate \
  --config_path=./configs/car.local_000008.config \
  --model_dir=../CLOCs_SecCas_pretrained \
  --measure_time=True \
  --batch_size=1 \
  --pickle_result=False
```

这个命令已生成：

```text
CLOCs/CLOCs_SecCas_pretrained/eval_results/step_30950/000008.txt
```

生成官方单帧可视化：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision

/root/miniconda3/envs/clocs_legacy/bin/python scripts/visualize_clocs_official.py \
  --pred CLOCs/CLOCs_SecCas_pretrained/eval_results/step_30950/000008.txt \
  --out-dir outputs/clocs_official_000008
```

生成候选机制解释图：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision

/root/miniconda3/envs/motiondetection/bin/python scripts/clocs_reproduction_demo.py \
  --out-dir outputs/clocs_reproduction
```

脚本会读取：

```text
PointPainting/detector/data/kitti/training/image_2/000008.png
PointPainting/detector/data/kitti/training/velodyne/000008.bin
PointPainting/detector/data/kitti/training/calib/000008.txt
mmdetection3d/demo/data/kitti/000008.pkl
outputs/pointpainting_detector_300iter/predictions_000008.npz
```

## 为什么没有做完整 KITTI AP

CLOCs README 中的完整官方 evaluate 需要：

- `cascade_rcnn_sigmoid_data`：2D detection candidates。
- `second_model.zip`：SECOND pretrained weights。
- `CLOCs_SecCas_pretrained.zip`：CLOCs fusion pretrained weights。
- KITTI full infos 和 reduced point cloud 文件。
- 旧版 SECOND-1.5 / spconv v1 环境。

本次已下载前三项和 `kitti_infos_val.pkl`，并在 `clocs_legacy` 隔离环境中跑通了单帧推理。官方 evaluate 在生成 label 后继续计算 AP 时会因为 mini KITTI 不包含完整验证集标注而停止，因此这里只使用生成的 KITTI label 做可视化。另一个限制是当前环境不是官方 spconv v1，而是 spconv2/CUDA wheel 兼容运行，导出的 txt 有重复行。因此报告中应表述为“官方 checkpoint 单帧流程跑通”，不要表述为 leaderboard 指标复现。

## PPT 汇报口径

可以这样讲：

> 我复现了 CLOCs 的官方单帧推理流程。CLOCs 不直接在原始点云或图像特征上融合，而是先分别取得 2D 检测器和 3D 检测器的候选框。这里使用官方提供的 SECOND checkpoint、CLOCs fusion checkpoint 和 Cascade-RCNN 2D candidates，在 KITTI demo 单帧上输出了融合后的 3D 检测结果。图中橙色框是 2D candidates，青色框是 CLOCs 融合后的 3D candidate 投影，右侧 BEV 展示同一预测在鸟瞰空间的位置。

需要补充限制：

> 本次没有做完整 KITTI AP 复现，因为本地缺完整 KITTI benchmark 数据，并且当前环境是 spconv2 兼容运行，不是论文的 spconv v1 环境。因此本页结果用于说明官方 checkpoint 流程和候选级融合机制，不代表 leaderboard 指标。

如果需要和 PointPainting / BEVFusion 对比，可以说：

- PointPainting 是点级融合：把图像语义分数涂到每个 LiDAR 点上。
- BEVFusion 是 BEV 特征级融合：把多模态特征统一到鸟瞰空间。
- CLOCs 是候选级后融合：不改底层 detector，而是在 2D/3D detection candidates 之间建立关联并重打分。
