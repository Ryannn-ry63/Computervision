# PointPainting MMSeg DeepLabV3+ Reproduction Notes

## Result

- Used the cloned `PointPainting/painting/painting.py` pipeline with `--seg-net 1`.
- Semantic model: the repository's built-in MMSegmentation 0.12.0 DeepLabV3+ branch with the provided Cityscapes checkpoint.
- Isolated environment: `/root/miniconda3/envs/pointpainting_legacy` with Python 3.8, PyTorch 1.13.1 CPU, and MMCV 1.3.18.
- Painted lidar file: `outputs/pointpainting_mmseg_legacy/painted_lidar/000008.npy`.
- Painted lidar shape: `(17153, 8)`. Columns are `x,y,z,intensity,s0,s1,s2,s3`.
- Projected painted points in image: `17153`.
- Label counts: `{'background': 10728, 'bicycle/cyclist': 6, 'car': 6389, 'person': 30}`.

## Commands

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision/PointPainting/painting

/root/miniconda3/envs/pointpainting_legacy/bin/python painting.py \
  --seg-net 1 \
  --training-path ../detector/data/kitti/training \
  --save-path ../../outputs/pointpainting_mmseg_legacy/painted_lidar \
  --sample-idx 000008 \
  --device cpu

cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision

/root/miniconda3/envs/pointpainting_legacy/bin/python scripts/visualize_official_pointpainting.py \
  --image PointPainting/detector/data/kitti/training/image_2/000008.png \
  --painted outputs/pointpainting_mmseg_legacy/painted_lidar/000008.npy \
  --calib PointPainting/detector/data/kitti/training/calib/000008.txt \
  --out-dir outputs/pointpainting_mmseg_legacy
```

## Compatibility Notes

- The BEVFusion environment uses newer OpenMMLab dependencies, so this run used a separate `pointpainting_legacy` environment.
- The original DeepLabV3+ config uses `SyncBN`, which expects GPU tensors. A CPU-specific config was added that replaces `SyncBN` with `BN`.
- The repository's bundled MMSegmentation imports some unused decode heads that require `mmcv-full` ops. For this DeepLabV3+ run, unused `PointHead` import was removed and optional ops imports were guarded so `DepthwiseSeparableASPPHead` and `FCNHead` can run without `mmcv-full`.
- This result validates the repository's built-in MMSeg DeepLabV3+ painting path on one KITTI sample; it is not a full KITTI benchmark evaluation.
