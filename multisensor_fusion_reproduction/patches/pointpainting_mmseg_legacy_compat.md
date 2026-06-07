# PointPainting MMSeg Legacy Compatibility Notes

This note records the minimal compatibility changes used to run the cloned PointPainting repository's built-in MMSegmentation DeepLabV3+ painting branch on the current server.

## Environment

- Conda env: `/root/miniconda3/envs/pointpainting_legacy`
- Python: 3.8
- PyTorch: 1.13.1 CPU
- Torchvision: 0.14.1 CPU
- MMCV: 1.3.18
- MMSegmentation: bundled `PointPainting/painting/mmseg`, version 0.12.0

## Code Compatibility Changes

1. `PointPainting/painting/mmseg/models/decode_heads/__init__.py`
   - Removed unused `PointHead` import from the package-level decode head import list.
   - Reason: importing `PointHead` requires `mmcv.ops.point_sample`, which requires `mmcv-full`; DeepLabV3+ does not use `PointHead`.

2. `PointPainting/painting/mmseg/models/decode_heads/cc_head.py`
   - Changed optional `CrissCrossAttention` import guard to catch both `ImportError` and `ModuleNotFoundError`.
   - Reason: CPU-only MMCV may expose `mmcv.ops` but fail loading compiled ops; CCHead is not used by DeepLabV3+.

3. `PointPainting/painting/mmseg/models/decode_heads/psa_head.py`
   - Changed optional `PSAMask` import guard to catch both `ImportError` and `ModuleNotFoundError`.
   - Reason: PSAHead is not used by DeepLabV3+.

4. `PointPainting/painting/mmseg/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_cpu.py`
   - Added a CPU-specific config inheriting the original DeepLabV3+ config and overriding `SyncBN` with `BN`.
   - Reason: PyTorch `SyncBatchNorm` expects GPU tensors during inference.

5. `PointPainting/painting/painting.py`
   - When `--device cpu` is used with `--seg-net 1`, the script selects the CPU-specific DeepLabV3+ config.

## Validation Command

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision/PointPainting/painting

/root/miniconda3/envs/pointpainting_legacy/bin/python painting.py \
  --seg-net 1 \
  --training-path ../detector/data/kitti/training \
  --save-path ../../outputs/pointpainting_mmseg_legacy/painted_lidar \
  --sample-idx 000008 \
  --device cpu
```

## Output

- `outputs/pointpainting_mmseg_legacy/painted_lidar/000008.npy`
- Shape: `(17153, 8)`
- Label counts: `background=10728`, `bicycle/cyclist=6`, `car=6389`, `person=30`

These changes are compatibility guards for unused modules and CPU inference. They do not change the PointPainting painting logic: image segmentation scores are still projected to LiDAR points and appended as point features.
