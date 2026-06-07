# CLOCs Reproduction Notes

## Result

- Generated `clocs_panel.png` as the report-ready summary image.
- Visualized the CLOCs candidate-level fusion idea on KITTI demo frame `000008`.
- 3D candidates come from `outputs/pointpainting_detector_300iter/predictions_000008.npz`.
- 2D candidates are approximated from KITTI annotation boxes because the cloned CLOCs repo does not include Cascade-RCNN candidate files.
- Pair features follow the CLOCs code path: 2D IoU, 3D candidate score, 2D candidate score, and distance to LiDAR.

## Outputs

- `clocs_candidates_image.png`: camera-view 2D candidates, projected 3D candidates, and best 2D/3D links.
- `clocs_fused_bev.png`: BEV point cloud with 3D candidates and before/after scores.
- `clocs_pair_iou_matrix.png`: 2D/3D candidate pair IoU matrix.
- `clocs_score_fusion.png`: confidence scores before and after CLOCs-style fusion.
- `fusion_summary.json`: numeric summary.

## Command

```bash
/root/miniconda3/envs/motiondetection/bin/python scripts/clocs_reproduction_demo.py \
  --out-dir outputs/clocs_reproduction
```

## Limitation

This is not an official KITTI AP reproduction. Full CLOCs evaluation requires downloaded SECOND weights, Cascade-RCNN 2D candidate files, KITTI info files, spconv v1, and a trained CLOCs fusion checkpoint.
