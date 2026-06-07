# PointPainting Reproduction Notes

## Result

- Generated `camera_raw.png`, `semantic_mask_overlay.png`, `pointpainting_projection.png`, `painted_bev_semantic.png`, `pointpainting_panel.png`, and `painted_points.npy`.
- Painted 17238 projected LiDAR points out of 17238 total points.
- This is a lightweight PointPainting-style reproduction: the core projection and point-feature augmentation are reproduced; the 2D semantic input is a deterministic local proxy so the demo does not require downloading a segmentation checkpoint.

## Command

```bash
python scripts/pointpainting_demo.py \
  --image mmdetection3d/demo/data/kitti/000008.png \
  --pointcloud mmdetection3d/demo/data/kitti/000008.bin \
  --calib mmdetection3d/demo/data/kitti/000008.pkl \
  --out-dir outputs/pointpainting_reproduction
```

## PPT Explanation

PointPainting first projects LiDAR points into the camera image with calibration matrices, samples per-pixel semantic scores from the image branch, and appends those scores to the original point features. The painted points can then be consumed by a standard LiDAR detector. This visualization shows the intermediate painted point cloud rather than a trained detector output.
