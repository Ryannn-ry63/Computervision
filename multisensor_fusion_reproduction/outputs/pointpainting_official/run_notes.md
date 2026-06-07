# Official PointPainting Reproduction Notes

## Result

- Used the cloned `PointPainting/painting/painting.py` pipeline with `--seg-net 0`.
- Semantic model: torchvision DeepLabV3 ResNet101 pretrained on COCO/VOC labels, which is the repo's `deeplabv3` branch.
- Painted lidar file: `outputs/pointpainting_official/painted_lidar/000008.npy`.
- Painted lidar shape: `(17153, 8)`. Columns are `x,y,z,intensity,s0,s1,s2,s3`.
- Projected painted points in image: `17153`.
- Label counts: `{'background': 11099, 'bicycle/cyclist': 0, 'car': 6054, 'person': 0}`.

## Commands

```bash
cd PointPainting/painting
/root/miniconda3/envs/motiondetection/bin/python painting.py \
  --seg-net 0 \
  --training-path ../detector/data/kitti/training \
  --save-path ../../outputs/pointpainting_official/painted_lidar \
  --sample-idx 000008

cd ../..
/root/miniconda3/envs/motiondetection/bin/python scripts/visualize_official_pointpainting.py \
  --image PointPainting/detector/data/kitti/training/image_2/000008.png \
  --painted outputs/pointpainting_official/painted_lidar/000008.npy \
  --calib PointPainting/detector/data/kitti/training/calib/000008.txt \
  --out-dir outputs/pointpainting_official
```

## Limitation

The cloned repo's DeepLabV3+ branch requires `mmcv<=1.4.0`, but the current environment has `mmcv==2.1.0` for BEVFusion. To avoid breaking that environment, this run used the repo's torchvision DeepLabV3 branch. Detector inference still needs a trained `pointpillar_painted` checkpoint and OpenPCDet installation.

## Detector Follow-up

The README loads detector checkpoints with:

```bash
cd PointPainting/detector/tools
python demo.py --cfg_file cfgs/kitti_models/pointpillar_painted.yaml \
  --ckpt ${your trained ckpt} \
  --data_path ${painted .npy file} \
  --ext .npy
```

No official `pointpillar_painted` detector checkpoint was found in the cloned repo. The detector branch was made runnable in the current environment by compiling the PointPillar-required OpenPCDet ops and avoiding unused PointNet2 imports.

For a complete local pipeline test, a single-frame overfit checkpoint was trained on KITTI demo frame `000008`:

```bash
/root/miniconda3/envs/motiondetection/bin/python scripts/run_pointpainting_detector.py \
  --mode overfit \
  --iters 300 \
  --lr 0.003 \
  --score-thresh 0.05 \
  --output-dir outputs/pointpainting_detector_300iter
```

Outputs:

- `outputs/pointpainting_detector_300iter/pointpillar_painted_000008_overfit.pth`
- `outputs/pointpainting_detector_300iter/pointpainting_detector_panel.png`
- `outputs/pointpainting_detector_300iter/detector_summary.json`

At score threshold `0.05`, the overfit checkpoint predicts 6 `Car` boxes with scores around `0.42-0.65`. This is a pipeline reproduction/debug result, not a KITTI benchmark result.
