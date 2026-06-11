# CLOCs legacy compatibility note

## Purpose

This note records the isolated CLOCs reproduction environment used for the report attachment. The goal was to run the official single-frame CLOCs inference path without changing the existing `motiondetection` environment.

## Environment

- Conda env: `/root/miniconda3/envs/clocs_legacy`
- Python: 3.7.16
- PyTorch: 1.7.1+cu110
- torchvision: 0.8.2+cu110
- spconv CUDA wheel: `spconv-cu118==2.3.6`
- cumm CUDA wheel: `cumm-cu118==0.4.11`
- Other key packages: `protobuf==3.20.3`, `numba==0.56.4`, `scikit-image==0.19.3`, `shapely==1.8.5.post1`

The original CLOCs README asks for SECOND-1.5 with `spconv v1.0`. A matching `spconv v1` wheel was not available for the current server, so the run uses the existing spconv2 compatibility patches and the converted SECOND checkpoint.

## Code Patch

`CLOCs/second/pytorch/train.py` was patched according to the README's PyTorch compatibility note:

```python
opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.to(torch.bool)
```

This replaces the old `dir_labels.byte()` expression, which raises a bool/byte type error under newer PyTorch behavior.

## Run Command

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

The command writes:

```text
CLOCs/CLOCs_SecCas_pretrained/eval_results/step_30950/000008.txt
```

The script continued into official AP evaluation after label export and stopped because the local mini KITTI subset does not contain a full validation set annotation list. Therefore the report uses the saved single-frame prediction and visualization, not KITTI benchmark AP.

## Visualization

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/computervision

/root/miniconda3/envs/clocs_legacy/bin/python scripts/visualize_clocs_official.py \
  --pred CLOCs/CLOCs_SecCas_pretrained/eval_results/step_30950/000008.txt \
  --out-dir outputs/clocs_official_000008
```

The visualization summary reported 100 exported rows and 1 unique prediction after deduplication, with score `0.8681`.
