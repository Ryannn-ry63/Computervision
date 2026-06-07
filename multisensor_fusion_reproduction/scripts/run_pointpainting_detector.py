#!/usr/bin/env python3
"""Headless PointPainting detector runner for the local KITTI demo sample.

The upstream demo.py requires mayavi. This script keeps the same OpenPCDet
checkpoint format but produces image files that can be used in reports.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_ROOT = ROOT / "PointPainting" / "detector"
if str(DETECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DETECTOR_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import box_utils, calibration_kitti, common_utils  # noqa: E402


MMDET_KITTI_LABELS = {
    0: "Pedestrian",
    1: "Cyclist",
    2: "Car",
}


class SingleFramePaintedDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, points_path, calib_path, ann_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=ann_path is not None,
            root_path=points_path.parent,
            logger=logger,
        )
        self.points_path = points_path
        self.calib_path = calib_path
        self.ann_path = ann_path
        self.calib = calibration_kitti.Calibration(calib_path)
        self.gt_names, self.gt_boxes_lidar = self._load_gt()

    def __len__(self):
        return 1

    def _load_gt(self):
        if self.ann_path is None:
            return None, None

        import pickle

        meta = pickle.load(open(self.ann_path, "rb"))
        instances = meta["data_list"][0]["instances"]
        names = []
        camera_boxes = []
        for inst in instances:
            label = int(inst.get("bbox_label_3d", -1))
            name = MMDET_KITTI_LABELS.get(label)
            if name not in self.class_names:
                continue
            box = np.asarray(inst["bbox_3d"], dtype=np.float32)
            if not np.isfinite(box).all() or box[2] < 0:
                continue
            names.append(name)
            camera_boxes.append(box)

        if not camera_boxes:
            raise RuntimeError(f"No valid KITTI 3D boxes found in {self.ann_path}")

        camera_boxes = np.stack(camera_boxes, axis=0)
        gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(camera_boxes, self.calib).astype(np.float32)
        return np.asarray(names), gt_boxes_lidar

    def __getitem__(self, index):
        points = np.load(self.points_path).astype(np.float32)
        input_dict = {
            "points": points,
            "frame_id": self.points_path.stem,
            "calib": self.calib,
        }
        if self.gt_boxes_lidar is not None:
            input_dict.update(
                {
                    "gt_names": self.gt_names.copy(),
                    "gt_boxes": self.gt_boxes_lidar.copy(),
                }
            )
        data_dict = self.prepare_data(input_dict)
        data_dict["image_shape"] = np.asarray([375, 1242], dtype=np.int32)
        return data_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["infer", "overfit"], default="infer")
    parser.add_argument(
        "--cfg-file",
        default=str(DETECTOR_ROOT / "tools" / "cfgs" / "kitti_models" / "pointpillar_painted.yaml"),
    )
    parser.add_argument(
        "--points",
        default=str(ROOT / "outputs" / "pointpainting_official" / "painted_lidar" / "000008.npy"),
    )
    parser.add_argument(
        "--calib",
        default=str(DETECTOR_ROOT / "data" / "kitti" / "training" / "calib" / "000008.txt"),
    )
    parser.add_argument(
        "--ann",
        default=str(ROOT / "mmdetection3d" / "demo" / "data" / "kitti" / "000008.pkl"),
    )
    parser.add_argument(
        "--image",
        default=str(DETECTOR_ROOT / "data" / "kitti" / "training" / "image_2" / "000008.png"),
    )
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "pointpainting_detector"))
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_cfg(cfg_file, score_thresh):
    old_cwd = Path.cwd()
    os.chdir(DETECTOR_ROOT / "tools")
    try:
        cfg_from_yaml_file(str(Path(cfg_file).resolve()), cfg)
    finally:
        os.chdir(old_cwd)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = score_thresh
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        "gt_sampling",
        "random_world_flip",
        "random_world_rotation",
        "random_world_scaling",
    ]
    return cfg


def build_dataset(args, logger, training):
    ann_path = Path(args.ann) if training else None
    return SingleFramePaintedDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        points_path=Path(args.points),
        calib_path=Path(args.calib),
        ann_path=ann_path,
        logger=logger,
    )


def run_inference(model, dataset, device):
    model.to(device).eval()
    data = dataset[0]
    batch = dataset.collate_batch([data])
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred_dicts, _ = model(batch)
    return data, pred_dicts[0]


def overfit(args, logger):
    dataset = build_dataset(args, logger, training=True)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    losses = []
    for i in range(1, args.iters + 1):
        data = dataset[0]
        batch = dataset.collate_batch([data])
        load_data_to_gpu(batch)
        optimizer.zero_grad(set_to_none=True)
        ret_dict, tb_dict, _ = model(batch)
        loss = ret_dict["loss"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if hasattr(model, "update_global_step"):
            model.update_global_step()
        losses.append(float(loss.detach().cpu()))
        if i == 1 or i % 10 == 0 or i == args.iters:
            print(f"iter {i:04d}/{args.iters} loss={losses[-1]:.4f} tb={tb_dict}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "pointpillar_painted_000008_overfit.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": 0,
            "it": args.iters,
            "version": "local-single-frame-overfit",
        },
        ckpt_path,
    )
    np.savetxt(output_dir / "overfit_loss.txt", np.asarray(losses), fmt="%.8f")
    print(f"saved ckpt: {ckpt_path}")
    return ckpt_path


def draw_bev(points, pred, gt_boxes, out_path):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    sem = points[:, 4:8]
    colors = np.array(["#6b7280", "#ef4444", "#22c55e", "#3b82f6"])
    labels = np.argmax(sem, axis=1) if sem.shape[1] else np.zeros(points.shape[0], dtype=np.int64)
    ax.scatter(points[:, 0], points[:, 1], s=0.25, c=colors[labels], alpha=0.55, linewidths=0)

    def add_boxes(boxes, color, label):
        if boxes is None or len(boxes) == 0:
            return
        corners = box_utils.boxes_to_corners_3d(boxes[:, :7])
        for i, box_corners in enumerate(corners):
            xy = box_corners[[0, 1, 2, 3, 0], :2]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.8, label=label if i == 0 else None)

    pred_boxes = pred["pred_boxes"].detach().cpu().numpy()
    add_boxes(gt_boxes, "#f59e0b", "GT")
    add_boxes(pred_boxes, "#06b6d4", "Pred")

    ax.set_xlim(0, 70)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    if len(pred_boxes) or gt_boxes is not None:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_image_boxes(image_path, calib, pred, out_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    pred_boxes = pred["pred_boxes"].detach().cpu().numpy()
    pred_scores = pred["pred_scores"].detach().cpu().numpy()
    pred_labels = pred["pred_labels"].detach().cpu().numpy()
    if len(pred_boxes):
        boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            boxes_camera, calib, image_shape=np.asarray(image.size[::-1])
        )
        for box, score, label in zip(boxes_img, pred_scores, pred_labels):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline=(0, 188, 212), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), f"{cfg.CLASS_NAMES[int(label) - 1]} {score:.2f}", fill=(0, 188, 212))
    image.save(out_path)


def save_outputs(args, dataset, data, pred):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = data["points"]
    scores = pred["pred_scores"].detach().cpu().numpy()
    labels = pred["pred_labels"].detach().cpu().numpy()
    boxes = pred["pred_boxes"].detach().cpu().numpy()
    np.savez(output_dir / "predictions_000008.npz", boxes=boxes, scores=scores, labels=labels)
    draw_bev(points, pred, dataset.gt_boxes_lidar, output_dir / "pointpainting_detector_bev.png")
    draw_image_boxes(Path(args.image), dataset.calib, pred, output_dir / "pointpainting_detector_image.png")
    summary = {
        "checkpoint": args.ckpt,
        "num_predictions": int(len(scores)),
        "scores": scores[:20].tolist(),
        "labels": [cfg.CLASS_NAMES[int(x) - 1] for x in labels[:20]],
        "note": "The overfit checkpoint is trained only on KITTI demo frame 000008 for pipeline reproduction.",
    }
    (output_dir / "detector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    load_cfg(args.cfg_file, args.score_thresh)
    logger = common_utils.create_logger()

    ckpt_path = args.ckpt
    if args.mode == "overfit":
        ckpt_path = str(overfit(args, logger))

    dataset = build_dataset(args, logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    if ckpt_path:
        model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        args.ckpt = ckpt_path
    data, pred = run_inference(model, dataset, args.device)
    save_outputs(args, dataset, data, pred)


if __name__ == "__main__":
    main()
