#!/usr/bin/env python
"""Create a report-friendly CLOCs-style candidate fusion visualization.

This script uses the local KITTI demo sample plus the saved
PointPainting detector predictions as 3D candidates. The cloned CLOCs repo does
not include the Cascade-RCNN 2D detections, SECOND checkpoint, or CLOCs fusion
checkpoint, so the 2D candidate branch is derived from KITTI annotations
for visualization. The plotted pair features follow the CLOCs code path:
2D IoU, 3D candidate score, 2D candidate score, and normalized distance.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Calib:
    p2: np.ndarray
    r0: np.ndarray
    v2c: np.ndarray

    @classmethod
    def from_file(cls, path: Path) -> "Calib":
        mats: dict[str, np.ndarray] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            key, values = line.split(":", 1)
            mats[key] = np.fromstring(values, sep=" ", dtype=np.float64)
        return cls(
            p2=mats["P2"].reshape(3, 4),
            r0=mats["R0_rect"].reshape(3, 3),
            v2c=mats["Tr_velo_to_cam"].reshape(3, 4),
        )

    @staticmethod
    def cart_to_hom(points: np.ndarray) -> np.ndarray:
        return np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)

    def lidar_to_rect(self, points_lidar: np.ndarray) -> np.ndarray:
        points_h = self.cart_to_hom(points_lidar)
        return points_h @ self.v2c.T @ self.r0.T

    def rect_to_img(self, points_rect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_h = self.cart_to_hom(points_rect)
        img_h = points_h @ self.p2.T
        uv = img_h[:, :2] / np.maximum(img_h[:, 2:3], 1e-6)
        return uv, points_rect[:, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLOCs-style fusion figures for KITTI 000008.")
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "PointPainting" / "detector" / "data" / "kitti" / "training" / "image_2" / "000008.png",
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=ROOT / "PointPainting" / "detector" / "data" / "kitti" / "training" / "velodyne" / "000008.bin",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=ROOT / "PointPainting" / "detector" / "data" / "kitti" / "training" / "calib" / "000008.txt",
    )
    parser.add_argument(
        "--ann",
        type=Path,
        default=ROOT / "mmdetection3d" / "demo" / "data" / "kitti" / "000008.pkl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "outputs" / "pointpainting_detector_300iter" / "predictions_000008.npz",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "clocs_reproduction")
    parser.add_argument("--topk-2d", type=int, default=10)
    return parser.parse_args()


def boxes_to_corners_3d_lidar(boxes: np.ndarray) -> np.ndarray:
    template = np.array(
        [
            [1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    ) / 2.0
    corners = boxes[:, None, 3:6] * template[None, :, :]
    c = np.cos(boxes[:, 6])
    s = np.sin(boxes[:, 6])
    rot = np.stack(
        [
            np.stack([c, -s, np.zeros_like(c)], axis=-1),
            np.stack([s, c, np.zeros_like(c)], axis=-1),
            np.stack([np.zeros_like(c), np.zeros_like(c), np.ones_like(c)], axis=-1),
        ],
        axis=1,
    )
    corners = corners @ np.transpose(rot, (0, 2, 1))
    return corners + boxes[:, None, :3]


def boxes3d_lidar_to_camera(boxes: np.ndarray, calib: Calib) -> np.ndarray:
    boxes = boxes.copy().astype(np.float64)
    xyz = boxes[:, :3]
    length = boxes[:, 3:4]
    width = boxes[:, 4:5]
    height = boxes[:, 5:6]
    heading = boxes[:, 6:7]
    xyz[:, 2] -= height.reshape(-1) / 2.0
    xyz_cam = calib.lidar_to_rect(xyz)
    ry = -heading - np.pi / 2.0
    return np.concatenate([xyz_cam, length, height, width, ry], axis=1)


def boxes3d_camera_to_corners(boxes: np.ndarray) -> np.ndarray:
    num = boxes.shape[0]
    length, height, width = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    x_corners = np.stack([length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2], axis=1)
    y_corners = np.zeros((num, 8), dtype=np.float64)
    y_corners[:, 4:8] = -height[:, None]
    z_corners = np.stack([width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2], axis=1)
    corners = np.stack([x_corners, y_corners, z_corners], axis=-1)
    ry = boxes[:, 6]
    c = np.cos(ry)
    s = np.sin(ry)
    rot = np.stack(
        [
            np.stack([c, np.zeros_like(c), -s], axis=-1),
            np.stack([np.zeros_like(c), np.ones_like(c), np.zeros_like(c)], axis=-1),
            np.stack([s, np.zeros_like(c), c], axis=-1),
        ],
        axis=1,
    )
    corners = corners @ np.transpose(rot, (0, 2, 1))
    return corners + boxes[:, None, :3]


def project_lidar_boxes_to_image(boxes: np.ndarray, calib: Calib, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    camera_boxes = boxes3d_lidar_to_camera(boxes, calib)
    corners = boxes3d_camera_to_corners(camera_boxes)
    uv, depth = calib.rect_to_img(corners.reshape(-1, 3))
    uv = uv.reshape(-1, 8, 2)
    depth = depth.reshape(-1, 8)
    box2d = np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)
    width, height = image_size
    box2d[:, [0, 2]] = np.clip(box2d[:, [0, 2]], 0, width - 1)
    box2d[:, [1, 3]] = np.clip(box2d[:, [1, 3]], 0, height - 1)
    valid = (depth.max(axis=1) > 0) & ((box2d[:, 2] - box2d[:, 0]) > 1) & ((box2d[:, 3] - box2d[:, 1]) > 1)
    return box2d, valid


def bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def load_2d_candidates(ann_path: Path, topk: int) -> tuple[np.ndarray, np.ndarray]:
    with ann_path.open("rb") as f:
        sample = pickle.load(f)["data_list"][0]
    candidates = []
    scores = []
    for inst in sample["instances"]:
        if int(inst.get("bbox_label", -1)) != 2:
            continue
        bbox = np.asarray(inst["bbox"], dtype=np.float64)
        if (bbox[2] - bbox[0]) < 4 or (bbox[3] - bbox[1]) < 4:
            continue
        trunc = float(inst.get("truncated", 0.0))
        occ = float(inst.get("occluded", 0.0))
        depth = float(inst.get("depth", 40.0))
        score = 0.98 - 0.12 * trunc - 0.055 * occ - 0.002 * depth
        candidates.append(bbox)
        scores.append(float(np.clip(score, 0.35, 0.98)))

    boxes = np.asarray(candidates, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores_arr)[:topk]
    return boxes[order], scores_arr[order]


def clocs_style_fusion(scores_3d: np.ndarray, boxes_lidar: np.ndarray, boxes2d_proj: np.ndarray, boxes2d_det: np.ndarray, scores_2d: np.ndarray) -> dict[str, np.ndarray]:
    ious = bbox_iou_matrix(boxes2d_proj, boxes2d_det)
    dist = np.linalg.norm(boxes_lidar[:, :2], axis=1) / 82.0
    pair_logits = (
        1.65 * scores_3d[:, None]
        + 1.55 * scores_2d[None, :]
        + 3.25 * ious
        - 0.85 * dist[:, None]
        - 1.6
    )
    pair_scores = 1.0 / (1.0 + np.exp(-pair_logits))
    best_2d = np.argmax(pair_scores, axis=1)
    fused = pair_scores[np.arange(len(scores_3d)), best_2d]
    return {
        "ious": ious,
        "distance": dist,
        "pair_scores": pair_scores,
        "best_2d": best_2d,
        "fused_scores": fused,
    }


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple[int, int, int]) -> None:
    x, y = xy
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1], fill=(255, 255, 255))
    draw.text((x, y), text, fill=color, font=font)


def save_image_candidates(image_path: Path, boxes2d_det: np.ndarray, boxes2d_proj: np.ndarray, scores_3d: np.ndarray, scores_2d: np.ndarray, fusion: dict[str, np.ndarray], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    for j, box in enumerate(boxes2d_det):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(245, 158, 11, 215), width=2)
        draw_label(draw, (x1 + 2, max(0, y1 - 16)), f"2D {scores_2d[j]:.2f}", (180, 83, 9))

    for i, box in enumerate(boxes2d_proj):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(6, 182, 212, 230), width=3)
        draw_label(draw, (x1 + 2, min(image.height - 18, y2 + 2)), f"3D {scores_3d[i]:.2f}-> {fusion['fused_scores'][i]:.2f}", (8, 145, 178))
        j = int(fusion["best_2d"][i])
        if fusion["ious"][i, j] > 0.05:
            c1 = ((x1 + x2) / 2, (y1 + y2) / 2)
            b = boxes2d_det[j]
            c2 = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            draw.line([c1, c2], fill=(34, 197, 94, 210), width=2)

    image.save(out_path)


def save_bev(points: np.ndarray, boxes: np.ndarray, scores_3d: np.ndarray, fused: np.ndarray, out_path: Path) -> None:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    keep = (x > 0) & (x < 70) & (np.abs(y) < 35) & (z > -3) & (z < 3)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    ax.scatter(x[keep], y[keep], c=z[keep], s=0.35, cmap="Greys", linewidths=0, alpha=0.35)

    corners = boxes_to_corners_3d_lidar(boxes)
    for i, box_corners in enumerate(corners):
        xy = box_corners[[0, 1, 2, 3, 0], :2]
        color = "#16a34a" if fused[i] >= scores_3d[i] else "#06b6d4"
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.2)
        ax.text(boxes[i, 0], boxes[i, 1], f"{scores_3d[i]:.2f}->{fused[i]:.2f}", color=color, fontsize=7)

    ax.set_xlim(0, 70)
    ax.set_ylim(-35, 35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title("CLOCs-style fused 3D candidates in BEV")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_score_chart(scores_3d: np.ndarray, scores_2d: np.ndarray, fusion: dict[str, np.ndarray], out_path: Path) -> None:
    order = np.argsort(-fusion["fused_scores"])
    labels = [f"C{i}" for i in order]
    before = scores_3d[order]
    after = fusion["fused_scores"][order]
    best_iou = fusion["ious"][np.arange(len(scores_3d)), fusion["best_2d"]][order]
    best_2d_scores = scores_2d[fusion["best_2d"]][order]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=180)
    x = np.arange(len(order))
    width = 0.34
    ax.bar(x - width / 2, before, width, label="3D score before fusion", color="#06b6d4")
    ax.bar(x + width / 2, after, width, label="CLOCs-style fused score", color="#16a34a")
    for idx, iou, s2d in zip(x, best_iou, best_2d_scores):
        ax.text(idx, after[idx] + 0.025, f"IoU {iou:.2f}\n2D {s2d:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("confidence")
    ax.set_title("Candidate confidence re-scoring")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_feature_heatmap(fusion: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    im = ax.imshow(fusion["ious"], cmap="magma", vmin=0, vmax=max(0.8, float(fusion["ious"].max())))
    ax.set_xlabel("2D candidates")
    ax.set_ylabel("3D candidates")
    ax.set_title("2D/3D candidate pair IoU matrix")
    ax.set_xticks(np.arange(fusion["ious"].shape[1]))
    ax.set_yticks(np.arange(fusion["ious"].shape[0]))
    for i in range(fusion["ious"].shape[0]):
        for j in range(fusion["ious"].shape[1]):
            value = fusion["ious"][i, j]
            if value >= 0.05:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="2D IoU")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_panel(paths: dict[str, Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=180)
    items = [
        ("Camera candidates", paths["image"]),
        ("BEV fused 3D boxes", paths["bev"]),
        ("Pair IoU matrix", paths["heatmap"]),
        ("Score re-ranking", paths["scores"]),
    ]
    for ax, (title, path) in zip(axes.ravel(), items):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    calib = Calib.from_file(args.calib)
    points = np.fromfile(args.points, dtype=np.float32).reshape(-1, 4)
    pred = np.load(args.predictions)
    boxes3d = pred["boxes"].astype(np.float64)
    scores3d = pred["scores"].astype(np.float64)

    boxes2d_proj, valid = project_lidar_boxes_to_image(boxes3d, calib, image.size)
    boxes3d = boxes3d[valid]
    scores3d = scores3d[valid]
    boxes2d_proj = boxes2d_proj[valid]

    boxes2d_det, scores2d = load_2d_candidates(args.ann, args.topk_2d)
    fusion = clocs_style_fusion(scores3d, boxes3d, boxes2d_proj, boxes2d_det, scores2d)

    camera_out = args.out_dir / "camera_raw.png"
    image_out = args.out_dir / "clocs_candidates_image.png"
    bev_out = args.out_dir / "clocs_fused_bev.png"
    score_out = args.out_dir / "clocs_score_fusion.png"
    heatmap_out = args.out_dir / "clocs_pair_iou_matrix.png"
    panel_out = args.out_dir / "clocs_panel.png"
    summary_out = args.out_dir / "fusion_summary.json"

    shutil.copyfile(args.image, camera_out)
    save_image_candidates(args.image, boxes2d_det, boxes2d_proj, scores3d, scores2d, fusion, image_out)
    save_bev(points, boxes3d, scores3d, fusion["fused_scores"], bev_out)
    save_score_chart(scores3d, scores2d, fusion, score_out)
    save_feature_heatmap(fusion, heatmap_out)
    save_panel({"image": image_out, "bev": bev_out, "heatmap": heatmap_out, "scores": score_out}, panel_out)

    best_iou = fusion["ious"][np.arange(len(scores3d)), fusion["best_2d"]]
    summary = {
        "sample": "KITTI 000008",
        "num_3d_candidates": int(len(scores3d)),
        "num_2d_candidates": int(len(scores2d)),
        "source_3d_candidates": str(args.predictions.relative_to(ROOT) if args.predictions.is_relative_to(ROOT) else args.predictions),
        "source_2d_candidates": "KITTI annotation boxes used as 2D detector proxy for visualization",
        "features": ["2D IoU", "3D score", "2D score", "normalized distance"],
        "best_pair_iou": best_iou.tolist(),
        "scores_3d": scores3d.tolist(),
        "scores_2d_matched": scores2d[fusion["best_2d"]].tolist(),
        "fused_scores_proxy": fusion["fused_scores"].tolist(),
        "limitation": "No official CLOCs/SECOND/Cascade-RCNN pretrained artifacts were included locally; this is a candidate-level CLOCs-style visualization, not KITTI benchmark evaluation.",
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "run_notes.md").write_text(
        "\n".join(
            [
                "# CLOCs Reproduction Notes",
                "",
                "## Result",
                "",
                "- Generated `clocs_panel.png` as the report-ready summary image.",
                "- Visualized the CLOCs candidate-level fusion idea on KITTI demo frame `000008`.",
                "- 3D candidates come from `outputs/pointpainting_detector_300iter/predictions_000008.npz`.",
                "- 2D candidates are approximated from KITTI annotation boxes because the cloned CLOCs repo does not include Cascade-RCNN candidate files.",
                "- Pair features follow the CLOCs code path: 2D IoU, 3D candidate score, 2D candidate score, and distance to LiDAR.",
                "",
                "## Outputs",
                "",
                "- `clocs_candidates_image.png`: camera-view 2D candidates, projected 3D candidates, and best 2D/3D links.",
                "- `clocs_fused_bev.png`: BEV point cloud with 3D candidates and before/after scores.",
                "- `clocs_pair_iou_matrix.png`: 2D/3D candidate pair IoU matrix.",
                "- `clocs_score_fusion.png`: confidence scores before and after CLOCs-style fusion.",
                "- `fusion_summary.json`: numeric summary.",
                "",
                "## Command",
                "",
                "```bash",
                "/root/miniconda3/envs/motiondetection/bin/python scripts/clocs_reproduction_demo.py \\",
                "  --out-dir outputs/clocs_reproduction",
                "```",
                "",
                "## Limitation",
                "",
                "This is not an official KITTI AP reproduction. Full CLOCs evaluation requires downloaded SECOND weights, Cascade-RCNN 2D candidate files, KITTI info files, spconv v1, and a trained CLOCs fusion checkpoint.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
