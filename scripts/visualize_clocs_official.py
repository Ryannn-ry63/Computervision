#!/usr/bin/env python
"""Visualize the locally executed official CLOCs single-frame output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=ROOT / "CLOCs" / "mini_kitti" / "training" / "image_2" / "000008.png")
    parser.add_argument("--points", type=Path, default=ROOT / "CLOCs" / "mini_kitti" / "training" / "velodyne_reduced" / "000008.bin")
    parser.add_argument("--calib", type=Path, default=ROOT / "CLOCs" / "mini_kitti" / "training" / "calib" / "000008.txt")
    parser.add_argument("--pred", type=Path, default=ROOT / "CLOCs" / "CLOCs_SecCas_pretrained" / "eval_results" / "step_30950" / "000008.txt")
    parser.add_argument("--det2d", type=Path, default=ROOT / "CLOCs" / "d2_detection_data" / "data" / "000008.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "clocs_official_000008")
    return parser.parse_args()


def read_calib(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, values = line.split(":", 1)
        mats[key] = np.fromstring(values, sep=" ", dtype=np.float64)
    return mats["P2"].reshape(3, 4), mats["R0_rect"].reshape(3, 3), mats["Tr_velo_to_cam"].reshape(3, 4)


def parse_kitti_labels(path: Path) -> list[dict[str, object]]:
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 16:
            continue
        labels.append(
            {
                "name": parts[0],
                "bbox": np.asarray([float(x) for x in parts[4:8]], dtype=np.float64),
                "dims_hwl": np.asarray([float(x) for x in parts[8:11]], dtype=np.float64),
                "loc": np.asarray([float(x) for x in parts[11:14]], dtype=np.float64),
                "ry": float(parts[14]),
                "score": float(parts[15]) if len(parts) > 15 else 0.0,
                "raw": line,
            }
        )
    return labels


def dedupe_labels(labels: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    unique = []
    for item in labels:
        bbox = tuple(np.round(item["bbox"], 2))
        loc = tuple(np.round(item["loc"], 2))
        dims = tuple(np.round(item["dims_hwl"], 2))
        key = (item["name"], bbox, loc, dims, round(float(item["ry"]), 2), round(float(item["score"]), 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def parse_2d(path: Path, max_items: int = 12) -> list[tuple[np.ndarray, float]]:
    items = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 16 or parts[0] != "Car":
            continue
        items.append((np.asarray([float(x) for x in parts[4:8]], dtype=np.float64), float(parts[15])))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:max_items]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple[int, int, int]) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1], fill=(255, 255, 255))
    draw.text(xy, text, fill=color, font=font)


def save_image(image_path: Path, det2d: list[tuple[np.ndarray, float]], labels: list[dict[str, object]], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for box, score in det2d:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(245, 158, 11, 180), width=2)
        draw_label(draw, (x1 + 2, max(0, y1 - 16)), f"2D {score:.2f}", (180, 83, 9))
    for item in labels:
        x1, y1, x2, y2 = item["bbox"].tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(6, 182, 212, 240), width=3)
        draw_label(draw, (x1 + 2, min(image.height - 18, y2 + 2)), f"CLOCs {float(item['score']):.2f}", (8, 145, 178))
    image.save(out_path)


def camera_box_to_lidar_box(label: dict[str, object], r0: np.ndarray, tr: np.ndarray) -> np.ndarray:
    loc = np.asarray(label["loc"], dtype=np.float64).reshape(1, 3)
    h, w, length = np.asarray(label["dims_hwl"], dtype=np.float64)
    r0_ext = np.eye(4)
    r0_ext[:3, :3] = r0
    tr_ext = np.eye(4)
    tr_ext[:3, :4] = tr
    loc_h = np.concatenate([loc, np.ones((1, 1))], axis=1)
    xyz_lidar = loc_h @ np.linalg.inv((r0_ext @ tr_ext).T)
    xyz_lidar = xyz_lidar[:, :3]
    xyz_lidar[:, 2] += h / 2.0
    heading = -float(label["ry"]) - np.pi / 2.0
    return np.asarray([xyz_lidar[0, 0], xyz_lidar[0, 1], xyz_lidar[0, 2], length, w, h, heading], dtype=np.float64)


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    template = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]], dtype=np.float64) / 2.0
    corners = boxes[:, None, 3:6] * template[None, :, :]
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rot = np.stack([np.stack([c, -s, c * 0], -1), np.stack([s, c, c * 0], -1), np.stack([c * 0, c * 0, c * 0 + 1], -1)], 1)
    return corners @ np.transpose(rot, (0, 2, 1)) + boxes[:, None, :3]


def save_bev(points_path: Path, labels: list[dict[str, object]], calib_path: Path, out_path: Path) -> None:
    _, r0, tr = read_calib(calib_path)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    keep = (points[:, 0] > 0) & (points[:, 0] < 70) & (np.abs(points[:, 1]) < 35) & (points[:, 2] > -3) & (points[:, 2] < 3)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    ax.scatter(points[keep, 0], points[keep, 1], c=points[keep, 2], cmap="Greys", s=0.35, linewidths=0, alpha=0.35)
    if labels:
        boxes = np.stack([camera_box_to_lidar_box(x, r0, tr) for x in labels], axis=0)
        corners = boxes_to_corners_3d(boxes)
        for item, box, box_corners in zip(labels, boxes, corners):
            xy = box_corners[[0, 1, 2, 3, 0], :2]
            ax.plot(xy[:, 0], xy[:, 1], color="#06b6d4", linewidth=2.2)
            ax.text(box[0], box[1], f"{float(item['score']):.2f}", color="#0891b2", fontsize=8)
    ax.set_xlim(0, 70)
    ax.set_ylim(-35, 35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title("Official CLOCs output in BEV")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_summary(total: int, unique: int, labels: list[dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.axis("off")
    lines = [
        "Official CLOCs checkpoint run",
        "",
        f"Raw exported predictions: {total}",
        f"Unique predictions shown: {unique}",
        "SECOND checkpoint: voxelnet-30950",
        "Fusion checkpoint: fusion_layer-11136",
        "2D candidates: official Cascade-RCNN sigmoid",
        "",
        "Note: AP evaluation was skipped for this mini-KITTI",
        "single frame because full KITTI labels are not local.",
    ]
    if labels:
        lines.insert(4, f"Top score: {float(labels[0]['score']):.4f}")
    ax.text(0.02, 0.96, "\n".join(lines), ha="left", va="top", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_panel(paths: dict[str, Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=180)
    items = [
        ("Camera + candidates", paths["image"]),
        ("BEV official output", paths["bev"]),
        ("Run summary", paths["summary"]),
        ("Original camera", paths["camera"]),
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
    raw_labels = parse_kitti_labels(args.pred)
    labels = dedupe_labels(raw_labels)
    labels.sort(key=lambda x: float(x["score"]), reverse=True)
    det2d = parse_2d(args.det2d)

    camera_out = args.out_dir / "camera_raw.png"
    image_out = args.out_dir / "official_clocs_image.png"
    bev_out = args.out_dir / "official_clocs_bev.png"
    summary_out = args.out_dir / "official_clocs_summary.png"
    panel_out = args.out_dir / "official_clocs_panel.png"
    Image.open(args.image).convert("RGB").save(camera_out)
    save_image(args.image, det2d, labels, image_out)
    save_bev(args.points, labels, args.calib, bev_out)
    save_summary(len(raw_labels), len(labels), labels, summary_out)
    save_panel({"image": image_out, "bev": bev_out, "summary": summary_out, "camera": camera_out}, panel_out)

    summary = {
        "raw_predictions": len(raw_labels),
        "unique_predictions": len(labels),
        "scores": [float(x["score"]) for x in labels],
        "prediction_file": str(args.pred),
        "note": "Visualization deduplicates identical KITTI rows exported by the compatibility run.",
    }
    (args.out_dir / "official_clocs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "run_notes.md").write_text(
        "\n".join(
            [
                "# Official CLOCs Single-frame Run",
                "",
                "Used downloaded official SECOND, CLOCs fusion, and Cascade-RCNN candidate files.",
                f"Raw prediction rows: {len(raw_labels)}.",
                f"Unique prediction rows visualized: {len(labels)}.",
                "The final KITTI AP evaluation failed because this mini-KITTI setup does not include full ground-truth labels.",
                "The exported CLOCs txt had duplicate rows under the spconv2 compatibility run, so the report image shows deduplicated predictions.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
