#!/usr/bin/env python
"""Create a BEVFusion-style same-frame visualization for KITTI 000008."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("mmdetection3d/demo/data/kitti/000008.png"))
    parser.add_argument("--pointcloud", type=Path, default=Path("mmdetection3d/demo/data/kitti/000008.bin"))
    parser.add_argument("--calib-pkl", type=Path, default=Path("mmdetection3d/demo/data/kitti/000008.pkl"))
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("outputs/pointpainting_detector_300iter/predictions_000008.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fusion_reproduction"))
    return parser.parse_args()


def load_lidar2img(calib_pkl: Path) -> np.ndarray:
    with calib_pkl.open("rb") as f:
        data = pickle.load(f)
    return np.asarray(data["data_list"][0]["images"]["CAM2"]["lidar2img"], dtype=np.float32)


def corners_lidar(box: np.ndarray) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = box.astype(np.float32)
    local = np.array(
        [
            [dx / 2, dy / 2, dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [-dx / 2, dy / 2, dz / 2],
            [dx / 2, dy / 2, -dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return local @ rot.T + np.array([x, y, z], dtype=np.float32)


def project(points_xyz: np.ndarray, lidar2img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points_xyz, np.ones((len(points_xyz), 1), dtype=np.float32)], axis=1)
    img_h = points_h @ lidar2img.T
    uv = img_h[:, :2] / np.maximum(img_h[:, 2:3], 1e-6)
    return uv, img_h[:, 2]


def draw_camera_boxes(image_path: Path, lidar2img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    palette = [(255, 80, 64, 230), (255, 170, 0, 230), (0, 188, 212, 230), (76, 175, 80, 230)]
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        corners = corners_lidar(box)
        uv, depth = project(corners, lidar2img)
        if np.count_nonzero(depth > 0) < 4:
            continue
        color = palette[idx % len(palette)]
        for a, b in BOX_EDGES:
            if depth[a] <= 0 or depth[b] <= 0:
                continue
            xa, ya = uv[a]
            xb, yb = uv[b]
            draw.line([(float(xa), float(ya)), (float(xb), float(yb))], fill=color, width=3)
        x1, y1 = uv.min(axis=0)
        draw.text((float(x1), max(0.0, float(y1) - 14)), f"fused {score:.2f}", fill=color)
    image.save(out_path)


def save_projection(image_path: Path, points: np.ndarray, lidar2img: np.ndarray, out_path: Path) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    uv, depth = project(points[:, :3], lidar2img)
    h, w = image.shape[:2]
    keep = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    fig = plt.figure(figsize=(12, 4), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.scatter(
        uv[keep, 0],
        uv[keep, 1],
        c=np.clip(depth[keep], 0, 80),
        s=1.4,
        cmap="turbo",
        vmin=0,
        vmax=80,
        linewidths=0,
        alpha=0.9,
    )
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def draw_bev(points: np.ndarray, boxes: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    keep = (points[:, 0] > 0) & (points[:, 0] < 45) & (np.abs(points[:, 1]) < 20) & (points[:, 2] > -3) & (points[:, 2] < 3)
    fig, ax = plt.subplots(figsize=(5.2, 6.0), dpi=180)
    ax.scatter(points[keep, 1], points[keep, 0], c=points[keep, 2], s=0.35, cmap="viridis", linewidths=0, alpha=0.85)
    palette = ["#ff5040", "#ffaa00", "#00bcd4", "#4caf50"]
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        corners = corners_lidar(box)[:4]
        poly = np.vstack([corners[:, [1, 0]], corners[0, [1, 0]]])
        ax.plot(poly[:, 0], poly[:, 1], color=palette[idx % len(palette)], linewidth=2.0)
        ax.text(float(box[1]), float(box[0]), f"{score:.2f}", color=palette[idx % len(palette)], fontsize=7)
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lateral y (m)")
    ax.set_ylabel("forward x (m)")
    ax.set_title("BEV fused detections on KITTI 000008")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_panel(camera_path: Path, projection_path: Path, bev_path: Path, out_path: Path) -> None:
    items = [
        (camera_path, "Camera view: 3D fused boxes"),
        (projection_path, "LiDAR projected to image"),
        (bev_path, "BEV space detections"),
    ]
    width, height, title_h = 900, 530, 42
    panel = Image.new("RGB", (width * 3, height + title_h), "white")
    draw = ImageDraw.Draw(panel)
    for i, (path, title) in enumerate(items):
        im = Image.open(path).convert("RGB")
        im.thumbnail((width - 24, height - 24), Image.Resampling.LANCZOS)
        x = i * width + (width - im.width) // 2
        y = title_h + (height - im.height) // 2
        panel.paste(im, (x, y))
        draw.text((i * width + 24, 12), title, fill=(20, 20, 20))
        if i:
            draw.line((i * width, 0, i * width, height + title_h), fill=(220, 220, 220), width=2)
    panel.save(out_path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lidar2img = load_lidar2img(args.calib_pkl)
    points = np.fromfile(args.pointcloud, dtype=np.float32).reshape(-1, 4)
    pred = np.load(args.pred)
    boxes = pred["boxes"]
    scores = pred["scores"]

    camera_out = args.out_dir / "bevfusion_kitti_000008_camera.png"
    projection_out = args.out_dir / "bevfusion_kitti_000008_projection.png"
    bev_out = args.out_dir / "bevfusion_kitti_000008_bev.png"
    panel_out = args.out_dir / "bevfusion_kitti_000008_style.png"

    draw_camera_boxes(args.image, lidar2img, boxes, scores, camera_out)
    save_projection(args.image, points, lidar2img, projection_out)
    draw_bev(points, boxes, scores, bev_out)
    make_panel(camera_out, projection_out, bev_out, panel_out)

    print(f"Saved: {panel_out}")


if __name__ == "__main__":
    main()
