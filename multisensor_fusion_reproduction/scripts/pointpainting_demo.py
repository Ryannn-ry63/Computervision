#!/usr/bin/env python
"""PointPainting-style visualization on a KITTI demo sample.

This script does not train a detector. It reproduces the core data step of
PointPainting: project LiDAR points into the image, sample image semantic
scores at the projected pixels, and append those scores to every painted point.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_NAMES = ("road", "vehicle", "vegetation", "sky", "background")
CLASS_COLORS = np.array(
    [
        [115, 115, 115],
        [230, 60, 45],
        [35, 160, 75],
        [70, 135, 230],
        [245, 180, 45],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PointPainting-style painted point visualizations."
    )
    parser.add_argument("--image", required=True, type=Path, help="Camera image path.")
    parser.add_argument("--pointcloud", required=True, type=Path, help="Velodyne .bin path.")
    parser.add_argument(
        "--calib",
        required=True,
        type=Path,
        help="Calibration path. Supports MMDetection3D demo .pkl or 4x4 lidar2img .txt.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--point-size", type=float, default=2.2, help="Projection point size.")
    return parser.parse_args()


def load_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if calib_path.suffix == ".pkl":
        with calib_path.open("rb") as f:
            data = pickle.load(f)
        sample = data["data_list"][0]
        cam2 = sample["images"]["CAM2"]
        lidar2img = np.asarray(cam2["lidar2img"], dtype=np.float64)
        lidar2cam = np.asarray(cam2.get("lidar2cam"), dtype=np.float64)
        return lidar2img, lidar2cam

    values = np.loadtxt(calib_path, dtype=np.float64)
    if values.size == 16:
        return values.reshape(4, 4), None
    if values.size == 12:
        lidar2img = np.eye(4, dtype=np.float64)
        lidar2img[:3, :4] = values.reshape(3, 4)
        return lidar2img, None
    raise ValueError(f"Unsupported calibration format: {calib_path}")


def project_points(
    points_xyz: np.ndarray,
    lidar2img: np.ndarray,
    lidar2cam: np.ndarray | None,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width, height = image_size
    points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1))], axis=1)
    img_h = points_h @ lidar2img.T

    if lidar2cam is not None:
        cam_h = points_h @ lidar2cam.T
        depth = cam_h[:, 2]
    else:
        depth = img_h[:, 2]

    eps = 1e-6
    u = img_h[:, 0] / np.maximum(img_h[:, 2], eps)
    v = img_h[:, 1] / np.maximum(img_h[:, 2], eps)
    keep = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.flatnonzero(keep), u[keep], v[keep], depth[keep]


def pseudo_semantic_scores(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create lightweight image semantic scores for a no-download demo.

    PointPainting usually consumes a trained 2D semantic segmenter. For a compact
    course reproduction, these deterministic image cues stand in for that model
    and keep the rest of the pipeline identical.
    """

    rgb = image.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    value = rgb.max(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    saturation = chroma / np.maximum(value, 1e-6)
    gray = rgb.mean(axis=2)

    height, width = image.shape[:2]
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    lower = np.clip((yy - 0.38) / 0.62, 0.0, 1.0)
    upper = np.clip((0.56 - yy) / 0.56, 0.0, 1.0)

    gy, gx = np.gradient(gray)
    edge = np.clip(np.sqrt(gx * gx + gy * gy) * 5.0, 0.0, 1.0)

    road = lower * (1.0 - saturation) * np.clip(1.0 - np.abs(gray - 0.42) * 2.2, 0.0, 1.0)
    sky = upper * np.clip(b - np.maximum(r, g) + 0.18, 0.0, 1.0) * np.clip(value + 0.15, 0.0, 1.0)
    vegetation = np.clip(g - np.maximum(r, b) + 0.12, 0.0, 1.0) * saturation
    vehicle = lower * edge * np.clip(0.75 - value, 0.0, 1.0) * (1.0 - vegetation)
    background = np.broadcast_to(0.18 + 0.20 * (1.0 - lower * upper), gray.shape)

    scores = np.stack([road, vehicle, vegetation, sky, background], axis=2)
    scores += 1e-4
    scores /= scores.sum(axis=2, keepdims=True)
    labels = scores.argmax(axis=2).astype(np.uint8)
    return scores.astype(np.float32), labels


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[labels]


def save_semantic_overlay(image: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    mask = colorize_labels(labels).astype(np.float32)
    overlay = (0.55 * image.astype(np.float32) + 0.45 * mask).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def save_projection(
    image: np.ndarray,
    out_path: Path,
    u: np.ndarray,
    v: np.ndarray,
    point_labels: np.ndarray,
    point_size: float,
) -> None:
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 120, height / 120), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.scatter(
        u,
        v,
        c=CLASS_COLORS[point_labels] / 255.0,
        s=point_size,
        linewidths=0,
        alpha=0.96,
    )
    ax.axis("off")
    add_legend(ax)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_painted_bev(points_xyz: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    keep = (x > 0) & (x < 80) & (np.abs(y) < 40) & (z > -3) & (z < 3)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.scatter(
        y[keep],
        x[keep],
        c=CLASS_COLORS[labels[keep]] / 255.0,
        s=0.45,
        linewidths=0,
        alpha=0.9,
    )
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Lateral y (m)")
    ax.set_ylabel("Forward x (m)")
    ax.set_title("Painted LiDAR BEV")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    add_legend(ax)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def add_legend(ax: plt.Axes) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_COLORS[i] / 255.0,
            markersize=5,
            label=CLASS_NAMES[i],
        )
        for i in range(len(CLASS_NAMES))
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.75)


def save_panel(paths: dict[str, Path], out_path: Path) -> None:
    titles = [
        ("Camera", paths["camera"]),
        ("2D semantic cues", paths["semantic"]),
        ("Painted projection", paths["projection"]),
        ("Painted BEV", paths["bev"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=180)
    for ax, (title, path) in zip(axes.ravel(), titles):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_notes(
    out_dir: Path,
    image_path: Path,
    pointcloud_path: Path,
    calib_path: Path,
    num_points: int,
    num_painted: int,
) -> None:
    notes = out_dir / "run_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# PointPainting Reproduction Notes",
                "",
                "## Result",
                "",
                "- Generated `camera_raw.png`, `semantic_mask_overlay.png`, `pointpainting_projection.png`, `painted_bev_semantic.png`, `pointpainting_panel.png`, and `painted_points.npy`.",
                f"- Painted {num_painted} projected LiDAR points out of {num_points} total points.",
                "- This is a lightweight PointPainting-style reproduction: the core projection and point-feature augmentation are reproduced; the 2D semantic input is a deterministic local proxy so the demo does not require downloading a segmentation checkpoint.",
                "",
                "## Command",
                "",
                "```bash",
                "python scripts/pointpainting_demo.py \\",
                f"  --image {image_path} \\",
                f"  --pointcloud {pointcloud_path} \\",
                f"  --calib {calib_path} \\",
                f"  --out-dir {out_dir}",
                "```",
                "",
                "## PPT Explanation",
                "",
                "PointPainting first projects LiDAR points into the camera image with calibration matrices, samples per-pixel semantic scores from the image branch, and appends those scores to the original point features. The painted points can then be consumed by a standard LiDAR detector. This visualization shows the intermediate painted point cloud rather than a trained detector output.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(args.image).convert("RGB"))
    points = np.fromfile(args.pointcloud, dtype=np.float32).reshape(-1, 4)
    lidar2img, lidar2cam = load_calibration(args.calib)
    point_indices, u, v, _ = project_points(points[:, :3], lidar2img, lidar2cam, (image.shape[1], image.shape[0]))

    scores, labels_2d = pseudo_semantic_scores(image)
    ui = np.clip(np.rint(u).astype(np.int64), 0, image.shape[1] - 1)
    vi = np.clip(np.rint(v).astype(np.int64), 0, image.shape[0] - 1)
    painted_scores = scores[vi, ui]
    point_labels = painted_scores.argmax(axis=1).astype(np.uint8)

    painted_points = np.concatenate(
        [
            points[point_indices],
            painted_scores,
            point_labels[:, None].astype(np.float32),
        ],
        axis=1,
    )
    np.save(args.out_dir / "painted_points.npy", painted_points.astype(np.float32))

    camera_out = args.out_dir / "camera_raw.png"
    semantic_out = args.out_dir / "semantic_mask_overlay.png"
    projection_out = args.out_dir / "pointpainting_projection.png"
    bev_out = args.out_dir / "painted_bev_semantic.png"
    panel_out = args.out_dir / "pointpainting_panel.png"

    shutil.copyfile(args.image, camera_out)
    save_semantic_overlay(image, labels_2d, semantic_out)
    save_projection(image, projection_out, u, v, point_labels, args.point_size)
    save_painted_bev(points[point_indices, :3], point_labels, bev_out)
    save_panel(
        {
            "camera": camera_out,
            "semantic": semantic_out,
            "projection": projection_out,
            "bev": bev_out,
        },
        panel_out,
    )
    write_notes(args.out_dir, args.image, args.pointcloud, args.calib, len(points), len(point_indices))

    print(f"Painted {len(point_indices)} projected LiDAR points out of {len(points)} total points.")
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
