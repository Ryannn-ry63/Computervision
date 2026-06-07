#!/usr/bin/env python
"""Visualize LiDAR-camera alignment for a KITTI-style sample."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project KITTI LiDAR points into the camera image and draw a BEV point cloud."
    )
    parser.add_argument("--image", required=True, type=Path, help="Camera image path.")
    parser.add_argument("--pointcloud", required=True, type=Path, help="Velodyne .bin path.")
    parser.add_argument(
        "--calib",
        required=True,
        type=Path,
        help="Calibration path. Supports mmdetection3d demo .pkl or 4x4 lidar2img .txt.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Depth color range in meters.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Projection point size.")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return u[keep], v[keep], depth[keep]


def save_projection(
    image_path: Path,
    out_path: Path,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    max_depth: float,
    point_size: float,
) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]

    fig = plt.figure(figsize=(width / 120, height / 120), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    sc = ax.scatter(
        u,
        v,
        c=np.clip(depth, 0, max_depth),
        s=point_size,
        cmap="turbo",
        vmin=0,
        vmax=max_depth,
        linewidths=0,
        alpha=0.95,
    )
    ax.axis("off")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Depth (m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_bev(points: np.ndarray, out_path: Path) -> None:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    keep = (x > 0) & (x < 80) & (np.abs(y) < 40) & (z > -3) & (z < 3)
    x = x[keep]
    y = y[keep]
    z = z[keep]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    sc = ax.scatter(y, x, c=z, s=0.35, cmap="viridis", linewidths=0, alpha=0.9)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Lateral y (m)")
    ax.set_ylabel("Forward x (m)")
    ax.set_title("LiDAR BEV point cloud")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Height z (m)")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    points = np.fromfile(args.pointcloud, dtype=np.float32).reshape(-1, 4)
    lidar2img, lidar2cam = load_calibration(args.calib)
    u, v, depth = project_points(points[:, :3], lidar2img, lidar2cam, image.size)

    shutil.copyfile(args.image, args.out_dir / "camera_raw.png")
    save_projection(
        args.image,
        args.out_dir / "lidar_projection_depth.png",
        u,
        v,
        depth,
        args.max_depth,
        args.point_size,
    )
    save_bev(points[:, :3], args.out_dir / "bev_pointcloud.png")

    notes = args.out_dir / "run_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# Fusion Reproduction Notes",
                "",
                "## Result",
                "",
                "- Generated `camera_raw.png`, `lidar_projection_depth.png`, and `bev_pointcloud.png`.",
                "- Data source: MMDetection3D built-in KITTI demo sample `000008`.",
                "- Method: projected Velodyne LiDAR points into the CAM2 image using the sample calibration matrix; colors encode depth.",
                "",
                "## Command",
                "",
                "```bash",
                "python scripts/kitti_lidar_projection.py \\",
                "  --image mmdetection3d/demo/data/kitti/000008.png \\",
                "  --pointcloud mmdetection3d/demo/data/kitti/000008.bin \\",
                "  --calib mmdetection3d/demo/data/kitti/000008.pkl \\",
                "  --out-dir outputs/fusion_reproduction",
                "```",
                "",
                "## Environment Observation",
                "",
                "- `motiondetection` has PyTorch/CUDA available.",
                "- The cloned MMDetection3D tree is a newer MMEngine-based version, while this environment currently has older OpenMMLab packages, so BEVFusion demo is left as an optional follow-up after dependency alignment.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Projected {len(depth)} LiDAR points into the image.")
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
