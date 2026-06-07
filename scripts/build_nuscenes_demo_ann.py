#!/usr/bin/env python
"""Build a one-frame nuScenes annotation pkl for MMDetection3D demos."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_table(meta_dir: Path, name: str) -> list[dict]:
    with (meta_dir / f"{name}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def transform_matrix(translation: list[float], rotation: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix.astype(np.float32)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def main() -> None:
    args = parse_args()
    meta_dir = args.dataroot / args.version

    samples = load_table(meta_dir, "sample")
    sample_data_rows = load_table(meta_dir, "sample_data")
    sample_data = {row["token"]: row for row in sample_data_rows}
    calibrated = {
        row["token"]: row for row in load_table(meta_dir, "calibrated_sensor")
    }
    ego_poses = {row["token"]: row for row in load_table(meta_dir, "ego_pose")}
    sensors = {row["token"]: row for row in load_table(meta_dir, "sensor")}

    data_by_sample: dict[str, dict[str, str]] = {}
    for row in sample_data_rows:
        if not row.get("is_key_frame", False):
            continue
        sensor = sensors[calibrated[row["calibrated_sensor_token"]]["sensor_token"]]
        data_by_sample.setdefault(row["sample_token"], {})[sensor["channel"]] = row[
            "token"
        ]

    candidates = []
    for sample in samples:
        data = sample.get("data") or data_by_sample.get(sample["token"], {})
        if "LIDAR_TOP" in data and all(cam in data for cam in CAMERAS):
            lidar_sd = sample_data[data["LIDAR_TOP"]]
            lidar_path = args.dataroot / lidar_sd["filename"]
            if lidar_path.exists() and all(
                (args.dataroot / sample_data[data[cam]]["filename"]).exists()
                for cam in CAMERAS
            ):
                candidates.append(sample)

    if not candidates:
        raise RuntimeError("No sample with LIDAR_TOP and six cameras was found.")
    sample = candidates[args.sample_index % len(candidates)]
    data = sample.get("data") or data_by_sample[sample["token"]]

    lidar_sd = sample_data[data["LIDAR_TOP"]]
    lidar_calib = calibrated[lidar_sd["calibrated_sensor_token"]]
    lidar_pose = ego_poses[lidar_sd["ego_pose_token"]]
    lidar2ego = transform_matrix(lidar_calib["translation"], lidar_calib["rotation"])
    lidar_ego2global = transform_matrix(lidar_pose["translation"], lidar_pose["rotation"])
    lidar2global = lidar_ego2global @ lidar2ego

    images = {}
    for cam in CAMERAS:
        cam_sd = sample_data[data[cam]]
        cam_calib = calibrated[cam_sd["calibrated_sensor_token"]]
        cam_pose = ego_poses[cam_sd["ego_pose_token"]]
        cam2ego = transform_matrix(cam_calib["translation"], cam_calib["rotation"])
        cam_ego2global = transform_matrix(cam_pose["translation"], cam_pose["rotation"])
        global2cam = np.linalg.inv(cam2ego) @ np.linalg.inv(cam_ego2global)
        lidar2cam = global2cam @ lidar2global

        images[cam] = {
            "img_path": cam_sd["filename"],
            "cam2img": cam_calib["camera_intrinsic"],
            "cam2ego": cam2ego.tolist(),
            "sample_data_token": cam_sd["token"],
            "timestamp": cam_sd["timestamp"] / 1e6,
            "lidar2cam": lidar2cam.astype(np.float32).tolist(),
        }

    ann = {
        "metainfo": {"DATASET": "NuScenes"},
        "data_list": [
            {
                "sample_idx": args.sample_index,
                "token": sample["token"],
                "timestamp": sample["timestamp"] / 1e6,
                "ego2global": lidar_ego2global.tolist(),
                "images": images,
                "lidar_points": {
                    "num_pts_feats": 5,
                    "lidar_path": lidar_sd["filename"],
                    "lidar2ego": lidar2ego.tolist(),
                },
                "instances": [],
                "cam_instances": {cam: [] for cam in CAMERAS},
            }
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(ann, f)

    print(f"sample_index={args.sample_index % len(candidates)}")
    print(f"sample_token={sample['token']}")
    print(f"lidar={lidar_sd['filename']}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
