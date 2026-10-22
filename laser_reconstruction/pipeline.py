"""Paired-frame reconstruction and JSON calibration loading."""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np

from .core import DetectionConfig, homography_from_points, reconstruct_frame
from .cloud import write_xyz


def calibration_from_json(path: str | Path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("src_pixels", "dst_plane_mm"):
        if key not in config:
            raise ValueError(f"Missing calibration key: {key}")
    return homography_from_points(config["src_pixels"], config["dst_plane_mm"])


def reconstruct_sequence(background_dir: str | Path, laser_dir: str | Path,
                         matrix: np.ndarray, *, frames: int = 360,
                         start_index: int = 0, angle_step_deg: float = 1.0,
                         cfg: DetectionConfig | None = None, strict: bool = False):
    """Process sequential 0000.jpg, 0001.jpg... pairs.

    Missing/unreadable pairs are skipped by default to match the archived script;
    use strict=True to fail rather than silently skipping frames.
    """
    if frames < 1 or start_index < 0 or not np.isfinite(angle_step_deg):
        raise ValueError("frames >= 1, start_index >= 0 and finite angle_step_deg required")
    background_dir, laser_dir = Path(background_dir), Path(laser_dir)
    collected = []
    summary = {"requested_frames": frames, "processed_frames": 0, "skipped_frames": [],
               "no_laser_frames": [], "points": 0}
    for index in range(start_index, start_index + frames):
        name = f"{index:04d}.jpg"
        background_path, laser_path = background_dir / name, laser_dir / name
        if not background_path.is_file() or not laser_path.is_file():
            if strict:
                raise FileNotFoundError(f"Missing paired images for {name}")
            summary["skipped_frames"].append(index)
            continue
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        laser_on = cv2.imread(str(laser_path), cv2.IMREAD_COLOR)
        if background is None or laser_on is None:
            if strict:
                raise ValueError(f"Unreadable paired images for {name}")
            summary["skipped_frames"].append(index)
            continue
        if background.shape != laser_on.shape:
            if strict:
                raise ValueError(f"Mismatched paired images for {name}")
            summary["skipped_frames"].append(index)
            continue
        frame = reconstruct_frame(background, laser_on, matrix, index * angle_step_deg, cfg)
        summary["processed_frames"] += 1
        if not len(frame):
            summary["no_laser_frames"].append(index)
            continue
        collected.append(frame)
        summary["points"] += len(frame)
    cloud = np.concatenate(collected, axis=0) if collected else np.empty((0, 6), dtype=np.float64)
    return cloud, summary
