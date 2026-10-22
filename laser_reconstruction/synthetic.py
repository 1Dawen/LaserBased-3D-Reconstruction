"""Generate artificial paired images for a reproducible software smoke test.

This is NOT a re-creation of the unavailable physical laser-scanning experiment.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np


def generate_demo(directory: str | Path, *, frames: int = 24, height: int = 128,
                  width: int = 192) -> Path:
    if frames < 4 or height < 50 or width < 50:
        raise ValueError("frames >= 4, height >= 50, width >= 50 required")
    root = Path(directory)
    background_dir, laser_dir = root / "colour", root / "laser"
    background_dir.mkdir(parents=True, exist_ok=True)
    laser_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(40)
    yy, xx = np.mgrid[:height, :width]
    # Stable synthetic object colors; generated solely for pipeline testing.
    background = np.stack([
        35 + xx * 0.15 + yy * 0.04,
        45 + yy * 0.23 + xx * 0.08,
        65 + xx * 0.27 + yy * 0.10,
    ], axis=-1).clip(0, 255).astype(np.uint8)
    background += rng.integers(0, 3, size=(height, width, 3), dtype=np.uint8)
    for frame in range(frames):
        on = background.copy()
        for row in range(12, height - 12):
            # A curved stripe with a slow periodic displacement across frames.
            column = int(round(width * 0.73 + 25 * np.sin(row / 20) + 10 * np.cos(2 * np.pi * frame / frames)))
            cv2.line(on, (column - 2, row), (column + 2, row), (0, 255, 0), 1)
        cv2.imwrite(str(background_dir / f"{frame:04d}.jpg"), background, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(str(laser_dir / f"{frame:04d}.jpg"), on, [cv2.IMWRITE_JPEG_QUALITY, 97])
    config = {
        "description": "Synthetic pixel-to-mm rectangle used only for automated tests",
        "units": "millimeters",
        "src_pixels": [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        "dst_plane_mm": [[-20, 70], [20, 70], [20, 0], [-20, 0]],
    }
    calib = root / "calibration.json"
    calib.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return calib
