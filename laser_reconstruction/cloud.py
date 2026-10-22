"""Validate, write, read and summarize colored ASCII XYZ point clouds."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def validate_cloud(cloud: np.ndarray) -> np.ndarray:
    arr = np.asarray(cloud, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("Expected N rows of X Y Z R G B")
    if not np.isfinite(arr).all() or (arr[:, 3:] < 0).any() or (arr[:, 3:] > 255).any():
        raise ValueError("XYZRGB values must be finite and RGB must be between 0 and 255")
    if not np.allclose(arr[:, 3:], np.rint(arr[:, 3:]), atol=1e-9):
        raise ValueError("RGB channels must be integers")
    return arr


def write_xyz(path: str | Path, cloud: np.ndarray) -> None:
    values = validate_cloud(cloud)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for x, y, z, r, g, b in values:
            file.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")


def load_xyz(path: str | Path) -> np.ndarray:
    with Path(path).open(encoding="utf-8") as file:
        rows = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    values = np.loadtxt(rows, dtype=np.float64, ndmin=2)
    return validate_cloud(values)


def point_stats(cloud: np.ndarray) -> dict:
    values = validate_cloud(cloud)
    if not len(values):
        raise ValueError("Cloud is empty")
    return {"points": len(values), "xyz_min_mm": values[:, :3].min(0).round(3).tolist(),
            "xyz_max_mm": values[:, :3].max(0).round(3).tolist(),
            "rgb_min": values[:, 3:].min(0).astype(int).tolist(),
            "rgb_max": values[:, 3:].max(0).astype(int).tolist()}
