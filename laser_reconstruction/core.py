"""Image subtraction, row-wise laser detection, planar mapping and 3D rotation.

The processing stages follow the archived academic notebook while exposing
parameters and validation. Images and calibration points are user supplied.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    threshold: int = 10
    median_kernel: int = 5
    sigma: float = 3.0
    channel: str = "grayscale"  # reproduces source notebook; optional 'green'

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if self.median_kernel < 1 or self.median_kernel % 2 == 0:
            raise ValueError("median_kernel must be a positive odd number")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.channel not in {"grayscale", "green"}:
            raise ValueError("channel must be 'grayscale' or 'green'")


def validate_images(background: np.ndarray, laser_on: np.ndarray) -> None:
    if not isinstance(background, np.ndarray) or not isinstance(laser_on, np.ndarray):
        raise ValueError("Images must be NumPy arrays")
    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError("Expected 3-channel BGR images")
    if background.shape != laser_on.shape:
        raise ValueError("Background and laser-on images must have the same dimensions")
    if background.dtype != np.uint8 or laser_on.dtype != np.uint8:
        raise ValueError("Images must be uint8")


def laser_mask(background: np.ndarray, laser_on: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Subtract paired frames and identify changed pixels.

    'grayscale' follows the original notebook; 'green' uses only the green
    channel's positive increase to reject many non-green lighting changes.
    """
    validate_images(background, laser_on)
    if cfg.channel == "grayscale":
        diff = cv2.absdiff(laser_on, background)
        intensity = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        intensity = np.clip(
            laser_on[..., 1].astype(np.int16) - background[..., 1].astype(np.int16),
            0, 255,
        ).astype(np.uint8)
    _, mask = cv2.threshold(intensity, cfg.threshold, 255, cv2.THRESH_BINARY)
    if cfg.median_kernel > 1:
        mask = cv2.medianBlur(mask, cfg.median_kernel)
    return mask


def extract_stripe(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Return one (row, column) coordinate per detected scan line.

    For each row, compute a Gaussian-weighted mean centered on the mean
    illuminated column, preserving the original notebook's method.
    """
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("mask must be a 2D uint8 array")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rows = np.flatnonzero(np.any(mask > 0, axis=1))
    result = np.empty((len(rows), 2), dtype=np.int32)
    for i, row in enumerate(rows):
        cols = np.flatnonzero(mask[row] > 0).astype(np.float64)
        weights = np.exp(-0.5 * ((cols - cols.mean()) / sigma) ** 2)
        center = int(np.rint(np.dot(cols, weights) / weights.sum()))
        result[i] = (int(row), center)
    return result


def homography_from_points(src_pixels, dst_plane_mm) -> np.ndarray:
    """Fit a pixel -> millimeter homography from at least 4 point pairs."""
    src = np.asarray(src_pixels, dtype=np.float64)
    dst = np.asarray(dst_plane_mm, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("Both calibration arrays must contain matching (N, 2) points (N >= 4)")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Calibration points must be finite")
    if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2 or np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2:
        raise ValueError("Calibration coordinates must not be collinear")
    matrix, _ = cv2.findHomography(src, dst, method=0)
    if matrix is None or not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("Homography cannot be computed from supplied calibration points")
    mapped = map_to_plane(src, matrix)
    if not np.allclose(mapped, dst, atol=1e-3):
        raise ValueError("Calibration points result in inconsistent projection")
    return matrix


def map_to_plane(pixel_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Homography must be 3x3")
    if len(pixel_xy) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(pixel_xy.reshape(-1, 1, 2), matrix).reshape(-1, 2).astype(np.float64)


def rotate_plane(plane_xy_mm: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate (X_plane, Y_plane, Z=0) about the Y axis, as in original code."""
    plane = np.asarray(plane_xy_mm, dtype=np.float64).reshape(-1, 2)
    theta = np.deg2rad(angle_deg)
    return np.column_stack((plane[:, 0] * np.cos(theta), plane[:, 1], plane[:, 0] * np.sin(theta)))


def reconstruct_frame(background: np.ndarray, laser_on: np.ndarray, matrix: np.ndarray,
                      angle_deg: float, cfg: DetectionConfig | None = None) -> np.ndarray:
    """Convert one paired frame into an (N, 6) XYZRGB point cloud.

    Physical units are those defined by dst_plane_mm in the calibration file.
    """
    cfg = cfg or DetectionConfig()
    mask = laser_mask(background, laser_on, cfg)
    row_col = extract_stripe(mask, cfg.sigma)
    if len(row_col) == 0:
        return np.empty((0, 6), dtype=np.float64)
    pixel_xy = row_col[:, [1, 0]].astype(np.float64)
    plane = map_to_plane(pixel_xy, matrix)
    xyz = rotate_plane(plane, angle_deg)
    bgr = background[row_col[:, 0], row_col[:, 1]].astype(np.float64)
    rgb = bgr[:, ::-1]
    return np.column_stack((xyz, rgb))
