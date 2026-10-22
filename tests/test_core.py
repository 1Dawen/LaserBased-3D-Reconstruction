import numpy as np
import pytest

from laser_reconstruction.core import (
    DetectionConfig, extract_stripe, homography_from_points, laser_mask,
    map_to_plane, reconstruct_frame, rotate_plane,
)


def test_config_rejects_invalid_threshold_and_kernel():
    with pytest.raises(ValueError):
        DetectionConfig(threshold=256)
    with pytest.raises(ValueError):
        DetectionConfig(median_kernel=4)
    with pytest.raises(ValueError):
        DetectionConfig(sigma=0)


def test_mask_detects_added_green_stripe_without_background_noise():
    off = np.full((22, 24, 3), (15, 20, 30), dtype=np.uint8)
    on = off.copy()
    on[3:19, 9:14, :] = (0, 255, 0)
    gray = laser_mask(off, on, DetectionConfig(threshold=10, median_kernel=1))
    green = laser_mask(off, on, DetectionConfig(threshold=10, median_kernel=1, channel="green"))
    assert gray[10, 11] == 255
    assert green[10, 11] == 255
    assert gray[0, 0] == 0
    assert green[0, 0] == 0


def test_row_center_gaussian_weighting():
    mask = np.zeros((6, 30), dtype=np.uint8)
    mask[1, 10:15] = 255
    mask[4, 16:21] = 255
    assert extract_stripe(mask).tolist() == [[1, 12], [4, 18]]


def test_empty_mask_returns_empty_positions():
    assert extract_stripe(np.zeros((8, 9), dtype=np.uint8)).shape == (0, 2)


def test_calibration_exact_rectangle():
    src = [[0, 0], [10, 0], [10, 20], [0, 20]]
    dst = [[-5, 30], [5, 30], [5, 0], [-5, 0]]
    matrix = homography_from_points(src, dst)
    mapped = map_to_plane(np.array([[5, 10]]), matrix)
    np.testing.assert_allclose(mapped, [[0, 15]], atol=1e-6)


def test_rejects_collinear_calibration():
    with pytest.raises(ValueError, match="collinear"):
        homography_from_points([[1, 1], [2, 2], [3, 3], [4, 4]],
                               [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_world_rotation_90_degrees():
    points = rotate_plane([[10, 12], [-10, 20]], 90)
    np.testing.assert_allclose(points, [[0, 12, 10], [0, 20, -10]], atol=1e-9)


def test_frame_maps_positions_and_samples_background_rgb():
    off = np.zeros((12, 20, 3), dtype=np.uint8)
    off[..., :] = [10, 20, 30]  # BGR
    on = off.copy()
    on[3:9, 7:12] = [0, 250, 0]
    matrix = homography_from_points([[0, 0], [19, 0], [19, 11], [0, 11]],
                                    [[0, 11], [19, 11], [19, 0], [0, 0]])
    cloud = reconstruct_frame(off, on, matrix, 0, DetectionConfig(median_kernel=1))
    assert len(cloud) == 6
    np.testing.assert_allclose(cloud[:, 0], 9.0, atol=1e-6)
    np.testing.assert_allclose(cloud[:, 2], 0, atol=1e-6)
    assert (cloud[:, 3:] == [30, 20, 10]).all()


def test_identical_frames_produce_no_cloud():
    off = np.ones((10, 10, 3), dtype=np.uint8)
    matrix = np.eye(3)
    cloud = reconstruct_frame(off, off.copy(), matrix, 0)
    assert cloud.shape == (0, 6)


def test_rejects_mismatched_frame_dimensions():
    with pytest.raises(ValueError, match="same dimensions"):
        laser_mask(np.zeros((9, 9, 3), dtype=np.uint8),
                   np.zeros((9, 8, 3), dtype=np.uint8), DetectionConfig())
