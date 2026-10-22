import numpy as np
import pytest

from laser_reconstruction.cloud import load_xyz, point_stats, validate_cloud, write_xyz


def test_roundtrip_xyzrgb_file(tmp_path):
    original = np.array([[1.5, 2.25, 3.125, 255, 3, 8], [0, -1, 4, 0, 45, 245]], dtype=float)
    out = tmp_path / "a" / "scan.xyz"
    write_xyz(out, original)
    np.testing.assert_allclose(load_xyz(out), original)
    assert point_stats(original)["points"] == 2


def test_rejects_invalid_point_color():
    with pytest.raises(ValueError, match="RGB"):
        validate_cloud(np.array([[0, 0, 0, 300, 20, 10]]))


def test_rejects_non_integer_rgb():
    with pytest.raises(ValueError, match="integers"):
        validate_cloud(np.array([[0, 0, 0, 20.5, 20, 10]]))


def test_empty_cloud_is_valid_file(tmp_path):
    path = tmp_path / "empty.xyz"
    write_xyz(path, np.empty((0, 6)))
    assert load_xyz(path).shape == (0, 6)
    with pytest.raises(ValueError, match="empty"):
        point_stats(load_xyz(path))
