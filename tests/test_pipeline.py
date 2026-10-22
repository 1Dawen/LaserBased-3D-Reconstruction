import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from laser_reconstruction.pipeline import calibration_from_json, reconstruct_sequence
from laser_reconstruction.synthetic import generate_demo

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_sequence_end_to_end(tmp_path):
    calib = generate_demo(tmp_path / "demo", frames=8)
    matrix = calibration_from_json(calib)
    cloud, summary = reconstruct_sequence(calib.parent / "colour", calib.parent / "laser",
                                          matrix, frames=8, angle_step_deg=45, strict=True)
    assert summary["processed_frames"] == 8
    assert summary["skipped_frames"] == []
    assert 200 < summary["points"] == len(cloud)
    assert np.isfinite(cloud).all()
    assert (cloud[:, 3:] >= 0).all() and (cloud[:, 3:] <= 255).all()


def test_nonexistent_images_raise_in_strict_mode(tmp_path):
    calib = generate_demo(tmp_path / "demo", frames=4)
    matrix = calibration_from_json(calib)
    with pytest.raises(FileNotFoundError):
        reconstruct_sequence(calib.parent / "colour", calib.parent / "laser",
                             matrix, frames=5, strict=True)


def test_missing_frames_reported_if_not_strict(tmp_path):
    calib = generate_demo(tmp_path / "demo", frames=4)
    matrix = calibration_from_json(calib)
    _, summary = reconstruct_sequence(calib.parent / "colour", calib.parent / "laser",
                                      matrix, frames=6, strict=False)
    assert summary["processed_frames"] == 4
    assert summary["skipped_frames"] == [4, 5]


def test_calibration_requires_correspondences(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"src_pixels": [[0, 0]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="dst_plane_mm"):
        calibration_from_json(path)


def test_cli_smoke_and_image_output(tmp_path):
    demo = tmp_path / "demo"
    calib = generate_demo(demo, frames=4)
    dest = tmp_path / "out" / "scan.xyz"
    subprocess.run([sys.executable, str(ROOT / "reconstruct.py"),
                    "--background-dir", str(demo / "colour"),
                    "--laser-dir", str(demo / "laser"),
                    "--calibration", str(calib), "--frames", "4", "--angle-step", "90",
                    "--output", str(dest), "--strict"], check=True, cwd=ROOT,
                   capture_output=True, text=True)
    assert dest.exists() and dest.with_suffix(".json").exists()
    from laser_reconstruction.cloud import load_xyz
    assert len(load_xyz(dest)) > 80
