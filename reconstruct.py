"""Reconstruct colored 3D points from numbered background/laser image pairs."""
import argparse
import json
from pathlib import Path

from laser_reconstruction.cloud import write_xyz
from laser_reconstruction.core import DetectionConfig
from laser_reconstruction.pipeline import calibration_from_json, reconstruct_sequence


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--background-dir", type=Path, required=True)
    ap.add_argument("--laser-dir", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/reconstructed.xyz"))
    ap.add_argument("--frames", type=int, default=360)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--angle-step", type=float, default=1.0)
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--median-kernel", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--channel", choices=["grayscale", "green"], default="grayscale")
    ap.add_argument("--strict", action="store_true", help="Fail on missing or unreadable image pair")
    args = ap.parse_args()
    calibration = calibration_from_json(args.calibration)
    cfg = DetectionConfig(threshold=args.threshold, median_kernel=args.median_kernel,
                          sigma=args.sigma, channel=args.channel)
    cloud, summary = reconstruct_sequence(args.background_dir, args.laser_dir, calibration,
                        frames=args.frames, start_index=args.start_index,
                        angle_step_deg=args.angle_step, cfg=cfg, strict=args.strict)
    if not len(cloud):
        raise SystemExit("No 3D points found. Check paired images, calibration and threshold.")
    write_xyz(args.output, cloud)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **summary}, indent=2))


if __name__ == "__main__":
    main()
