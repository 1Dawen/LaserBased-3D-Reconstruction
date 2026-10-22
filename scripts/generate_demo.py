import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laser_reconstruction.synthetic import generate_demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate fictional paired scanning frames")
    ap.add_argument("--output", type=Path, default=Path("data/demo"))
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()
    print("Synthetic calibration:", generate_demo(args.output, frames=args.frames))
