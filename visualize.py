"""Render a deterministic static preview of an XYZRGB cloud (headless)."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from laser_reconstruction.cloud import load_xyz, point_stats


def render_preview(cloud, destination: str | Path, *, points: int = 16000,
                   seed: int = 40, azimuth: float = -55, elevation: float = 16, title: str = "Point cloud"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if points < 1:
        raise ValueError("points must be positive")
    data = np.asarray(cloud)
    if not len(data):
        raise ValueError("Cannot render empty point cloud")
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(data), size=points, replace=False) if len(data) > points
           else np.arange(len(data)))
    sample = data[idx]
    xyz = sample[:, :3]
    # Display authentic archived color. Brighten only the preview to make the dark scan visible.
    rgb = np.clip((sample[:, 3:] / 255.0) ** 0.7, 0, 1)
    fig = plt.figure(figsize=(8, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=rgb, s=0.55,
               edgecolors="none", rasterized=True, alpha=0.85)
    ax.set(xlabel="X (mm)", ylabel="Z (mm)", zlabel="Y (mm)")
    ax.view_init(elev=elevation, azim=azimuth)
    ext = np.ptp(xyz, axis=0)
    ax.set_box_aspect((max(ext[0], 1), max(ext[2], 1), max(ext[1], 1)))
    ax.set_title(f"{title} | {len(data):,} points", fontsize=13, pad=8)
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/point_cloud.png"))
    ap.add_argument("--points", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--title", type=str, default="Point cloud")
    ap.add_argument("--azimuth", type=float, default=-55)
    ap.add_argument("--elevation", type=float, default=16)
    args = ap.parse_args()
    cloud = load_xyz(args.input)
    print(point_stats(cloud))
    print("Preview:", render_preview(cloud, args.output, points=args.points, seed=args.seed,
                                     azimuth=args.azimuth, elevation=args.elevation, title=args.title))


if __name__ == "__main__":
    main()
