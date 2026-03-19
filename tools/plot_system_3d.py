#!/usr/bin/env python3
"""
plot_system_3d.py

Visualize the full NBT system in 3D from HDF5 snapshots.

Usage:
  python3 tools/plot_system_3d.py simulation.hd5
  python3 tools/plot_system_3d.py simulation.hd5 --save system3d.mp4 --fps 30 --subsample 20000

Dependencies:
  python3 -m pip install --user h5py numpy matplotlib
"""

import argparse
import math
import sys
from typing import List, Optional, Tuple

try:
    import h5py
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as exc:
    print("Missing dependency:", exc)
    print("Install with: python3 -m pip install --user h5py numpy matplotlib")
    sys.exit(2)


def load_snapshots(path: str) -> Tuple[List[int], List[float], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load snapshots from file.

    Returns:
      steps: list[int]
      times: list[float]
      positions: list[np.ndarray] shape (N,3)
      centers: list[np.ndarray] shape (3,)
      masses: list[np.ndarray] shape (N,) or empty if missing
    """
    with h5py.File(path, "r") as h5:
        if "snapshots" not in h5:
            raise RuntimeError("HDF5 file does not contain '/snapshots' group")

        entries = []
        for key in h5["snapshots"].keys():
            group = h5["snapshots"][key]
            step = int(group.attrs.get("step", -1))
            sim_time = float(group.attrs.get("simulation_time", math.nan))
            entries.append((step, sim_time, key))

        if not entries:
            raise RuntimeError("No snapshots found in file")

        entries.sort(key=lambda item: item[0])

        steps: List[int] = []
        times: List[float] = []
        positions: List[np.ndarray] = []
        centers: List[np.ndarray] = []
        masses: List[np.ndarray] = []

        for step, sim_time, key in entries:
            group = h5["snapshots"][key]
            if "positions" not in group:
                raise RuntimeError(f"Snapshot {key} does not contain 'positions' dataset")

            pos = np.asarray(group["positions"])
            com = np.asarray(group.attrs.get("center_of_mass", [np.nan, np.nan, np.nan]))
            mass = np.asarray(group["masses"]) if "masses" in group else np.array([])

            steps.append(step)
            times.append(sim_time)
            positions.append(pos)
            centers.append(com)
            masses.append(mass)

    return steps, times, positions, centers, masses


def subsample_frame(
    pos: np.ndarray,
    mass: np.ndarray,
    max_points: int,
    skip_core: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return subsampled positions/masses for one frame."""
    start = 1 if skip_core and pos.shape[0] > 0 else 0
    pos_view = pos[start:]
    mass_view = mass[start:] if mass.size == pos.shape[0] else np.array([])

    n = pos_view.shape[0]
    if max_points <= 0 or n <= max_points:
        return pos_view, mass_view

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return pos_view[idx], (mass_view[idx] if mass_view.size == n else np.array([]))


def compute_global_limits(frames: List[np.ndarray], padding: float = 0.05) -> Tuple[float, float, float, float, float, float]:
    """Compute stable cubic axis limits across all frames."""
    non_empty = [frame for frame in frames if frame.size > 0]
    if not non_empty:
        return -1.0, 1.0, -1.0, 1.0, -1.0, 1.0

    stacked = np.concatenate(non_empty, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) * 0.5
    extents = (maxs - mins) * 0.5
    radius = float(max(extents.max(), 1e-6))
    radius *= 1.0 + max(0.0, padding)

    return (
        float(center[0] - radius),
        float(center[0] + radius),
        float(center[1] - radius),
        float(center[1] + radius),
        float(center[2] - radius),
        float(center[2] + radius),
    )


def build_color_values(pos: np.ndarray, mass: np.ndarray, color_by: str) -> np.ndarray:
    """Build scalar colors per point for scatter."""
    if pos.size == 0:
        return np.array([])

    if color_by == "mass" and mass.size == pos.shape[0]:
        return mass.astype(float)

    if color_by == "z":
        return pos[:, 2].astype(float)

    # Default: radial distance in 3D.
    return np.linalg.norm(pos, axis=1)


def compute_density_strength(pos: np.ndarray, bins: int = 48) -> np.ndarray:
    """Estimate local density per point with a fast 3D voxel count."""
    if pos.size == 0:
        return np.array([])

    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    norm = (pos - mins) / spans

    idx = np.clip((norm * bins).astype(int), 0, bins - 1)
    linear = idx[:, 0] + bins * (idx[:, 1] + bins * idx[:, 2])

    counts = np.bincount(linear, minlength=bins**3)
    local_counts = counts[linear].astype(float)

    local_log = np.log1p(local_counts)
    scale = np.percentile(local_log, 95.0)
    if scale <= 0.0:
        return np.zeros_like(local_log)
    return np.clip(local_log / scale, 0.0, 1.0)


def build_emissive_rgba(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map local density to glow/core RGBA so dense regions approach white."""
    if pos.size == 0:
        empty = np.zeros((0, 4), dtype=float)
        return empty, empty

    d = compute_density_strength(pos)
    t = np.power(d, 0.75)

    sparse_blue = np.array([0.58, 0.83, 1.00], dtype=float)
    dense_white = np.array([1.00, 1.00, 1.00], dtype=float)

    core_rgb = sparse_blue[None, :] * (1.0 - t[:, None]) + dense_white[None, :] * t[:, None]
    core_alpha = 0.10 + 0.42 * t
    core_rgba = np.column_stack((core_rgb, core_alpha))

    glow_sparse = np.array([0.33, 0.70, 1.00], dtype=float)
    glow_dense = np.array([0.88, 0.95, 1.00], dtype=float)
    glow_t = np.power(d, 0.55)
    glow_rgb = glow_sparse[None, :] * (1.0 - glow_t[:, None]) + glow_dense[None, :] * glow_t[:, None]
    glow_alpha = 0.025 + 0.11 * glow_t
    glow_rgba = np.column_stack((glow_rgb, glow_alpha))

    return glow_rgba, core_rgba


def make_animation(
    steps: List[int],
    times: List[float],
    positions: List[np.ndarray],
    centers: List[np.ndarray],
    masses: List[np.ndarray],
    out_path: Optional[str],
    show: bool,
    interval_ms: int,
    subsample: int,
    skip_core: bool,
    show_com: bool,
    size: float,
    cmap: str,
    color_by: str,
    elev: float,
    azim: float,
    padding: float,
    track_com: bool,
    max_frames: int,
    fps: int,
    dpi: int,
) -> None:
    """Create and optionally save/show a 3D animation."""
    frame_count = len(positions) if max_frames <= 0 else min(len(positions), max_frames)
    if frame_count == 0:
        print("No frames to render.")
        return

    sampled_positions: List[np.ndarray] = []
    sampled_masses: List[np.ndarray] = []
    centered_positions: List[np.ndarray] = []

    for i in range(frame_count):
        pos, mass = subsample_frame(positions[i], masses[i], subsample, skip_core, seed=12345 + i)
        sampled_positions.append(pos)
        sampled_masses.append(mass)

        if track_com and not np.any(np.isnan(centers[i])):
            centered_positions.append(pos - centers[i])
        else:
            centered_positions.append(pos)

    xmin, xmax, ymin, ymax, zmin, zmax = compute_global_limits(centered_positions, padding=padding)

    fig = plt.figure(figsize=(7, 7), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    ax.view_init(elev=elev, azim=azim)
    ax.mouse_init(rotate_btn=1, zoom_btn=3)

    initial_pos = centered_positions[0]
    x0 = initial_pos[:, 0] if initial_pos.size > 0 else []
    y0 = initial_pos[:, 1] if initial_pos.size > 0 else []
    z0 = initial_pos[:, 2] if initial_pos.size > 0 else []
    initial_glow_rgba, initial_core_rgba = build_emissive_rgba(initial_pos)

    # Two layers make points read as faint emissive stars.
    star_glow = ax.scatter(
        x0,
        y0,
        z0,
        c=initial_glow_rgba,
        s=max(1.5, size * 2.6),
        linewidths=0,
        depthshade=False,
    )
    star_core = ax.scatter(
        x0,
        y0,
        z0,
        c=initial_core_rgba,
        s=max(1.0, size * 0.9),
        linewidths=0,
        depthshade=False,
    )

    com_artist = None
    if show_com:
        com_artist = ax.scatter([], [], [], c="#e8f7ff", marker="+", s=60, alpha=0.9)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.xaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.set_title("NBT system (3D)", color="#cfefff")

    def update(frame_idx: int):
        pos = centered_positions[frame_idx]
        glow_rgba, core_rgba = build_emissive_rgba(pos)

        if pos.size == 0:
            empty = np.array([])
            star_glow._offsets3d = (empty, empty, empty)
            star_core._offsets3d = (empty, empty, empty)
        else:
            star_glow._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            star_core._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            star_glow.set_facecolors(glow_rgba)
            star_glow.set_edgecolors(glow_rgba)
            star_core.set_facecolors(core_rgba)
            star_core.set_edgecolors(core_rgba)

        if show_com and com_artist is not None:
            com = centers[frame_idx]
            if np.any(np.isnan(com)):
                com_artist._offsets3d = (np.array([]), np.array([]), np.array([]))
            elif track_com:
                com_artist._offsets3d = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
            else:
                com_artist._offsets3d = (np.array([float(com[0])]), np.array([float(com[1])]), np.array([float(com[2])]))

        ax.set_title(f"NBT system (3D) | step={steps[frame_idx]} time={times[frame_idx]:.3f}", color="#cfefff")
        if com_artist is None:
            return star_glow, star_core
        return star_glow, star_core, com_artist

    if not out_path and not show:
        # Allow command-line smoke tests without opening a window.
        update(0)
        plt.close(fig)
        return

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=max(1, interval_ms),
        blit=False,
    )

    if out_path:
        ext = out_path.rsplit(".", 1)[-1].lower() if "." in out_path else ""
        print(f"Saving animation to {out_path} ... this may take a while")
        try:
            if ext in ("gif", "gifv"):
                ani.save(out_path, writer="pillow", dpi=dpi, fps=max(1, fps))
            else:
                ani.save(out_path, writer="ffmpeg", dpi=dpi, fps=max(1, fps))
            print("Saved animation.")
        except Exception as exc:
            print("Failed to save animation:", exc)

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the full NBT system in 3D from HDF5 snapshots")
    parser.add_argument("file", help="Path to HDF5 snapshot file")
    parser.add_argument("--save", "-s", default=None, help="Save animation to this file (mp4 or gif)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not open the interactive window")
    parser.add_argument("--interval", type=int, default=120, help="Frame interval in milliseconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for saved output")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved output")
    parser.add_argument("--frames", type=int, default=0, help="Max number of frames to animate (0 = all)")
    parser.add_argument("--subsample", type=int, default=20000, help="Max particles to plot per frame (0 = all)")
    parser.add_argument("--skip-core", action="store_true", help="Skip particle index 0 (massive core)")
    parser.add_argument("--no-com", dest="show_com", action="store_false", help="Do not plot center-of-mass marker")
    parser.add_argument("--track-com", action="store_true", help="Center each frame on center-of-mass")
    parser.add_argument("--size", type=float, default=3.0, help="Marker size")
    parser.add_argument("--padding", type=float, default=0.05, help="Extra global axis padding fraction")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    parser.add_argument("--color-by", choices=["radius", "mass", "z"], default="radius", help="Point coloring scalar")
    parser.add_argument("--elev", type=float, default=25.0, help="Camera elevation angle")
    parser.add_argument("--azim", type=float, default=45.0, help="Camera azimuth angle")
    parser.set_defaults(show=True, show_com=True)

    args = parser.parse_args()

    try:
        steps, times, positions, centers, masses = load_snapshots(args.file)
    except Exception as exc:
        print("Error loading HDF5:", exc)
        sys.exit(1)

    make_animation(
        steps=steps,
        times=times,
        positions=positions,
        centers=centers,
        masses=masses,
        out_path=args.save,
        show=args.show,
        interval_ms=args.interval,
        subsample=args.subsample,
        skip_core=args.skip_core,
        show_com=args.show_com,
        size=args.size,
        cmap=args.cmap,
        color_by=args.color_by,
        elev=args.elev,
        azim=args.azim,
        padding=args.padding,
        track_com=args.track_com,
        max_frames=args.frames,
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

