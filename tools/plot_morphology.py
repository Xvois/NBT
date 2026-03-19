#!/usr/bin/env python3
"""
plot_morphology.py

Small utility to visualize galaxy morphology from NBT HDF5 snapshots.

Features:
- Reads /snapshots/step_* groups and plots particle positions (x,z) per snapshot
- Keeps axis limits consistent across frames for smooth animations
- Options to subsample points, skip the massive core (index 0), mark center-of-mass
- Can show interactively or save as an MP4/GIF using matplotlib animation writers

Usage:
  python3 tools/plot_morphology.py simulation.hd5 --save movie.mp4 --interval 100 --subsample 5000

Dependencies:
  python3 -m pip install --user h5py numpy matplotlib

"""
import argparse
import sys
import math
from typing import List, Tuple

try:
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except Exception as e:
    print("Missing dependency:", e)
    print("Install with: python3 -m pip install --user h5py numpy matplotlib")
    sys.exit(2)


def load_snapshots(path: str) -> Tuple[List[int], List[float], List[np.ndarray], List[np.ndarray]]:
    """Load snapshots from file.

    Returns:
      steps: list of int
      times: list of float (simulation_time attribute)
      positions: list of ndarray shape (N,3)
      centers: list of fVector3 (center_of_mass attribute) as ndarray shape (3,)
    """
    with h5py.File(path, 'r') as f:
        if 'snapshots' not in f:
            raise RuntimeError("HDF5 file does not contain '/snapshots' group")

        entries = []
        for key in f['snapshots'].keys():
            try:
                g = f['snapshots'][key]
            except Exception:
                continue
            step = int(g.attrs.get('step', -1))
            time = float(g.attrs.get('simulation_time', math.nan))
            entries.append((step, time, key))

        if not entries:
            raise RuntimeError('No snapshots found in file')

        entries.sort(key=lambda e: e[0])

        steps = []
        times = []
        positions = []
        centers = []

        for step, time, key in entries:
            g = f['snapshots'][key]
            if 'positions' not in g:
                raise RuntimeError(f"Snapshot {key} does not contain 'positions' dataset")
            pos = np.array(g['positions'])  # shape (particleCount,3)
            # read center_of_mass attribute if present
            if 'center_of_mass' in g.attrs:
                com = np.array(g.attrs.get('center_of_mass'))
            else:
                com = np.array([np.nan, np.nan, np.nan])

            steps.append(step)
            times.append(time)
            positions.append(pos)
            centers.append(com)

        return steps, times, positions, centers


def compute_axis_limits(positions_list: List[np.ndarray], padding: float = 0.05, plane: str = 'xz') -> Tuple[float, float, float, float]:
    """Compute consistent axis limits across a list of position arrays.

    plane: 'xz' or 'xy' or 'yz' (default xz: plot x vs z)
    Returns: (xmin, xmax, ymin, ymax)
    """
    idx_map = {'x': 0, 'y': 1, 'z': 2}
    if plane == 'xz':
        i0, i1 = 0, 2
    elif plane == 'xy':
        i0, i1 = 0, 1
    elif plane == 'yz':
        i0, i1 = 1, 2
    else:
        raise ValueError('unknown plane')

    all_x = np.concatenate([p[:, i0] for p in positions_list if p.size > 0])
    all_y = np.concatenate([p[:, i1] for p in positions_list if p.size > 0])

    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))

    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

    xmin -= dx * padding
    xmax += dx * padding
    ymin -= dy * padding
    ymax += dy * padding

    # Make axes square by expanding the smaller range
    xlen = xmax - xmin
    ylen = ymax - ymin
    if xlen > ylen:
        extra = (xlen - ylen) / 2.0
        ymin -= extra
        ymax += extra
    else:
        extra = (ylen - xlen) / 2.0
        xmin -= extra
        xmax += extra

    return xmin, xmax, ymin, ymax


def subsample_positions(pos: np.ndarray, max_points: int, skip_core: bool = True) -> np.ndarray:
    """Return subsampled positions (Nx3) limited to max_points.

    If skip_core is True, remove index 0 before sampling.
    """
    if skip_core and pos.shape[0] > 0:
        pos = pos[1:]
    n = pos.shape[0]
    if max_points <= 0 or n <= max_points:
        return pos
    idx = np.random.default_rng(12345).choice(n, size=max_points, replace=False)
    return pos[idx]


def make_animation(steps: List[int], times: List[float], positions: List[np.ndarray], centers: List[np.ndarray],
                   out_path: str = None, show: bool = True, interval_ms: int = 200, subsample: int = 5000,
                   skip_core: bool = True, plane: str = 'xz', cmap: str = 'viridis', size: float = 1.0):
    # prepare figure
    plane_map = {'xz': ('x', 'z'), 'xy': ('x', 'y'), 'yz': ('y', 'z')}
    if plane not in plane_map:
        raise ValueError('plane must be one of xz, xy, yz')

    idx0 = {'x': 0, 'y': 1, 'z': 2}[plane_map[plane][0]]
    idx1 = {'x': 0, 'y': 1, 'z': 2}[plane_map[plane][1]]

    sampled = [subsample_positions(p, subsample, skip_core) for p in positions]

    xmin, xmax, ymin, ymax = compute_axis_limits(sampled, padding=0.05, plane=plane)

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=max(0.1, size), c=[], cmap=cmap, lw=0, alpha=0.9)
    com_point, = ax.plot([], [], 'r+', markersize=10, markeredgewidth=1.5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(plane_map[plane][0])
    ax.set_ylabel(plane_map[plane][1])
    ax.set_title('Galaxy morphology')
    ax.grid(True, alpha=0.2)

    # precompute colors (optional: by radius)
    color_arrays = []
    for p in sampled:
        if p.size == 0:
            color_arrays.append(np.array([]))
            continue
        # color by radius from origin in plane
        radii = np.sqrt(p[:, idx0] ** 2 + p[:, idx1] ** 2)
        color_arrays.append(radii)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        com_point.set_data([], [])
        return scat, com_point

    def update(frame):
        p = sampled[frame]
        if p.size == 0:
            offsets = np.empty((0, 2))
            scat.set_offsets(offsets)
            scat.set_array(np.array([]))
            com_point.set_data([], [])
            ax.set_title(f'Step {steps[frame]} time={times[frame]:.3f}')
            return scat, com_point

        offsets = np.column_stack([p[:, idx0], p[:, idx1]])
        scat.set_offsets(offsets)
        # normalize colors for display
        col = color_arrays[frame]
        if col.size > 0:
            # clip and normalize
            col = np.clip(col, 0.0, np.percentile(col, 99.0))
            scat.set_array(col)
        else:
            scat.set_array(np.array([]))

        com = centers[frame]
        if not np.any(np.isnan(com)):
            comx = float(com[idx0])
            comy = float(com[idx1])
            com_point.set_data([comx], [comy])
        else:
            com_point.set_data([], [])

        ax.set_title(f'Step {steps[frame]} time={times[frame]:.3f}')
        return scat, com_point

    ani = animation.FuncAnimation(fig, update, frames=len(sampled), init_func=init, blit=True, interval=interval_ms)

    if out_path:
        ext = out_path.split('.')[-1].lower()
        print(f"Saving animation to {out_path} ... this may take a while")
        try:
            if ext in ('mp4', 'm4v'):
                Writer = animation.writers['ffmpeg'] if 'ffmpeg' in animation.writers.list() else None
                if Writer is None:
                    print('ffmpeg writer not found; trying ffmpeg fallback; ensure ffmpeg is installed for MP4 output')
                ani.save(out_path, writer='ffmpeg', dpi=150)
            elif ext in ('gif', 'gifv'):
                ani.save(out_path, writer='pillow', dpi=150)
            else:
                # fallback: save as mp4
                ani.save(out_path, writer='ffmpeg', dpi=150)
            print('Saved animation.')
        except Exception as e:
            print('Failed to save animation:', e)

    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description='Visualize galaxy morphology from NBT HDF5 snapshots')
    p.add_argument('file', help='Path to HDF5 snapshot file')
    p.add_argument('--save', '-s', help='Save animation to this file (mp4 or gif). If absent, only shows interactively.', default=None)
    p.add_argument('--no-show', dest='show', action='store_false', help="Don't open interactive window")
    p.add_argument('--interval', type=int, default=200, help='Interval between frames in ms')
    p.add_argument('--subsample', type=int, default=5000, help='Max particles to plot per frame (0 = no subsample)')
    p.add_argument('--skip-core', dest='skip_core', action='store_true', help='Skip particle index 0 (massive core) from display')
    p.add_argument('--plane', choices=['xz', 'xy', 'yz'], default='xz', help='Which plane to plot (default: xz)')
    p.add_argument('--no-com', dest='show_com', action='store_false', help="Don't plot center-of-mass marker")
    p.add_argument('--size', type=float, default=1.0, help='Marker size scale')
    args = p.parse_args()

    try:
        steps, times, positions, centers = load_snapshots(args.file)
    except Exception as e:
        print('Error loading HDF5:', e)
        sys.exit(1)

    make_animation(steps, times, positions, centers, out_path=args.save, show=args.show, interval_ms=args.interval,
                   subsample=args.subsample, skip_core=args.skip_core, plane=args.plane, size=args.size)


if __name__ == '__main__':
    main()

