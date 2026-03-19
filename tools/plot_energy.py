import argparse
import sys
from typing import Tuple

try:
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing dependency:", e)
    print("Install with: python3 -m pip install --user h5py numpy matplotlib")
    sys.exit(2)


def load_energy_series(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load snapshots and return arrays: steps, times, total_energy, (kinetic, potential)

    Returns:
      steps: shape (N,) int
      times: shape (N,) float
      total: shape (N,) float
      kin_pot: shape (N,2) float (kinetic, potential)
    """
    with h5py.File(path, 'r') as f:
        if 'snapshots' not in f:
            raise RuntimeError("HDF5 file does not contain '/snapshots' group")

        entries = []
        for key in f['snapshots'].keys():
            g = f['snapshots'][key]
            # Read attributes with fallbacks
            step = int(g.attrs.get('step', -1))
            time = float(g.attrs.get('simulation_time', np.nan))
            total = float(g.attrs.get('total_energy', np.nan))
            kin = float(g.attrs.get('kinetic_energy', np.nan))
            pot = float(g.attrs.get('potential_energy', np.nan))
            entries.append((step, time, total, kin, pot))

        if not entries:
            raise RuntimeError('No snapshots found in file')

        # Sort by step (stable) so ordering is deterministic
        entries.sort(key=lambda e: e[0])
        arr = np.array(entries)
        steps = arr[:, 0].astype(int)
        times = arr[:, 1].astype(float)
        total = arr[:, 2].astype(float)
        kin = arr[:, 3].astype(float)
        pot = arr[:, 4].astype(float)

        return steps, times, total, np.stack([kin, pot], axis=1)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = np.ones(window) / float(window)
    return np.convolve(x, w, mode='same')


def main():
    p = argparse.ArgumentParser(description='Plot total energy vs time from NBT HDF5 snapshots')
    p.add_argument('file', help='Path to HDF5 snapshot file')
    p.add_argument('--save', '-s', help='Save plot to this PNG file (also shows by default)', default=None)
    p.add_argument('--no-show', dest='show', action='store_false', help="Don't open interactive window")
    p.add_argument('--relative', action='store_true', help='Plot relative drift (E/E0 - 1) instead of absolute energy')
    p.add_argument('--x', choices=['time', 'step'], default='time', help='Use simulation_time or step on x-axis')
    p.add_argument('--smooth', type=int, default=1, help='Apply simple moving-average smoothing (window size)')
    args = p.parse_args()

    try:
        steps, times, total, kinpot = load_energy_series(args.file)
    except Exception as e:
        print('Error loading HDF5:', e)
        sys.exit(1)

    x = times if args.x == 'time' else steps
    y = total.copy()
    if args.smooth and args.smooth > 1:
        y = smooth(y, args.smooth)

    # Basic statistics
    E0 = float(total[0])
    Efinal = float(total[-1])
    abs_change = Efinal - E0
    rel_change = (Efinal - E0) / E0 if E0 != 0 else float('nan')

    # Linear trend (least squares) for drift per x-unit
    try:
        A = np.vstack([x.astype(float), np.ones_like(x, dtype=float)]).T
        m, c = np.linalg.lstsq(A, total.astype(float), rcond=None)[0]
    except Exception:
        m, c = float('nan'), float('nan')

    print(f"Snapshots: {len(x)} | first step={steps[0]} last step={steps[-1]}")
    print(f"Initial energy E0 = {E0:.6e}")
    print(f"Final energy  Efinal = {Efinal:.6e}")
    print(f"Absolute change = {abs_change:.6e}")
    print(f"Relative change = {rel_change:.6e}")
    print(f"Linear trend: slope = {m:.6e} per x-unit (intercept {c:.6e})")

    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios':[3,1]})

    if args.relative:
        yplot = (y / E0) - 1.0
        ax1.plot(x, yplot, '-k', label='relative drift (E/E0 - 1)')
        ax1.set_ylabel('Relative drift')
    else:
        ax1.plot(x, y, '-k', label='total energy')
        ax1.set_ylabel('Total energy')

    # Also plot kinetic and potential energies (unsmoothed) in the top panel for context
    ax1.plot(x, kinpot[:,0], '--r', alpha=0.7, label='kinetic')
    ax1.plot(x, kinpot[:,1], '--b', alpha=0.7, label='potential')
    ax1.legend(loc='best', fontsize='small')
    ax1.grid(True)

    # bottom panel: relative drift (always helpful)
    rel = (total / E0) - 1.0
    if args.smooth and args.smooth > 1:
        rel = smooth(rel, args.smooth)
    ax2.plot(x, rel, '-k')
    ax2.set_xlabel('simulation_time' if args.x == 'time' else 'step')
    ax2.set_ylabel('E/E0 - 1')
    ax2.grid(True)

    fig.suptitle('Energy diagnostics from NBT snapshots')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f'Saved plot to {args.save}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()

