"""scan_ray_ranges.py — scan WKBeam HDF5 ray files for coordinate extents.

Usage
-----
    python scan_ray_ranges.py "path/to/run_*/traces_*.hdf5" [--margin 0.01]

Uses energy-weighted percentiles (TracesWfct as weights, p0.1%–p99.9%) rather
than raw min/max, so rays that traverse the full orbit but carry negligible
energy do not blow out the grid.  A small margin is then added, and hard
physical limits are enforced (EC waves: |N| < 1, theta in [-pi, pi]).

Psi/rho is intentionally excluded: its grid comes from LUKE (WKBacca_grids.mat).
"""

import argparse
import glob
import math
import sys
import h5py
import numpy as np

# ── Target bin widths ───────────────────────────────────────────────────────
TARGET_BIN_WIDTHS = {
    'theta': 0.05,    # rad  (~3°)
    'Npar':  0.005,   # dimensionless
    'Nperp': 0.005,   # dimensionless
    'phiN':  0.05,    # rad  (diagnostic only)
}

# ── Hard physical limits applied after margin ────────────────────────────────
# EC waves: |N| < 1 by definition; theta is a poloidal angle in [-pi, pi].
# phiN is diagnostic-only (not passed to D_RF) and its sign convention varies
# by WKBeam version, so no hard limit is applied to it.
PHYSICAL_LIMITS = {
    'theta': (-math.pi, math.pi),
    'Npar':  (-1.0,     1.0),
    'Nperp': (0.0,      1.0),
}

# Energy-weighted percentile cutoffs (fraction of total Wfct discarded per side)
PERCENTILE_LO = 0.001   # 0.1 %
PERCENTILE_HI = 0.999   # 99.9 %

FIELDS = {
    'theta': 'TracesTheta',
    'Npar':  'TracesNparallel',
    'Nperp': 'TracesNperpendicular',
    'phiN':  'TracesphiN',
}


def weighted_percentile(x, w, q_lo, q_hi):
    """Return the q_lo and q_hi energy-weighted quantiles of 1-D array x."""
    order = np.argsort(x)
    x_s   = x[order]
    cdf   = np.cumsum(w[order]) / w.sum()
    lo    = float(x_s[np.searchsorted(cdf, q_lo)])
    hi    = float(x_s[min(np.searchsorted(cdf, q_hi), len(x_s) - 1)])
    return lo, hi


def scan(pattern: str, margin_frac: float = 0.01):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No files matched: {pattern}", file=sys.stderr)
        sys.exit(1)

    # Collect all coordinate data and Wfct across files.
    buf      = {k: [] for k in FIELDS}
    wfct_buf = []

    for fname in files:
        try:
            with h5py.File(fname, 'r') as f:
                wfct = f['TracesWfct'][()].ravel() if 'TracesWfct' in f else None
                for key, ds_name in FIELDS.items():
                    if ds_name in f:
                        buf[key].append(f[ds_name][()].ravel())
                if wfct is not None:
                    wfct_buf.append(wfct)
        except Exception as exc:
            print(f"  WARNING: could not read {fname}: {exc}", file=sys.stderr)

    have_wfct = bool(wfct_buf)
    if not have_wfct:
        print("WARNING: TracesWfct missing — falling back to raw min/max", file=sys.stderr)
    wfct_all = np.concatenate(wfct_buf) if have_wfct else None

    print(f"\nScanned {len(files)} file(s)  (margin = {margin_frac*100:.0f}%,  "
          f"Wfct percentile cut = {PERCENTILE_LO*100:.1f}% / {PERCENTILE_HI*100:.1f}%)\n")
    print(f"{'dim':8s}  {'raw lo':>9s}  {'raw hi':>9s}  "
          f"{'p0.1%':>9s}  {'p99.9%':>9s}  "
          f"{'out lo':>9s}  {'out hi':>9s}  {'n_bins':>7s}")
    print("-" * 84)

    results = {}
    for key in FIELDS:
        if not buf[key]:
            print(f"{key:8s}  (dataset not found in any file)")
            continue

        x      = np.concatenate(buf[key])
        raw_lo = float(x.min())
        raw_hi = float(x.max())

        # Energy-weighted percentile range (falls back to raw if no Wfct)
        if have_wfct:
            p_lo, p_hi = weighted_percentile(x, wfct_all, PERCENTILE_LO, PERCENTILE_HI)
        else:
            p_lo, p_hi = raw_lo, raw_hi

        # Margin on the percentile-trimmed span
        span   = p_hi - p_lo
        margin = margin_frac * span if span > 0 else 0.01
        lo_m   = p_lo - margin
        hi_m   = p_hi + margin

        # Hard physical clipping
        phys_lo, phys_hi = PHYSICAL_LIMITS.get(key, (-np.inf, np.inf))
        lo_m = max(lo_m, phys_lo)
        hi_m = min(hi_m, phys_hi)

        bw     = TARGET_BIN_WIDTHS.get(key)
        n_bins = math.ceil((hi_m - lo_m) / bw) if (bw and hi_m > lo_m) else '—'

        print(f"{key:8s}  {raw_lo:9.4f}  {raw_hi:9.4f}  "
              f"{p_lo:9.4f}  {p_hi:9.4f}  "
              f"{lo_m:9.4f}  {hi_m:9.4f}  {str(n_bins):>7s}")

        results[key] = dict(lo=lo_m, hi=hi_m, n_bins=n_bins, bin_width=bw)

        if key == 'phiN':
            # phiN convention varies; warn only if clearly outside [-2π, 2π]
            if p_lo < -2 * math.pi - 0.01 or p_hi > 2 * math.pi + 0.01:
                print("  ** WARNING: phiN outside [-2π, 2π] — check coordinate transform! **")

    print("\n── Copy-paste config block ─────────────────────────────────────────────")
    print("idata.bin_width = {")
    for key in ('theta', 'Npar', 'Nperp'):
        bw = TARGET_BIN_WIDTHS.get(key, 'None')
        print(f"    '{key}': {bw},")
    print("}")
    print("\nidata.min / idata.max (non-psi dims):")
    for key in ('theta', 'Npar', 'Nperp'):
        if key in results:
            r = results[key]
            print(f"    {key}: [{r['lo']:.4f}, {r['hi']:.4f}]  →  {r['n_bins']} bins")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pattern', help='Glob pattern for HDF5 ray files')
    parser.add_argument('--margin', type=float, default=0.01,
                        help='Fractional margin on each side of the energy-weighted range '
                             '(default 0.01 = 1%%)')
    parser.add_argument('--json', action='store_true',
                        help='Append a machine-readable JSON summary as the last stdout line')
    args = parser.parse_args()
    results = scan(args.pattern, args.margin)
    if args.json:
        import json
        out = {}
        for key in ('theta', 'Npar', 'Nperp'):
            if key in results:
                r = results[key]
                out[key] = {'lo': r['lo'], 'hi': r['hi'], 'n_bins': r['n_bins']}
        print(json.dumps(out))


if __name__ == '__main__':
    main()
