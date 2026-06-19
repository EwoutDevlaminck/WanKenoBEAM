"""scan_ray_ranges.py — scan WKBeam HDF5 ray files for coordinate extents.

Usage
-----
    python scan_ray_ranges.py "path/to/run_*/traces_*.hdf5" [--margin 0.05]

The script makes one pass over all matching files, collects the global min/max
of each binnable coordinate, applies a configurable margin, and prints a
copy-paste–ready config block showing suggested bin counts for the target bin
widths defined in TARGET_BIN_WIDTHS below.

Psi/rho is intentionally excluded from the range calculation: its grid is
always defined by LUKE (WKBacca_grids.mat) and must not be set via bin_width.

The phi_N range is theoretically [0, 2pi]; the scan verifies this and warns
if values fall outside that range (which indicates a coordinate-transform bug).
"""

import argparse
import glob
import math
import sys
import h5py
import numpy as np

# ── Target bin widths (edit per scenario) ──────────────────────────────────
# Keys must match the dataset names listed in FIELDS below.
TARGET_BIN_WIDTHS = {
    'theta': 0.05,    # rad  (~3°)
    'Npar':  0.005,   # dimensionless
    'Nperp': 0.005,   # dimensionless
    'phiN':  0.05,    # rad  (diagnostic only — phiN bins not passed to D_RF)
}
# ───────────────────────────────────────────────────────────────────────────

FIELDS = {
    'theta': 'TracesTheta',
    'Npar':  'TracesNparallel',
    'Nperp': 'TracesNperpendicular',
    'phiN':  'TracesphiN',
}

PHIPHI_EXPECTED = (0.0, 2 * math.pi)


def scan(pattern: str, margin_frac: float = 0.05):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No files matched: {pattern}", file=sys.stderr)
        sys.exit(1)

    lo = {k: np.inf  for k in FIELDS}
    hi = {k: -np.inf for k in FIELDS}

    for fname in files:
        try:
            with h5py.File(fname, 'r') as f:
                for key, ds_name in FIELDS.items():
                    if ds_name not in f:
                        continue
                    data = f[ds_name][()]
                    lo[key] = min(lo[key], float(data.min()))
                    hi[key] = max(hi[key], float(data.max()))
        except Exception as exc:
            print(f"  WARNING: could not read {fname}: {exc}", file=sys.stderr)

    print(f"\nScanned {len(files)} file(s)  (margin = {margin_frac*100:.0f}%)\n")
    print(f"{'dim':8s}  {'raw min':>12s}  {'raw max':>12s}  "
          f"{'marg min':>12s}  {'marg max':>12s}  {'n_bins':>8s}  bin_width")
    print("-" * 80)

    results = {}
    for key in FIELDS:
        if np.isinf(lo[key]):
            print(f"{key:8s}  (dataset not found in any file)")
            continue

        span   = hi[key] - lo[key]
        margin = margin_frac * span if span > 0 else 0.05
        lo_m   = lo[key] - margin
        hi_m   = hi[key] + margin
        bw     = TARGET_BIN_WIDTHS.get(key, None)
        n_bins = math.ceil((hi_m - lo_m) / bw) if bw else '—'

        print(f"{key:8s}  {lo[key]:12.4f}  {hi[key]:12.4f}  "
              f"{lo_m:12.4f}  {hi_m:12.4f}  {str(n_bins):>8s}  {bw}")

        results[key] = dict(lo=lo_m, hi=hi_m, n_bins=n_bins, bin_width=bw)

        if key == 'phiN':
            if lo[key] < PHIPHI_EXPECTED[0] - 0.01 or hi[key] > PHIPHI_EXPECTED[1] + 0.01:
                print(f"  ** WARNING: phiN outside expected [0, 2π] — check coordinate transform! **")

    print("\n── Copy-paste config block (idata.bin_width) ──────────────────────────")
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
    parser.add_argument('--margin', type=float, default=0.05,
                        help='Fractional margin added to each side (default 0.05)')
    args = parser.parse_args()
    scan(args.pattern, args.margin)


if __name__ == '__main__':
    main()
