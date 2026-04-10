"""
compare_runs.py
===============
Compare three WKBeam ray-tracing runs (same rays, different absorption modules):

    L4_Farina_file0.hdf5    -- Farina (warmdamp) fully-relativistic Maxwellian
    L4_QLabs_M_file0.hdf5   -- qlabs loaded with LUKE Maxwellian (XXfM)
    L4_QLabs_EDF_file0.hdf5 -- qlabs loaded with LUKE actual EDF (XXf0)

Because freeze_random_numbers = True was set, all three runs trace identical
rays.  The only difference is the absorption coefficient, so the Wfct decay
curves can be overlaid directly.

Usage:
    cd StandardCases/TCV_88612_1.25_fluct
    python compare_runs.py

Output:
    compare_runs.png   -- multi-panel figure
    compare_runs.txt   -- summary statistics
"""

import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
_WKBEAM_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
if _WKBEAM_ROOT not in sys.path:
    sys.path.insert(0, _WKBEAM_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTDIR = os.path.join(_HERE, 'output')

RUNS = {
    'Farina':    os.path.join(OUTDIR, 'L4_Farina_file0.hdf5'),
    'qlabs(MJ)': os.path.join(OUTDIR, 'L4_QLabs_M_file0.hdf5'),
    'qlabs(EDF)':os.path.join(OUTDIR, 'L4_QLabs_EDF_file0.hdf5'),
}
COLORS = {
    'Farina':    'C0',
    'qlabs(MJ)': 'C1',
    'qlabs(EDF)':'C2',
}
LS = {
    'Farina':    '-',
    'qlabs(MJ)': '--',
    'qlabs(EDF)':':',
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_run(fpath):
    """Return a dict of arrays from a WKBeam output HDF5 file."""
    with h5py.File(fpath, 'r') as f:
        out = {}
        out['Wfct']    = f['TracesWfct'][:]          # (nRays, npt)
        out['Psi']     = f['TracesPsi'][:]            # (nRays, npt)
        out['Npar']    = f['TracesNparallel'][:]      # (nRays, npt)
        out['Nperp']   = f['TracesNperpendicular'][:] # (nRays, npt)
        out['time']    = f['TracesTime'][:]           # (nRays, npt)
        XYZ            = f['TracesXYZ'][:]            # (nRays, 3, npt)
        out['X'] = XYZ[:, 0, :]
        out['Y'] = XYZ[:, 1, :]
        out['Z'] = XYZ[:, 2, :]
        if 'TracesCorrectionFactor' in f:
            out['xi'] = f['TracesCorrectionFactor'][:]
        if 'TracesTheta' in f:
            out['Theta'] = f['TracesTheta'][:]
    return out

# Check all files exist
missing = [k for k, v in RUNS.items() if not os.path.isfile(v)]
if missing:
    print("ERROR: Output files not found for runs:", missing)
    print("Expected paths:")
    for k in missing:
        print(f"  {RUNS[k]}")
    print("\nRun the three simulations first, then re-run this script.")
    sys.exit(1)

data = {name: load_run(path) for name, path in RUNS.items()}
nRays = data['Farina']['Wfct'].shape[0]
print(f"Loaded {len(data)} runs, {nRays} rays each.")

# ---------------------------------------------------------------------------
# Per-ray active mask: steps where all coordinates are non-zero
# ---------------------------------------------------------------------------
def active_mask(d, iray):
    """Boolean mask of valid (non-zero) steps for ray iray."""
    return d['Wfct'][iray] != 0.0

def absorbed_fraction(d, iray):
    """1 - Wfct_last / Wfct_first for active steps."""
    mask = active_mask(d, iray)
    if mask.sum() < 2:
        return float('nan')
    W = d['Wfct'][iray, mask]
    return 1.0 - W[-1] / W[0]

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
lines = []
lines.append("="*60)
lines.append(f"Absorbed power fraction per ray (1 - W_final/W_initial)")
lines.append("="*60)
header = f"{'Ray':>4}  " + "  ".join(f"{k:>12}" for k in RUNS)
lines.append(header)
lines.append("-"*60)

afrac = {name: [] for name in RUNS}
for iray in range(nRays):
    row = f"{iray:>4}  "
    for name, d in data.items():
        af = absorbed_fraction(d, iray)
        afrac[name].append(af)
        row += f"  {af:>12.4f}"
    lines.append(row)

lines.append("-"*60)
means_row = f"{'mean':>4}  " + "  ".join(f"{np.nanmean(afrac[k]):>12.4f}" for k in RUNS)
lines.append(means_row)
lines.append("="*60)

lines.append("")
lines.append("Ratio qlabs(EDF) / Farina per ray:")
for iray in range(nRays):
    farina = afrac['Farina'][iray]
    edf    = afrac['qlabs(EDF)'][iray]
    ratio  = edf / farina if (np.isfinite(farina) and farina > 0) else float('nan')
    lines.append(f"  Ray {iray}: {ratio:.4f}")

for line in lines:
    print(line)

txt_out = os.path.join(_HERE, 'compare_runs.txt')
with open(txt_out, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print(f"\nSummary written → {txt_out}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    'WKBeam absorption comparison: Farina vs qlabs(MJ) vs qlabs(EDF)\n'
    'TCV 88612, t=1.25s — same rays (freeze_random_numbers=True), scattering off',
    fontsize=11
)
gs = fig.add_gridspec(3, 5, hspace=0.45, wspace=0.35)

for iray in range(5):
    # --- Row 0: Wfct vs step index ---
    ax = fig.add_subplot(gs[0, iray])
    for name, d in data.items():
        mask = active_mask(d, iray)
        steps = np.where(mask)[0]
        W = d['Wfct'][iray, mask]
        W_norm = W / W[0] if W[0] > 0 else W
        ax.plot(steps, W_norm, color=COLORS[name], ls=LS[name], lw=1.5, label=name)
    ax.set(xlabel='Step', ylabel='W/W₀',
           title=f'Ray {iray} — Wfct decay')
    ax.set_ylim(0, 1.05)
    if iray == 0:
        ax.legend(fontsize=7)
    ax.grid(True)

    # --- Row 1: Wfct vs psi ---
    ax = fig.add_subplot(gs[1, iray])
    for name, d in data.items():
        mask = active_mask(d, iray)
        psi = d['Psi'][iray, mask]
        W = d['Wfct'][iray, mask]
        W_norm = W / W[0] if W[0] > 0 else W
        ax.plot(psi, W_norm, color=COLORS[name], ls=LS[name], lw=1.5, label=name)
    ax.set(xlabel='ψ', ylabel='W/W₀',
           title=f'Ray {iray} — Wfct vs ψ')
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # --- Row 2: N‖ along ray ---
    ax = fig.add_subplot(gs[2, iray])
    # N‖ is the same for all runs (same rays), just use Farina
    d_ref = data['Farina']
    mask = active_mask(d_ref, iray)
    steps = np.where(mask)[0]
    ax.plot(steps, d_ref['Npar'][iray, mask],  lw=1.5, label='N‖', color='C3')
    ax.plot(steps, d_ref['Nperp'][iray, mask], lw=1.5, label='N⊥', color='C4', ls='--')
    ax.set(xlabel='Step', ylabel='N',
           title=f'Ray {iray} — N‖, N⊥')
    if iray == 0:
        ax.legend(fontsize=7)
    ax.grid(True)

# --- Extra summary panel: absorbed fraction bar chart ---
fig2, ax2 = plt.subplots(figsize=(7, 4))
x = np.arange(nRays)
w = 0.25
for i, (name, vals) in enumerate(afrac.items()):
    ax2.bar(x + (i - 1)*w, vals, width=w, label=name,
            color=COLORS[name], alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels([f'Ray {i}' for i in range(nRays)])
ax2.set(ylabel='Absorbed fraction', title='Absorbed power fraction per ray')
ax2.legend()
ax2.grid(True, axis='y')
fig2.tight_layout()
out2 = os.path.join(_HERE, 'compare_absorbed_fraction.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Bar chart saved → {out2}")

plt.figure(fig.number)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = os.path.join(_HERE, 'compare_runs.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Ray traces figure saved → {out1}")

# --- Psi-integrated absorption comparison (all rays combined) ---
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
fig3.suptitle('Absorption profile vs ψ (all rays, dW/dψ)', fontsize=11)

psi_bins = np.linspace(0.0, 1.0, 41)
psi_mid  = 0.5 * (psi_bins[:-1] + psi_bins[1:])

for name, d in data.items():
    dW_dpsi = np.zeros(len(psi_mid))
    for iray in range(nRays):
        mask = active_mask(d, iray)
        if mask.sum() < 3:
            continue
        psi_ray = d['Psi'][iray, mask]
        W_ray   = d['Wfct'][iray, mask]
        # absorbed power per step = -d(W)/d(step); bin by psi
        dW = -np.diff(W_ray)       # positive where absorption occurs
        psi_mid_step = 0.5 * (psi_ray[:-1] + psi_ray[1:])
        counts, _ = np.histogram(psi_mid_step, bins=psi_bins, weights=dW)
        dW_dpsi += counts
    axes3[0].plot(psi_mid, dW_dpsi, color=COLORS[name], ls=LS[name],
                  lw=2, label=name)

axes3[0].set(xlabel='ψ', ylabel='Σ dW/bin', title='dW/dψ (raw, all rays)')
axes3[0].legend(); axes3[0].grid(True)

# Normalised: divide each run by its total to compare shapes
for name, d in data.items():
    dW_dpsi = np.zeros(len(psi_mid))
    for iray in range(nRays):
        mask = active_mask(d, iray)
        if mask.sum() < 3:
            continue
        psi_ray = d['Psi'][iray, mask]
        W_ray   = d['Wfct'][iray, mask]
        dW = -np.diff(W_ray)
        psi_mid_step = 0.5 * (psi_ray[:-1] + psi_ray[1:])
        counts, _ = np.histogram(psi_mid_step, bins=psi_bins, weights=dW)
        dW_dpsi += counts
    total = dW_dpsi.sum()
    if total > 0:
        axes3[1].plot(psi_mid, dW_dpsi / total, color=COLORS[name], ls=LS[name],
                      lw=2, label=name)

axes3[1].set(xlabel='ψ', ylabel='dW/dψ (normalised)', title='Absorption shape comparison')
axes3[1].legend(); axes3[1].grid(True)

fig3.tight_layout()
out3 = os.path.join(_HERE, 'compare_absorption_profile.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f"Absorption profile figure saved → {out3}")

# ---------------------------------------------------------------------------
# 2-D poloidal trajectory plot coloured by W/W₀ (viridis)
# with flux-surface contours from the actual equilibrium
# ---------------------------------------------------------------------------
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium, IntSample, StixParamSample
from Tools.PlotData.CommonPlotting import plotting_functions

# Load equilibrium once from the Farina config (identical for all three runs)
_config = os.path.join(_HERE, 'RayTracing_Farina.txt')
print(f"\nLoading equilibrium from {_config} ...")
_idata = InputData(_config)
Eq = TokamakEquilibrium(_idata)
print("Equilibrium loaded.")

# Build a 2D (R, Z) grid covering the full equilibrium domain.
# IntSample(R1d, Z1d, f) returns shape (nZ, nR) — correct for ax.contour(R1d, Z1d, data).
_Rmin = float(Eq.Rgrid[0,  0])
_Rmax = float(Eq.Rgrid[-1, 0])
_Zmin = float(Eq.zgrid[0,  0])
_Zmax = float(Eq.zgrid[0, -1])
_nR   = int((_Rmax - _Rmin) / (_idata.rmin / 80.))
_nZ   = int((_Zmax - _Zmin) / (_idata.rmin / 80.))
_R1d  = np.linspace(_Rmin, _Rmax, _nR)
_Z1d  = np.linspace(_Zmin, _Zmax, _nZ)

print(f"Sampling psi and StixY on {_nR}×{_nZ} grid ...")
psi_2d = IntSample(_R1d, _Z1d, Eq.PsiInt.eval)    # shape (nZ, nR)
rho_2d = np.sqrt(np.clip(psi_2d, 0, None))
_, StixY, _ = StixParamSample(_R1d, _Z1d, Eq, _idata.freq)  # shape (nZ, nR)
print("Done.")

# Optional vessel outline
_vessel_R, _vessel_Z = None, None
try:
    from Tools.PlotData.PlotVessel.plotVessel import plotVessel
    _Rv_in, _Rv_out, _Zv_in, _Zv_out, _Zt, _Rt = plotVessel(_idata)
    _vessel_R = _Rt
    _vessel_Z = _Zt
    print("Vessel outline loaded.")
except Exception as _e:
    print(f"Vessel not available ({_e}), skipping.")

cmap_ray = cm.viridis

fig4, axes4 = plt.subplots(1, len(RUNS), figsize=(5 * len(RUNS), 6), sharey=True)
fig4.suptitle(
    '2D ray trajectories in poloidal plane — coloured by W/W₀\n'
    '(viridis: yellow = full power, purple = absorbed)',
    fontsize=11
)

# Axis limits: ray extent plus margin, clipped to equilibrium domain
all_R_pts, all_Z_pts = [], []
for name, d in data.items():
    for iray in range(nRays):
        mask = active_mask(d, iray)
        if mask.sum() < 2:
            continue
        all_R_pts.extend(np.sqrt(d['X'][iray, mask]**2 + d['Y'][iray, mask]**2))
        all_Z_pts.extend(d['Z'][iray, mask])

_R_pad, _Z_pad = 8.0, 8.0
Rlim = (max(min(all_R_pts) - _R_pad, _Rmin),
        min(max(all_R_pts) + _R_pad, _Rmax))
Zlim = (max(min(all_Z_pts) - _Z_pad, _Zmin),
        min(max(all_Z_pts) + _Z_pad, _Zmax))

for ax, (name, d) in zip(axes4, data.items()):

    # --- flux surface contours (background) ---
    # rho_2d has shape (nZ, nR) — correct for contour(R1d, Z1d, data)
    ax.contourf(_R1d, _Z1d, rho_2d,
                levels=np.linspace(0, 1.05, 22),
                cmap='Blues_r', alpha=0.35, zorder=0)
    ax.contour(_R1d, _Z1d, rho_2d,
               levels=np.arange(0.1, 1.0, 0.1),
               colors='grey', linewidths=0.7, linestyles='dashed', zorder=1)
    ax.contour(_R1d, _Z1d, rho_2d,
               levels=[1.0],
               colors='black', linewidths=1.2, zorder=2)

    # --- cyclotron resonances (lime green lines: n=1 dotted, n=2 dash-dot, n=3 dashed) ---
    plotting_functions.add_cyclotron_resonances(_R1d, _Z1d, StixY, ax)

    # --- vessel outline ---
    if _vessel_R is not None:
        ax.plot(_vessel_R, _vessel_Z, color='dimgrey', lw=1.5, zorder=3)

    # --- coloured ray segments ---
    for iray in range(nRays):
        mask = active_mask(d, iray)
        if mask.sum() < 2:
            continue
        R  = np.sqrt(d['X'][iray, mask]**2 + d['Y'][iray, mask]**2)
        Z  = d['Z'][iray, mask]
        W  = d['Wfct'][iray, mask]
        W0 = W[0] if W[0] > 0 else 1.0
        c  = W / W0

        pts      = np.stack([R, Z], axis=1)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        c_seg    = 0.5 * (c[:-1] + c[1:])

        lc = LineCollection(segments, cmap=cmap_ray, norm=plt.Normalize(0, 1),
                            linewidth=2.5, zorder=4)
        lc.set_array(c_seg)
        ax.add_collection(lc)

        ax.plot(R[0], Z[0], 'o', ms=4, color='white', mec='black', mew=0.6, zorder=5)

    ax.set_xlim(Rlim)
    ax.set_ylim(Zlim)
    ax.set_aspect('equal')
    ax.set_xlabel('R [cm]')
    if ax is axes4[0]:
        ax.set_ylabel('Z [cm]')
    ax.set_title(name, fontsize=10)

# Shared colourbar for W/W₀
sm = cm.ScalarMappable(cmap=cmap_ray, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig4.colorbar(sm, ax=axes4.tolist(), shrink=0.75, pad=0.02)
cbar.set_label('W / W₀  (1 = full power, 0 = absorbed)', fontsize=9)

fig4.tight_layout(rect=[0, 0, 1, 0.93])
out4 = os.path.join(_HERE, 'compare_trajectories.png')
fig4.savefig(out4, dpi=150, bbox_inches='tight')
print(f"Trajectory figure saved → {out4}")
