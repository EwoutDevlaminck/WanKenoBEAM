"""
test_all_absorbers.py
---------------------
Maxwellian recovery test for all three EC absorption modules:

    1. warmdamp   (Farina, fully-relativistic warm plasma — ecdisp)
    2. dampbq     (Westerhof, weakly-relativistic — westerino)
    3. qlabs      (arbitrary EDF — qlabs)  << loaded with a Maxwell-Jüttner EDF

For modules 1 and 2, a Maxwellian is hard-wired internally.
For module 3, we explicitly build a Maxwell-Jüttner EDF on a (p,xi,psi) grid
and pass it through qlabs_init.  The ratio qlabs/warmdamp should be ≈ 1
everywhere (within numerical quadrature + spline accuracy).

Physical parameters:  170 GHz O-mode, analytical ITER-like profiles.
Temperature is held UNIFORM at Te0 = 15 keV so that the EDF loaded into
qlabs exactly matches the bulk temperature passed to each module.  This is
the precondition for the ratio = 1 recovery test.

Usage:
    cd WanKenoBEAM/RayTracing/lib
    python test_all_absorbers.py

All three shared libraries (farinaECabsorption.so, westerinoECabsorption.so,
qlabsECabsorption.so) must have been built with their respective Makefiles.
If qlabs is not compiled yet, the script runs without it and prints a warning.
"""

import sys
import os
import warnings
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless; change to 'TkAgg' for interactive
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Add each module directory to sys.path so the .so files are found
# ---------------------------------------------------------------------------
_LIB = os.path.dirname(os.path.abspath(__file__))
for _sub in ('ecdisp', 'westerino', 'qlabs'):
    sys.path.insert(0, os.path.join(_LIB, _sub))

# ---------------------------------------------------------------------------
# Import absorption modules
# ---------------------------------------------------------------------------
from farinaECabsorption   import warmdamp
from westerinoECabsorption import dampbq

_qlabs_ok = False
try:
    from qlabsECabsorption import qlabs_init, qlabs_absorption
    _qlabs_ok = True
    print("qlabs module loaded.")
except ImportError as _e:
    warnings.warn(
        f"qlabs not found ({_e}). "
        "Build it with 'make build' in lib/qlabs/. "
        "The script will continue without the qlabs panel.",
        stacklevel=2,
    )

# ===========================================================================
# 1.  Physical and numerical parameters
#     (same profile as ecdisp/test_abs.py and westerino/test_abs.py)
# ===========================================================================
nptx = 300        # spatial grid points
nptn = 100        # N‖ grid points
xmax = 150.0      # spatial range [cm]

f_GHz = 170.0     # beam frequency [GHz]
ne0   = 1.0e14    # peak electron density [cm⁻³]
Lne   = 151.0     # density scale length [cm]
B0    = 5.5       # peak magnetic field [T]
Te0   = 15.0      # electron temperature [keV]  — uniform throughout scan
R0    = 621.0     # major radius [cm]
me    = 9.1e-31   # electron mass [kg]
mode  = +1        # O-mode

x   = np.linspace(-xmax, +xmax, nptx)
Nll = np.linspace(-1.0,  +1.0,  nptn)

# Plasma profiles
ne = ne0 * (1.0 - (x / Lne)**4)**0.1    # electron density [cm⁻³]
B  = 1.0e4 * B0 / (1.0 + x / R0)        # magnetic field [Gauss]
Te = np.full_like(x, Te0)                # uniform temperature [keV]

# Derived frequencies
omega   = 2.0 * np.pi * f_GHz * 1.0e9   # angular frequency [rad/s]
omega_p = 5.64e4  * np.sqrt(ne)          # plasma frequency [rad/s]
omega_c = 1.76e7  * B                    # cyclotron frequency [rad/s]

# Thermal speed conventions:
#   Farina  (warmdamp):   BTH = vth/c = sqrt(Te[keV]/511)          [1D convention]
#   Westerhof (dampbq):   vTe = sqrt(2 kBTe/me)/c                   [3D rms convention]
#     numerical: sqrt(3.2e-16 * Te / me) / c  where 3.2e-16 ≈ 2 × 1.602e-16 J/keV
BTH      = np.sqrt(Te0 / 511.0)                    # = sqrt(1/mu), dimensionless
vTe_west = np.sqrt(3.2e-16 * Te0 / me) / 3.0e8   # Westerhof vTe/c = sqrt(2)*BTH

print(f"BTH (Farina/qlabs)   = {BTH:.5f}")
print(f"vTe (Westerhof)      = {vTe_west:.5f}  (= sqrt(2)*BTH = {np.sqrt(2)*BTH:.5f})")

# ===========================================================================
# 2.  Build Maxwell-Jüttner EDF for qlabs Maxwellian-recovery test
#
#     f_MJ(p, xi, psi) = exp( -mu * (gamma(p) - 1) )
#         p     = |p| / (m_e v_th)    [LUKE normalisation]
#         xi    = p‖/|p| at B_min     (isotropic → no xi dependence here)
#         psi   = normalised poloidal flux (uniform test → no psi dependence)
#         gamma = sqrt( 1 + (BTH * p)^2 )
#         mu    = 511 / Te0 = m_e c^2 / (kB Te0)
#
#     qlabs_init computes log(f_MJ) and natural-spline derivatives internally.
#     Normalization is irrelevant: it cancels in the ratio I_QL / I_Maxw.
# ===========================================================================
if _qlabs_ok:
    mu_edf  = 511.0 / Te0    # relativistic temperature parameter

    # EDF grid sizes — generous enough to cover resonance ellipse at Te0, 170 GHz
    np_edf  = 100    # momentum points: 0 … p_max = 10 (m_e v_th) = 10*BTH (m_e c)
    nxi_edf = 33     # pitch-angle points: -1 … +1
    nps_edf = 2      # poloidal-flux points: trivial (uniform plasma)

    # Build grid in thermal units first (natural for the MJ formula),
    # then convert to p/(m_e c) for qlabs_init (which now requires p_mc).
    p_grid_th = np.linspace(0.0, 10.0, np_edf)          # p / (m_e v_th)
    p_grid_mc = p_grid_th * BTH                          # p / (m_e c)  ← qlabs unit
    xi_grid   = np.linspace(-1.0, 1.0, nxi_edf)
    psi_grid  = np.array([0.0, 1.0])

    # Maxwell-Jüttner: gamma = sqrt(1 + (p_mc)^2) = sqrt(1 + (BTH * p_th)^2)
    # isotropic → no xi or psi dependence
    gamma_1d = np.sqrt(1.0 + p_grid_mc**2)              # shape (np_edf,)
    f_mj_1d  = np.exp(-mu_edf * (gamma_1d - 1.0))       # shape (np_edf,)

    # Broadcast to (np_edf, nxi_edf, nps_edf) and make Fortran-contiguous
    edf_mj = np.asfortranarray(
        np.broadcast_to(f_mj_1d[:, np.newaxis, np.newaxis],
                        (np_edf, nxi_edf, nps_edf)).copy()
    )

    qlabs_init(p_grid_mc, xi_grid, psi_grid, edf_mj, np_edf, nxi_edf, nps_edf)
    print(f"qlabs EDF initialised: MJ at Te0={Te0} keV, "
          f"mu={mu_edf:.2f}, p_mc=[0,{p_grid_mc[-1]:.3f}] (m_e c), "
          f"np={np_edf}, nxi={nxi_edf}")

# ===========================================================================
# 3.  Spatial / N‖ scan
# ===========================================================================
ImNperp_farina = np.zeros((nptx, nptn))
ImNperp_wester = np.zeros((nptx, nptn))
ImNperp_qlabs  = np.zeros((nptx, nptn))

icalled = 0    # Westerhof initialisation flag (0 on first call, 1 thereafter)

print(f"Scanning {nptx} × {nptn} grid …", end='', flush=True)
for ix in range(nptx):
    alpha  = (omega_p[ix] / omega)**2      # X = (ωpe/ω)²
    beta   = (omega_c[ix] / omega)**2      # Y² = (ωce/ω)²
    Te_loc = Te[ix]                        # [keV] — uniform here

    # O-mode cold refractive index: Nr = sqrt(1 - alpha)
    alpha_clamped = min(alpha, 1.0)
    Nr = np.sqrt(1.0 - alpha_clamped)
    if Nr < 1.0e-6:
        continue    # at or beyond O-mode cutoff

    for iN in range(nptn):
        npar = Nll[iN]
        if abs(npar) >= Nr:
            continue    # evanescent

        theta = np.arccos(npar / Nr)    # angle between N and B [rad]

        # --- Farina (warmdamp) ---
        ImNperp_farina[ix, iN] = warmdamp(alpha, beta, Nr, theta, Te_loc, mode)

        # --- Westerhof (dampbq) ---
        #     interface: dampbq(theta, Nr, alpha, beta, vTe, mode, icalled)
        #     Fortran line 253: PNI = AIMAG(CR) * SIN(PTHETA)
        #     so raw output = Im(N_perp) * sin(theta). Must divide to get Im(N_perp).
        _west_raw = dampbq(theta, Nr, alpha, beta, vTe_west, mode, icalled)
        _sin_th   = math.sin(theta)
        ImNperp_wester[ix, iN] = _west_raw / _sin_th if _sin_th > 1e-6 else _west_raw
        if icalled == 0:
            icalled = 1

        # --- qlabs (Maxwell-Jüttner loaded above) ---
        if _qlabs_ok:
            ImNperp_qlabs[ix, iN] = qlabs_absorption(
                alpha, beta, Nr, theta, Te_loc, mode,
                0.5,    # psi_local — any value in [0,1] for uniform test
                1.0,    # b_over_bmin — no trapping (midplane)
            )

print(" done.")

# ===========================================================================
# 4.  Summary statistics
# ===========================================================================
_thresh = 1.0e-8 * ImNperp_farina.max()
mask = ImNperp_farina > _thresh           # Farina non-zero
if _qlabs_ok:
    mask_both = mask & (ImNperp_qlabs > _thresh)  # both modules non-zero

print(f"\n--- Maxwellian recovery statistics ---")
print(f"  Max Im(N⊥) Farina  : {ImNperp_farina.max():.4e}")
print(f"  Max Im(N⊥) Westerhof: {ImNperp_wester.max():.4e}")

if _qlabs_ok:
    print(f"  Max Im(N⊥) qlabs   : {ImNperp_qlabs.max():.4e}")

    # Safe element-wise division: avoid division-by-zero RuntimeWarning from
    # np.where evaluating both branches.  _safe_div returns 0 where denom ≈ 0.
    def _safe_div(num, denom):
        out = np.zeros_like(num)
        ok = np.abs(denom) > 0.0
        out[ok] = num[ok] / denom[ok]
        return out

    # --- Single-mask ratio: Farina non-zero only ---
    # qlabs returns 0 where N‖² + OMC² − 1 ≤ 0 (CNST ≤ 0), i.e. near N‖ ≈ 0
    # at the fundamental resonance. This is a known limitation of the Shkarofsky
    # formalism (shared with damping.f), not a code bug. The mean will be < 1
    # because of those zero-qlabs points.
    _r_full = _safe_div(ImNperp_qlabs, ImNperp_farina)
    r_single = _r_full[mask]
    print(f"\n  [A] qlabs/Farina where Farina > threshold  ({mask.sum()} points):")
    print(f"      mean = {r_single.mean():.6f}  std = {r_single.std():.2e}")
    print(f"      max  = {r_single.max():.6f}   min = {r_single.min():.6f}")

    # --- Double-mask ratio: both modules non-zero (Maxwellian recovery) ---
    r_both = _r_full[mask_both]
    n_good = int((r_both > 0.9).sum())
    print(f"\n  [B] qlabs/Farina where BOTH > threshold  ({mask_both.sum()} points):")
    print(f"      max    = {r_both.max():.6f}   median = {np.median(r_both):.6f}")
    print(f"      mean   = {r_both.mean():.6f}   std    = {r_both.std():.2e}")
    print(f"      ratio > 0.9: {n_good}/{len(r_both)} points ({100.0*n_good/max(len(r_both),1):.1f}%)")
    print(f"      << ideal: mean=1, std≈0, all points > 0.9 >>")

    # Fraction of Farina-active points where qlabs is also active
    pct = 100.0 * mask_both.sum() / mask.sum() if mask.sum() > 0 else 0.0
    print(f"\n  qlabs active fraction: {pct:.1f}% of Farina-active points")

diff_wf = ImNperp_wester - ImNperp_farina
if mask.any():
    print(f"\n  Westerhof vs Farina  max|diff|: {np.abs(diff_wf[mask]).max():.4e}")
    _r_wf = _safe_div(diff_wf, ImNperp_farina)
    print(f"  Westerhof/Farina−1  mean: {_r_wf[mask].mean():.4e}")

# ===========================================================================
# 5.  Plots
# ===========================================================================
_ncols = 3
_nrows = 4 if _qlabs_ok else 3
fig, axes = plt.subplots(_nrows, _ncols, figsize=(16, 4 * _nrows))
fig.suptitle(
    f'EC absorption comparison — 170 GHz O-mode, Te = {Te0} keV (uniform)\n'
    'Maxwellian recovery test: all modules should give identical Im(N⊥)',
    fontsize=12,
)

def _filled(ax, data, title, cmap='viridis', vmin=None, vmax=None):
    cs = ax.contourf(x, Nll, data.T, levels=20, cmap=cmap,
                     vmin=vmin, vmax=vmax)
    ax.set_xlabel('x = R−R₀ [cm]')
    ax.set_ylabel('N‖')
    ax.set_title(title, fontsize=10)
    fig.colorbar(cs, ax=ax)

# Row 0 — plasma profiles
axes[0, 0].plot(x, ne);  axes[0, 0].set(xlabel='x [cm]', ylabel='cm⁻³', title='n_e')
axes[0, 1].plot(x, B);   axes[0, 1].set(xlabel='x [cm]', ylabel='Gauss', title='B')
axes[0, 2].plot(x, Te);  axes[0, 2].set(xlabel='x [cm]', ylabel='keV', title=f'T_e (uniform = {Te0} keV)')
for ax in axes[0]:
    ax.grid(True)

# Row 1 — absorption contours
_filled(axes[1, 0], ImNperp_farina, 'Im(N⊥) — Farina (warmdamp)')
_filled(axes[1, 1], ImNperp_wester, 'Im(N⊥) — Westerhof (dampbq)')
if _qlabs_ok:
    _filled(axes[1, 2], ImNperp_qlabs, 'Im(N⊥) — qlabs  [MJ EDF]')
else:
    axes[1, 2].text(0.5, 0.5, 'qlabs not compiled\n(run make build in lib/qlabs)',
                    ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=11)
    axes[1, 2].set_title('qlabs (unavailable)', fontsize=10)

# Row 2 — differences relative to Farina
_lim_wf = max(np.abs(diff_wf).max(), 1e-20)
_filled(axes[2, 0], diff_wf,
        'Westerhof − Farina', cmap='RdBu_r', vmin=-_lim_wf, vmax=_lim_wf)

if _qlabs_ok:
    diff_qf = ImNperp_qlabs - ImNperp_farina
    _lim_qf = max(np.abs(diff_qf).max(), 1e-20)
    _filled(axes[2, 1], diff_qf,
            'qlabs − Farina  (0 = exact recovery)', cmap='RdBu_r', vmin=-_lim_qf, vmax=_lim_qf)

    # ratio qlabs / Farina (safe division — mask already excludes near-zero Farina)
    _ratio_plot = np.where(mask, _r_full, np.nan)
    cs_r = axes[2, 2].contourf(x, Nll, _ratio_plot.T,
                                levels=np.linspace(0.9, 1.1, 41), cmap='RdBu_r',
                                extend='both')
    axes[2, 2].set_title('qlabs / Farina  (ideal = 1.00)', fontsize=10)
    axes[2, 2].set_xlabel('x [cm]'); axes[2, 2].set_ylabel('N‖')
    fig.colorbar(cs_r, ax=axes[2, 2])
else:
    for col in (1, 2):
        axes[2, col].set_visible(False)

# Row 3 — line cuts at peak absorption
if _nrows == 4:
    ix_peak = int(np.argmax(ImNperp_farina.sum(axis=1)))
    x_peak  = x[ix_peak]

    axes[3, 0].plot(Nll, ImNperp_farina[ix_peak, :],  lw=2,   label='Farina')
    axes[3, 0].plot(Nll, ImNperp_wester[ix_peak, :],  lw=1.5, ls='--', label='Westerhof')
    axes[3, 0].plot(Nll, ImNperp_qlabs[ix_peak,  :],  lw=1.5, ls=':',  label='qlabs (MJ)')
    axes[3, 0].set(xlabel='N‖', ylabel='Im(N⊥)',
                   title=f'Im(N⊥) at x = {x_peak:.1f} cm (peak absorption)')
    axes[3, 0].legend(); axes[3, 0].grid(True)

    axes[3, 1].plot(Nll, ImNperp_qlabs[ix_peak, :] - ImNperp_farina[ix_peak, :])
    axes[3, 1].axhline(0, color='k', lw=0.8)
    axes[3, 1].set(xlabel='N‖', ylabel='qlabs − Farina',
                   title='Residual at peak x (absolute)')
    axes[3, 1].grid(True)

    # relative residual (safe: only divide where Farina is significant)
    _f_line = ImNperp_farina[ix_peak, :]
    _q_line = ImNperp_qlabs[ix_peak, :]
    _active = _f_line > 1e-8 * _f_line.max()
    _rel    = np.where(_active,
                       _safe_div(_q_line - _f_line, _f_line), np.nan)
    axes[3, 2].plot(Nll, _rel)
    axes[3, 2].axhline(0, color='k', lw=0.8)
    axes[3, 2].set(xlabel='N‖', ylabel='(qlabs−Farina)/Farina',
                   title='Relative residual at peak x')
    axes[3, 2].grid(True)

plt.tight_layout()
_out = os.path.join(_LIB, 'test_all_absorbers.png')
plt.savefig(_out, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {_out}")
plt.show()
