"""
test_vs_damping_ref.py
======================
Compares qlabs against the reference Shkarofsky & Shoucri (2011) damping.f code
using the non-Maxwellian distribution function provided in distr.dat.

Three tiers of comparison:
  [1]  Maxwellian sanity check
       Both codes should reproduce the Maxwellian result:
         damping_ref(f_MJ) / damping_ref(f_MJ_ref) ≈ 1
         qlabs(f_MJ)  / warmdamp                   ≈ 1

  [2]  distr.dat with a single test point (damping.f hardcoded parameters)
       Compare the EDF correction ratio from both codes at:
         TE=4 keV, NH=2, OMC=0.95, PARN=0.6, PERPN=0.4, OMP2=1.0
       Metric: ratio_damping = dint11(EDF) / dint11(MJ)
               ratio_qlabs   = qlabs_absorption(EDF) / warmdamp
       These should agree within the integration accuracy (different methods:
       trapezoidal vs GL-35, cylindrical vs spherical coordinates).

  [3]  Multi-point scan
       Scan over PARN ∈ [-0.8, 0.8] and PERPN fixed, checking that the
       EDF correction factors from both codes agree in trend and magnitude.

Physical context:
  distr.dat is a 400×198 Fokker-Planck EDF for a quasi-linear wave-heated
  plasma.  The high-energy tail is depleted relative to a Maxwellian, so the
  absorption at the resonance ellipse is reduced.  Both codes should show this
  reduction; the exact ratio differs by a few percent due to:
    - Different integration methods (trapezoidal vs GL quadrature)
    - Different coordinate systems (cylindrical vs spherical + chain rule)
    - Spline vs natural spline derivatives

Usage:
    cd WanKenoBEAM/RayTracing/lib/test_damping_ref
    python test_vs_damping_ref.py

Build requirements:
    make build       in this directory   (damping_refECabsorption.so)
    make build       in lib/qlabs/       (qlabsECabsorption.so)
    make build       in lib/ecdisp/      (farinaECabsorption.so)
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE  = os.path.dirname(os.path.abspath(__file__))
_LIB   = os.path.dirname(_HERE)
for _d in [_HERE,
           os.path.join(_LIB, 'ecdisp'),
           os.path.join(_LIB, 'qlabs')]:
    sys.path.insert(0, _d)

# ---------------------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------------------
from damping_refECabsorption import damping_ref_ec
from farinaECabsorption       import warmdamp
from qlabsECabsorption        import qlabs_init, qlabs_absorption

print("All modules loaded.")

# ===========================================================================
# 1.  Load distr.dat  (400 × 198 Fokker-Planck EDF)
#
#     damping.f reads: ((DD(I,J), I=1,NX), J=1,NY)
#     → Fortran column-major: inner index I (p) is fastest.
#     In Python (row-major): shape (NY=198, NX=400); f[j, i] = DD(i+1, j+1).
#
#     Grid: p = (I-1)*PMAX/(NX-1) = (I-1)*30/399  [m_e v_th units, PMAX=30]
#           xi= -1 + (J-1)*2/(NY-1)               [ascending, -1 to +1]
# ===========================================================================
NX_dat = 400
NY_dat = 198
PMAX_dat = 30.0

p_dat  = np.linspace(0.0, PMAX_dat, NX_dat)
xi_dat = np.linspace(-1.0, 1.0, NY_dat)

# Read the file (free-format Fortran: space/newline delimited)
# Fortran drops the exponent letter when |exp| >= 100, giving e.g. "9.47-104".
# Fix all three forms: D±, d±, and bare ±NNN (no exponent letter).
import re as _re
with open(os.path.join(_HERE, 'distr.dat'), 'r') as fh:
    raw = fh.read()
# 1. Replace Fortran D/d exponent with E
raw = raw.replace('D', 'E').replace('d', 'e')
# 2. Fix bare large exponent: "1.234-104" → "1.234E-104" (digit immediately before ±)
raw = _re.sub(r'(\d)([-+])(\d{3,})', r'\1E\2\3', raw)
vals = np.array(raw.split(), dtype=np.float64)

# Re-order: Fortran READ gives vals[(j)*NX_dat + i] = DD(i+1, j+1)
# → f_dat[i, j] (Fortran column-major = Python transposed row-major)
f_dat_ftn = vals.reshape(NY_dat, NX_dat)   # shape (NY, NX), xi outer
f_sph_edf = f_dat_ftn.T                    # shape (NX, NY), p outer  [for qlabs_init]

# Verify grid stats
print(f"\ndistr.dat loaded: {NX_dat}×{NY_dat}, p∈[{p_dat[0]:.1f},{p_dat[-1]:.1f}],"
      f" xi∈[{xi_dat[0]:.1f},{xi_dat[-1]:.1f}]")
print(f"  f max = {f_sph_edf.max():.4e},  f min (nonzero) = {f_sph_edf[f_sph_edf>0].min():.4e}")

# ===========================================================================
# 2.  Build Maxwell-Jüttner EDF on the same grid  (Te=4 keV, same as damping.f)
# ===========================================================================
TE_ref  = 4.0                         # keV — hardcoded in damping.f
BTH_ref = np.sqrt(TE_ref) * 0.044237436  # damping.f line 50; = sqrt(Te/511)
# Verify: sqrt(4/511) = sqrt(0.007828) = 0.08848,  0.044237*2 = 0.088474 ✓
mu_ref  = 511.0 / TE_ref              # = 127.75

# Convert p grid to p/(m_e c) — qlabs_init requires this convention.
# p_dat is in p/(m_e v_th_ref) units; p_dat_mc = p_dat * BTH_ref = p/(m_e c).
p_dat_mc = p_dat * BTH_ref                                # shape (NX_dat,)

gamma_1d = np.sqrt(1.0 + p_dat_mc**2)                    # shape (NX_dat,)
f_mj_1d  = np.exp(-mu_ref * (gamma_1d - 1.0))            # shape (NX_dat,)
# Make (NX_dat, NY_dat): isotropic → broadcast over xi
f_sph_mj = np.broadcast_to(f_mj_1d[:, None], (NX_dat, NY_dat)).copy()

print(f"\nMaxwell-Jüttner at Te={TE_ref} keV: BTH={BTH_ref:.6f}, mu={mu_ref:.2f}")

# ===========================================================================
# 3.  damping.f reference parameters (hardcoded in damping.f)
#
#     TE=4, NH=2, OMC=0.95 (= 2*omega_ce/omega  → omega_ce/omega = 0.475)
#     PARN=0.6, PERPN=0.4, OMP2=1.0
# ===========================================================================
OMC_ref   = 0.95    # N*omega_ce/omega
PARN_ref  = 0.6
PERPN_ref = 0.4
NH_ref    = 2
OMP2_ref  = 1.0

# Derived qlabs/warmdamp parameters:
sqoc_ref  = OMC_ref / NH_ref              # omega_ce/omega = 0.475
oc_ref    = sqoc_ref**2                   # = 0.225625
op_ref    = OMP2_ref                      # = (omega_pe/omega)^2
Nr_ref    = np.sqrt(PARN_ref**2 + PERPN_ref**2)  # total refractive index
theta_ref = np.arccos(PARN_ref / Nr_ref)   # angle between N and B

CNST_ref  = PARN_ref**2 + OMC_ref**2 - 1.0
print(f"\ndamping.f parameters: TE={TE_ref}, NH={NH_ref}, OMC={OMC_ref}")
print(f"  PARN={PARN_ref}, PERPN={PERPN_ref}, Nr={Nr_ref:.4f}, "
      f"theta={np.degrees(theta_ref):.1f}°")
print(f"  oc={oc_ref:.4f}, op={op_ref:.4f}, CNST={CNST_ref:.4f}  (must be > 0)")

# psi_local and b_over_bmin are irrelevant for this test (no trapping)
PSI_LOC = 0.5
B_OVER_BMIN = 1.0

# ===========================================================================
# 4.  Tier 1 — Maxwellian sanity check
# ===========================================================================
print("\n" + "="*60)
print("TIER 1 — Maxwellian sanity check")
print("="*60)

# --- damping_ref with MJ EDF at damping.f parameters
f_sph_mj_fort = np.asfortranarray(f_sph_mj)
d11_mj, d12_mj, d13_mj, d22_mj, d32_mj, d33_mj = \
    damping_ref_ec(f_sph_mj_fort, PMAX_dat,
                   OMC_ref, PARN_ref, PERPN_ref, BTH_ref, NH_ref)

print(f"\ndamping_ref (MJ EDF):")
print(f"  dint11={d11_mj:.4e}  dint22={d22_mj:.4e}  dint33={d33_mj:.4e}")
print(f"  dint12={d12_mj:.4e}  dint13={d13_mj:.4e}  dint32={d32_mj:.4e}")

# --- warmdamp at same parameters
imn_warm = warmdamp(op_ref, oc_ref, Nr_ref, theta_ref, TE_ref, -1)  # X-mode
imn_warm_o = warmdamp(op_ref, oc_ref, Nr_ref, theta_ref, TE_ref, +1)  # O-mode
print(f"\nwarmdamp (X-mode) = {imn_warm:.4e},  warmdamp (O-mode) = {imn_warm_o:.4e}")

# --- qlabs loaded with MJ EDF
f_qlabs_mj = np.asfortranarray(
    np.broadcast_to(f_sph_mj[:, :, np.newaxis], (NX_dat, NY_dat, 2)).copy())
qlabs_init(p_dat_mc, xi_dat, np.array([0.0, 1.0]),
           f_qlabs_mj, NX_dat, NY_dat, 2)

imn_qlabs_mj_x = qlabs_absorption(op_ref, oc_ref, Nr_ref, theta_ref,
                                    TE_ref, -1, PSI_LOC, B_OVER_BMIN)
imn_qlabs_mj_o = qlabs_absorption(op_ref, oc_ref, Nr_ref, theta_ref,
                                    TE_ref, +1, PSI_LOC, B_OVER_BMIN)
_rx = imn_qlabs_mj_x/imn_warm    if imn_warm   > 0 else float('nan')
_ro = imn_qlabs_mj_o/imn_warm_o  if imn_warm_o > 0 else float('nan')
print(f"\nqlabs (MJ EDF, X-mode) = {imn_qlabs_mj_x:.4e}  vs warmdamp = {imn_warm:.4e}"
      f"   ratio = {_rx:.4f}")
print(f"qlabs (MJ EDF, O-mode) = {imn_qlabs_mj_o:.4e}  vs warmdamp = {imn_warm_o:.4e}"
      f"   ratio = {_ro:.4f}")
print("  << both ratios should be ≈ 1.000 for Maxwellian recovery >>")

# ===========================================================================
# 5.  Tier 2 — distr.dat non-Maxwellian EDF comparison at reference point
# ===========================================================================
print("\n" + "="*60)
print("TIER 2 — distr.dat EDF at damping.f reference parameters")
print("="*60)

# --- damping_ref with distr.dat EDF
f_sph_edf_fort = np.asfortranarray(f_sph_edf)
d11_edf, d12_edf, d13_edf, d22_edf, d32_edf, d33_edf = \
    damping_ref_ec(f_sph_edf_fort, PMAX_dat,
                   OMC_ref, PARN_ref, PERPN_ref, BTH_ref, NH_ref)

print(f"\ndamping_ref (distr.dat EDF):")
print(f"  dint11={d11_edf:.4e}  dint22={d22_edf:.4e}  dint33={d33_edf:.4e}")
print(f"  dint12={d12_edf:.4e}  dint13={d13_edf:.4e}  dint32={d32_edf:.4e}")

# Compute correction ratios from damping_ref:
# ratio = DINT_edf / DINT_mj  (how much absorption changes from MJ to actual EDF)
def safe_ratio(a, b, name):
    if abs(b) > 1e-50:
        r = a/b
        print(f"  ratio({name}) = {a:.4e} / {b:.4e} = {r:.4f}")
        return r
    else:
        print(f"  ratio({name}) = undefined (denominator near zero)")
        return np.nan

print(f"\nEDF/MJ correction ratios from damping_ref:")
r11 = safe_ratio(d11_edf, d11_mj, 'dint11')
r12 = safe_ratio(d12_edf, d12_mj, 'dint12')
r13 = safe_ratio(d13_edf, d13_mj, 'dint13')
r22 = safe_ratio(d22_edf, d22_mj, 'dint22')
r32 = safe_ratio(d32_edf, d32_mj, 'dint32')
r33 = safe_ratio(d33_edf, d33_mj, 'dint33')

# --- qlabs loaded with distr.dat EDF
f_qlabs_edf = np.asfortranarray(
    np.broadcast_to(f_sph_edf[:, :, np.newaxis], (NX_dat, NY_dat, 2)).copy())
qlabs_init(p_dat_mc, xi_dat, np.array([0.0, 1.0]),
           f_qlabs_edf, NX_dat, NY_dat, 2)

imn_qlabs_edf_x = qlabs_absorption(op_ref, oc_ref, Nr_ref, theta_ref,
                                     TE_ref, -1, PSI_LOC, B_OVER_BMIN)
imn_qlabs_edf_o = qlabs_absorption(op_ref, oc_ref, Nr_ref, theta_ref,
                                     TE_ref, +1, PSI_LOC, B_OVER_BMIN)

# qlabs correction ratio = absorption(EDF) / warmdamp(MJ)
ratio_qlabs_x = imn_qlabs_edf_x / imn_warm    if imn_warm    > 0 else np.nan
ratio_qlabs_o = imn_qlabs_edf_o / imn_warm_o  if imn_warm_o  > 0 else np.nan

print(f"\nqlabs (distr.dat) vs warmdamp:")
print(f"  X-mode:  qlabs={imn_qlabs_edf_x:.4e}  warmdamp={imn_warm:.4e}"
      f"   ratio={ratio_qlabs_x:.4f}")
print(f"  O-mode:  qlabs={imn_qlabs_edf_o:.4e}  warmdamp={imn_warm_o:.4e}"
      f"   ratio={ratio_qlabs_o:.4f}")

print(f"\nComparison: damping_ref dint11 ratio vs qlabs X-mode ratio:")
print(f"  damping_ref r11 = {r11:.4f}")
print(f"  qlabs X-mode    = {ratio_qlabs_x:.4f}")
if not (np.isnan(r11) or np.isnan(ratio_qlabs_x)):
    diff_pct = abs(r11 - ratio_qlabs_x) / max(abs(r11), abs(ratio_qlabs_x), 1e-10) * 100
    print(f"  Difference: {diff_pct:.1f}%")
    print(f"  << Expected agreement within ~5-20% due to different integration")
    print(f"     methods (trapezoidal vs GL-35) and coordinate systems >>")

# ===========================================================================
# 6.  Tier 3 — PARN scan
# ===========================================================================
print("\n" + "="*60)
print("TIER 3 — PARN scan at fixed PERPN=0.4, TE=4 keV, NH=2")
print("="*60)

PARN_arr  = np.linspace(-0.8, 0.8, 33)
ratio_d11 = np.full(len(PARN_arr), np.nan)
ratio_qlabs_arr = np.full(len(PARN_arr), np.nan)

for idx, PARN_i in enumerate(PARN_arr):
    CNST_i = PARN_i**2 + OMC_ref**2 - 1.0
    if CNST_i <= 0.0:
        continue   # damping impossible at this N//
    Nr_i    = np.sqrt(PARN_i**2 + PERPN_ref**2)
    theta_i = np.arccos(PARN_i / Nr_i)

    # damping_ref
    de11, _, _, _, _, _ = damping_ref_ec(
        f_sph_edf_fort, PMAX_dat,
        OMC_ref, PARN_i, PERPN_ref, BTH_ref, NH_ref)
    dm11, _, _, _, _, _ = damping_ref_ec(
        f_sph_mj_fort, PMAX_dat,
        OMC_ref, PARN_i, PERPN_ref, BTH_ref, NH_ref)
    if abs(dm11) > 1e-50:
        ratio_d11[idx] = de11 / dm11

    # qlabs
    qlabs_init(p_dat_mc, xi_dat, np.array([0.0, 1.0]),
               f_qlabs_edf, NX_dat, NY_dat, 2)
    imn_e = qlabs_absorption(op_ref, oc_ref, Nr_i, theta_i, TE_ref, -1, PSI_LOC, B_OVER_BMIN)
    wd    = warmdamp(op_ref, oc_ref, Nr_i, theta_i, TE_ref, -1)
    if wd > 0:
        ratio_qlabs_arr[idx] = imn_e / wd

mask = np.isfinite(ratio_d11) & np.isfinite(ratio_qlabs_arr)
if mask.any():
    diff = ratio_d11[mask] - ratio_qlabs_arr[mask]
    print(f"\nScan results over {mask.sum()} active PARN points:")
    print(f"  damping_ref r11:   mean={ratio_d11[mask].mean():.4f}  "
          f"min={ratio_d11[mask].min():.4f}  max={ratio_d11[mask].max():.4f}")
    print(f"  qlabs X-mode:      mean={ratio_qlabs_arr[mask].mean():.4f}  "
          f"min={ratio_qlabs_arr[mask].min():.4f}  max={ratio_qlabs_arr[mask].max():.4f}")
    print(f"  |Δ ratio| mean={np.abs(diff).mean():.4f}  max={np.abs(diff).max():.4f}")
    print(f"  << ratios should be < 1 (EDF shows less absorption than MJ)")
    print(f"     and agree in sign and rough magnitude >>")
else:
    print("  No active points in scan.")

# ===========================================================================
# 7.  Plot
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    'qlabs vs Shkarofsky damping.f reference\n'
    f'distr.dat EDF, Te={TE_ref} keV, NH={NH_ref}, '
    f'OMC={OMC_ref}, PERPN={PERPN_ref}',
    fontsize=11)

# Panel (0,0): f(p) at xi=0 for distr.dat vs MJ
ixi0 = np.argmin(np.abs(xi_dat))
ax = axes[0, 0]
ax.semilogy(p_dat, f_sph_edf[:, ixi0], lw=2, label='distr.dat  (Fokker-Planck)')
ax.semilogy(p_dat, f_sph_mj[:, ixi0],  lw=1.5, ls='--', label=f'Maxwell-Jüttner Te={TE_ref} keV')
ax.set(xlabel='p / (m_e v_th)', ylabel='f(p, xi≈0)', title='EDF comparison at xi≈0')
ax.legend(); ax.grid(True)

# Panel (0,1): PARN scan — correction ratios
ax = axes[0, 1]
ax.plot(PARN_arr, ratio_d11,       lw=2,   label='damping_ref  dint11(EDF)/dint11(MJ)')
ax.plot(PARN_arr, ratio_qlabs_arr, lw=1.5, ls='--', label='qlabs  Im(N⊥)(EDF)/warmdamp')
ax.axhline(1.0, color='k', lw=0.8, ls=':', label='ratio = 1 (MJ limit)')
ax.set(xlabel='N‖', ylabel='EDF / MJ ratio',
       title=f'Correction factor vs N‖  (PERPN={PERPN_ref})')
ax.legend(fontsize=9); ax.grid(True)

# Panel (1,0): Absolute dint11 comparison (EDF vs MJ)
PARN_scan2 = PARN_arr[mask]
d11_edf_arr = np.full(len(PARN_arr), np.nan)
d11_mj_arr  = np.full(len(PARN_arr), np.nan)
for idx, PARN_i in enumerate(PARN_arr):
    CNST_i = PARN_i**2 + OMC_ref**2 - 1.0
    if CNST_i <= 0.0: continue
    de11, _, _, _, _, _ = damping_ref_ec(f_sph_edf_fort, PMAX_dat,
                                          OMC_ref, PARN_i, PERPN_ref, BTH_ref, NH_ref)
    dm11, _, _, _, _, _ = damping_ref_ec(f_sph_mj_fort, PMAX_dat,
                                          OMC_ref, PARN_i, PERPN_ref, BTH_ref, NH_ref)
    d11_edf_arr[idx] = de11
    d11_mj_arr[idx]  = dm11

ax = axes[1, 0]
ax.plot(PARN_arr, d11_edf_arr, lw=2,   label='damping_ref  dint11 (EDF)')
ax.plot(PARN_arr, d11_mj_arr,  lw=1.5, ls='--', label='damping_ref  dint11 (MJ)')
ax.set(xlabel='N‖', ylabel='dint11 [arb. units]',
       title='Absolute tensor element dint11 vs N‖')
ax.legend(fontsize=9); ax.grid(True)

# Panel (1,1): All 6 tensor element ratios at the reference point
elements = {'dint11': (d11_edf, d11_mj),
            'dint12': (d12_edf, d12_mj),
            'dint13': (d13_edf, d13_mj),
            'dint22': (d22_edf, d22_mj),
            'dint32': (d32_edf, d32_mj),
            'dint33': (d33_edf, d33_mj)}
ax = axes[1, 1]
names  = list(elements.keys())
ratios = [a/b if abs(b) > 1e-50 else np.nan for (a,b) in elements.values()]
colors = ['C0' if r > 0 else 'C3' for r in ratios]
bars   = ax.bar(names, ratios, color=colors)
ax.axhline(1.0, color='k', lw=0.8, ls='--')
ax.set(ylabel='EDF / MJ', title=f'All tensor element ratios at ref. point\n'
       f'(N‖={PARN_ref}, N⊥={PERPN_ref}, TE={TE_ref} keV)')
ax.tick_params(axis='x', rotation=15)
for bar, r in zip(bars, ratios):
    if np.isfinite(r):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r:.3f}', ha='center', va='bottom', fontsize=9)
ax.grid(True, axis='y')

plt.tight_layout()
_out = os.path.join(_HERE, 'test_vs_damping_ref.png')
plt.savefig(_out, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {_out}")
plt.show()

print("\n--- Test summary ---")
if not np.isnan(r11) and not np.isnan(ratio_qlabs_x):
    print(f"Reference point (N‖={PARN_ref}, PERPN={PERPN_ref}):")
    print(f"  damping_ref dint11 ratio = {r11:.4f}")
    print(f"  qlabs X-mode ratio       = {ratio_qlabs_x:.4f}")
    agree = abs(r11 - ratio_qlabs_x) < 0.25   # within 25% is reasonable
    print(f"  Agreement within 25%: {'PASS' if agree else 'CHECK'}")
print("See test_vs_damping_ref.png for visual comparison.")
