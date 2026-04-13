# Session Notes — 2026-04-09

## Summary

Four main tasks completed: accurate Bmin/Bmax computation, a BminInt consistency fix,
full qlabs integration into WanKenoBeam, and a p_mc refactor of the qlabs Fortran modules.

---

## 1. Accurate `_compute_Bmin_Bmax` in `PlasmaEquilibrium.py`

**Problem:** The old implementation used `np.digitize` grid-binning — inaccurate, rejected.

**New approach** (`CommonModules/PlasmaEquilibrium.py`):
- For each of 60 flux surfaces (psi ∈ [0.01, 0.98]), sample B at 48 poloidal angles via
  `flux_to_grid_coord(psi, theta)`.
- Build a cubic `interp1d` of B(theta), then use `minimize_scalar(method='bounded')` for
  accurate Bmin. For Bmax: rotate the array by N/2 and shift theta-range to [0, 2π] to
  avoid the ±π boundary discontinuity.
- At psi=0, anchor both to B on the magnetic axis.
- `BminInt`, `BmaxInt` are `UnivariateSpline(s=0, k=3)` built from the 61-point arrays.

This mirrors the `minmaxB` logic in `QL_diff_aux.py`.

---

## 2. BminInt Consistency Fix in `QL_diff_aux.py`

**Problem:** After `BminInt` was introduced, `D_RF` was updated to use `Eq.BminInt(psi_l)`
for B0. But `BminInt` uses a *different* theta grid than `ptB_Int_at_psi`. When
`BminInt < min(ptB)`, all `B_ratio = ptB/B0 > 1` → `fsolve` in `Trapping_boundary` finds
no root for theta_T → NaN in `ksi_vals` → `ValueError: 'x' must be finite` in `npar_tree.query`.

**Fix:** `D_RF` reverted to always calling `minmaxB(ptB_Int_at_psi, theta_h)`, which is
self-consistent with the `ptB` values used in the rest of the function.

`Trapping_boundary` retains optional `B0_in`, `Bmax_in` kwargs but they are not used from `D_RF`.

---

## 3. qlabs Integration into WanKenoBeam

**New file:** `RayTracing/lib/qlabs/qlabs_loader.py`
- Reads a LUKE EDF `.mat` file (struct: `XXf0`, `p`, `xi`, `psi`, `beta`).
- Converts `p_th = p/(m_e v_th_ref)` → `p_mc = p_th * beta = p/(m_e c)`.
- Returns `(p_mc, xi_grid, psi_grid, f0)` ready for `qlabs_init`.

**`RayTracing/modules/maintrace.py`:** Added `absorptionModule == 2` block before the ray
loop that calls `load_luke_edf(idata.qlabs_edf_file)` then `qlabs_init(...)`.

**`RayTracing/modules/trace_one_ray.py`:** Added `absorptionModule == 2` branch in the
absorption step:
```python
if self.absorptionModule == 2:
    b_over_bmin = max(1.0, Bnorm / float(self.Eq.BminInt(psi)))
    absImN = qlabs_absorption(parAlpha, parBeta,
                              Nnorm, math.acos(Nparallel / Nnorm),
                              Te, sigma, psi, b_over_bmin)
```
The `max(1.0, ...)` clamp handles any UnivariateSpline oscillation near the axis.

EDF file for TCV shot 88612, t=1250 ms:
`/home/devlamin/BOBAFET/Data/TCV88612_t1250_C/runs/_WKBeam_test/EDF.mat`
Grid: `XXf0(100, 122, 16)`, `p(1,100)` up to ~15 thermal momenta, `xi(1,122)`, `psi(1,16)`,
`beta = v_th/c ≈ 0.0681`.

---

## 4. p_mc Refactor of Fortran qlabs Modules

**Why:** The LUKE EDF p-grid is in thermal units `p_th = p/(m_e v_th_ref)`.
This grid does NOT change with psi (it's normalised to a reference temperature).
If qlabs evaluated f at the local `z = p/(m_e v_th_local)`, the grid-to-z mapping
would shift with local Te — wrong. The fix: store everything on a fixed `p_mc = p/(m_e c)`
grid and pass p_mc to qlabs_init.

### `qlabs_edf.f90`
- Module comment updated: p grid is `p/(m_e c)`.
- Interface of `edf_eval` renamed: `dfdp_par → dfdu_par`, `dfdp_perp → dfdu_perp`,
  `dlogf_dp → dlogf_du`. Two early-return lines also renamed.
- Chain rule formula unchanged (it's dimensionless and coordinate-independent).

### `qlabs.f90` — `resonance_integral`

Key changes:
```fortran
! Before: p_sph = u_total / BTH   (converted to p_th for edf_eval)
! After:  p_sph = u_total          (p_mc — edf_eval now uses p/(m_e c) grid)

! Before: DERIVF_mj = -f_mj / gamma_res * (z_perp_vth * OMC + z_par_vth * ZPERP * PARN)
! After:  DERIVF_mj = -f_mj * mu_bulk / gamma_res * (ZPERP * OMC + ZPAR * ZPERP * PARN)
```

The `mu_bulk` factor is essential: in p_mc coordinates,
`∂f_MJ/∂u_perp = -f_MJ * mu * u_perp / gamma`, so mu must appear explicitly.
Without it, the ratio I_QL/I_MJ = mu ≈ 34 instead of 1 for a Maxwellian EDF.

Removed variables: `z_par_vth`, `z_perp_vth`, `BTH` (no longer needed in resonance_integral).

---

## 5. Test Results

### `test_all_absorbers.py`

Fix: `p_grid` (in p_th units) → `p_grid_mc = p_grid * BTH` before `qlabs_init`.
Also `gamma_1d = sqrt(1 + p_grid_mc**2)` (was `sqrt(1 + (BTH*p_grid_th)^2)` — same physics,
cleaner expression).

Result: mean ratio qlabs/warmdamp = **0.9995**, std = 3.66e-4. **PASS.**

### `test_vs_damping_ref.py`

Fix: added `p_dat_mc = p_dat * BTH_ref` after the `BTH_ref` definition; replaced all
three `qlabs_init(p_dat, ...)` calls with `qlabs_init(p_dat_mc, ...)`.
Simplified `gamma_1d = sqrt(1 + p_dat_mc**2)`.

Results:
- **Tier 1 (Maxwellian recovery):** qlabs/warmdamp X-mode = **0.9996** ≈ 1. PASS.
- **Tier 2 (non-MJ distr.dat):** damping_ref r11 = 0.0727, qlabs X-mode = 0.0874,
  difference = **16.8%** — within expected 5-20% (different integration methods). PASS.
- **Tier 3 (PARN scan):** Both codes show ratios ≪ 1 (depleted tail EDF), mean |Δ| = 0.006.
  Good agreement in sign and magnitude. PASS.

---

## Files Modified

| File | Change |
|------|--------|
| `CommonModules/PlasmaEquilibrium.py` | Replaced `_compute_Bmin_Bmax` with flux-surface sampling + `minimize_scalar` |
| `QL_diffusion/QL_diff_aux.py` | Reverted D_RF to use `minmaxB` (not `Eq.BminInt`) for B0 consistency |
| `RayTracing/lib/qlabs/qlabs_edf.f90` | Renamed derivative variables to `dfdu_*`; updated module comment for p_mc |
| `RayTracing/lib/qlabs/qlabs.f90` | p_sph = u_total (no BTH division); DERIVF_mj includes explicit mu_bulk |
| `RayTracing/lib/qlabs/qlabs_loader.py` | **New** — loads LUKE EDF .mat, converts p_th → p_mc |
| `RayTracing/modules/maintrace.py` | absorptionModule==2 block: load EDF, call qlabs_init |
| `RayTracing/modules/trace_one_ray.py` | absorptionModule==2 branch: qlabs_absorption with b_over_bmin |
| `RayTracing/lib/test_all_absorbers.py` | p_grid_mc conversion; gamma_1d uses p_mc |
| `RayTracing/lib/test_damping_ref/test_vs_damping_ref.py` | p_dat_mc conversion; all qlabs_init calls use p_mc |

---

## Next Steps (after session 1)

- Run a full WanKenoBeam trace with `absorptionModule = 2` and the TCV EDF file to verify
  end-to-end qlabs absorption in a real geometry.
- Set up the QL driver workflow: WanKenoBeam → QL_diffusion → LUKE → new EDF → repeat.
- Consider adding `qlabs_edf_file` to the standard input template and documentation.

---

# Session Notes — 2026-04-09 (continued, session 2)

## Problem: qlabs giving ~6% of Farina absorption

When running WKBeam with the qlabs module (ratio method) using the LUKE Maxwellian EDF (XXfM),
computed absorption was only ~6% of the Farina reference run.

---

## Root Cause

The qlabs ratio method computes:

    Im(N⊥) = Im(N⊥)_Maxw × (I_QL / I_Maxw)

For ratio=1 with a Maxwellian EDF, the loaded f_ql and the internal reference
`f_mj_ref = exp(-mu*(γ-1))` must agree in absolute value on the resonance ellipse.
Since `exp(-mu*(γ-1)) → 1` at p→0, the loaded EDF must satisfy `f(p_min, xi, psi) ≈ 1`.

**LUKE normalization convention:**

    2π ∫ f_LUKE(p_th, xi, psi) p_th² dp_th dxi = n_e(psi) / n_e_ref

At the smallest grid momentum, `f_LUKE(p_min) ≈ N_norm(psi)`, varying widely with psi:

| psi   | ne/ne_ref | Te_local  | N_norm  |
|-------|-----------|-----------|---------|
| 0.005 | 0.8902    | 2.198 keV | ~0.0636 |
| 0.091 | 1.0000    | 2.297 keV | ~0.0668 |
| 0.491 | 0.7448    | 0.644 keV | ~0.337  |
| 0.905 | 0.4356    | 0.188 keV | ~1.252  |

At the hot core (psi≈0), N_norm ≈ 0.064, so I_QL/I_Maxw ≈ 0.064 → ~6% absorption.

---

## Fix: Per-psi normalization in `qlabs_loader.py`

**Design principle**: Keep `qlabs.f90` general and convention-agnostic. Apply LUKE-specific
normalization at the Python loader layer (analogous to YodaU).

In `RayTracing/lib/qlabs/qlabs_loader.py`, `load_luke_edf()` now divides each psi-slice
by the mean of f at the first (smallest) momentum grid point:

```python
N_norm = f0[0, :, :].mean(axis=0)          # shape (npsi,)
N_norm = np.where(N_norm > 0.0, N_norm, 1.0)
f0 = f0 / N_norm[np.newaxis, np.newaxis, :]
```

This ensures `f(p_min, xi, psi) ≈ 1` at every psi, matching `exp(-mu*(γ-1)) ≈ 1` at p→0.

**Result:**
- XXfM / Farina ≈ 0.975 (97.5%) — the 2.5% residual is expected because p_min ≠ 0
- All three tiers of `test_vs_damping_ref.py` continue to PASS (test bypasses loader)

---

## Why NOT the YodaU normalization

YodaU uses: `f_normalized = f_LUKE × ne_ref / (beta³ × ne(psi))`

This makes f integrate to 1 over p_mc space — correct for YodaU's direct absorption integral,
but wrong for the qlabs ratio method. At the core, this factor is ~3168, so f(p_min) >> 1
while f_mj_ref(p_min) = 1, giving ratio ≈ 7.93 instead of 1.

---

## Why NOT fixing normalization inside `qlabs.f90`

A test was made: compute N_norm from `p_g(1)` inside Fortran and use
`f_mj_ref = N_norm × exp(-mu*(γ-1))`. This passed the XXfM/Farina test (~97.6%) but broke
Tier 2/3 of `test_vs_damping_ref.py` because the `distr.dat` reference EDF is Jüttner-
normalized (f(p_min) ≈ 0.0635), and the Fortran fix incorrectly rescaled its reference.

**Key insight**: Normalization is distribution-format-specific. It belongs in the loader,
not in the general qlabs physics module.

---

## Files Modified in Session 2

| File | Change |
|------|--------|
| `RayTracing/lib/qlabs/qlabs_loader.py` | Replaced YodaU normalization with per-psi `f[0,:,ipsi].mean()` normalization; updated docstring |
| `RayTracing/lib/qlabs/qlabs.f90` | Reverted N_norm block; restored `f_mj = exp(-mu_bulk*(gamma_res-1))` |
| `RayTracing/lib/qlabs/qlabs_edf.f90` | Minor revert of comment on `p_g` declaration |
| `StandardCases/TCV_88612_1.25_fluct/compare_runs.py` | Added `fig5`: two-panel ratio-colored trajectory plot |

### `compare_runs.py` fig5 details
- Left panel: `W_qlabs(MJ) / W_Farina` (each normalized to its own first step)
- Right panel: `W_qlabs(EDF) / W_qlabs(MJ)`
- Colormap: RdYlGn, range [0.8, 1.2]
- Saved to `compare_ratio_trajectories.png`

---

## Pending Next Steps

1. Re-run three WKBeam simulations: Farina, QLabs_MJ, QLabs_EDF with fixed qlabs_loader.py
2. Run `compare_runs.py` to generate all figures including new ratio-trajectory plot
3. Validate that XXf0 (quasi-linear EDF) gives physically meaningful absorption vs. MJ
