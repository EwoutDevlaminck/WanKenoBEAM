# QL_diffusion module — restructuring log

## 1. Pipeline summary

```
WKBeam ray-tracing (binned HDF5)
        │
        ▼
  [QL_diff_driver]  ← QL config file, WKBacca_grids.mat
   rank 0: load data, build task queue, bcast shared state
   rank k: recv psi slice → D_RF(psi_l) → send result back
   rank 0: collect, write QLdiff_*.h5
        │
        ▼
  [QL_diff_coefficient / D_RF]
   D_RF(psi_l):
     1. Trapping boundary (B_min, B_max, Trapksi0, theta_T) on theta_trap (360-pt)
     2. Full-orbit CB kernel for lambda_q normalisation
     3. build_psi_edens → PsiEdens (unified dense/sparse container)
     4. _bounce_integral called twice (ksi0_h, then ksi0_w):
          passing: integrate over all theta_h via get_slice
          trapped: integrate over theta_T_m … theta_T_M via interpolate_trapped
        → D_RF_nobounce at each theta point
     5. Optional Gaussian smooth + trapped symmetrisation
        │
        ▼
  [QL_diff_coefficient / D_RF_nobounce]
   resonance lookup via KDTree, Bessel terms, polarisation cache
        │
        ▼
  [QL_diff_aux]
   helper functions: config_quantities, Trapping_boundary, D_RF_prefactor, ...
```

---

## 2. Current file structure (as of Development/phiN\_coupling, uncommitted)

The restructuring is complete but not yet committed.  The three new files
exist as untracked/modified working-tree changes.

```
QL_diffusion/
  QL_diff_aux.py          469 lines  pure helpers (physics + I/O), no psi loop
  QL_diff_coefficient.py  713 lines  PsiEdens + D_RF_nobounce + _bounce_integral + D_RF
  QL_diff_driver.py       466 lines  MPI orchestration + I/O  (replaces QL_diff_calc.py)
  QL_diff_restructure.md             this file
```

`QL_diff_calc.py` is deleted (working-tree deletion, uncommitted).
`WKBeam.py` entry-point dict has been updated to `QL_diffusion.QL_diff_driver`.
`CommonModules/BiSplineDer.py` has `eval_vec` added to both `BiSpline` and `UniBiSpline`.

---

## 3. Function inventory (current state)

### `QL_diff_aux.py` — pure helpers (469 lines)

| Function | Lines (approx.) | Role |
|---|---|---|
| `_compute_Edens(...)` | 37–76 | Dense Wfct → J m⁻³ N⁻² energy density |
| `read_h5file(filename)` | 79–115 | Read binned HDF5; returns `None` for `BinnedTraces` when absent (sparse-only files) |
| `config_quantities(psi, theta, omega, Eq)` | 117–181 | B, Ne, Te, Stix on (psi, theta) mesh — geometry loop scalar (brentq), all field evals vectorised via `eval_vec` + inline Stix |
| `minmaxB(BInt_at_psi, theta)` | 182–200 | B_min / B_max on a flux surface via scipy minimize |
| `Trapping_boundary(ksi0, BInt, ...)` | 201–242 | TrapB, Trapksi0, theta_roots per (psi, ksi0) array |
| `pTe_from_Te(Te)` | 243–251 | Thermal momentum from T_e [keV] → normalised to m_e c |
| `gamma(p, pTe)` | 252–259 | Relativistic factor from p_norm and p_Te |
| `N_par_resonant(inv_kp, p_Te, Gamma, X, harm)` | 260–267 | Resonant N‖ from resonance condition |
| `polarisation(N2, K_angle, P, R, L, S)` | 268–293 | Cold-plasma polarisation (diag + cross terms) |
| `A_perp(nperp, p_norm, pTe, ksi, X)` | 294–300 | Bessel argument k⊥ ρ_L |
| `D_RF_prefactor(...)` | 301–330 | C_RF normalisation factor (p, ksi0, Ne, Te, ω) |
| `_load_sparse_edens(...)` | 331–405 | COO sparse group → nested edens dict |
| `_interpolate_sparse_slice(...)` | 406–457 | Linear interpolate sparse dict at a theta value |
| `_dense_to_sparse_slice(Wfct_2d)` | 458–469 | Dense [n_npar, n_nperp] → sparse dict (u1=u2=0) |

Deleted from the old aux: `bessel_integrand` (was dead code), `D_RF_nobounce`, `bounce_sum`, `D_RF`.

---

### `QL_diff_coefficient.py` — computation kernel (713 lines)

| Symbol | Kind | Role |
|---|---|---|
| `PsiEdens` | dataclass | Unified energy-density container for one psi surface; `slices` dict maps theta_idx → sparse slice; `has_phi_N` flag; `_dense_interp` for trapped dense path |
| `PsiEdens.get_slice(t_idx)` | method | O(1) lookup of sparse slice for passing particles |
| `PsiEdens.interpolate_trapped(theta_val)` | method | Linear interpolation (sparse) or `RegularGridInterpolator` (dense) for trapped-particle arcs |
| `build_psi_edens(l, Edens, edens_sparse, ...)` | function | Constructs `PsiEdens` for index `l`; dense path pre-converts all theta slices once before returning |
| `D_RF_nobounce(p_norm_w, ksi, ..., npar_tree, ...)` | function | Un-bounce-averaged integrand at one (psi, theta, ksi); single resonance search for both p grids; polarisation cache per (i_npar, i_nperp) cell |
| `_bounce_sum(d_theta, CB, integrand)` | function | Weighted theta-quadrature sum (trapped lambda_q only) |
| `_bounce_integral(ctx, ksi0_grid, ...)` | function | Bounce-averages D_RF_nobounce over one ksi0 grid; writes directly into caller's output-array views; handles both passing and trapped branches; called twice (ksi0_h, ksi0_w) |
| `D_RF(psi, theta_w, ..., edens_sparse=None, ...)` | function | Outer psi loop; builds KDTree once; builds `PsiEdens` per surface; bundles per-surface state into `SimpleNamespace ctx`; calls `_bounce_integral` twice; applies Gaussian smooth + trapped symmetrisation |

Key structural changes relative to the old `D_RF` in `QL_diff_aux.py`:
- **`PsiEdens`**: collapses `use_sparse` branching — no `if use_sparse` inside theta or ksi loops.
- **`_bounce_integral`**: eliminates ksi0_h/ksi0_w code duplication; ~620-line D_RF body → ~250 lines of D_RF + ~220 lines of `_bounce_integral`.
- **`SimpleNamespace ctx`**: replaces the ~20-argument scatter across private calls.
- **Dense pre-conversion**: `build_psi_edens` pre-converts all passing theta slices in one pass (§4.2).
- **KDTree**: built once at the top of `D_RF`, before the psi loop (effective for serial calls; see §4.7 note below).
- **D_rf allocation**: `D_rf_primary`/`D_rf_secondary` allocated once per `j` (ksi0), not per harmonic (§4.4).

---

### `QL_diff_driver.py` — MPI orchestration (466 lines)

| Function | Role |
|---|---|
| `_load_momentum_grids(idata)` | Load p/ksi grids from manual config or `WKBacca_grids.mat`; also reads `ne_ref`, `Te_ref`, `lnc_e_ref` from mat file when present |
| `_sparsify(arr)` | Fortran-order flatten + threshold → (values, mask) for HDF5 sparse storage |
| `_theta_cost(data_slice, use_sparse)` | Count occupied theta bins in a psi slice — used for cost-weighted task sorting |
| `_save_results(...)` | Transpose to LUKE axis order (n_p, n_ksi, n_psi, n_harm), sparsify, write all DRF0 arrays + absorption profile to HDF5 |
| `call_QLdiff(input_file)` | MPI master/worker orchestration (see below) |

`call_QLdiff` changes relative to the old `QL_diff_calc.py`:
- **Task ordering by theta cost** (§4.5): task queue sorted by descending occupied-theta-bin count via `_theta_cost`; psi-index ordering dropped.
- **Equilibrium per worker** (§4.6): only `configfile` path is broadcast; each worker independently calls `TokamakEquilibrium(InputData(configfile))` — no `comm.bcast(Eq)`.
- **Unified task packet**: both sparse and dense branches send `(idx, psi_val, data_slice)` — branching happens once at pack (rank 0) and once at unpack (workers), not scattered through the receive loop.

---

## 4. Optimisation status

| § | Description | Status |
|---|---|---|
| 4.1 Step A | `eval_vec` added to `BiSpline` and `UniBiSpline` in `BiSplineDer.py` | ✅ done |
| 4.1 Step B | `config_quantities` vectorised: geometry loop stays scalar, field evals + Stix are batched NumPy | ✅ done |
| 4.2 | Dense slice pre-conversion moved out of theta loop into `build_psi_edens` | ✅ done |
| 4.3 | ksi0_h/w duplication eliminated via `_bounce_integral` | ✅ done |
| 4.4 | `D_rf_primary`/`D_rf_secondary` allocated once per ksi0 (not per harmonic) | ✅ done |
| 4.5 | MPI task ordering by occupied theta bins (`_theta_cost`) | ✅ done |
| 4.6 | Equilibrium reconstructed per worker — no `bcast(Eq)` | ✅ done |
| 4.7 | KDTree pre-built once per worker, not per psi slice | ✅ done |

`D_RF` accepts an optional `npar_tree=None` parameter; when provided it is reused directly (no rebuild).  `call_QLdiff` builds one tree per worker process before the receive loop and passes it on every `D_RF` call.

---

## 5. Unified input-data handling (implemented as PsiEdens)

`PsiEdens` replaces the `use_sparse` flag that previously forced branching at every level.

```python
@dataclass
class PsiEdens:
    slices: dict           # {theta_idx: {i_npar: {i_nperp: (W, u1_re, u2_re)}}}
    theta_h: np.ndarray    # theta bin centres
    has_phi_N: bool        # True for genuine φ_N Fourier moments (sparse path)
    _dense_interp: ...     # RegularGridInterpolator for trapped dense path
    _npar, _nperp: ...     # grids needed by dense interpolant
```

Both dense and sparse inputs produce a `PsiEdens` via `build_psi_edens`.  Inside `D_RF` and `D_RF_nobounce` there is a single code path:
- Passing particles: `psi_edens.get_slice(t)` — O(1) dict lookup.
- Trapped particles: `psi_edens.interpolate_trapped(theta_val)` — linear interp (sparse) or `RegularGridInterpolator` query (dense).

---

## 6. Entry point

`WKBeam.py` updated:
```python
# was:
'QLdiff': {'procedure': 'call_QLdiff', 'module': 'QL_diffusion.QL_diff_calc'}
# now:
'QLdiff': {'procedure': 'call_QLdiff', 'module': 'QL_diffusion.QL_diff_driver'}
```

All imports in driver and coefficient files are explicit named imports; no wildcard `import *`.

---

## 7. Verification (pending)

1. **Unit tests** (existing): `For_QL_coupling/tests/test_dense_sparse_parity.py`,
   `test_nobounce_fourier.py`, `test_sparse_roundtrip.py` — must all pass without
   modification (they import by function name, not by module).
2. **End-to-end**: Run a single-psi QLdiff job on the TCV test case
   (`EC23_WKBeam_sims/TCV_88612_1.25_QLtest_phiN/iter_0/`) with `mpirun -n 4`
   and compare `QLdiff_L4.h5` DRF0_wh/hw/hh values against a pre-restructure
   baseline.
3. **Performance**: Compare wall-clock time on the full TCV run before and after
   the `config_quantities` vectorisation and (once wired) KDTree pre-build (§4.1, §4.7).

---

## 8. Remaining work

1. **Run unit tests** to confirm no regression from the restructure.
2. **End-to-end comparison** on TCV test case.
3. **Performance check**: wall-clock time before/after `config_quantities` vectorisation and KDTree pre-build (§4.1, §4.7).
