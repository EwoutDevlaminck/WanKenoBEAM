"""Load a LUKE EDF .mat file and prepare arrays for qlabs_init.

The LUKE EDF file is a MATLAB struct with fields:
  XXf0  (np, nxi, npsi)  -- distribution function f(p, xi, psi)
  XXfM  (np, nxi, npsi)  -- Maxwell-Juettner distribution
  p     (1, np)           -- momentum grid in p/(m_e v_th_ref) thermal units
  xi    (1, nxi)          -- pitch-angle cosine at B_min
  psi   (1, npsi)         -- normalised poloidal flux
  beta                    -- scalar  v_th_ref/c = sqrt(Te_ref[keV] / 511)

LUKE normalization convention:
  2π ∫ f_LUKE(p_th, xi, psi) p_th² dp_th dxi = n_e(psi) / n_e_ref

qlabs ratio method requirement:
  For Im(N⊥) = Im(N⊥)_Maxw × (I_QL / I_Maxw), the ratio equals 1 when
  f_ql is a Maxwellian.  This requires f_ql and the internal reference
  f_mj = exp(-mu*(γ-1)) to agree in absolute value at each (p, xi, psi).

  Since exp(-mu*(γ-1)) → 1 at p → 0, and f_LUKE at small p is
    f_LUKE(p_min, xi, psi) = N_norm(psi) × exp(-mu_local*(γ(p_min)-1)) ≈ N_norm(psi)
  we normalize by dividing each psi slice by N_norm(psi), so that the
  loaded EDF starts at ≈ 1 (matching exp(-mu*(γ-1)) at p_min):

    f_normalized(p, xi, psi) = f_LUKE(p, xi, psi) / N_norm(psi)

  where N_norm(psi) = average of f_LUKE(p_min, :, psi) over the xi grid.

  This is correct for both the Maxwellian (XXfM) and the quasi-linear (XXf0)
  distributions, and keeps qlabs.f90 general (no convention assumptions).
"""

import numpy as np
import scipy.io


def load_luke_edf(mat_file, maxwellian=False):
    """Read a LUKE EDF .mat file and return (p_mc, xi_grid, psi_grid, f0).

    The returned f0 is normalized so that f(p_min, xi, psi) ≈ 1 at each psi,
    matching the convention of qlabs' internal Maxwellian reference
    f_mj = exp(-mu*(γ-1)) which also equals 1 at p → 0.

    Parameters
    ----------
    mat_file : str
        Path to the .mat file containing the 'EDF' struct.
    maxwellian : bool
        If True, load XXfM; otherwise load XXf0.

    Returns
    -------
    p_mc : ndarray, shape (np,)
        Momentum grid in p/(m_e c) units.
    xi_grid : ndarray, shape (nxi,)
        Pitch-angle cosine grid at B_min, spanning [-1, 1].
    psi_grid : ndarray, shape (npsi,)
        Normalised poloidal flux grid, spanning [0, 1].
    f0 : ndarray, shape (np, nxi, npsi), Fortran order
        Distribution function, normalized so f(p_min, xi, psi) ≈ 1.
    """
    mat = scipy.io.loadmat(mat_file)
    edf = mat['EDF'][0, 0]

    p_th    = edf['p'].ravel().astype(np.float64)           # (np,)  p/(m_e v_th)
    xi_grid = edf['xi'].ravel().astype(np.float64)          # (nxi,)
    psi_grid= edf['psi'].ravel().astype(np.float64)         # (npsi,)
    beta    = float(edf['beta'][0, 0])                      # v_th_ref/c

    if maxwellian:
        f0 = edf['XXfM'].astype(np.float64)                 # (np, nxi, npsi)
    else:
        f0 = edf['XXf0'].astype(np.float64)                 # (np, nxi, npsi)

    # Convert momentum grid to relativistic units p/(m_e c)
    p_mc = p_th * beta

    # Per-psi normalization: at the smallest grid momentum, exp(-mu*(γ-1)) ≈ 1,
    # so f_LUKE(p_min, xi, psi) ≈ N_norm(psi).  Dividing by N_norm makes the
    # loaded EDF start at ≈ 1, matching the qlabs reference f_mj = exp(-mu*(γ-1)).
    # Use the mean over xi at the first p point to reduce noise.
    N_norm = f0[0, :, :].mean(axis=0)                      # (npsi,)
    N_norm = np.where(N_norm > 0.0, N_norm, 1.0)           # guard against zeros
    f0 = f0 / N_norm[np.newaxis, np.newaxis, :]             # (np, nxi, npsi)

    return p_mc, xi_grid, psi_grid, np.asfortranarray(f0)
