"""Load a LUKE EDF .mat file and prepare arrays for qlabs_init.

The LUKE EDF file is a MATLAB struct with fields:
  XXf0  (np, nxi, npsi)  -- distribution function f(p, xi, psi)
  p     (1, np)          -- momentum grid in p/(m_e v_th_ref) thermal units
  xi    (1, nxi)         -- pitch-angle cosine at B_min
  psi   (1, npsi)        -- normalised poloidal flux
  beta                   -- scalar  v_th_ref/c = sqrt(Te_ref[keV] / 511)

qlabs_init expects p in p/(m_e c) units.  The conversion is:
  p_mc[i] = p_th[i] * beta

This module does that conversion and returns Fortran-contiguous arrays
ready to pass directly to qlabs_init(p_mc, xi, psi, f0).
"""

import numpy as np
import scipy.io


def load_luke_edf(mat_file, maxwellian=False):
    """Read a LUKE EDF .mat file and return (p_mc, xi_grid, psi_grid, f0).

    Parameters
    ----------
    mat_file : str
        Path to the .mat file containing the 'EDF' struct.

    Returns
    -------
    p_mc : ndarray, shape (np,)
        Momentum grid in p/(m_e c) units.
    xi_grid : ndarray, shape (nxi,)
        Pitch-angle cosine grid at B_min, spanning [-1, 1].
    psi_grid : ndarray, shape (npsi,)
        Normalised poloidal flux grid, spanning [0, 1].
    f0 : ndarray, shape (np, nxi, npsi), Fortran order
        Distribution function values.  Fortran-contiguous layout is
        required by f2py for the edf_in(np, nxi, npsi) dummy argument.
    """
    mat = scipy.io.loadmat(mat_file)
    edf = mat['EDF'][0, 0]

    # Arrays are stored directly in the struct (no extra object wrapping).
    # p, xi, psi come as (1, N) row vectors; XXf0 is already (np, nxi, npsi).
    # beta is a (1, 1) scalar array.
    p_th     = edf['p'].ravel().astype(np.float64)          # (np,)  p/(m_e v_th)
    xi_grid  = edf['xi'].ravel().astype(np.float64)         # (nxi,)
    psi_grid = edf['psi'].ravel().astype(np.float64)        # (npsi,)
    if maxwellian:
        f0       = np.asfortranarray(edf['XXfM'].astype(np.float64))  # (np,nxi,npsi)
    else:
        f0       = np.asfortranarray(edf['XXf0'].astype(np.float64))  # (np,nxi,npsi)
    beta     = float(edf['beta'][0, 0])                     # v_th_ref/c

    # Convert momentum grid to relativistic units p/(m_e c)
    p_mc = p_th * beta

    return p_mc, xi_grid, psi_grid, f0
