import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.special as sp
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize, fsolve
from scipy.ndimage import gaussian_filter


from CommonModules.input_data import InputData 
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
import RayTracing.modules.dispersion_matrix_cfunctions as disp
import CommonModules.physics_constants as phys

eps = np.finfo(np.float32).eps

# electron mass
m_e         = 9.10938356e-31 # kg
# speed of light
c           = 299792458 # m/s
# electron charge
e           = 1.60217662e-19 # C
# momentum conversion
e_over_me_c2 = e / (m_e * c**2) 


#-------------------------------#
#---Import the QL input from WKBeam---#
#-------------------------------#

def _compute_Edens(Wfct, psi, d_psi, theta, d_theta, Nperp, d_nperp, d_npar, Eq):
    """Convert the wave kinetic energy Wfct to k-space energy density Edens.

    Computes the volume-averaged energy density per phase-space cell,
    normalised so that Edens is in units of J/N^2 (joules per unit N-space volume).

    Parameters
    ----------
    Wfct    : ndarray, shape (n_psi, n_theta, n_npar, n_nperp, ...)
    psi     : 1-D array of flux surface centres
    d_psi   : 1-D array of psi bin widths (len = n_psi)
    theta   : 1-D array of poloidal angle centres
    d_theta : scalar bin width in theta
    Nperp   : 1-D array of Nperp bin centres
    d_nperp : scalar Nperp bin width
    d_npar  : scalar Npar bin width
    Eq      : TokamakEquilibrium instance

    Returns
    -------
    Edens : ndarray, same shape as Wfct[..., 0]
    """
    # Phase-space volume element: dV = 2π * 1e-6 (m³) * d_psi * d_theta * J(psi, theta)
    # where J is the Jacobian of the (psi, theta) → (R, Z) mapping
    ptV = np.zeros((len(psi), len(theta)))
    for l, psi_val in enumerate(psi):
        for t, theta_val in enumerate(theta):
            ptV[l, t] = 2*np.pi * 1e-6 * d_psi[l] * d_theta * Eq.volume_element_J(theta_val, psi_val)

    # k-space volume element per phi_N: dV_N = Nperp * d_Nperp * d_Npar  (cylindrical in N-space)
    dV_N =  Nperp * d_nperp * d_npar

    # Normalise: integrate Wfct over real-space volume, then divide by k-space volume
    # Factor 1/c[cm/s] * 1e6 converts WKBeam units to MJ/cm^3.
    # The binning makes this MJ. Then divide by dV_N to get J/N² and by volume to get J/m³/N^2.
    Edens = Wfct[:, :, :, :, 0] / ptV[:, :, None, None]
    Edens /= dV_N[None, None, None, :]
    Edens *= 1e6 / (100*c)

    return Edens


def read_h5file(filename):
    """Read a RhoThetaN-binned HDF5 file and return the data arrays.

    Returns
    -------
    WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux,
    rhobins, Thetabins, Nparallelbins, Nperpbins
    """
    with h5py.File(filename, 'r') as file:
        WhatToResolve = file['WhatToResolve'][()]
        FreqGHz       = file['FreqGHz'][()]
        mode          = file['Mode'][()]
        # BinnedTraces is absent in sparse-only output files; the caller
        # must not use the returned None when use_sparse=True.
        Wfct = file['BinnedTraces'][()] if 'BinnedTraces' in file else None

        Absorption = file['Absorption'][()] if 'Absorption'    in file else None
        EnergyFlux = file['VelocityField'][()] if 'VelocityField' in file else None

        uniform_bins = bool(file['uniform_bins'][()]) if 'uniform_bins' in file else True

        if uniform_bins:
            rhobins       = np.linspace(file['rhomin'][()],       file['rhomax'][()],       int(file['nmbrrho'][()]) + 1)
            Thetabins     = np.linspace(file['Thetamin'][()],     file['Thetamax'][()],     int(file['nmbrTheta'][()]) + 1)
            Nparallelbins = np.linspace(file['Nparallelmin'][()], file['Nparallelmax'][()], int(file['nmbrNparallel'][()]) + 1)
            Nperpbins     = np.linspace(file['Nperpmin'][()],     file['Nperpmax'][()],     int(file['nmbrNperp'][()]) + 1)
        else:
            rhobins       = file['rhobins'][()]
            Thetabins     = file['Thetabins'][()]
            Nparallelbins = file['Nparallelbins'][()]
            Nperpbins     = file['Nperpbins'][()]

    return WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, rhobins, Thetabins, Nparallelbins, Nperpbins

#-------------------------------#
#--- Calculate configuration space quantities on psi, theta grid---#
#-------------------------------#

def config_quantities(psi, theta, omega, Eq):
    """Evaluate magnetic-field and plasma quantities on a (psi, theta) mesh.

    Parameters
    ----------
    psi   : 1-D array  Normalised poloidal flux surface centres.
    theta : 1-D array  Poloidal angle values [rad].
    omega : float      Wave angular frequency [rad/s].
    Eq    : TokamakEquilibrium

    Returns
    -------
    ptR, ptZ         : (n_psi, n_theta) arrays  Position [m].
    ptBt, ptBR, ptBz : (n_psi, n_theta) arrays  Field components [T].
    ptB              : (n_psi, n_theta) array   |B| [T].
    ptNe             : (n_psi, n_theta) array   Electron density [1e19 m⁻³].
    ptTe             : (n_psi, n_theta) array   Electron temperature [keV].
    P, X, R, L, S   : (n_psi, n_theta) arrays  Stix parameters.
    """
    n_psi, n_theta = len(psi), len(theta)

    # --- Geometry: (R, Z) via root-finding — inherently scalar, loop is unavoidable ---
    ptR = np.empty((n_psi, n_theta))
    ptZ = np.empty((n_psi, n_theta))
    for l, psi_l in enumerate(psi):
        for t, theta_t in enumerate(theta):
            ptR[l, t], ptZ[l, t] = Eq.flux_to_grid_coord(psi_l, theta_t)

    # --- Field evaluations: batch over all points at once ---
    # WKBeam stores (R, Z) in cm internally; convert to m for physics.
    R_flat = ptR.ravel()   # cm
    Z_flat = ptZ.ravel()   # cm

    ptBt = Eq.BtInt.eval_vec(R_flat, Z_flat).reshape(n_psi, n_theta)   # T
    ptBR = Eq.BRInt.eval_vec(R_flat, Z_flat).reshape(n_psi, n_theta)   # T
    ptBz = Eq.BzInt.eval_vec(R_flat, Z_flat).reshape(n_psi, n_theta)   # T
    ptNe = Eq.NeInt.eval_vec(R_flat, Z_flat).reshape(n_psi, n_theta)   # 1e19 m⁻³
    ptTe = Eq.TeInt.eval_vec(R_flat, Z_flat).reshape(n_psi, n_theta)   # keV
    ptB  = np.sqrt(ptBt**2 + ptBR**2 + ptBz**2)

    # --- Stix parameters: vectorised (same formulas as Cython disParam* functions) ---
    # Physical constants (SI)
    _echarge  = 1.60217662e-19   # C
    _emass    = 9.10938356e-31   # kg
    _epsilon0 = 8.854187817e-12  # F/m
    omega_pe = _echarge * np.sqrt(ptNe * 1e19 / _epsilon0 / _emass)  # rad/s
    omega_ce = _echarge * ptB / _emass                                 # rad/s

    P = 1.0 - omega_pe**2 / omega**2
    X = omega_ce / omega
    R = (P + X) / (1.0 + X)
    L = (P - X) / (1.0 - X)
    S = 0.5 * (R + L)

    # Convert position from cm (WKBeam internal) to m
    return ptR / 100, ptZ / 100, ptBt, ptBR, ptBz, ptB, ptNe, ptTe, P, X, R, L, S

#-------------------------------#
#---Functions for trapping boundary---#
#-------------------------------#

#   - B_bounce(psi, ksi0): at what field a particle will bounce
#   - Ksi_trapping(psi): The boundary value for given psi. Particles with smaller ksi will be trapped
#   - theta_T,m and theta_T,M(psi,ksi0): minimal and maximal angle reached by particles (where they meet B_bounce)

def minmaxB(BInt_at_psi, theta):

    minusB_at_psi = -BInt_at_psi(theta)
    # Artificial shift to avoid problems of minimization, we shift the first half of the values and put them at the end.
    # If the optimum happens to lie beyond the original max value, we know it is actually at the theta value that was
    # shifted by 2*pi
    minusB_at_psi_shift = np.concatenate((minusB_at_psi[len(theta)//2:], minusB_at_psi[:len(theta)//2]))
    theta_shift = np.concatenate((theta[len(theta)//2:], theta[:len(theta)//2] + 2*np.pi))

    minusB_at_psiInt = interp1d(theta_shift, minusB_at_psi_shift, kind='cubic')

    minimum = minimize(BInt_at_psi, 0.)
    maximum = minimize(minusB_at_psiInt, np.pi)

    # Return the minimum and maximum
    return BInt_at_psi(minimum.x), -minusB_at_psiInt(maximum.x)

#-------------------------------#

def Trapping_boundary(ksi0, BInt_at_psi, theta_grid=[], B0_in=None, Bmax_in=None, eps = np.finfo(np.float32).eps):
    """Compute the trapping boundary pitch angle and bounce theta limits.

    Parameters
    ----------
    ksi0        : array of pitch-angle cosines (at B_min) to classify
    BInt_at_psi : callable B(theta) on this flux surface (still needed for
                  the theta_root fsolve even when B0/Bmax are precomputed)
    theta_grid  : 1-D array of poloidal angles used by the internal minimiser
                  (only required when B0_in or Bmax_in is None)
    B0_in       : optional precomputed B_min on this flux surface; if given,
                  skips the internal minmaxB call (use Eq.BminInt(psi))
    Bmax_in     : optional precomputed B_max; paired with B0_in
    """
    TrapB = np.zeros_like(ksi0)
    theta_roots = np.zeros((len(ksi0), 2))

    if B0_in is not None and Bmax_in is not None:
        B0, Bmax = B0_in, Bmax_in
    else:
        B0, Bmax = minmaxB(BInt_at_psi, theta_grid)
    TrapB = B0/(1-ksi0**2)# + eps) # Might revision, is eps needed?
    Trapksi0 = np.sqrt(1-B0/Bmax)

    for j, ksi0_val in enumerate(ksi0):
        if abs(ksi0_val) <= Trapksi0:

            def deltaB(x):
                return BInt_at_psi(x) - TrapB[j]

            theta_roots[j] = fsolve(deltaB, [-np.pi/2, np.pi/2])
        else:
            theta_roots[j, 0] = -np.pi
            theta_roots[j, 1] = np.pi

    return TrapB, Trapksi0, theta_roots

#-------------------------------#
#---Helper functions for the calculation of D_RF(psi, theta, p, ksi)---#
#-------------------------------#

#@jit(nopython=True)
def pTe_from_Te(Te):
    """
    Thermal momentum from temperature, normalised to m_e*c
    Te in keV
    
    """
    return np.sqrt(Te / 511)

#@jit(nopython=True)
def gamma(p, pTe):
    """
    Relativistic factor, for p a grid of momenta, normalized to the thermal momentum.
    pTe is the thermal momentum, normalised to m_e*c itself, making the calculation easy
    """
    return np.sqrt(1 + (p*pTe)**2)

#@jit(nopython=True)
def N_par_resonant(inv_kp, p_Te, Gamma, X, harm):
    """
    Calculate the resonant n_par. p_norm and Gamma are of shape (n_p), StixY is a scalar.
    Returns an array of the same shape as p_norm
    """
    return (Gamma - harm*X)/p_Te *inv_kp

#@jit(nopython=True)
def polarisation(N2, K_angle, P, R, L, S):
    """Return polarisation diagonal terms and signed cross-products.

    All quantities are real for a propagating cold-plasma wave (real N²,
    real Stix parameters), so e+, e-, e‖ are all real-valued amplitudes.

    Returns
    -------
    pol_diag  : (3,) array  [|e+|², |e-|², |e‖|²]  — strictly non-negative
    pol_cross : (3,) array  [e+·e-, e-·e‖, e+·e‖]  — signed real products
    """
    PlusOverMinus = (N2 - R)/(N2 - L)
    ParOverMinus  = -(N2 - S)/(N2 - L) * (N2*np.cos(K_angle)*np.sin(K_angle))/(P - N2*np.sin(K_angle)**2)

    emin2  = 1.0 / (1.0 + PlusOverMinus**2 + ParOverMinus**2)
    eplus2 = PlusOverMinus**2 * emin2
    epar2  = ParOverMinus**2  * emin2

    # Signed cross-products (real because all amplitudes are real)
    ep_em  = PlusOverMinus * emin2                       # e+ × e-
    em_epar = ParOverMinus  * emin2                      # e- × e‖
    ep_epar = PlusOverMinus * ParOverMinus * emin2       # e+ × e‖

    return np.array([eplus2, emin2, epar2]), np.array([ep_em, em_epar, ep_epar])

#@jit(nopython=True)
def A_perp(nperp, p_norm, pTe, ksi, X):
    return - nperp * p_norm * pTe * np.sqrt(1-ksi**2) / X

#-------------------------------#
#---Functions for the prefactor of bounce averaged D_RF matrices---#
#-------------------------------#

def D_RF_prefactor(p_norm, ksi0, Ne_ref, Te_ref, omega, eps, lnc_e_ref=None):
    """Compute the momentum-pitch prefactor C_RF for the QL diffusion operator.

    Parameters
    ----------
    Ne_ref     : float  Reference electron density [1e19 m⁻³].
    Te_ref     : float  Reference electron temperature [keV].
    lnc_e_ref  : float, optional
        Coulomb logarithm from LUKE.  When provided it replaces the analytic
        formula (DKE eq. 6.50) to keep the Python prefactor consistent with
        the LUKE collision operator.  Units: dimensionless.
    """
    p_Te     = pTe_from_Te(Te_ref)
    Gamma_Te = gamma(1, p_Te)
    if lnc_e_ref is not None:
        coulomb_log = lnc_e_ref
    else:
        coulomb_log = 31.3 - 0.5 * np.log(1e19*Ne_ref) + np.log(1e3*Te_ref)  # DKE 6.50
    Gamma    = gamma(p_norm, p_Te)
    P_norm, Ksi0 = np.meshgrid(p_norm, ksi0)
    inv_kabsp = 1.0 / (abs(Ksi0) * P_norm + eps)
    omega_pe = disp.disParamomegaP(Ne_ref)
    prefac   = 8*np.pi**2 * Gamma * inv_kabsp / (m_e * omega_pe**2 * coulomb_log * Gamma_Te**3) * (c/omega)

    return prefac.T

#-------------------------------#
# Sparse energy-density loader
#-------------------------------#

def _load_sparse_edens(h5path, psi, d_psi, theta_h, d_theta, npar, d_npar, nperp, d_nperp, Eq):
    """Read the COO sparse group from a WKBeam HDF5 file and return an energy-density dict.

    The function applies the same unit conversion as _compute_Edens so that
    the returned values are in J m⁻³ and can be passed directly to D_RF_nobounce.

    Parameters
    ----------
    h5path          : str   Path to the binned HDF5 file.
    psi, d_psi      : arrays  Radial half-grid and bin widths [a.u.].
    theta_h, d_theta: arrays  Poloidal half-grid and bin widths [rad].
    npar, d_npar    : array, float  Npar half-grid and bin width.
    nperp, d_nperp  : array, float  Nperp half-grid and bin width.
    Eq              : Equilibrium  WKBeam equilibrium object.

    Returns
    -------
    edens_sparse : dict
        Nested dict ``{l: {t_idx: {i_npar: {i_nperp: (W, u1_re, u2_re)}}}}``.
        ``W`` is the energy density in J m⁻³.  ``u1_re`` / ``u2_re`` are the
        cos(φ_N) / cos(2φ_N) Fourier moments with the same J m⁻³ pre-factor.
    """
    factor = 1e6 / (100.0 * c)   # same as _compute_Edens

    # Precompute dV_N[i_nperp] = nperp * d_nperp * d_npar
    dV_N = nperp * d_nperp * d_npar

    # Precompute ptV[l, t] = 2π × 1e-6 × d_psi[l] × d_theta[t] × J(theta, psi)
    # d_theta is a 1-D array (bin widths); Jacobian evaluated at each (psi, theta) centre.
    ptV = np.zeros((len(psi), len(theta_h)))
    for l, psi_l in enumerate(psi):
        for t, theta_l in enumerate(theta_h):
            dt = d_theta[t] if hasattr(d_theta, '__len__') else d_theta
            ptV[l, t] = 2.0*np.pi * 1e-6 * d_psi[l] * dt * Eq.volume_element_J(theta_l, psi_l)

    edens_sparse = {}

    with h5py.File(h5path, 'r') as fid:
        if 'sparse' not in fid:
            raise KeyError(f"No 'sparse' group found in {h5path}. "
                           "Re-run binning with idata.sparse_output = True.")
        grp = fid['sparse']
        shape   = grp['shape'][()]                       # [n_psi, n_theta, n_npar, n_nperp]
        indices = grp['indices'][()]                     # (4, nnz) int32
        BT      = grp['values_BinnedTraces'][()]         # (nnz,) float
        has_u1  = 'values_cos_phiN'  in grp
        has_u2  = 'values_cos2_phiN' in grp
        u1_arr  = grp['values_cos_phiN'][()]  if has_u1 else np.zeros_like(BT)
        u2_arr  = grp['values_cos2_phiN'][()] if has_u2 else np.zeros_like(BT)

    i_psi_arr, i_theta_arr, i_npar_arr, i_nperp_arr = indices

    for k in range(len(BT)):
        l   = int(i_psi_arr[k])
        t   = int(i_theta_arr[k])
        inp = int(i_npar_arr[k])
        inpp= int(i_nperp_arr[k])

        denom = ptV[l, t] * dV_N[inpp]
        if denom == 0.0:
            continue
        conv = factor / denom

        W     = float(BT[k])    * conv
        u1_re = float(u1_arr[k]) * conv
        u2_re = float(u2_arr[k]) * conv

        edens_sparse.setdefault(l, {})
        edens_sparse[l].setdefault(t, {})
        edens_sparse[l][t].setdefault(inp, {})
        edens_sparse[l][t][inp][inpp] = (W, u1_re, u2_re)

    return edens_sparse


def _interpolate_sparse_slice(theta_val, sparse_l, theta_h):
    """Return a linearly interpolated sparse (i_npar -> i_nperp -> (W,u1,u2)) slice.

    Finds the one or two occupied theta bins in ``sparse_l`` that bracket
    ``theta_val`` and linearly interpolates.  If only one side is occupied,
    uses that value unchanged (nearest-neighbour at the edge).

    Parameters
    ----------
    theta_val : float        Target theta value [rad].
    sparse_l  : dict         ``edens_sparse[l]`` for a single psi surface.
    theta_h   : 1-D array    Theta bin centres corresponding to the integer keys.

    Returns
    -------
    result : dict  {i_npar: {i_nperp: (W, u1_re, u2_re)}}  — may be empty.
    """
    occupied = sorted(sparse_l.keys())
    if not occupied:
        return {}

    occ_vals = theta_h[occupied]
    idx = int(np.searchsorted(occ_vals, theta_val))

    if idx == 0:
        return sparse_l[occupied[0]]
    if idx >= len(occupied):
        return sparse_l[occupied[-1]]

    i0, i1 = occupied[idx-1], occupied[idx]
    t0, t1 = occ_vals[idx-1], occ_vals[idx]
    alpha = float((theta_val - t0) / (t1 - t0))

    all_npar = set(sparse_l[i0]) | set(sparse_l[i1])
    result = {}
    for inp in all_npar:
        sub0 = sparse_l[i0].get(inp, {})
        sub1 = sparse_l[i1].get(inp, {})
        all_nperp = set(sub0) | set(sub1)
        sub_interp = {}
        for inpp in all_nperp:
            v0 = sub0.get(inpp, (0.0, 0.0, 0.0))
            v1 = sub1.get(inpp, (0.0, 0.0, 0.0))
            sub_interp[inpp] = (
                (1.0-alpha)*v0[0] + alpha*v1[0],
                (1.0-alpha)*v0[1] + alpha*v1[1],
                (1.0-alpha)*v0[2] + alpha*v1[2],
            )
        result[inp] = sub_interp
    return result


def _dense_to_sparse_slice(Wfct_2d):
    """Convert a dense [n_npar, n_nperp] Wfct slice to a sparse-dict slice.

    Used only by the backward-compatible dense path in D_RF.  Fourier moments
    u1_re / u2_re are set to zero (dense format carries no φ_N information).
    """
    result = {}
    rows, cols = np.where(Wfct_2d > 0)
    for inp, inpp in zip(rows.tolist(), cols.tolist()):
        result.setdefault(inp, {})[inpp] = (float(Wfct_2d[inp, inpp]), 0.0, 0.0)
    return result

