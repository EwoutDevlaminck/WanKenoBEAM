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
        Wfct          = file['BinnedTraces'][()]

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

    ptR, ptZ = np.zeros([len(psi), len(theta)]), np.zeros([len(psi), len(theta)])
    ptBt = np.zeros_like(ptR)
    ptBR = np.zeros_like(ptR)
    ptBz = np.zeros_like(ptR)
    ptB = np.zeros_like(ptR)
    ptNe = np.zeros_like(ptR)
    ptTe = np.zeros_like(ptR)

    P, X, R, L, S = np.zeros_like(ptR), np.zeros_like(ptR), np.zeros_like(ptR), np.zeros_like(ptR), np.zeros_like(ptR)

    for l, psi_l in enumerate(psi):
        for t, theta_t in enumerate(theta):
            ptR[l, t], ptZ[l, t] = Eq.flux_to_grid_coord(psi_l, theta_t)

            ptNe[l, t] = Eq.NeInt.eval(ptR[l, t], ptZ[l, t]) # 1e19 m⁻3
            ptTe[l, t] = Eq.TeInt.eval(ptR[l, t], ptZ[l, t]) # keV

            ptBt[l, t] = Eq.BtInt.eval(ptR[l, t], ptZ[l, t]) # T
            ptBR[l, t] = Eq.BRInt.eval(ptR[l, t], ptZ[l, t])
            ptBz[l, t] = Eq.BzInt.eval(ptR[l, t], ptZ[l, t])

            ptB[l, t] = np.sqrt(ptBt[l, t]**2 + ptBR[l, t]**2 + ptBz[l, t]**2)

            omega_pe = disp.disParamomegaP(ptNe[l, t])
            omega_ce = disp.disParamOmega(ptB[l, t])

            P[l, t] = 1 - omega_pe**2 / (omega**2)
            X[l, t] = omega_ce / omega

            R[l, t] = (P[l, t] + X[l, t])/(1 + X[l, t])
            L[l, t] = (P[l, t] - X[l, t])/(1 - X[l, t])
            S[l, t] = .5 * (R[l, t] + L[l, t])

    #Everything inside the WKbeam code is in cm, so we needed to keep R and Z in cm, but now we want it in m!
    return ptR/100, ptZ/100, ptBt, ptBR, ptBz, ptB, ptNe, ptTe, P, X, R, L, S

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

# Import the bessel functions
import scipy.special as sp

def bessel_integrand(n, x):
    return sp.jn(n, x)**2

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


#-------------------------------#
#---Function for the calculation of D_RF(psi, theta, p, ksi)---#
#----------------------------

def D_RF_nobounce(p_norm_w, ksi, npar, nperp, sparse_slice, Te, P, X, R, L, S, harm, eps,
                  npar_tree, d_npar, d_nperp, p_norm_h=None):
    """Calculate the un-bounce-averaged D_RF integrand on the momentum grid(s).

    Both the whole-grid (p_norm_w) and the optional half-grid (p_norm_h) are
    processed in a single pass, sharing the resonance search and a cache of
    polarisation terms.  Polarisation only depends on (Npar, Nperp), not on
    p_norm, so it is computed at most once per unique (i_npar, i_nperp) cell
    across both grids.

    Parameters
    ----------
    p_norm_w     : 1-D array   Momentum whole-grid (always required).
    ksi          : float       Local pitch-angle cosine at this theta point.
    npar, nperp  : 1-D arrays  Parallel / perpendicular refractive-index grids.
    sparse_slice : dict        {i_npar: {i_nperp: (W, u1_re, u2_re)}} energy
                               density slice at this (psi, theta).  W in J m⁻³;
                               u1_re = W × <cos φ_N>, u2_re = W × <cos 2φ_N>.
    Te           : float       Electron temperature [keV] — used for thermal momentum.
    P, X, R, L, S : float     Stix parameters at this (psi, theta) point.
    harm         : int         Cyclotron harmonic number.
    eps          : float       Small regularisation offset (avoids division by zero).
    npar_tree    : KDTree      Pre-built spatial index over the Npar grid.
    d_npar       : float       Uniform Npar bin width.
    d_nperp      : float       Uniform Nperp bin width.
    p_norm_h     : 1-D array, optional
        Momentum half-grid.  When supplied, both grids are processed together
        and the function returns a tuple ``(result_w, result_h)``.  When
        omitted, only ``p_norm_w`` is processed and a single array is returned.

    Notes on polarisation
    ---------------------
    The full Fourier-averaged power weight per cell is:

        w̃ = W·T₀ + 2·C₂·u2_re + 2·C₁·u1_re

    where T₀ collects the diagonal (φ_N-independent) polarisation terms and
    C₁, C₂ are the Fourier cross-coupling coefficients.  All quantities are
    real for a cold-plasma propagating wave.
    """
    dist_bound = d_npar / 2          # half-bin tolerance for resonance matching
    p_Te       = pTe_from_Te(Te)     # thermal momentum normalised to m_e*c

    # Concatenate both p grids so the resonance search is done in a single pass.
    # n_w marks the split point between the two grids in the concatenated array.
    if p_norm_h is not None:
        p_norm_all = np.concatenate([p_norm_w, p_norm_h])
    else:
        p_norm_all = p_norm_w
    n_w = len(p_norm_w)

    if not sparse_slice:
        # Empty slice — no beam energy at this (psi, theta)
        if p_norm_h is not None:
            return np.zeros(n_w), np.zeros(len(p_norm_h))
        return np.zeros(n_w)

    # --- Resonance condition: for each p, find which Npar bin it resonates with ---
    inv_kp          = 1.0 / (ksi * p_norm_all + eps)
    Gamma           = gamma(p_norm_all, p_Te)
    resonance_N_par = N_par_resonant(inv_kp, p_Te, Gamma, X, harm)

    # Query the pre-built KDTree: points further than d_npar/2 from any bin
    # centre are not in resonance and are skipped entirely.
    dist_N_par, ind_N_par = npar_tree.query(
        np.expand_dims(resonance_N_par, axis=-1), k=2, distance_upper_bound=dist_bound)
    res_condition_N_par = np.where(np.isinf(dist_N_par), -1, ind_N_par)

    # i_res:     indices into p_norm_all of resonant p values
    # n_par_res: corresponding column in res_condition_N_par (k=0 or 1 neighbour)
    i_res, n_par_res = np.where(res_condition_N_par != -1)

    D_RF_integrand = np.zeros(len(p_norm_all))

    # Cache geometry-only (pol_diag, pol_cross, nperp_weight) per (i_npar, i_nperp).
    # Polarisation depends only on N², wave angle, and Stix params — not on p_norm.
    pol_cache = {}

    ksi2_ratio     = ksi**2 / (1.0 - ksi**2 + eps)          # (k‖/k⊥)² factor
    ksi_perp_ratio = ksi / np.sqrt(1.0 - ksi**2 + eps)       # k‖/k⊥  (without 1/√2)

    for i, m in zip(i_res, n_par_res):
        i_npar = res_condition_N_par[i, m]
        if i_npar not in sparse_slice:
            continue

        for i_nperp, (W_val, u1_re_val, u2_re_val) in sparse_slice[i_npar].items():
            if W_val <= 0.0:
                continue

            # Look up or compute geometry-only cache entry
            key = (i_npar, i_nperp)
            if key not in pol_cache:
                N2      = nperp[i_nperp]**2 + npar[i_npar]**2
                K_angle = np.arctan2(nperp[i_nperp], npar[i_npar])
                pol_diag, pol_cross = polarisation(N2, K_angle, P, R, L, S)
                nperp_weight        = d_nperp * nperp[i_nperp]
                pol_cache[key]      = (pol_diag, pol_cross, nperp_weight)

            pol_diag, pol_cross, nperp_weight = pol_cache[key]

            # Bessel argument (depends on p_norm[i])
            a_perp = A_perp(nperp[i_nperp], p_norm_all[i], p_Te, ksi, X)
            Jm1    = sp.jn(harm-1, a_perp)
            Jp1    = sp.jn(harm+1, a_perp)
            Jn     = sp.jn(harm,   a_perp)

            # Diagonal polarisation term (φ_N-independent)
            T0 = (0.5*(pol_diag[0]*Jm1**2 + pol_diag[1]*Jp1**2)
                  + ksi2_ratio * pol_diag[2] * Jn**2)

            # φ_N Fourier cross-coupling coefficients
            # C2 enters with cos(2φ_N) → u2_re;  C1 with cos(φ_N) → u1_re
            C2 = 0.5 * pol_cross[0] * Jm1 * Jp1
            C1 = (ksi_perp_ratio / np.sqrt(2.0)
                  * (pol_cross[2]*Jm1*Jn + pol_cross[1]*Jp1*Jn))

            Pol_term = W_val*T0 + 2.0*C2*u2_re_val + 2.0*C1*u1_re_val
            D_RF_integrand[i] += nperp_weight * Pol_term

    if p_norm_h is not None:
        return D_RF_integrand[:n_w], D_RF_integrand[n_w:]
    return D_RF_integrand


#-------------------------------#
#---Function for a bounce sum---#
#-------------------------------#

def bounce_sum(d_theta_grid_j, CB_j, Func, passing, sigma_dep=False):
    if passing or not sigma_dep:
        # Even if it is trapped, when there's no explicit sigma dependence, we can just sum over the grid
        # Instead of summing over both signs of ksi
        return np.nansum(d_theta_grid_j/(2*np.pi) * CB_j * Func)
    else:
        return 1/2* (np.nansum(d_theta_grid_j/(2*np.pi) * CB_j * Func) + np.sum(d_theta_grid_j/(2*np.pi) * CB_j * -Func))
    
#-------------------------------#
#---Function for the calculation of D_RF---#
#---THIS IS THE MAIN FUNCTION---#
#-------------------------------#

def D_RF(psi, theta_w, p_norm_w, p_norm_h, ksi0_w, ksi0_h, npar, nperp, Edens, Eq, Ne_ref, Te_ref, n=[2, 3], FreqGHz=82.7, DKE_calc=False, gaussian_smooth=False, gaussian_sigma=2, symmetrise_trapped=True, eps=np.finfo(np.float32).eps, edens_sparse=None, lnc_e_ref=None):

    """
    The main function to calculate the RF diffusion coefficients.
    It takes in the following arguments:

        psi: np.array [l]
            The radial coordinate, already on the half grid as required
        theta_w: np.array [t]
            The poloidal coordinate, on the whole grid made of the edges of the bins
        p_norm_w: np.array [i]
            The normalised (to the thermal momentum) momentum grid (whole grid)
        ksi0_w: np.array [j]
            The pitch angle grid (whole grid)
        npar: np.array [length given by WKBeam binning]
            The parallel refractive index
        nperp: np.array [length given by WKBeam binning]
            The perpendicular refractive index
        Edens: np.array [l, t, npar, nperp]
            The electric power density in [J/m^3/[N-volume]] from WKbeam
        Eq: equilibrium object
            The equilibrium object, containing the equilibrium quantities
            From WKBeam too
        n: list of harmonics to take into account
        FreqGHZ: float [GHZ]
        DK_calc: bool, wether to calculate the DKE diffusion coefficients

    Returns:
    
            DRF0_wh: np.array [l, i_w, j_h]
                The FP diffusion coefficient D_RF0 on the whole-half grid
            DRF0D_wh: np.array [l, i_w, j_h]
                The DKE diffusion coefficient D_RF0D on the whole-half grid
            DRF0F_wh: np.array [l, i_w, j_h]
                The DKE convection coefficient D_RF0F on the whole-half grid
            DRF0_hw: np.array [l, i_h, j_w] 
                The FP diffusion coefficient D_RF0 on the half-whole grid
            DRF0D_hw: np.array [l, i_h, j_w]
                The DKE diffusion coefficient D_RF0D on the half-whole grid
            DRF0F_hw: np.array [l, i_h, j_w]
                The DKE convection coefficient D_RF0F on the half-whole grid
            DRF0_hh: np.array [l, i_h, j_h]
                The FP diffusion coefficient D_RF0 on the half-half grid
            DRF0D_hh: np.array [l, i_h, j_h]
                The DKE diffusion coefficient D_RF0D on the half-half grid
    """

    # Timekeeping
    tic_internal = time.time()
    #---------------------------------#
    #---Calculate the quantities needed for the calculation---#
    #---------------------------------#

    omega = phys.AngularFrequency(FreqGHz)

    use_sparse = edens_sparse is not None

    # Precaution to not have exactly 0 values in the grid for ksi, as this would be
    # infintely trapped particles
    # Hopefully will not be needed in final implementation, if LUKE grids are ok.
    ksi0_h[abs(ksi0_h)<1e-4] = 1e-4
    ksi0_w[abs(ksi0_w)<1e-4] = 1e-4


    # For theta. Psi is a half grid already, theta is a full grid

    d_theta = np.diff(theta_w)
    theta_h = theta_w[:-1] + d_theta/2

    # Precalculate quantities in configuration space (binned theta grid).
    # Used for passing-particle B values (to index into the same bins as the Edens),
    # Stix parameters, and global geometry.
    Rp, Zp = Eq.magn_axis_coord_Rz /100 #m
    ptR, ptZ, ptBt, ptBR, ptBz, ptB, ptNe, ptTe, P, X, R, L, S = config_quantities(psi, theta_h, omega, Eq)

    # Dedicated fine geometry grid (360 pts) for accurate trapping-boundary detection.
    # Always computed — used by both dense and sparse paths.  Never used for Edens.
    theta_trap = np.linspace(-np.pi, np.pi, 361)[:-1]
    ptR_tr, ptZ_tr, _, ptBR_tr, ptBz_tr, ptB_tr, _, _, _, _, _, _, _ = \
        config_quantities(psi, theta_trap, omega, Eq)

    # --- Quantities that are constant across all psi and ksi values ---

    # KDTree over the Npar grid: built once and reused in every D_RF_nobounce call
    npar_tree = KDTree(npar.reshape(-1, 1))
    d_npar    = npar[1] - npar[0]
    d_nperp   = nperp[1] - nperp[0]

    # Normalisation prefactors for the three staggered grids.
    # These depend only on Ne_ref, Te_ref, omega (all constants), so they are
    # computed once here rather than inside the psi loop.
    C_RF_wh = D_RF_prefactor(p_norm_w, ksi0_h, Ne_ref, Te_ref, omega, eps, lnc_e_ref)
    C_RF_hw = D_RF_prefactor(p_norm_h, ksi0_w, Ne_ref, Te_ref, omega, eps, lnc_e_ref)
    C_RF_hh = D_RF_prefactor(p_norm_h, ksi0_h, Ne_ref, Te_ref, omega, eps, lnc_e_ref)

    #--------------------------------#
    #---Initialision of grids---#
    #--------------------------------#

    # Trapping boundaries on psi, ksi grids
    # Trapksi [l, -], theta_T [l, j, 2]
    Trapksi0_w, theta_T_w = np.zeros((len(psi), 1)), np.zeros((len(psi), len(ksi0_w), 2))
    Trapksi0_h, theta_T_h = np.zeros((len(psi), 1)), np.zeros((len(psi), len(ksi0_h), 2))

    # Initialise the final 8 matrices, the D_RF matrices [l, i, j]
    # Also the lambda*q matrices [l, j]
    lambda_q_h = np.zeros((len(psi), len(ksi0_h)))
    lambda_q_w = np.zeros((len(psi), len(ksi0_w)))
                          
    DRF0_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(n)))
    DRF0_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(n)))
    DRF0_hh = np.zeros((len(psi), len(p_norm_h), len(ksi0_h), len(n)))

    if DKE_calc:
        DRF0D_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(n)))
        DRF0D_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(n)))
        DRF0D_hh = np.zeros((len(psi), len(p_norm_h), len(ksi0_h), len(n))) 

        DRF0F_wh = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(n)))
        DRF0F_hw = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(n)))
    else:
        DRF0D_wh, DRF0D_hw, DRF0D_hh = np.zeros_like(psi), np.zeros_like(psi), np.zeros_like(psi)
        DRF0F_wh, DRF0F_hw = np.zeros_like(psi), np.zeros_like(psi)



    #--------------------------------#
    # ---Calculation split into the psi grid---#
    #--------------------------------#

    for l, psi_l in enumerate(psi):
        # The calculation is split completely into the psi grid,
        # as the calculation is independent for every psi value

        if use_sparse:
            # --- Sparse path: use the fine theta_trap grid for trapping detection ---
            ptB_Int_at_psi = interp1d(theta_trap, ptB_tr[l, :],
                                      fill_value=np.amax(ptB_tr[l, :]), bounds_error=False)
            B0_psi, Bmax_psi = minmaxB(ptB_Int_at_psi, theta_trap)

            _, Trapksi0_w[l], theta_T_w[l] = Trapping_boundary(ksi0_w, ptB_Int_at_psi, theta_trap,
                                                                 B0_in=B0_psi, Bmax_in=Bmax_psi)
            _, Trapksi0_h[l], theta_T_h[l] = Trapping_boundary(ksi0_h, ptB_Int_at_psi, theta_trap,
                                                                 B0_in=B0_psi, Bmax_in=Bmax_psi)

            # Fine-grid geometry interpolants for trapped-particle orbits
            ptB_interp  = interp1d(theta_trap, ptB_tr[l, :])
            ptBR_interp = interp1d(theta_trap, ptBR_tr[l, :])
            ptBz_interp = interp1d(theta_trap, ptBz_tr[l, :])
            ptR_interp  = interp1d(theta_trap, ptR_tr[l, :])
            ptZ_interp  = interp1d(theta_trap, ptZ_tr[l, :])

            # Set of theta bin indices with nonzero beam energy at this psi
            sparse_l = edens_sparse.get(l, {})
            occupied_theta_set = set(sparse_l.keys())
        else:
            # --- Dense path: use the fine theta_trap grid for trapping detection ---
            # Edens is still interpolated on theta_h below; theta_trap is only for geometry.
            ptB_Int_at_psi = interp1d(theta_trap, ptB_tr[l, :],
                                      fill_value=np.amax(ptB_tr[l, :]), bounds_error=False)
            B0_psi, Bmax_psi = minmaxB(ptB_Int_at_psi, theta_trap)

            _, Trapksi0_w[l], theta_T_w[l] = Trapping_boundary(ksi0_w, ptB_Int_at_psi, theta_trap,
                                                                 B0_in=B0_psi, Bmax_in=Bmax_psi)
            _, Trapksi0_h[l], theta_T_h[l] = Trapping_boundary(ksi0_h, ptB_Int_at_psi, theta_trap,
                                                                 B0_in=B0_psi, Bmax_in=Bmax_psi)

            # Per-field geometry interpolants on the fine grid, matching the sparse path.
            ptB_interp  = interp1d(theta_trap, ptB_tr[l, :])
            ptBR_interp = interp1d(theta_trap, ptBR_tr[l, :])
            ptBz_interp = interp1d(theta_trap, ptBz_tr[l, :])
            ptR_interp  = interp1d(theta_trap, ptR_tr[l, :])
            ptZ_interp  = interp1d(theta_trap, ptZ_tr[l, :])

        #--------------------------------#
        #---Bounce averaging calculation-#
        #--------------------------------#

        # Full-orbit bounce normalization kernel (360-pt theta_trap, always −π…π).
        # lambda_q for passing particles must integrate over the entire orbit,
        # not just the scanned theta range, to avoid a 1/theta_range bias in D_RF.
        R_axis_tr_l  = ptR_tr[l, :] - Rp
        Z_axis_tr_l  = ptZ_tr[l, :] - Zp
        CB_full_orbit = (ptB_tr[l, :] * (R_axis_tr_l**2 + Z_axis_tr_l**2)
                         / (Rp * np.abs(ptBR_tr[l, :] * Z_axis_tr_l
                                        - ptBz_tr[l, :] * R_axis_tr_l)))
        d_theta_full = 2 * np.pi / len(theta_trap)   # uniform step of theta_trap

        if not use_sparse:
            # Dense path: Wfct slice and 3-D interpolant for trapped particles.
            Edens_at_psi     = Edens[l, :, :, :]
            Edens_Int_at_psi = RegularGridInterpolator((theta_h, npar, nperp), Edens_at_psi, bounds_error=False, fill_value=None)


        #--------------------------------#
        #---ksi0 half grid calculation---#
        #--------------------------------#

        for j, ksi0_val in enumerate(ksi0_h):  
            # Now, we iterate over ksi first, defining the accessible 
            # theta grid and calculating the D_RF_nobounce matrices for these points
            # 
  
            if abs(ksi0_val) > Trapksi0_h[l]:
                # Passing!
                # Then it is rather easy, all theta values are accessible
                passing = True      

                theta_grid_j_h = theta_h
                d_theta_grid_j_h = d_theta


                # B0 is the minimum B on this flux surface — already computed once per psi
                B_at_psi_j_h      = ptB[l, :]
                BR_at_psi_j_h     = ptBR[l, :]
                Bz_at_psi_j_h     = ptBz[l, :]
                R_axis_at_psi_j_h = ptR[l, :] - Rp
                Z_axis_at_psi_j_h = ptZ[l, :] - Zp

                # Calculate the internal factor of the bounce integral [t]
                CB_j_h = B_at_psi_j_h * (R_axis_at_psi_j_h**2 + Z_axis_at_psi_j_h**2)\
                  / (Rp * abs(BR_at_psi_j_h*Z_axis_at_psi_j_h - Bz_at_psi_j_h*R_axis_at_psi_j_h))

                # B/B0 ratio and local ksi along the orbit [t]
                B_ratio_h         = B_at_psi_j_h / B0_psi
                ksi_vals          = np.sign(ksi0_val) * np.sqrt(1 - B_ratio_h * (1 - ksi0_val**2))
                ksi0_over_ksi_j_h = ksi0_val / ksi_vals

                # Lambda*q: always integrate over the full orbit so lambda_q is
                # independent of the scanned theta range (avoids 1/range bias).
                ksi_tr_h = np.sign(ksi0_val) * np.sqrt(
                    np.clip(1 - ptB_tr[l, :] / B0_psi * (1 - ksi0_val**2), eps**2, None))
                lambda_q_h[l, j] = np.nansum(
                    d_theta_full / (2 * np.pi) * CB_full_orbit * (ksi0_val / ksi_tr_h))

                # Define the integrands for the different bounce integrals [t]
                # See the notes for the derivation of these
                DRF0_integrand = ksi0_over_ksi_j_h**2 * B_ratio_h
                if DKE_calc:
                    DRF0D_integrand = ksi0_over_ksi_j_h
                    DRF0F_integrand = (B_ratio_h - 1) * ksi0_over_ksi_j_h**3

                # D_RF_nobounce is called once per theta point with both p grids
                # simultaneously (p_norm_h passed as keyword arg).  The function
                # shares the resonance search and polarisation cache across both
                # grids, returning (result_wh, result_hh) in a single pass.
                D_rf_lj_wh = np.zeros((len(theta_h), len(p_norm_w), len(n)))
                D_rf_lj_hh = np.zeros((len(theta_h), len(p_norm_h), len(n)))

                for n_idx, harm in enumerate(n):
                    for t, theta_val in enumerate(theta_grid_j_h):
                        if use_sparse:
                            has_beam = t in occupied_theta_set
                            slice_t  = sparse_l.get(t, {}) if has_beam else {}
                        else:
                            has_beam = Edens_at_psi[t].max() > 0
                            slice_t  = _dense_to_sparse_slice(Edens_at_psi[t]) if has_beam else {}
                        if not has_beam:
                            continue
                        D_rf_lj_wh[t, :, n_idx], D_rf_lj_hh[t, :, n_idx] = \
                            D_RF_nobounce(p_norm_w, ksi_vals[t], npar, nperp,
                                slice_t, Te_ref,
                                P[l, 0], X[l, t], R[l, t], L[l, t], S[l, t], harm, eps,
                                npar_tree, d_npar, d_nperp, p_norm_h=p_norm_h)
            
                    # Vectorised bounce integrals over the full p dimension at once.
                    # w_base[t] = d_theta[t] / (2π) * CB[t] — the theta-quadrature weight.
                    # Multiplying by the integrand and summing over t replaces the p loop.
                    w_base = d_theta_grid_j_h / (2 * np.pi) * CB_j_h   # shape [t]

                    DRF0_wh[l, :, j, n_idx] = np.nansum(
                        (w_base * DRF0_integrand)[:, None] * D_rf_lj_wh[:, :, n_idx], axis=0)
                    DRF0_hh[l, :, j, n_idx] = np.nansum(
                        (w_base * DRF0_integrand)[:, None] * D_rf_lj_hh[:, :, n_idx], axis=0)
                    if DKE_calc:
                        DRF0D_wh[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0D_integrand)[:, None] * D_rf_lj_wh[:, :, n_idx], axis=0)
                        DRF0D_hh[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0D_integrand)[:, None] * D_rf_lj_hh[:, :, n_idx], axis=0)
                        DRF0F_wh[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0F_integrand)[:, None] * D_rf_lj_wh[:, :, n_idx], axis=0)

                    # Apply prefactor and lambda*q normalisation
                    DRF0_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]
                    DRF0_hh[l, :, j, n_idx] *= C_RF_hh[:, j] / lambda_q_h[l, j]
                    if DKE_calc:
                        DRF0D_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]
                        DRF0D_hh[l, :, j, n_idx] *= C_RF_hh[:, j] / lambda_q_h[l, j]
                        DRF0F_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]


            else:
                # Trapped!
                passing = False

                # The theta roots are the boundaries of the region where the particles are trapped
                theta_T_m, theta_T_M = theta_T_h[l, j]

                # Both dense and sparse use theta_w (bin edges) as boundaries.
                # Midpoints then land on theta_h bin centres, so Edens lookups are exact
                # and identical between the two paths — required for 1e-11 parity.
                theta_w_clip = theta_w[(theta_w >= theta_T_m) & (theta_w <= theta_T_M)]
                theta_w_aux = np.concatenate(([theta_T_m], theta_w_clip, [theta_T_M]))

                d_theta_grid_j_h = np.diff(theta_w_aux)
                theta_grid_j_h   = theta_w_aux[:-1] + d_theta_grid_j_h/2

                theta_ref_lo = theta_h[0]
                theta_ref_hi = theta_h[-1]
                if theta_grid_j_h[-1] > theta_ref_hi:
                    theta_grid_j_h[-1] = theta_ref_hi
                if theta_grid_j_h[0]  < theta_ref_lo:
                    theta_grid_j_h[0]  = theta_ref_lo

                # From here, most is the same as the passing case, but fields and
                # Edens must be interpolated onto the restricted theta grid.

                # Use the per-psi interpolants built before the ksi loop
                # (fine-trap interpolants for sparse, coarse binned interpolants for dense)
                B_at_psi_j_h      = ptB_interp(theta_grid_j_h)
                BR_at_psi_j_h     = ptBR_interp(theta_grid_j_h)
                Bz_at_psi_j_h     = ptBz_interp(theta_grid_j_h)
                R_axis_at_psi_j_h = ptR_interp(theta_grid_j_h) - Rp
                Z_axis_at_psi_j_h = ptZ_interp(theta_grid_j_h) - Zp

                # Stix parameters on the restricted theta grid (only theta-dependent ones)
                # -0.0 % (2π) == 2π in IEEE 754, which breaks __maximum_r__.
                # Adding 0.0 converts -0.0 → +0.0 without changing any other value.
                theta_grid_j_h = theta_grid_j_h + 0.0
                _, _, _, _, _, _, _, _, _, X_h, R_h, L_h, S_h = config_quantities([psi_l], theta_grid_j_h, omega, Eq)

                # Bounce integral kernel [t]
                CB_j_h = B_at_psi_j_h * (R_axis_at_psi_j_h**2 + Z_axis_at_psi_j_h**2)\
                  / (Rp * abs(BR_at_psi_j_h*Z_axis_at_psi_j_h - Bz_at_psi_j_h*R_axis_at_psi_j_h))

                # B/B0 ratio and local ksi along the (restricted) orbit [t]
                B_ratio_h         = B_at_psi_j_h / B0_psi
                ksi_vals          = np.sign(ksi0_val) * np.sqrt(1 - B_ratio_h * (1 - ksi0_val**2))
                ksi0_over_ksi_j_h = ksi0_val / ksi_vals

                # Lambda*q normalisation factor for the bounce average
                lambda_q_h[l, j] = bounce_sum(d_theta_grid_j_h, CB_j_h, ksi0_over_ksi_j_h, passing, False)

                # Bounce integrand for DRF0 (DKE terms not computed for trapped particles)
                DRF0_integrand = ksi0_over_ksi_j_h**2 * B_ratio_h

                if not use_sparse:
                    # Dense path: interpolate Edens onto the restricted theta grid
                    Theta_grid_j_h, Npar_grid_j_h, Nperp_grid_j_h = np.meshgrid(
                        theta_grid_j_h, npar, nperp, indexing='ij')
                    Edens_interp_lj_h = Edens_Int_at_psi((Theta_grid_j_h, Npar_grid_j_h, Nperp_grid_j_h))

                # D_RF_nobounce called once per theta with both p grids merged
                D_rf_lj_wh = np.zeros((len(theta_grid_j_h), len(p_norm_w), len(n)))
                D_rf_lj_hh = np.zeros((len(theta_grid_j_h), len(p_norm_h), len(n)))

                for n_idx, harm in enumerate(n):
                    for t, theta_val in enumerate(theta_grid_j_h):
                        if use_sparse:
                            slice_t = _interpolate_sparse_slice(theta_val, sparse_l, theta_h)
                            if not slice_t:
                                continue
                        else:
                            if Edens_interp_lj_h[t].max() <= 0:
                                continue
                            slice_t = _dense_to_sparse_slice(Edens_interp_lj_h[t])
                        D_rf_lj_wh[t, :, n_idx], D_rf_lj_hh[t, :, n_idx] = \
                            D_RF_nobounce(p_norm_w, ksi_vals[t], npar, nperp,
                                slice_t, Te_ref,
                                P[l, 0], X_h[0, t], R_h[0, t], L_h[0, t], S_h[0, t], harm, eps,
                                npar_tree, d_npar, d_nperp, p_norm_h=p_norm_h)
                        
                    # Vectorised bounce integrals over p (same pattern as passing case)
                    w_base = d_theta_grid_j_h / (2 * np.pi) * CB_j_h   # shape [t]

                    DRF0_wh[l, :, j, n_idx] = np.nansum(
                        (w_base * DRF0_integrand)[:, None] * D_rf_lj_wh[:, :, n_idx], axis=0)
                    DRF0_hh[l, :, j, n_idx] = np.nansum(
                        (w_base * DRF0_integrand)[:, None] * D_rf_lj_hh[:, :, n_idx], axis=0)
                    if DKE_calc:
                        DRF0D_wh[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0D_integrand)[:, None] * D_rf_lj_wh[:, :, n_idx], axis=0)
                        DRF0D_hh[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0D_integrand)[:, None] * D_rf_lj_hh[:, :, n_idx], axis=0)
                        DRF0F_wh[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0F_integrand)[:, None] * D_rf_lj_wh[:, :, n_idx], axis=0)

                    # Apply prefactor and lambda*q normalisation
                    DRF0_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]
                    DRF0_hh[l, :, j, n_idx] *= C_RF_hh[:, j] / lambda_q_h[l, j]
                    if DKE_calc:
                        DRF0D_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]
                        DRF0D_hh[l, :, j, n_idx] *= C_RF_hh[:, j] / lambda_q_h[l, j]
                        DRF0F_wh[l, :, j, n_idx] *= C_RF_wh[:, j] / lambda_q_h[l, j]
                
                
        #--------------------------------#
        #---ksi0 whole grid calculation--#
        #--------------------------------#
        for j, ksi0_val in enumerate(ksi0_w):
            if abs(ksi0_val) > Trapksi0_w[l]:

                passing = True

                theta_grid_j_w = theta_h
                d_theta_grid_j_w = d_theta


                # B0 already computed once per psi
                B_at_psi_j_w      = ptB[l, :]
                BR_at_psi_j_w     = ptBR[l, :]
                Bz_at_psi_j_w     = ptBz[l, :]
                R_axis_at_psi_j_w = ptR[l, :] - Rp
                Z_axis_at_psi_j_w = ptZ[l, :] - Zp

                # Bounce integral kernel [t]
                CB_j_w = B_at_psi_j_w * (R_axis_at_psi_j_w**2 + Z_axis_at_psi_j_w**2)\
                  / (Rp * abs(BR_at_psi_j_w*Z_axis_at_psi_j_w - Bz_at_psi_j_w*R_axis_at_psi_j_w))

                # B/B0 ratio and local ksi along the orbit [t]
                B_ratio_w         = B_at_psi_j_w / B0_psi
                ksi_vals          = np.sign(ksi0_val) * np.sqrt(1 - B_ratio_w * (1 - ksi0_val**2))
                ksi0_over_ksi_j_w = ksi0_val / ksi_vals

                # Lambda*q: full orbit, same rationale as ksi0_h loop above.
                ksi_tr_w = np.sign(ksi0_val) * np.sqrt(
                    np.clip(1 - ptB_tr[l, :] / B0_psi * (1 - ksi0_val**2), eps**2, None))
                lambda_q_w[l, j] = np.nansum(
                    d_theta_full / (2 * np.pi) * CB_full_orbit * (ksi0_val / ksi_tr_w))

                # Bounce integrands
                DRF0_integrand = ksi0_over_ksi_j_w**2 * B_ratio_w
                if DKE_calc:
                    DRF0D_integrand = ksi0_over_ksi_j_w
                    DRF0F_integrand = (B_ratio_w - 1) * ksi0_over_ksi_j_w**3

                D_rf_lj_hw = np.zeros((len(theta_h), len(p_norm_h), len(n)))

                for n_idx, harm in enumerate(n):
                    for t, theta_val in enumerate(theta_grid_j_w):
                        if use_sparse:
                            has_beam = t in occupied_theta_set
                            slice_t  = sparse_l.get(t, {}) if has_beam else {}
                        else:
                            has_beam = Edens_at_psi[t].max() > 0
                            slice_t  = _dense_to_sparse_slice(Edens_at_psi[t]) if has_beam else {}
                        if not has_beam:
                            continue
                        D_rf_lj_hw[t, :, n_idx] = \
                            D_RF_nobounce(p_norm_h, ksi_vals[t], npar, nperp,
                                slice_t, Te_ref,
                                P[l, 0], X[l, t], R[l, t], L[l, t], S[l, t], harm, eps,
                                npar_tree, d_npar, d_nperp)

                    # Vectorised bounce integrals over p
                    w_base = d_theta_grid_j_w / (2 * np.pi) * CB_j_w   # shape [t]

                    DRF0_hw[l, :, j, n_idx] = np.nansum(
                        (w_base * DRF0_integrand)[:, None] * D_rf_lj_hw[:, :, n_idx], axis=0)
                    if DKE_calc:
                        DRF0D_hw[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0D_integrand)[:, None] * D_rf_lj_hw[:, :, n_idx], axis=0)
                        DRF0F_hw[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0F_integrand)[:, None] * D_rf_lj_hw[:, :, n_idx], axis=0)

                    # Apply prefactor and lambda*q normalisation
                    DRF0_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]
                    if DKE_calc:
                        DRF0D_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]
                        DRF0F_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]


            else:
                # Trapped!
                passing = False

                # The theta roots are the boundaries of the region where the particles are trapped
                theta_T_m, theta_T_M = theta_T_w[l, j]

                # Both dense and sparse use theta_w (bin edges) as boundaries.
                theta_w_clip = theta_w[(theta_w >= theta_T_m) & (theta_w <= theta_T_M)]
                theta_w_aux = np.concatenate(([theta_T_m], theta_w_clip, [theta_T_M]))

                d_theta_grid_j_w = np.diff(theta_w_aux)
                theta_grid_j_w   = theta_w_aux[:-1] + d_theta_grid_j_w/2

                theta_ref_lo = theta_h[0]
                theta_ref_hi = theta_h[-1]
                if theta_grid_j_w[-1] > theta_ref_hi:
                    theta_grid_j_w[-1] = theta_ref_hi
                if theta_grid_j_w[0]  < theta_ref_lo:
                    theta_grid_j_w[0]  = theta_ref_lo

                # From here, most is the same as the passing case, but fields and
                # Edens must be interpolated onto the restricted theta grid.

                # Use the per-psi interpolants built before the ksi loop
                B_at_psi_j_w      = ptB_interp(theta_grid_j_w)
                BR_at_psi_j_w     = ptBR_interp(theta_grid_j_w)
                Bz_at_psi_j_w     = ptBz_interp(theta_grid_j_w)
                R_axis_at_psi_j_w = ptR_interp(theta_grid_j_w) - Rp
                Z_axis_at_psi_j_w = ptZ_interp(theta_grid_j_w) - Zp

                # -0.0 % (2π) == 2π in IEEE 754, which breaks __maximum_r__.
                theta_grid_j_w = theta_grid_j_w + 0.0
                # Stix parameters on the restricted theta grid
                _, _, _, _, _, _, _, _, _, X_w, R_w, L_w, S_w = config_quantities([psi_l], theta_grid_j_w, omega, Eq)

                # Bounce integral kernel [t]
                CB_j_w = B_at_psi_j_w * (R_axis_at_psi_j_w**2 + Z_axis_at_psi_j_w**2)\
                  / (Rp * abs(BR_at_psi_j_w*Z_axis_at_psi_j_w - Bz_at_psi_j_w*R_axis_at_psi_j_w))

                # B/B0 ratio and local ksi along the (restricted) orbit [t]
                B_ratio_w         = B_at_psi_j_w / B0_psi
                ksi_vals          = np.sign(ksi0_val) * np.sqrt(1 - B_ratio_w * (1 - ksi0_val**2))
                ksi0_over_ksi_j_w = ksi0_val / ksi_vals

                # Lambda*q normalisation factor
                lambda_q_w[l, j] = bounce_sum(d_theta_grid_j_w, CB_j_w, ksi0_over_ksi_j_w, passing, False)

                # Bounce integrand for DRF0
                DRF0_integrand = ksi0_over_ksi_j_w**2 * B_ratio_w

                if not use_sparse:
                    Theta_grid_j_w, Npar_grid_j_w, Nperp_grid_j_w = np.meshgrid(
                        theta_grid_j_w, npar, nperp, indexing='ij')
                    Edens_interp_lj_w = Edens_Int_at_psi((Theta_grid_j_w, Npar_grid_j_w, Nperp_grid_j_w))

                D_rf_lj_hw = np.zeros((len(theta_grid_j_w), len(p_norm_h), len(n)))

                for n_idx, harm in enumerate(n):
                    for t, theta_val in enumerate(theta_grid_j_w):
                        if use_sparse:
                            slice_t = _interpolate_sparse_slice(theta_val, sparse_l, theta_h)
                            if not slice_t:
                                continue
                        else:
                            if Edens_interp_lj_w[t].max() <= 0:
                                continue
                            slice_t = _dense_to_sparse_slice(Edens_interp_lj_w[t])
                        D_rf_lj_hw[t, :, n_idx] = \
                            D_RF_nobounce(p_norm_h, ksi_vals[t], npar, nperp,
                                slice_t, Te_ref,
                                P[l, 0], X_w[0, t], R_w[0, t], L_w[0, t], S_w[0, t], harm, eps,
                                npar_tree, d_npar, d_nperp)
                        
                    # Vectorised bounce integrals over p (same pattern as passing case)
                    w_base = d_theta_grid_j_w / (2 * np.pi) * CB_j_w   # shape [t]

                    DRF0_hw[l, :, j, n_idx] = np.nansum(
                        (w_base * DRF0_integrand)[:, None] * D_rf_lj_hw[:, :, n_idx], axis=0)
                    if DKE_calc:
                        DRF0D_hw[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0D_integrand)[:, None] * D_rf_lj_hw[:, :, n_idx], axis=0)
                        DRF0F_hw[l, :, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            (w_base * DRF0F_integrand)[:, None] * D_rf_lj_hw[:, :, n_idx], axis=0)

                    # Apply prefactor and lambda*q normalisation
                    DRF0_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]
                    if DKE_calc:
                        DRF0D_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]
                        DRF0F_hw[l, :, j, n_idx] *= C_RF_hw[:, j] / lambda_q_w[l, j]

        #---------------------------------------------------------#
        # Optional Gaussian smoothing in (p, ksi) space per psi  #
        # Enable via gaussian_smooth=True in the D_RF call.       #
        #---------------------------------------------------------#
        if gaussian_smooth:
            for n_idx, harm in enumerate(n):
                DRF0_wh[l, :, :, n_idx] = gaussian_filter(DRF0_wh[l, :, :, n_idx], sigma=gaussian_sigma)
                DRF0_hh[l, :, :, n_idx] = gaussian_filter(DRF0_hh[l, :, :, n_idx], sigma=gaussian_sigma)
                DRF0_hw[l, :, :, n_idx] = gaussian_filter(DRF0_hw[l, :, :, n_idx], sigma=gaussian_sigma)

                if DKE_calc:
                    DRF0D_wh[l, :, :, n_idx] = gaussian_filter(DRF0D_wh[l, :, :, n_idx], sigma=gaussian_sigma)
                    DRF0D_hh[l, :, :, n_idx] = gaussian_filter(DRF0D_hh[l, :, :, n_idx], sigma=gaussian_sigma)
                    DRF0D_hw[l, :, :, n_idx] = gaussian_filter(DRF0D_hw[l, :, :, n_idx], sigma=gaussian_sigma)

                    DRF0F_wh[l, :, :, n_idx] = gaussian_filter(DRF0F_wh[l, :, :, n_idx], sigma=gaussian_sigma)
                    DRF0F_hw[l, :, :, n_idx] = gaussian_filter(DRF0F_hw[l, :, :, n_idx], sigma=gaussian_sigma)

        #---------------------------------------------------------#
        # Optional ξ₀ ↔ −ξ₀ symmetrisation in the trapped region #
        #---------------------------------------------------------#
        if symmetrise_trapped:
            # Trapped particles bounce between two mirror points and therefore
            # visit both signs of ξ along their orbit.  Consequently they can
            # resonate with waves at both +N_∥ and −N_∥, but the bounce integral
            # above only sees one sign (determined by sign(ξ₀)).  Averaging
            # D_RF(ξ₀) with D_RF(−ξ₀) for every trapped ξ₀ corrects this,
            # enforcing the physical symmetry D_RF(ξ₀) = D_RF(−ξ₀) in the
            # trapped region (equivalent to the mhu-flip in rfdiff_dke_jd.m).
            # Note: DKE drag/convection terms are odd in ξ₀ and are left unchanged.

            trap_h = float(Trapksi0_h[l])
            trap_w = float(Trapksi0_w[l])

            # ksi0_h grid  →  DRF0_wh and DRF0_hh
            for j in range(len(ksi0_h)):
                if abs(ksi0_h[j]) <= trap_h:
                    j_m = int(np.argmin(np.abs(ksi0_h + ksi0_h[j])))
                    if j < j_m:  # process each (j, j_mirror) pair exactly once
                        avg = (DRF0_wh[l, :, j, :] + DRF0_wh[l, :, j_m, :]) / 2
                        DRF0_wh[l, :, j, :]  = avg
                        DRF0_wh[l, :, j_m, :] = avg
                        avg = (DRF0_hh[l, :, j, :] + DRF0_hh[l, :, j_m, :]) / 2
                        DRF0_hh[l, :, j, :]  = avg
                        DRF0_hh[l, :, j_m, :] = avg

            # ksi0_w grid  →  DRF0_hw
            for j in range(len(ksi0_w)):
                if abs(ksi0_w[j]) <= trap_w:
                    j_m = int(np.argmin(np.abs(ksi0_w + ksi0_w[j])))
                    if j < j_m:
                        avg = (DRF0_hw[l, :, j, :] + DRF0_hw[l, :, j_m, :]) / 2
                        DRF0_hw[l, :, j, :]  = avg
                        DRF0_hw[l, :, j_m, :] = avg
                
    return DRF0_wh, DRF0D_wh, DRF0F_wh, DRF0_hw, DRF0D_hw, DRF0F_hw, DRF0_hh, DRF0D_hh, Trapksi0_h, Trapksi0_w

#-------------------------------#
#---End of function definitions---#
#-------------------------------#