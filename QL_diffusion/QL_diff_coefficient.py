"""Quasi-linear RF diffusion coefficient computation.

Provides the two main computational kernels:

  D_RF_nobounce  — un-bounce-averaged integrand at one (psi, theta, ksi) point
  D_RF           — full bounce-averaged tensor over all (psi, ksi) combinations

Both functions operate on a unified PsiEdens container that abstracts over the
dense (theta-grid Wfct array) and sparse (phi_N COO) input formats.  The
use_sparse flag is evaluated once per psi surface during PsiEdens construction;
it never appears inside the theta or ksi loops.
"""

import time
import numpy as np
import scipy.special as sp
from dataclasses import dataclass, field
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from types import SimpleNamespace

import CommonModules.physics_constants as phys
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
from QL_diffusion.QL_diff_aux import (
    config_quantities,
    minmaxB,
    Trapping_boundary,
    pTe_from_Te,
    gamma,
    N_par_resonant,
    polarisation,
    A_perp,
    D_RF_prefactor,
    _dense_to_sparse_slice,
    _interpolate_sparse_slice,
)

_EPS_F32 = np.finfo(np.float32).eps


# ---------------------------------------------------------------------------
# Unified energy-density container (one psi surface)
# ---------------------------------------------------------------------------

@dataclass
class PsiEdens:
    """Energy density for one psi surface, unified for dense and sparse input.

    Attributes
    ----------
    slices : dict
        {theta_idx (int): sparse_slice_dict} mapping theta bin index to a
        ``{i_npar: {i_nperp: (W, u1_re, u2_re)}}`` dict.  Non-occupied bins
        are absent from the dict (O(1) test for beam presence).
    theta_h : 1-D ndarray
        Theta bin centres [rad] — used by ``interpolate_trapped`` to map float
        theta values back to bin indices.
    has_phi_N : bool
        True when u1_re / u2_re carry genuine Fourier moments (sparse path).
        False for dense input where u1 = u2 = 0.
    _dense_interp : RegularGridInterpolator or None
        Present for the dense trapped path; None for the sparse path.
    _npar, _nperp : 1-D ndarrays or None
        Npar/Nperp grids needed by the dense trapped interpolant.
    """
    slices: dict
    theta_h: np.ndarray
    has_phi_N: bool = False
    _dense_interp: object = field(default=None, repr=False)
    _npar: np.ndarray = field(default=None, repr=False)
    _nperp: np.ndarray = field(default=None, repr=False)

    def get_slice(self, t_idx: int) -> dict:
        """Return the sparse dict for theta bin t_idx (passing path, O(1))."""
        return self.slices.get(t_idx, {})

    def interpolate_trapped(self, theta_val: float) -> dict:
        """Return interpolated sparse dict at an arbitrary theta value (trapped path).

        For the sparse input the 1-D linear interpolation between adjacent
        occupied theta bins is used.  For the dense input the pre-built
        RegularGridInterpolator is queried and the result is converted to a
        sparse dict on the fly.
        """
        if self._dense_interp is not None:
            # Dense: 3-D (theta, npar, nperp) interpolation → sparse dict
            npar, nperp = self._npar, self._nperp
            Npar_g, Nperp_g = np.meshgrid(npar, nperp, indexing='ij')
            pts = np.column_stack([
                np.full(npar.size * nperp.size, theta_val),
                Npar_g.ravel(),
                Nperp_g.ravel(),
            ])
            vals = self._dense_interp(pts).reshape(npar.size, nperp.size)
            return _dense_to_sparse_slice(vals)
        else:
            return _interpolate_sparse_slice(theta_val, self.slices, self.theta_h)


def build_psi_edens(l: int, Edens, edens_sparse, theta_h: np.ndarray,
                    npar: np.ndarray, nperp: np.ndarray, use_sparse: bool) -> PsiEdens:
    """Construct a PsiEdens for psi surface index l.

    Parameters
    ----------
    l            : int   Index into the psi grid.
    Edens        : ndarray or None  Shape (n_psi, n_theta, n_npar, n_nperp).
                   Required when use_sparse=False.
    edens_sparse : dict or None   Full sparse dict {l: {t: ...}}.
                   Required when use_sparse=True.
    theta_h      : 1-D ndarray  Theta bin centres.
    npar, nperp  : 1-D ndarrays  Npar / Nperp grids.
    use_sparse   : bool

    Returns
    -------
    PsiEdens
    """
    if use_sparse:
        sparse_l = edens_sparse.get(l, {})
        return PsiEdens(slices=sparse_l, theta_h=theta_h, has_phi_N=True)
    else:
        Edens_at_psi = Edens[l]                            # (n_theta, n_npar, n_nperp)
        # Pre-convert all passing theta slices once — avoids per-theta conversion
        # inside the ksi loop.  Only non-empty slices are stored.
        slices = {
            t: _dense_to_sparse_slice(Edens_at_psi[t])
            for t in range(len(theta_h))
            if Edens_at_psi[t].max() > 0
        }
        # Build 3-D interpolant for the trapped path (theta may fall between bins)
        dense_interp = RegularGridInterpolator(
            (theta_h, npar, nperp), Edens_at_psi,
            bounds_error=False, fill_value=None,
        )
        return PsiEdens(
            slices=slices,
            theta_h=theta_h,
            has_phi_N=False,
            _dense_interp=dense_interp,
            _npar=npar,
            _nperp=nperp,
        )


# ---------------------------------------------------------------------------
# Un-bounce-averaged integrand
# ---------------------------------------------------------------------------

def D_RF_nobounce(p_norm_w, ksi, npar, nperp, sparse_slice, Te, P, X, R, L, S,
                  harm, eps, npar_tree, d_npar, d_nperp, p_norm_h=None):
    """Un-bounce-averaged D_RF integrand at one (psi, theta, ksi) point.

    Both the whole-grid (p_norm_w) and the optional half-grid (p_norm_h) are
    processed in a single pass sharing the resonance search and a cache of
    polarisation terms.  Polarisation depends only on (Npar, Nperp), not on
    p_norm, so it is computed at most once per unique (i_npar, i_nperp) cell.

    Parameters
    ----------
    p_norm_w     : 1-D ndarray   Momentum whole-grid (normalised to m_e c).
    ksi          : float         Local pitch-angle cosine at this theta point.
    npar, nperp  : 1-D ndarrays  Parallel / perpendicular refractive-index grids.
    sparse_slice : dict          {i_npar: {i_nperp: (W, u1_re, u2_re)}} energy
                                 density at this (psi, theta).  W in J m⁻³.
    Te           : float         Electron temperature [keV].
    P, X, R, L, S : float       Stix cold-plasma parameters.
    harm         : int           Cyclotron harmonic number.
    eps          : float         Small regularisation offset.
    npar_tree    : KDTree        Pre-built spatial index over the Npar grid.
    d_npar       : float         Uniform Npar bin width.
    d_nperp      : float         Uniform Nperp bin width.
    p_norm_h     : 1-D ndarray, optional
        Momentum half-grid.  When given both grids share one resonance search
        and the function returns ``(result_w, result_h)``.  When omitted only
        ``result_w`` is returned.

    Returns
    -------
    result_w : 1-D ndarray of length len(p_norm_w)
    result_h : 1-D ndarray of length len(p_norm_h)   — only when p_norm_h given
    """
    dist_bound = d_npar / 2
    p_Te       = pTe_from_Te(Te)

    if p_norm_h is not None:
        p_norm_all = np.concatenate([p_norm_w, p_norm_h])
    else:
        p_norm_all = p_norm_w
    n_w = len(p_norm_w)

    if not sparse_slice:
        if p_norm_h is not None:
            return np.zeros(n_w), np.zeros(len(p_norm_h))
        return np.zeros(n_w)

    inv_kp          = 1.0 / (ksi * p_norm_all + eps)
    Gamma           = gamma(p_norm_all, p_Te)
    resonance_N_par = N_par_resonant(inv_kp, p_Te, Gamma, X, harm)

    dist_N_par, ind_N_par = npar_tree.query(
        np.expand_dims(resonance_N_par, axis=-1), k=2, distance_upper_bound=dist_bound)
    res_condition_N_par = np.where(np.isinf(dist_N_par), -1, ind_N_par)

    i_res, n_par_res = np.where(res_condition_N_par != -1)

    D_RF_integrand = np.zeros(len(p_norm_all))

    # Polarisation cache: keyed by (i_npar, i_nperp) — constant for a given
    # (psi, theta) point, independent of p_norm.
    pol_cache = {}

    ksi2_ratio     = ksi**2 / (1.0 - ksi**2 + eps)
    ksi_perp_ratio = ksi / np.sqrt(1.0 - ksi**2 + eps)

    for i, m in zip(i_res, n_par_res):
        i_npar = res_condition_N_par[i, m]
        if i_npar not in sparse_slice:
            continue

        for i_nperp, (W_val, u1_re_val, u2_re_val) in sparse_slice[i_npar].items():
            if W_val <= 0.0:
                continue

            key = (i_npar, i_nperp)
            if key not in pol_cache:
                N2      = nperp[i_nperp]**2 + npar[i_npar]**2
                K_angle = np.arctan2(nperp[i_nperp], npar[i_npar])
                pol_diag, pol_cross = polarisation(N2, K_angle, P, R, L, S)
                nperp_weight        = d_nperp * nperp[i_nperp]
                pol_cache[key]      = (pol_diag, pol_cross, nperp_weight)

            pol_diag, pol_cross, nperp_weight = pol_cache[key]

            a_perp = A_perp(nperp[i_nperp], p_norm_all[i], p_Te, ksi, X)
            Jm1    = sp.jn(harm - 1, a_perp)
            Jp1    = sp.jn(harm + 1, a_perp)
            Jn     = sp.jn(harm,     a_perp)

            T0 = (0.5 * (pol_diag[0] * Jm1**2 + pol_diag[1] * Jp1**2)
                  + ksi2_ratio * pol_diag[2] * Jn**2)
            C2 = 0.5 * pol_cross[0] * Jm1 * Jp1
            C1 = (ksi_perp_ratio / np.sqrt(2.0)
                  * (pol_cross[2] * Jm1 * Jn + pol_cross[1] * Jp1 * Jn))

            D_RF_integrand[i] += nperp_weight * (W_val * T0 + 2.0 * C2 * u2_re_val
                                                 + 2.0 * C1 * u1_re_val)

    if p_norm_h is not None:
        return D_RF_integrand[:n_w], D_RF_integrand[n_w:]
    return D_RF_integrand


# ---------------------------------------------------------------------------
# Bounce-sum helper (trapped particles only)
# ---------------------------------------------------------------------------

def _bounce_sum(d_theta, CB, integrand):
    """Weighted theta-quadrature sum: sum_t (d_theta[t] / 2π) * CB[t] * integrand[t]."""
    return np.nansum(d_theta / (2 * np.pi) * CB * integrand)


def _bounce_integral(ctx,
                     ksi0_grid, Trapksi0_l, theta_T_l, lambda_q_l,
                     p_primary, p_secondary,
                     C_RF_primary, C_RF_secondary,
                     DRF0_primary, DRF0_secondary,
                     DRF0D_primary, DRF0D_secondary, DRF0F_primary):
    """Bounce-average D_RF_nobounce over one ksi0 grid (half- or whole-grid) for one psi surface.

    Writes results directly into the provided output-array views
    (DRF0_primary, DRF0_secondary, …) which are slices of the full D_RF
    output arrays for the current psi index.

    Parameters
    ----------
    ctx : SimpleNamespace
        Per-psi-surface shared state (geometry, Stix, psi_edens, options).
        Created once per psi surface in D_RF and reused for both ksi grids.
    ksi0_grid : 1-D ndarray [n_ksi]
        The ksi0 grid to loop over (ksi0_h or ksi0_w).
    Trapksi0_l : float
        Trapping boundary for this psi surface.
    theta_T_l : ndarray [n_ksi, 2]
        Bounce-arc limits [theta_min, theta_max] for each ksi0.
    lambda_q_l : 1-D ndarray [n_ksi]  (written in-place)
        Lambda_q normalisation factors.
    p_primary : 1-D ndarray
        Primary momentum grid (p_norm_w for ksi0_h, p_norm_h for ksi0_w).
    p_secondary : 1-D ndarray or None
        Secondary momentum grid (p_norm_h for ksi0_h, None for ksi0_w).
    C_RF_primary : ndarray [n_p_primary, n_ksi]
        Prefactor matrix for the primary p-grid.
    C_RF_secondary : ndarray [n_p_secondary, n_ksi] or None
        Prefactor matrix for the secondary p-grid.
    DRF0_primary : ndarray [n_p_primary, n_ksi, n_harm]  (written in-place)
    DRF0_secondary : ndarray [n_p_secondary, n_ksi, n_harm] or None  (written in-place)
    DRF0D_primary : ndarray or None  (written in-place when ctx.DKE_calc)
    DRF0D_secondary : ndarray or None  (written in-place when ctx.DKE_calc)
    DRF0F_primary : ndarray or None  (written in-place when ctx.DKE_calc)
    """
    n_theta_pass = len(ctx.theta_h)
    n_p_primary  = len(p_primary)
    n_p_secondary = len(p_secondary) if p_secondary is not None else 0
    n_harm        = len(ctx.n)

    for j, ksi0_val in enumerate(ksi0_grid):

        if np.abs(ksi0_val) > Trapksi0_l:
            # ----------------------------------------------------------------
            # Passing particle — integrate over full binned theta grid
            # ----------------------------------------------------------------
            R_axis_j = ctx.ptR_pass - ctx.Rp
            Z_axis_j = ctx.ptZ_pass - ctx.Zp
            CB_j = (ctx.ptB_pass * (R_axis_j**2 + Z_axis_j**2)
                    / (ctx.Rp * np.abs(ctx.ptBR_pass * Z_axis_j
                                       - ctx.ptBz_pass * R_axis_j)))

            B_ratio       = ctx.ptB_pass / ctx.B0_psi
            ksi_local     = np.sign(ksi0_val) * np.sqrt(1 - B_ratio * (1 - ksi0_val**2))
            ksi0_over_ksi = ksi0_val / ksi_local

            # Full-orbit lambda_q (avoids scanned-range bias)
            ksi_tr = np.sign(ksi0_val) * np.sqrt(
                np.clip(1 - ctx.ptB_tr / ctx.B0_psi * (1 - ksi0_val**2),
                        ctx.eps**2, None))
            lambda_q_l[j] = np.nansum(
                ctx.d_theta_full / (2 * np.pi) * ctx.CB_full_orbit * (ksi0_val / ksi_tr))

            DRF0_integrand = ksi0_over_ksi**2 * B_ratio
            if ctx.DKE_calc:
                DRF0D_integrand = ksi0_over_ksi
                DRF0F_integrand = (B_ratio - 1) * ksi0_over_ksi**3

            D_rf_primary   = np.zeros((n_theta_pass, n_p_primary, n_harm))
            D_rf_secondary = (np.zeros((n_theta_pass, n_p_secondary, n_harm))
                              if p_secondary is not None else None)
            w_base = ctx.d_theta / (2 * np.pi) * CB_j

            for n_idx, harm in enumerate(ctx.n):
                for t in range(n_theta_pass):
                    slice_t = ctx.psi_edens.get_slice(t)
                    if not slice_t:
                        continue
                    if p_secondary is not None:
                        D_rf_primary[t, :, n_idx], D_rf_secondary[t, :, n_idx] = \
                            D_RF_nobounce(p_primary, ksi_local[t],
                                         ctx.npar, ctx.nperp, slice_t, ctx.Te_ref,
                                         ctx.P_l, ctx.X_pass[t], ctx.R_pass[t],
                                         ctx.L_pass[t], ctx.S_pass[t],
                                         harm, ctx.eps, ctx.npar_tree,
                                         ctx.d_npar, ctx.d_nperp,
                                         p_norm_h=p_secondary)
                    else:
                        D_rf_primary[t, :, n_idx] = \
                            D_RF_nobounce(p_primary, ksi_local[t],
                                         ctx.npar, ctx.nperp, slice_t, ctx.Te_ref,
                                         ctx.P_l, ctx.X_pass[t], ctx.R_pass[t],
                                         ctx.L_pass[t], ctx.S_pass[t],
                                         harm, ctx.eps, ctx.npar_tree,
                                         ctx.d_npar, ctx.d_nperp)

                w_DRF0 = (w_base * DRF0_integrand)[:, None]
                DRF0_primary[:, j, n_idx] = np.nansum(
                    w_DRF0 * D_rf_primary[:, :, n_idx], axis=0)
                if D_rf_secondary is not None:
                    DRF0_secondary[:, j, n_idx] = np.nansum(
                        w_DRF0 * D_rf_secondary[:, :, n_idx], axis=0)
                if ctx.DKE_calc:
                    w_D = (w_base * DRF0D_integrand)[:, None]
                    w_F = (w_base * DRF0F_integrand)[:, None]
                    DRF0D_primary[:, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                        w_D * D_rf_primary[:, :, n_idx], axis=0)
                    if DRF0D_secondary is not None:
                        DRF0D_secondary[:, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            w_D * D_rf_secondary[:, :, n_idx], axis=0)
                    DRF0F_primary[:, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                        w_F * D_rf_primary[:, :, n_idx], axis=0)

                DRF0_primary[:, j, n_idx] *= C_RF_primary[:, j] / lambda_q_l[j]
                if DRF0_secondary is not None:
                    DRF0_secondary[:, j, n_idx] *= C_RF_secondary[:, j] / lambda_q_l[j]
                if ctx.DKE_calc:
                    DRF0D_primary[:, j, n_idx] *= C_RF_primary[:, j] / lambda_q_l[j]
                    if DRF0D_secondary is not None:
                        DRF0D_secondary[:, j, n_idx] *= C_RF_secondary[:, j] / lambda_q_l[j]
                    DRF0F_primary[:, j, n_idx] *= C_RF_primary[:, j] / lambda_q_l[j]

        else:
            # ----------------------------------------------------------------
            # Trapped particle — integrate over the bounce arc only
            # ----------------------------------------------------------------
            theta_T_m, theta_T_M = theta_T_l[j]

            theta_w_clip = ctx.theta_w[
                (ctx.theta_w >= theta_T_m) & (ctx.theta_w <= theta_T_M)]
            theta_w_aux  = np.concatenate(([theta_T_m], theta_w_clip, [theta_T_M]))
            d_theta_j    = np.diff(theta_w_aux)
            theta_grid_j = theta_w_aux[:-1] + d_theta_j / 2
            theta_grid_j = np.clip(theta_grid_j, ctx.theta_h[0], ctx.theta_h[-1]) + 0.0

            B_j    = ctx.ptB_interp(theta_grid_j)
            BR_j   = ctx.ptBR_interp(theta_grid_j)
            Bz_j   = ctx.ptBz_interp(theta_grid_j)
            R_ax_j = ctx.ptR_interp(theta_grid_j) - ctx.Rp
            Z_ax_j = ctx.ptZ_interp(theta_grid_j) - ctx.Zp

            CB_j = B_j * (R_ax_j**2 + Z_ax_j**2) / (
                ctx.Rp * np.abs(BR_j * Z_ax_j - Bz_j * R_ax_j))

            B_ratio       = B_j / ctx.B0_psi
            ksi_local     = np.sign(ksi0_val) * np.sqrt(1 - B_ratio * (1 - ksi0_val**2))
            ksi0_over_ksi = ksi0_val / ksi_local

            lambda_q_l[j] = _bounce_sum(d_theta_j, CB_j, ksi0_over_ksi)
            DRF0_integrand = ksi0_over_ksi**2 * B_ratio
            if ctx.DKE_calc:
                DRF0D_integrand = ksi0_over_ksi
                DRF0F_integrand = (B_ratio - 1) * ksi0_over_ksi**3

            _, _, _, _, _, _, _, _, _, X_j, R_j, L_j, S_j = \
                config_quantities([ctx.psi_l], theta_grid_j, ctx.omega, ctx.Eq)

            n_theta_trap  = len(theta_grid_j)
            D_rf_primary  = np.zeros((n_theta_trap, n_p_primary, n_harm))
            D_rf_secondary = (np.zeros((n_theta_trap, n_p_secondary, n_harm))
                              if p_secondary is not None else None)
            w_base = d_theta_j / (2 * np.pi) * CB_j

            for n_idx, harm in enumerate(ctx.n):
                for t, theta_val in enumerate(theta_grid_j):
                    slice_t = ctx.psi_edens.interpolate_trapped(theta_val)
                    if not slice_t:
                        continue
                    if p_secondary is not None:
                        D_rf_primary[t, :, n_idx], D_rf_secondary[t, :, n_idx] = \
                            D_RF_nobounce(p_primary, ksi_local[t],
                                         ctx.npar, ctx.nperp, slice_t, ctx.Te_ref,
                                         ctx.P_l, X_j[0, t], R_j[0, t],
                                         L_j[0, t], S_j[0, t],
                                         harm, ctx.eps, ctx.npar_tree,
                                         ctx.d_npar, ctx.d_nperp,
                                         p_norm_h=p_secondary)
                    else:
                        D_rf_primary[t, :, n_idx] = \
                            D_RF_nobounce(p_primary, ksi_local[t],
                                         ctx.npar, ctx.nperp, slice_t, ctx.Te_ref,
                                         ctx.P_l, X_j[0, t], R_j[0, t],
                                         L_j[0, t], S_j[0, t],
                                         harm, ctx.eps, ctx.npar_tree,
                                         ctx.d_npar, ctx.d_nperp)

                w_DRF0 = (w_base * DRF0_integrand)[:, None]
                DRF0_primary[:, j, n_idx] = np.nansum(
                    w_DRF0 * D_rf_primary[:, :, n_idx], axis=0)
                if D_rf_secondary is not None:
                    DRF0_secondary[:, j, n_idx] = np.nansum(
                        w_DRF0 * D_rf_secondary[:, :, n_idx], axis=0)
                if ctx.DKE_calc:
                    w_D = (w_base * DRF0D_integrand)[:, None]
                    w_F = (w_base * DRF0F_integrand)[:, None]
                    DRF0D_primary[:, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                        w_D * D_rf_primary[:, :, n_idx], axis=0)
                    if DRF0D_secondary is not None:
                        DRF0D_secondary[:, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                            w_D * D_rf_secondary[:, :, n_idx], axis=0)
                    DRF0F_primary[:, j, n_idx] = np.sign(ksi0_val) * np.nansum(
                        w_F * D_rf_primary[:, :, n_idx], axis=0)

                DRF0_primary[:, j, n_idx] *= C_RF_primary[:, j] / lambda_q_l[j]
                if DRF0_secondary is not None:
                    DRF0_secondary[:, j, n_idx] *= C_RF_secondary[:, j] / lambda_q_l[j]
                if ctx.DKE_calc:
                    DRF0D_primary[:, j, n_idx] *= C_RF_primary[:, j] / lambda_q_l[j]
                    if DRF0D_secondary is not None:
                        DRF0D_secondary[:, j, n_idx] *= C_RF_secondary[:, j] / lambda_q_l[j]
                    DRF0F_primary[:, j, n_idx] *= C_RF_primary[:, j] / lambda_q_l[j]


# ---------------------------------------------------------------------------
# Full bounce-averaged D_RF tensor
# ---------------------------------------------------------------------------

def D_RF(psi, theta_w, p_norm_w, p_norm_h, ksi0_w, ksi0_h,
         npar, nperp, Edens, Eq, Ne_ref, Te_ref,
         n=None, FreqGHz=82.7, DKE_calc=False,
         gaussian_smooth=False, gaussian_sigma=2, symmetrise_trapped=True,
         eps=_EPS_F32, edens_sparse=None, lnc_e_ref=None, npar_tree=None):
    """Compute the bounce-averaged quasi-linear RF diffusion tensor.

    Parameters
    ----------
    psi       : 1-D ndarray [n_psi]    Normalised poloidal flux surface centres.
    theta_w   : 1-D ndarray [n_theta+1] Poloidal angle bin edges [rad] (whole grid).
    p_norm_w  : 1-D ndarray [n_pw]     Momentum whole-grid (normalised to m_e c).
    p_norm_h  : 1-D ndarray [n_ph]     Momentum half-grid.
    ksi0_w    : 1-D ndarray [n_kw]     Pitch-angle whole-grid.
    ksi0_h    : 1-D ndarray [n_kh]     Pitch-angle half-grid.
    npar      : 1-D ndarray            Parallel refractive index bin centres.
    nperp     : 1-D ndarray            Perpendicular refractive index bin centres.
    Edens     : ndarray [n_psi, n_theta, n_npar, n_nperp] or None
        Dense energy density [J m⁻³ N⁻²].  Provide either Edens or edens_sparse.
    Eq        : TokamakEquilibrium
    Ne_ref    : float   Reference electron density [1e19 m⁻³].
    Te_ref    : float   Reference electron temperature [keV].
    n         : list of int   Cyclotron harmonics (default: [2, 3]).
    FreqGHz   : float         Wave frequency [GHz].
    DKE_calc  : bool          Compute DKE first-order drift/convection terms.
    gaussian_smooth    : bool   Apply Gaussian smoothing in (p, ksi) per psi.
    gaussian_sigma     : float  Sigma for Gaussian filter.
    symmetrise_trapped : bool   Average D_RF(ksi0) and D_RF(-ksi0) for trapped
                                particles (physically required for sigma-symmetric
                                bounce).
    eps       : float   Small regularisation offset.
    edens_sparse : dict or None   Sparse energy density {l: {t: ...}}.
    lnc_e_ref    : float or None   LUKE Coulomb logarithm (overrides analytic).
    npar_tree    : KDTree or None  Pre-built spatial index over the Npar grid.
                   When provided (e.g. from the MPI worker loop) the tree is
                   reused across all psi surfaces in this call.  When None a
                   new tree is built internally.

    Returns
    -------
    DRF0_wh, DRF0D_wh, DRF0F_wh  : ndarray [n_psi, n_pw, n_kh, n_harm]
    DRF0_hw, DRF0D_hw, DRF0F_hw  : ndarray [n_psi, n_ph, n_kw, n_harm]
    DRF0_hh, DRF0D_hh            : ndarray [n_psi, n_ph, n_kh, n_harm]
    Trapksi0_h, Trapksi0_w       : 1-D ndarray [n_psi]   Trapping boundaries.
    """
    if n is None:
        n = [2, 3]

    omega      = phys.AngularFrequency(FreqGHz)
    use_sparse = edens_sparse is not None

    # Defensive copy: avoid mutating caller's arrays
    ksi0_h = ksi0_h.copy()
    ksi0_w = ksi0_w.copy()
    ksi0_h[np.abs(ksi0_h) < 1e-4] = 1e-4
    ksi0_w[np.abs(ksi0_w) < 1e-4] = 1e-4

    # Theta bin centres and widths from whole-grid edges
    d_theta = np.diff(theta_w)
    theta_h = theta_w[:-1] + d_theta / 2

    # Configuration-space quantities on the binned theta grid
    Rp, Zp = Eq.magn_axis_coord_Rz / 100   # m
    ptR, ptZ, ptBt, ptBR, ptBz, ptB, ptNe, ptTe, P, X, R, L, S = \
        config_quantities(psi, theta_h, omega, Eq)

    # Fine geometry grid for trapping-boundary detection (always full orbit)
    theta_trap = np.linspace(-np.pi, np.pi, 361)[:-1]
    ptR_tr, ptZ_tr, _, ptBR_tr, ptBz_tr, ptB_tr, _, _, _, _, _, _, _ = \
        config_quantities(psi, theta_trap, omega, Eq)

    # KDTree over Npar grid — built once, shared across all psi/ksi/theta.
    # If a pre-built tree is supplied by the caller (e.g. the MPI worker loop)
    # it is reused directly to avoid redundant construction.
    if npar_tree is None:
        npar_tree = KDTree(npar.reshape(-1, 1))
    d_npar    = npar[1] - npar[0]
    d_nperp   = nperp[1] - nperp[0]

    # Normalisation prefactors — depend only on constants, computed once
    C_RF_wh = D_RF_prefactor(p_norm_w, ksi0_h, Ne_ref, Te_ref, omega, eps, lnc_e_ref)
    C_RF_hw = D_RF_prefactor(p_norm_h, ksi0_w, Ne_ref, Te_ref, omega, eps, lnc_e_ref)
    C_RF_hh = D_RF_prefactor(p_norm_h, ksi0_h, Ne_ref, Te_ref, omega, eps, lnc_e_ref)

    # Output arrays: (n_psi, n_p, n_ksi, n_harm)
    n_psi, n_pw, n_ph, n_kh, n_kw, n_harm = (
        len(psi), len(p_norm_w), len(p_norm_h),
        len(ksi0_h), len(ksi0_w), len(n))

    DRF0_wh = np.zeros((n_psi, n_pw, n_kh, n_harm))
    DRF0_hw = np.zeros((n_psi, n_ph, n_kw, n_harm))
    DRF0_hh = np.zeros((n_psi, n_ph, n_kh, n_harm))
    Trapksi0_h = np.zeros(n_psi)
    Trapksi0_w = np.zeros(n_psi)

    if DKE_calc:
        DRF0D_wh = np.zeros((n_psi, n_pw, n_kh, n_harm))
        DRF0D_hw = np.zeros((n_psi, n_ph, n_kw, n_harm))
        DRF0D_hh = np.zeros((n_psi, n_ph, n_kh, n_harm))
        DRF0F_wh = np.zeros((n_psi, n_pw, n_kh, n_harm))
        DRF0F_hw = np.zeros((n_psi, n_ph, n_kw, n_harm))
    else:
        DRF0D_wh = DRF0D_hw = DRF0D_hh = np.zeros(n_psi)
        DRF0F_wh = DRF0F_hw = np.zeros(n_psi)

    lambda_q_h = np.zeros((n_psi, n_kh))
    lambda_q_w = np.zeros((n_psi, n_kw))
    Trapksi0_h_arr = np.zeros((n_psi, 1))
    Trapksi0_w_arr = np.zeros((n_psi, 1))
    theta_T_h      = np.zeros((n_psi, n_kh, 2))
    theta_T_w      = np.zeros((n_psi, n_kw, 2))

    # -----------------------------------------------------------------------
    # Outer loop: psi surfaces (independent, distributed across MPI workers)
    # -----------------------------------------------------------------------
    for l, psi_l in enumerate(psi):

        # --- Trapping boundary on the fine theta_trap grid ---
        ptB_Int = interp1d(theta_trap, ptB_tr[l, :],
                           fill_value=np.amax(ptB_tr[l, :]), bounds_error=False)
        B0_psi, Bmax_psi = minmaxB(ptB_Int, theta_trap)

        _, Trapksi0_w_arr[l], theta_T_w[l] = Trapping_boundary(
            ksi0_w, ptB_Int, theta_trap, B0_in=B0_psi, Bmax_in=Bmax_psi)
        _, Trapksi0_h_arr[l], theta_T_h[l] = Trapping_boundary(
            ksi0_h, ptB_Int, theta_trap, B0_in=B0_psi, Bmax_in=Bmax_psi)
        Trapksi0_h[l] = float(Trapksi0_h_arr[l])
        Trapksi0_w[l] = float(Trapksi0_w_arr[l])

        # Fine-grid geometry interpolants for trapped particle bounce arcs
        ptB_interp  = interp1d(theta_trap, ptB_tr[l, :])
        ptBR_interp = interp1d(theta_trap, ptBR_tr[l, :])
        ptBz_interp = interp1d(theta_trap, ptBz_tr[l, :])
        ptR_interp  = interp1d(theta_trap, ptR_tr[l, :])
        ptZ_interp  = interp1d(theta_trap, ptZ_tr[l, :])

        # Full-orbit bounce kernel (360-pt) for passing lambda_q normalisation.
        # Using the full orbit avoids a 1/(scanned_theta_range) bias in D_RF.
        R_axis_tr = ptR_tr[l, :] - Rp
        Z_axis_tr = ptZ_tr[l, :] - Zp
        CB_full_orbit = (ptB_tr[l, :] * (R_axis_tr**2 + Z_axis_tr**2)
                         / (Rp * np.abs(ptBR_tr[l, :] * Z_axis_tr
                                        - ptBz_tr[l, :] * R_axis_tr)))
        d_theta_full = 2 * np.pi / len(theta_trap)

        # Build unified energy-density container for this psi surface
        psi_edens = build_psi_edens(l, Edens, edens_sparse, theta_h,
                                    npar, nperp, use_sparse)

        # Bundle all shared per-psi-surface state into a single namespace so
        # _bounce_integral doesn't need a 40-argument signature.
        ctx = SimpleNamespace(
            ptB_pass=ptB[l, :],    ptBR_pass=ptBR[l, :], ptBz_pass=ptBz[l, :],
            ptR_pass=ptR[l, :],    ptZ_pass=ptZ[l, :],
            ptB_tr=ptB_tr[l, :],
            X_pass=X[l, :], R_pass=R[l, :], L_pass=L[l, :], S_pass=S[l, :],
            P_l=float(P[l, 0]),
            ptB_interp=ptB_interp, ptBR_interp=ptBR_interp,
            ptBz_interp=ptBz_interp, ptR_interp=ptR_interp, ptZ_interp=ptZ_interp,
            CB_full_orbit=CB_full_orbit, d_theta_full=d_theta_full,
            theta_w=theta_w, theta_h=theta_h, d_theta=d_theta,
            B0_psi=B0_psi, Rp=Rp, Zp=Zp,
            psi_l=psi_l, psi_edens=psi_edens, omega=omega, Eq=Eq,
            npar=npar, nperp=nperp, npar_tree=npar_tree,
            d_npar=d_npar, d_nperp=d_nperp,
            n=n, DKE_calc=DKE_calc, eps=eps, Te_ref=Te_ref,
        )

        # ksi0 HALF-GRID loop  →  fills DRF0_wh, DRF0_hh  (p_norm_w primary)
        _bounce_integral(ctx,
                         ksi0_h, Trapksi0_h[l], theta_T_h[l], lambda_q_h[l],
                         p_norm_w, p_norm_h,
                         C_RF_wh, C_RF_hh,
                         DRF0_wh[l], DRF0_hh[l],
                         DRF0D_wh[l] if DKE_calc else None,
                         DRF0D_hh[l] if DKE_calc else None,
                         DRF0F_wh[l] if DKE_calc else None)

        # ksi0 WHOLE-GRID loop  →  fills DRF0_hw  (p_norm_h primary, no secondary)
        _bounce_integral(ctx,
                         ksi0_w, Trapksi0_w[l], theta_T_w[l], lambda_q_w[l],
                         p_norm_h, None,
                         C_RF_hw, None,
                         DRF0_hw[l], None,
                         DRF0D_hw[l] if DKE_calc else None,
                         None,
                         DRF0F_hw[l] if DKE_calc else None)

        # ===================================================================
        # Post-processing per psi surface
        # ===================================================================

        # Optional Gaussian smoothing in (p, ksi) space
        if gaussian_smooth:
            for n_idx in range(n_harm):
                DRF0_wh[l, :, :, n_idx] = gaussian_filter(DRF0_wh[l, :, :, n_idx], sigma=gaussian_sigma)
                DRF0_hh[l, :, :, n_idx] = gaussian_filter(DRF0_hh[l, :, :, n_idx], sigma=gaussian_sigma)
                DRF0_hw[l, :, :, n_idx] = gaussian_filter(DRF0_hw[l, :, :, n_idx], sigma=gaussian_sigma)
                if DKE_calc:
                    DRF0D_wh[l, :, :, n_idx] = gaussian_filter(DRF0D_wh[l, :, :, n_idx], sigma=gaussian_sigma)
                    DRF0D_hh[l, :, :, n_idx] = gaussian_filter(DRF0D_hh[l, :, :, n_idx], sigma=gaussian_sigma)
                    DRF0D_hw[l, :, :, n_idx] = gaussian_filter(DRF0D_hw[l, :, :, n_idx], sigma=gaussian_sigma)
                    DRF0F_wh[l, :, :, n_idx] = gaussian_filter(DRF0F_wh[l, :, :, n_idx], sigma=gaussian_sigma)
                    DRF0F_hw[l, :, :, n_idx] = gaussian_filter(DRF0F_hw[l, :, :, n_idx], sigma=gaussian_sigma)

        # Optional symmetrisation: D_RF(ksi0) = D_RF(-ksi0) for trapped particles.
        # Trapped particles bounce between mirror points and therefore sample both
        # signs of ksi along their orbit, but the bounce integral above only sees
        # one sign.  Averaging corrects this (mhu-flip in rfdiff_dke_jd.m).
        if symmetrise_trapped:
            trap_h = Trapksi0_h[l]
            trap_w = Trapksi0_w[l]

            for j in range(n_kh):
                if np.abs(ksi0_h[j]) <= trap_h:
                    j_m = int(np.argmin(np.abs(ksi0_h + ksi0_h[j])))
                    if j < j_m:
                        avg_wh = (DRF0_wh[l, :, j, :] + DRF0_wh[l, :, j_m, :]) / 2
                        DRF0_wh[l, :, j, :]  = avg_wh
                        DRF0_wh[l, :, j_m, :] = avg_wh
                        avg_hh = (DRF0_hh[l, :, j, :] + DRF0_hh[l, :, j_m, :]) / 2
                        DRF0_hh[l, :, j, :]  = avg_hh
                        DRF0_hh[l, :, j_m, :] = avg_hh

            for j in range(n_kw):
                if np.abs(ksi0_w[j]) <= trap_w:
                    j_m = int(np.argmin(np.abs(ksi0_w + ksi0_w[j])))
                    if j < j_m:
                        avg_hw = (DRF0_hw[l, :, j, :] + DRF0_hw[l, :, j_m, :]) / 2
                        DRF0_hw[l, :, j, :]  = avg_hw
                        DRF0_hw[l, :, j_m, :] = avg_hw

    return (DRF0_wh, DRF0D_wh, DRF0F_wh,
            DRF0_hw, DRF0D_hw, DRF0F_hw,
            DRF0_hh, DRF0D_hh,
            Trapksi0_h, Trapksi0_w)
