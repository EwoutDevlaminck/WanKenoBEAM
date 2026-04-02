"""Plotting routines for diagnosing how scattering fluctuations affect the beam.

The equilibrium and scattering models are re-instantiated from the RayTracing
configuration file, so the plots faithfully represent what the ray tracer
experienced.

Entry point (called by WKBeam.py):

    plot_beam_fluct([xz_binned.hdf5, ..., RayTracing.txt])

All but the last element of the input list are XZ-binned HDF5 files; the last
element is the RayTracing configuration file.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import h5py

from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import IntSample, StixParamSample
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium, TokamakEquilibrium2
from CommonModules.PlasmaEquilibrium import AxisymmetricEquilibrium
from RayTracing.modules.atanrightbranch import atanRightBranch
from RayTracing.modules.scattering.GaussianModel import GaussianModel_base
from RayTracing.modules.scattering.ShaferModel import ShaferModel_base
from Tools.PlotData.CommonPlotting import plotting_functions
from Tools.PlotData.PlotVessel.plotVessel import plotVessel

import CommonModules.physics_constants as PhysConst

# Normalised poloidal flux ψ beyond which fluctuations are not rendered.
# ψ = ρ_pol², so ψ < 1.6 corresponds to ρ < √1.6 ≈ 1.26  (26 % outside separatrix).
_PSI_FLUCT_CUTOFF = 1.6

# Colour sequence for successive beam overlays (teal palette)
_BEAM_COLOURS = ['#007480', '#413C3A', '#00A79F']


# ---------------------------------------------------------------------------
# Fluctuation sampler
# ---------------------------------------------------------------------------

def sample_fluct_envelop(R1d, Z1d, axis, envelope, Lpp, radial_coord, Eq):
    """Sample the fluctuation envelope and correlation length on a 2-D (R, Z) grid.

    Parameters
    ----------
    R1d, Z1d     : 1-D arrays    Grid coordinates [cm].
    axis         : (Raxis, Zaxis) Magnetic axis position [cm].
    envelope     : callable       envelope(Ne, rho, theta) → (δne/ne)²
    Lpp          : callable       Lpp(rho, theta, Ne, Te, Bnorm) → L⊥ [cm]
    radial_coord : callable       radial_coord(R, Z) → normalised radius ρ
    Eq           : equilibrium    Provides Ne, Te, B-field interpolants.

    Returns
    -------
    fluct_sample  : 2-D array (nR, nZ)   Fluctuation amplitude squared.
    length_sample : 2-D array (nR, nZ)   Perpendicular correlation length [cm].
    """
    Raxis, Zaxis = axis
    nptR, nptZ   = np.size(R1d), np.size(Z1d)
    fluct_sample  = np.empty([nptR, nptZ])
    length_sample = np.empty([nptR, nptZ])

    for iR in range(nptR):
        Rloc   = R1d[iR]
        deltaR = Rloc - Raxis
        for jZ in range(nptZ):
            Zloc   = Z1d[jZ]
            deltaZ = Zloc - Zaxis
            Ne    = Eq.NeInt.eval(Rloc, Zloc)
            Te    = Eq.TeInt.eval(Rloc, Zloc)
            Bnorm = np.sqrt(Eq.BtInt.eval(Rloc, Zloc)**2 +
                            Eq.BRInt.eval(Rloc, Zloc)**2 +
                            Eq.BzInt.eval(Rloc, Zloc)**2)
            rho   = radial_coord(Rloc, Zloc)
            theta = atanRightBranch(deltaZ, deltaR)
            fluct_sample[iR, jZ]  = envelope(Ne, rho, theta)
            length_sample[iR, jZ] = Lpp(rho, theta, Ne, Te, Bnorm)

    return fluct_sample, length_sample


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------

def _load_beam_data(filename):
    """Load one XZ-binned HDF5 file and convert to physical units.

    All returned spatial coordinates are in cm.
    Wfct is converted to energy density [J/m³];
    Absorption to power density [MW/m³] averaged over the toroidal angle.

    Parameters
    ----------
    filename : str   Path to the XZ-binned HDF5 file.

    Returns
    -------
    dict with keys:
        FreqGHz      float         Frequency [GHz].
        mode         int           Wave mode index.
        R_cm         1-D array     Bin-centre R coordinates [cm].
        Z_cm         1-D array     Bin-centre Z coordinates [cm].
        Wfct         2-D array     Mean energy density [J/m³], shape (nR, nZ).
        Absorption   2-D array     Mean absorbed power density [MW/m³], or None.
        P_abs        1-D array     Total absorbed power [MW], shape (2,) = [mean, rms],
                                   or None if absorption was not recorded.
        Velocity     4-D array     Velocity field (nR, nZ, ncomp, 2), or None.
        Abs_recorded bool
        Vel_recorded bool
        beam_extent  tuple         (Rmin, Rmax, Zmin, Zmax) [cm].
    """
    c = PhysConst.SpeedOfLight  # cm/s

    with h5py.File(filename, 'r') as fid:
        FreqGHz  = fid['FreqGHz'][()]
        mode     = fid['Mode'][()]
        Wfct_raw = fid['BinnedTraces'][()]

        Abs_recorded = 'Absorption' in fid
        Abs_raw      = fid['Absorption'][()] if Abs_recorded else None

        Vel_recorded = 'VelocityField' in fid
        Vel_raw      = fid['VelocityField'][()] if Vel_recorded else None

        uniform_bins = bool(fid['uniform_bins'][()]) if 'uniform_bins' in fid else True

        if uniform_bins:
            # Support both 'X' and 'R' naming conventions
            if 'Xmin' in fid:
                Rmin_b = fid['Xmin'][()];  Rmax_b = fid['Xmax'][()];  nR = int(fid['nmbrX'][()])
            else:
                Rmin_b = fid['Rmin'][()];  Rmax_b = fid['Rmax'][()];  nR = int(fid['nmbrR'][()])
            Zmin_b = fid['Zmin'][()];  Zmax_b = fid['Zmax'][()];  nZ = int(fid['nmbrZ'][()])
            resolveY    = 'Ymin' in fid
            resolveNpar = 'Nparallelmin' in fid
        else:
            Rbins = fid['Xbins'][()] if 'Xbins' in fid else fid['Rbins'][()]
            Zbins = fid['Zbins'][()]
            resolveY    = 'Ybins' in fid
            resolveNpar = 'Nparallelbins' in fid

    # Sum out unused dimensions (toroidal Y and Nparallel) if they were resolved
    if resolveY:
        Wfct_raw = np.sum(Wfct_raw, axis=1)
        if Abs_recorded: Abs_raw = np.sum(Abs_raw, axis=1)
        if Vel_recorded: Vel_raw = np.sum(Vel_raw, axis=1)
    if resolveNpar:
        Wfct_raw = np.sum(Wfct_raw, axis=2)
        if Abs_recorded: Abs_raw = np.sum(Abs_raw, axis=2)
        if Vel_recorded: Vel_raw = np.sum(Vel_raw, axis=2)

    # Build coordinate arrays [cm] and toroidal volume element [m³].
    # Coordinates are divided by 100 (cm → m) only for the unit-conversion step;
    # all arrays returned by this function stay in cm.
    if uniform_bins:
        R_cm = np.linspace(Rmin_b, Rmax_b, nR)
        Z_cm = np.linspace(Zmin_b, Zmax_b, nZ)
        dR_m = (Rmax_b - Rmin_b) / nR * 1e-2   # uniform bin width [m]
        dZ_m = (Zmax_b - Zmin_b) / nZ * 1e-2
        _, RR_m = np.meshgrid(Z_cm * 1e-2, R_cm * 1e-2)   # (nR, nZ) [m]
        vol_elem = dR_m * dZ_m * 2 * np.pi * RR_m          # (nR, nZ) [m³]
        beam_extent = (Rmin_b, Rmax_b, Zmin_b, Zmax_b)
    else:
        R_cm = (Rbins[:-1] + Rbins[1:]) / 2
        Z_cm = (Zbins[:-1] + Zbins[1:]) / 2
        dR_m = np.diff(Rbins) * 1e-2   # per-bin widths [m]
        dZ_m = np.diff(Zbins) * 1e-2
        _, RR_m = np.meshgrid(Z_cm * 1e-2, R_cm * 1e-2)
        vol_elem = np.outer(dR_m, dZ_m) * 2 * np.pi * RR_m  # (nR, nZ) [m³]
        beam_extent = (Rbins[0], Rbins[-1], Zbins[0], Zbins[-1])

    # Convert Wfct to energy density [J/m³].
    # WKBeam stores action density proportional to [MW]; the factor
    # 1e6 · 4π / (c [m/s]) converts to Joules in the full toroidal volume,
    # and dividing by vol_elem gives J/m³.
    unit_factor = 1e6 * 4 * np.pi / (c * 1e-2)   # c: cm/s → m/s
    print(f'  Total field energy = {np.sum(Wfct_raw) * unit_factor:.3e} J  '
          f'({filename})')
    Wfct = Wfct_raw * unit_factor / vol_elem[..., np.newaxis]

    if Abs_recorded:
        # Compute total absorbed power [MW] before normalisation
        P_abs = np.sum(Abs_raw, axis=(0, 1))   # shape (2,): [mean, rms]
        # Normalise to [MW/m³], then average over the toroidal angle (÷ 2πR)
        Absorption = (Abs_raw
                      / vol_elem[..., np.newaxis]
                      / (2 * np.pi * RR_m[..., np.newaxis]))
    else:
        Absorption = None
        P_abs      = None

    if Vel_recorded:
        # Velocity shape: (nR, nZ, ncomp, 2); vol_elem shape: (nR, nZ)
        Velocity = Vel_raw / vol_elem[:, :, np.newaxis, np.newaxis]
    else:
        Velocity = None

    return dict(
        FreqGHz      = FreqGHz,
        mode         = mode,
        R_cm         = R_cm,
        Z_cm         = Z_cm,
        Wfct         = Wfct[:, :, 0],              # mean channel (index 0)
        Absorption   = Absorption[:, :, 0] if Abs_recorded else None,
        P_abs        = P_abs,
        Velocity     = Velocity,
        Abs_recorded = Abs_recorded,
        Vel_recorded = Vel_recorded,
        beam_extent  = beam_extent,
    )


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_beam_fluct(inputdata):
    """Plot the 2-D beam propagation overlaid with the fluctuation amplitude profile.

    Parameters
    ----------
    inputdata : list
        All elements except the last are paths to XZ-binned HDF5 files.
        The last element is the path to the RayTracing configuration file,
        which is used to reconstruct the equilibrium and fluctuation model.
    """
    inputfilenames = inputdata[:-1]
    configfile     = inputdata[-1]

    # -----------------------------------------------------------------------
    # Load beam data
    # -----------------------------------------------------------------------
    print('Reading beam data...')
    beams = [_load_beam_data(f) for f in inputfilenames]

    any_abs = any(b['Abs_recorded'] for b in beams)
    if any_abs:
        P_abs_total      = sum(b['P_abs'] for b in beams if b['P_abs'] is not None)
        Absorption_total = sum(b['Absorption'] for b in beams if b['Absorption'] is not None)

    # -----------------------------------------------------------------------
    # Load equilibrium and fluctuation model from the RayTracing config
    # -----------------------------------------------------------------------
    idata = InputData(configfile)

    if idata.equilibrium == 'Tokamak':
        Eq      = TokamakEquilibrium(idata)
        figsize = (7, 8)
    elif idata.equilibrium == 'Tokamak2D':
        Eq      = TokamakEquilibrium2(idata)
        figsize = (7, 8)
    elif idata.equilibrium == 'Axisymmetric':
        Eq      = AxisymmetricEquilibrium(idata)
        figsize = (7, 8)
    else:
        raise ValueError("'equilibrium' must be 'Tokamak', 'Tokamak2D', or 'Axisymmetric'")

    # -----------------------------------------------------------------------
    # Set up the equilibrium grid
    # -----------------------------------------------------------------------
    has_flux_surfaces = idata.equilibrium in ('Tokamak', 'Tokamak2D')

    if has_flux_surfaces:
        Rmin, Rmax = Eq.Rgrid[0, 0],  Eq.Rgrid[-1, 0]
        Zmin, Zmax = Eq.zgrid[0, 0],  Eq.zgrid[0, -1]
        axis       = Eq.magn_axis_coord_Rz
    else:
        Rmin, Rmax = idata.rmaj - idata.rmin, idata.rmaj + idata.rmin
        Zmin, Zmax = -idata.rmin, idata.rmin
        axis       = [idata.rmaj, 0.]

    # Grid spacing ≈ rmin / 100  (keeps resolution machine-independent)
    nptR = int((Rmax - Rmin) / (idata.rmin / 100.))
    nptZ = int((Zmax - Zmin) / (idata.rmin / 100.))
    print(f'Equilibrium grid: nptR={nptR}, nptZ={nptZ}')
    R1d = np.linspace(Rmin, Rmax, nptR)
    Z1d = np.linspace(Zmin, Zmax, nptZ)

    # Sample equilibrium quantities on the 2-D grid
    Ne = IntSample(R1d, Z1d, Eq.NeInt.eval)

    if has_flux_surfaces:
        _, StixY, _ = StixParamSample(R1d, Z1d, Eq, idata.freq)
        psi         = IntSample(R1d, Z1d, Eq.PsiInt.eval)  # normalised poloidal flux
        equilibrium = psi
    else:
        equilibrium = Ne   # fall back to density for non-tokamak devices

    # -----------------------------------------------------------------------
    # Fluctuation model (identical to what the ray tracer used)
    # -----------------------------------------------------------------------
    rank = 0   # dummy MPI rank, not used in the base model classes
    if idata.scatteringGaussian:
        Fluct    = GaussianModel_base(idata, rank)
        envelope = lambda Ne, rho, theta: Fluct.scatteringDeltaneOverne(Ne, rho, theta)**2
    else:
        Fluct    = ShaferModel_base(idata, rank)
        envelope = lambda Ne, rho, theta: Fluct.ShapeModel(rho, theta)

    Lpp          = Fluct.scatteringLengthPerp
    radial_coord = lambda R, Z: np.sqrt(Eq.PsiInt.eval(R, Z))

    print('Sampling fluctuation envelope on equilibrium grid...')
    fluct, _ = sample_fluct_envelop(R1d, Z1d, axis, envelope, Lpp, radial_coord, Eq)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal')

    # -- Vessel --------------------------------------------------------------
    vessel_plotted = False
    try:
        Rv_in, Rv_out, Zv_in, Zv_out, Zt, Rt = plotVessel(idata)
        R_fill = np.concatenate([Rv_in,  np.flipud(Rv_out)])
        Z_fill = np.concatenate([Zv_in,  np.flipud(Zv_out)])
        ax.fill(R_fill,                               Z_fill,
                color=[0.4, 0.4, 0.4], edgecolor='none', zorder=13)
        ax.fill(np.concatenate([Rt, Rv_in]),
                np.concatenate([Zt, Zv_in]),
                color=[0.5, 0.5, 0.5], edgecolor='none', zorder=13)
        vessel_plotted = True
        print('Vessel plotted.')
    except Exception:
        print('No vessel data found, skipping.')

    # -- Background: fluctuation amplitude (or density when scattering is off) --
    if idata.scattering:
        # Display RMS absolute fluctuation δne = (δne/ne) · ne,
        # masked beyond ψ = _PSI_FLUCT_CUTOFF to avoid artefacts outside the plasma.
        delta_ne = np.where(equilibrium < _PSI_FLUCT_CUTOFF, fluct.T * Ne, 0.)
        c_bg = ax.pcolormesh(R1d, Z1d, delta_ne,
                             cmap='Reds', vmin=0., vmax=np.max(delta_ne), alpha=0.9, zorder=0)
        cb_bg = plt.colorbar(c_bg, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
        cb_bg.set_label(r'$\mathrm{RMS}\ \delta n_e\ [10^{19}\ \mathrm{m}^{-3}]$', size=13)
    else:
        c_bg = ax.pcolormesh(R1d, Z1d, Ne,
                             cmap='Reds', vmin=0., alpha=0.9, zorder=0)
        cb_bg = plt.colorbar(c_bg, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
        cb_bg.set_label(r'$n_e\ [10^{19}\ \mathrm{m}^{-3}]$', size=13)

    # -- Flux surface contours -----------------------------------------------
    if has_flux_surfaces:
        rho = np.sqrt(equilibrium)
        ax.contour(R1d, Z1d, rho, np.arange(0., np.amax(rho), 0.1),
                   colors='grey', linestyles='dashed', linewidths=1, zorder=3)
        ax.contour(R1d, Z1d, rho, [1.],
                   colors='black', linestyles='solid',  linewidths=1, zorder=6)

    # -- White mask outside the vessel boundary ------------------------------
    if vessel_plotted:
        from matplotlib.path import Path
        vessel_path = Path(np.column_stack([Rt, Zt]))
        RR_eq = np.tile(R1d, (nptZ, 1))
        ZZ_eq = np.tile(Z1d, (nptR, 1)).T
        outside = ~vessel_path.contains_points(
            np.column_stack([RR_eq.ravel(), ZZ_eq.ravel()])
        ).reshape(RR_eq.shape)
        c_trans  = clrs.colorConverter.to_rgba('white', alpha=0.)
        c_opaque = clrs.colorConverter.to_rgba('white', alpha=1.)
        mask_cmap = clrs.LinearSegmentedColormap.from_list(
            'vessel_mask', [c_trans, c_opaque], 512)
        ax.pcolormesh(RR_eq, ZZ_eq, outside, cmap=mask_cmap, zorder=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # -- Beam energy density overlays ----------------------------------------
    c_white_semi = clrs.colorConverter.to_rgba('white', alpha=0.8)
    for i, beam in enumerate(beams):
        # 2-D coordinate mesh for this beam's grid [cm]
        _, RR_b = np.meshgrid(beam['Z_cm'], beam['R_cm'])   # (nR, nZ), R values
        ZZ_b, _ = np.meshgrid(beam['Z_cm'], beam['R_cm'])   # (nR, nZ), Z values

        Q = beam['Wfct']
        cb_label = (f'$|E|^2\ (\\mathrm{{J/m}}^3)$\n'
                    f'$f = {beam["FreqGHz"]:.1f}\ \\mathrm{{GHz}}$')

        # Build a semi-transparent colourmap: low values fade into the background
        colour    = _BEAM_COLOURS[i % len(_BEAM_COLOURS)]
        cmap_full = clrs.LinearSegmentedColormap.from_list(
            'beam_full', [c_white_semi, colour], 512)
        cmap_beam = clrs.LinearSegmentedColormap.from_list(
            'beam', [cmap_full(0.25), colour], 512)
        cmap_beam.set_under(alpha=0.)

        upper = np.amax(Q)
        lower = upper * 1e-2   # display values down to 1 % of peak

        bm = ax.pcolormesh(RR_b, ZZ_b, Q,
                           vmin=lower, vmax=upper, cmap=cmap_beam, zorder=9)
        cb_beam = plt.colorbar(bm, ax=ax, orientation='vertical', pad=0.1, shrink=0.7)
        cb_beam.set_label(cb_label, size=10, labelpad=-30, y=1.08, rotation=0)

    # -- Absorption overlay --------------------------------------------------
    if any_abs:
        cmap_abs = matplotlib.colormaps['afmhot'].copy()
        cmap_abs.set_under(alpha=0.)
        peak = np.amax(Absorption_total)
        ab = ax.contourf(RR_b, ZZ_b, Absorption_total,
                         levels=100, vmin=peak / 20., cmap=cmap_abs, zorder=12)
        cb_abs = plt.colorbar(ab, ax=ax, orientation='vertical', pad=0.1, shrink=0.7)
        cb_abs.set_label(r'$P_\mathrm{abs}\ (\mathrm{MW/m}^3)$',
                         size=10, labelpad=-30, y=1.08, rotation=0)
        print(f'Total absorbed power: '
              f'{P_abs_total[0]:.3f} ± {P_abs_total[1]:.3f} MW')

    # -- Cyclotron resonances ------------------------------------------------
    if has_flux_surfaces:
        plotting_functions.add_cyclotron_resonances(R1d, Z1d, StixY, ax)

    # -- Labels and axis limits ----------------------------------------------
    ax.set_xlabel('$R$ [cm]')
    ax.set_ylabel('$Z$ [cm]')
    ax.set_title('Wave propagation\nthrough fluctuations', fontsize=16)

    beamView       = getattr(idata, 'beamView', False)
    Rmin_b, Rmax_b, Zmin_b, Zmax_b = beams[0]['beam_extent']

    if vessel_plotted:
        # Show the full vessel, but always include the beam region
        ax.set_xlim(np.amin(Rv_out), max(np.amax(Rv_out), Rmax_b))
        ax.set_ylim(np.amin(Zv_out), max(np.amax(Zv_out), Zmax_b))
    elif beamView:
        ax.set_xlim(Rmin_b, Rmax_b)
        ax.set_ylim(Zmin_b, Zmax_b)
    else:
        ax.set_xlim(Rmin, Rmax)
        ax.set_ylim(Zmin, Zmax)

    plt.tight_layout()
    plt.show()
#
# END OF FILE
