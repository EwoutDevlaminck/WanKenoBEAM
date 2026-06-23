"""This file reads a 2-D binned data file -- either an axisymmetric (R,Z)
(poloidal-plane) binning or a top-down (X,Y) binning -- and plots the beam
intensity and its statistical uncertainty, overlaid on the equilibrium.

Layout and beam colour/colormap conventions mirror beamFluct
(Tools/PlotData/PlotFluctuations/plotBeamFluctuations.py): same per-beam
colour sequence, same white-to-colour blended colormap, same absorption
overlay (afmhot). Unlike beamFluct, there is no fluctuation-amplitude
background and no vessel.
"""

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import IntSample, StixParamSample
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium, TokamakEquilibrium2
from Tools.PlotData.CommonPlotting import plotting_functions
from Tools.PlotData.PlotFluctuations.plotBeamFluctuations import _load_beam_data

# Colour sequence for successive beam overlays (same as beamFluct)
_BEAM_COLOURS = ['#007480', '#F39869', '#413C3A', '#00A79F']


def _load_beam_data_topdown(filename):
    """Load one (X,Y)-binned (top-down) HDF5 file.

    Unlike the axisymmetric (R,Z) case handled by _load_beam_data, an (X,Y)
    bin is not a toroidal shell of known volume -- Z is unresolved, so each
    bin mixes contributions from all Z. No toroidal-volume unit conversion
    is applied here: values are simple areal densities (native binning
    units / bin area), matching what this view already produced before this
    rewrite. Returned dict keys mirror _load_beam_data's, so the rest of the
    plotting code can treat both cases uniformly.
    """
    with h5py.File(filename, 'r') as fid:
        FreqGHz  = fid['FreqGHz'][()]
        mode     = fid['Mode'][()]
        Wfct_raw = fid['BinnedTraces'][()]

        Abs_recorded = 'Absorption' in fid
        Abs_raw      = fid['Absorption'][()] if Abs_recorded else None

        Xmin = fid['Xmin'][()]; Xmax = fid['Xmax'][()]; nX = int(fid['nmbrX'][()])
        Ymin = fid['Ymin'][()]; Ymax = fid['Ymax'][()]; nY = int(fid['nmbrY'][()])

        resolveNpar = 'Nparallelmin' in fid

    if resolveNpar:
        Wfct_raw = np.sum(Wfct_raw, axis=2)
        if Abs_recorded:
            Abs_raw = np.sum(Abs_raw, axis=2)

    X_cm = np.linspace(Xmin, Xmax, nX)
    Y_cm = np.linspace(Ymin, Ymax, nY)
    area = (Xmax - Xmin) / nX * (Ymax - Ymin) / nY

    Wfct = Wfct_raw / area

    if Abs_recorded:
        P_abs      = np.sum(Abs_raw, axis=(0, 1))   # shape (2,): [mean, rms]
        Absorption = Abs_raw / area
    else:
        Absorption = None
        P_abs      = None

    return dict(
        FreqGHz      = FreqGHz,
        mode         = mode,
        R_cm         = X_cm,
        Z_cm         = Y_cm,
        Wfct         = Wfct[:, :, 0],
        Wfct_unc     = Wfct[:, :, 1],
        Absorption   = Absorption[:, :, 0] if Abs_recorded else None,
        P_abs        = P_abs,
        Abs_recorded = Abs_recorded,
        beam_extent  = (Xmin, Xmax, Ymin, Ymax),
    )


def plot_binned(inputdata):
    """Plot the binned beam intensity and uncertainty, overlaid on the
    equilibrium.

    Parameters
    ----------
    inputdata : list
        All elements except the last are paths to binned HDF5 files
        (either (R,Z) or (X,Y) binning -- determined from the first file).
        The last element is the path to the RayTracing configuration file.
    """
    inputfilenames, configfile = inputdata[0:-1], inputdata[-1]

    idata = InputData(configfile)

    # -------------------------------------------------------------------
    # Figure out which view we're dealing with, from the first file, and
    # load all beams accordingly.
    # -------------------------------------------------------------------
    with h5py.File(inputfilenames[0], 'r') as fid:
        resolveZ_file = ('Zmin' in fid) or ('Zbins' in fid)
        resolveY_file = (('Ymin' in fid) or ('Ybins' in fid)) and not resolveZ_file

    if resolveZ_file:
        print('Reading data file(s) (R/X, Z binning)...')
        beams       = [_load_beam_data(f) for f in inputfilenames]
        axis_labels = ('$R$ [cm]', '$Z$ [cm]')
    elif resolveY_file:
        print('Reading data file(s) (X, Y top-down binning)...')
        beams       = [_load_beam_data_topdown(f) for f in inputfilenames]
        axis_labels = ('$X$ [cm]', '$Y$ [cm]')
    else:
        raise ValueError('plot_binned: could not determine whether the '
                          'binned file resolves (R,Z) or (X,Y).')

    any_abs = any(b['Abs_recorded'] for b in beams)
    if any_abs:
        P_abs_total = sum(b['P_abs'] for b in beams if b['P_abs'] is not None)

    # -------------------------------------------------------------------
    # Equilibrium (only for Tokamak/Tokamak2D devices)
    # -------------------------------------------------------------------
    has_equilibrium = idata.equilibrium in ('Tokamak', 'Tokamak2D')

    if has_equilibrium:
        Eq = TokamakEquilibrium(idata) if idata.equilibrium == 'Tokamak' \
            else TokamakEquilibrium2(idata)

        if resolveZ_file:
            # Dedicated equilibrium grid (independent of the binning
            # resolution), as in beamFluct.
            Rmin, Rmax = Eq.Rgrid[0, 0], Eq.Rgrid[-1, 0]
            Zmin, Zmax = Eq.zgrid[0, 0], Eq.zgrid[0, -1]
            nptR = int((Rmax - Rmin) / (idata.rmin / 100.))
            nptZ = int((Zmax - Zmin) / (idata.rmin / 100.))
            print('Equilibrium grid: nptR={}, nptZ={}'.format(nptR, nptZ))
            R1d = np.linspace(Rmin, Rmax, nptR)
            Z1d = np.linspace(Zmin, Zmax, nptZ)

            _, StixY, _ = StixParamSample(R1d, Z1d, Eq, idata.freq)
            psi = IntSample(R1d, Z1d, Eq.PsiInt.eval)
            rho = np.sqrt(psi)
        else:
            # Top-down view: no poloidal grid -- just the characteristic
            # radii, drawn as circles centred on X=Y=0.
            R_axis, R_lcfs_min, R_lcfs_max, R_harm2 = plotting_functions.topdown_radii(
                Eq, idata.freq, harm_n=2)

    # -------------------------------------------------------------------
    # Figure: intensity (left) and statistical uncertainty (right)
    # -------------------------------------------------------------------
    print('Plotting data...')

    plt.rcParams.update({'font.size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    panels = [
        (axes[0], 'Wfct',     'Wave intensity\n' + r'$|E|^2$',                True),
        (axes[1], 'Wfct_unc', 'Statistical uncertainty\non ' + r'$|E|^2$',    False),
    ]

    c_white_semi = clrs.colorConverter.to_rgba('white', alpha=0.8)

    RR_b = ZZ_b = None   # last beam's coordinate mesh, reused for the absorption overlay

    for ax, key, title, show_abs in panels:

        ax.set_aspect('equal')

        # -- Equilibrium overlay ---------------------------------------
        if has_equilibrium:
            if resolveZ_file:
                ax.contour(R1d, Z1d, rho, np.arange(0., np.amax(rho), 0.1),
                           colors='grey', linestyles='dashed', linewidths=1, zorder=3)
                ax.contour(R1d, Z1d, rho, [1.],
                           colors='black', linestyles='solid', linewidths=1, zorder=6)
            else:
                plotting_functions.add_topdown_circles(
                    ax, R_axis, R_lcfs_min, R_lcfs_max, R_harm2, harm_n=2)
                ax.legend(fontsize=8, loc='upper right')

        # -- Beam overlays (same colour/colormap convention as beamFluct) --
        for i, beam in enumerate(beams):
            _, RR_b = np.meshgrid(beam['Z_cm'], beam['R_cm'])   # (nR, nZ)
            ZZ_b, _ = np.meshgrid(beam['Z_cm'], beam['R_cm'])

            Q = beam[key]

            colour    = _BEAM_COLOURS[i % len(_BEAM_COLOURS)]
            cmap_full = clrs.LinearSegmentedColormap.from_list(
                'beam_full', [c_white_semi, colour], 512)
            cmap_beam = clrs.LinearSegmentedColormap.from_list(
                'beam', [cmap_full(0.25), colour], 512)
            cmap_beam.set_under(alpha=0.)

            upper = np.amax(Q)
            lower = upper * 1e-2   # display values down to 1 % of peak

            bm = ax.pcolormesh(RR_b, ZZ_b, Q, vmin=lower, vmax=upper,
                               cmap=cmap_beam, zorder=9)
            cb_beam = plt.colorbar(bm, ax=ax, orientation='vertical', pad=0.1, shrink=0.7)
            unit = r'$\mathrm{J/m}^3$' if resolveZ_file else 'a.u.'
            cb_beam.set_label(
                '$|E|^2$ ({})\n$f={:.1f}$ GHz'.format(unit, beam['FreqGHz']),
                size=9, labelpad=-30, y=1.08, rotation=0)

        # -- Absorption overlay (intensity panel only) -------------------
        if show_abs and any_abs:
            Absorption_total = sum(b['Absorption'] for b in beams
                                    if b['Absorption'] is not None)
            cmap_abs = matplotlib.colormaps['afmhot'].copy()
            cmap_abs.set_under(alpha=0.)
            peak = np.amax(Absorption_total)
            ab = ax.contourf(RR_b, ZZ_b, Absorption_total,
                             levels=100, vmin=peak / 50., cmap=cmap_abs, zorder=12)
            cb_abs = plt.colorbar(ab, ax=ax, orientation='vertical', pad=0.1, shrink=0.7)
            unit = r'\mathrm{MW/m}^3' if resolveZ_file else r'\mathrm{a.u.}'
            cb_abs.set_label(r'$P_\mathrm{{abs}}$ ({})'.format(unit),
                             size=9, labelpad=-30, y=1.08, rotation=0)

        # -- Cyclotron resonances (R,Z view only; the top-down view already
        #    drew the 2nd-harmonic circle as part of add_topdown_circles) --
        if has_equilibrium and resolveZ_file:
            plotting_functions.add_cyclotron_resonances(R1d, Z1d, StixY, ax)

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(title, fontsize=13)

        # -- Axis limits ---------------------------------------------------
        Rmin_b, Rmax_b, Zmin_b, Zmax_b = beams[0]['beam_extent']
        beamView = getattr(idata, 'beamView', False)
        if has_equilibrium and resolveZ_file and not beamView:
            ax.set_xlim(Rmin, Rmax)
            ax.set_ylim(Zmin, Zmax)
        else:
            ax.set_xlim(Rmin_b, Rmax_b)
            ax.set_ylim(Zmin_b, Zmax_b)

    if any_abs:
        print('Total absorbed power: {:.3f} ± {:.3f} MW'.format(
            P_abs_total[0], P_abs_total[1]))

    plt.tight_layout()
    plt.show()

    return
#
# END OF FILE
