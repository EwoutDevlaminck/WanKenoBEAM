"""Wrapper utilities for creating WKBeam simulation directories and config files.

Typical use (from wrapper.ipynb):

    from Wrapper.wrapper_aux import SimulationSetup

    sim = SimulationSetup(
        sim_dir='/path/to/TCV_87608_1.55_MyRun',
        sim_name='TCV_87608_1.55',
        rmaj=88., rmin=25., InputPower=1.0,
    )
    sim.create_dirs()

    sim.make_raytracing_config(
        freq=84, sigma=-1.,
        beamwidth1=2.087, beamwidth2=2.087,
        curvatureradius1=79.58, curvatureradius2=79.58,
        rayStartX=122.99, rayStartY=-4.29, rayStartZ=-0.30,
        antennatordeg=-14.0, antennapoldeg=-0.157,
        nmbrRays=1000,
    )

    sim.make_absorption_config(
        nmbr=[40], bin_min=[0.01], bin_max=[0.5],
        uniform_bins=False, grids_file='WKBacca_grids.mat', grids_key='rho_S',
    )

    sim.generate_run_script(n_trace=32, n_bin=8)
"""

import glob
import os
import shutil

import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STANDARD_CONFIGS_DIR = os.path.join(_THIS_DIR, 'StandardConfigs')
STANDARD_MODELS_DIR  = os.path.join(_THIS_DIR, 'StandardModels')


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_config_str(val):
    """Convert a Python value to a string that is valid in a WKBeam config file.

    Config files are executed with ``exec('self.' + line)`` inside InputData,
    with numpy available as ``np``.  Plain Python literals (str, list, float,
    bool, …) and numpy arrays are all handled.
    """
    if isinstance(val, np.ndarray):
        return repr(val.tolist())
    return repr(val)


def make_config(standard_file, output_file, overrides):
    """Merge parameter overrides into a standard config template and write it.

    Parameters
    ----------
    standard_file : str
        Path to the standard template (.txt).
    output_file : str
        Path to write the merged config file.
    overrides : dict
        Mapping of ``{parameter_name: Python_value}``.  Values are converted
        to valid config-file strings automatically (repr for most types).

    Any key in ``overrides`` that does not appear in the template is appended
    at the bottom with a warning, so nothing is silently lost.
    """
    with open(standard_file) as f:
        lines = f.readlines()

    used = set()
    out_lines = []
    for line in lines:
        stripped = line.strip()
        # Only replace top-level (non-indented) key = value lines that are not comments.
        # Skipping indented lines avoids replacing parameter names inside docstrings,
        # function bodies, or nested blocks.
        is_toplevel = bool(line) and not line[0].isspace()
        if is_toplevel and stripped and not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=')[0].strip()
            if key in overrides:
                out_lines.append(f'{key} = {_to_config_str(overrides[key])}\n')
                used.add(key)
                continue
        out_lines.append(line)

    # Keys in overrides that were not in the template → append with a warning
    unused = set(overrides) - used
    if unused:
        print(f'  WARNING ({os.path.basename(standard_file)}): the following '
              f'keys are not in the standard template and will be appended: '
              f'{sorted(unused)}')
        out_lines.append('\n# Parameters not present in the standard template\n')
        for key in sorted(unused):
            out_lines.append(f'{key} = {_to_config_str(overrides[key])}\n')

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(out_lines)
    print(f'  Written: {output_file}')


def create_dir(directory, subdirs=None):
    """Create *directory* (and optional *subdirs*) if they do not yet exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created: {directory}')
    if subdirs:
        for subdir in subdirs:
            subpath = os.path.join(directory, subdir)
            if not os.path.exists(subpath):
                os.makedirs(subpath)
                print(f'Created: {subpath}')


# ---------------------------------------------------------------------------
# EC launcher parameter loader
# ---------------------------------------------------------------------------

def load_ec_params(data_folder):
    """Read an ``ECparams_<shot>_<time>s.mat`` file and return WKBeam-ready params.

    The returned dict can be unpacked directly into
    :meth:`SimulationSetup.make_raytracing_config` for all launcher-specific
    parameters.  The shared ``SimulationSetup`` parameters that also come from
    this file (``InputPower``) are returned separately so they can be passed to
    the constructor.

    Parameters
    ----------
    data_folder : str
        Path to the folder that contains the ``ECparams_*.mat`` file.

    Returns
    -------
    params : dict
        Keys ready for ``make_raytracing_config``:
        ``freq``, ``beamwidth1``, ``beamwidth2``,
        ``curvatureradius1``, ``curvatureradius2``,
        ``rayStartX``, ``rayStartY``, ``rayStartZ``,
        ``antennatordeg``, ``antennapoldeg``.
    InputPower : float
        Input power in MW (converted from kW stored in the file).
    launcher_name : str
        E.g. ``'L4'`` for launcher 4 — use this as ``sim_name`` so that the
        ray-tracing output filename matches the launcher.

    Example
    -------
    ::

        params, InputPower, launcher_name = load_ec_params(data_folder)

        sim = SimulationSetup(
            sim_dir    = ...,
            sim_name   = launcher_name,
            InputPower = InputPower,
        )
        sim.make_raytracing_config(sigma=-1., nmbrRays=1000, **params)
    """
    # Locate the ECparams file
    matches = glob.glob(os.path.join(data_folder, 'ECparams_*.mat'))
    if not matches:
        raise FileNotFoundError(
            f'No ECparams_*.mat file found in {data_folder}')
    if len(matches) > 1:
        raise ValueError(
            f'Multiple ECparams files found in {data_folder}: {matches}')

    mat = loadmat(matches[0])
    b = mat['beams'][0, 0]   # the top-level struct

    def _scalar(field):
        return float(b[field].squeeze())

    launcher    = int(b['launchers'].squeeze())
    freq        = _scalar('frequencies')         # GHz
    beamwidth   = _scalar('beamwidth')           # cm (symmetric beam)
    curv_rad    = _scalar('curv_rad')            # cm (symmetric beam)
    antennapoldeg = _scalar('theta')             # injection poloidal angle [deg]
    antennatordeg = _scalar('phi')               # injection toroidal angle [deg]
    InputPower_kW = _scalar('inputpower')        # kW → convert to MW
    cp = b['centerpoint'].squeeze()
    rayStartX   = float(cp['x'].squeeze())       # cm
    rayStartY   = float(cp['y'].squeeze())       # cm
    rayStartZ   = float(cp['z'].squeeze())       # cm

    launcher_name = f'L{launcher}'
    InputPower_MW = InputPower_kW / 1000.0

    params = dict(
        freq             = freq,
        beamwidth1       = beamwidth,
        beamwidth2       = beamwidth,
        curvatureradius1 = curv_rad,
        curvatureradius2 = curv_rad,
        rayStartX        = rayStartX,
        rayStartY        = rayStartY,
        rayStartZ        = rayStartZ,
        antennatordeg    = antennatordeg,
        antennapoldeg    = antennapoldeg,
    )

    print(f'Loaded ECparams: launcher={launcher_name}, '
          f'freq={freq:.1f} GHz, InputPower={InputPower_MW:.3f} MW')
    return params, InputPower_MW, launcher_name


# ---------------------------------------------------------------------------
# SimulationSetup
# ---------------------------------------------------------------------------

class SimulationSetup:
    """Helper for creating a WKBeam simulation folder and all config files.

    Parameters
    ----------
    sim_dir : str
        Absolute path of the simulation root folder.
    sim_name : str
        Base name used as ``output_filename`` in RayTracing and as
        ``inputfilename`` in all binning configs.
    rmaj, rmin : float
        Tokamak major and minor radius in cm.  Used in Absorption and XZ
        configs for volume/plotting purposes.
    InputPower : float
        Input power in MW applied to all binning configs.
    nmbrFiles : str or list
        ``'all'`` (default) or an explicit list of file indices for binning.
    wkbeam_dir : str, optional
        Path to the WKBeam code directory (where ``WKBeam.py`` lives).
        Defaults to the parent of the Wrapper folder.
    tag : str, optional
        Short label used to identify this simulation run (e.g. ``'nominal'``,
        ``'high_fluct'``).  Becomes the ``label`` field in ``Absorption.txt``.
    """

    def __init__(self, sim_dir, sim_name,
                 rmaj=88., rmin=25.,
                 InputPower=1.0,
                 nmbrFiles='all',
                 wkbeam_dir=None,
                 tag=''):
        self.sim_dir    = sim_dir
        self.sim_name   = sim_name
        self.rmaj       = rmaj
        self.rmin       = rmin
        self.InputPower = InputPower
        self.nmbrFiles  = nmbrFiles
        self.tag        = tag
        self.wkbeam_dir = wkbeam_dir or os.path.dirname(_THIS_DIR)

        # Absolute sub-paths used in generated config files
        self.input_dir  = os.path.join(sim_dir, 'input') + '/'
        self.output_dir = os.path.join(sim_dir, 'output') + '/'
        self.equil_dir  = os.path.join(sim_dir, 'input') + '/'

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _out(self, filename):
        """Absolute path for a config file inside the simulation directory."""
        return os.path.join(self.sim_dir, filename)

    def _std(self, filename):
        """Absolute path for a standard template file."""
        return os.path.join(STANDARD_CONFIGS_DIR, filename)

    def _shared_binning(self):
        """Parameters common to every binning config file."""
        return {
            'inputdirectory':       self.output_dir,
            'outputdirectory':      self.output_dir,
            'equilibriumdirectory': self.equil_dir,
            'inputfilename':        [self.sim_name],
            'nmbrFiles':            self.nmbrFiles,
            'InputPower':           self.InputPower,
            'rmaj':                 self.rmaj,
            'rmin':                 self.rmin,
        }

    # -----------------------------------------------------------------------
    # Directory setup
    # -----------------------------------------------------------------------

    def create_dirs(self, extra_subdirs=None):
        """Create the simulation folder with standard sub-directories.

        Standard subdirs: ``input/``, ``output/``, ``plots/``.
        Pass *extra_subdirs* to add more.
        """
        subdirs = ['input', 'output', 'plots']
        if extra_subdirs:
            subdirs += list(extra_subdirs)
        create_dir(self.sim_dir, subdirs)
        return self

    def copy_input_files(self, data_folder):
        """Copy equilibrium and profile files from *data_folder* into ``input/``.

        Copies every file in *data_folder* except ``*.fig`` and
        ``ECparams_*.mat`` (those are MATLAB figures / launcher parameters that
        WKBeam does not need at run time).

        Parameters
        ----------
        data_folder : str
            Source folder (e.g. the shot/time folder prepared by MATLAB).
        """
        dest = os.path.join(self.sim_dir, 'input')
        os.makedirs(dest, exist_ok=True)
        copied = []
        for src_path in sorted(glob.glob(os.path.join(data_folder, '*'))):
            fname = os.path.basename(src_path)
            if fname.endswith('.fig') or fname.startswith('ECparams_'):
                continue
            shutil.copy2(src_path, os.path.join(dest, fname))
            copied.append(fname)
        print(f'Copied to {dest}/: {", ".join(copied)}')
        return self

    def make_fluct_amplitude(self, ballooning=0.5):
        """Generate ``input/fluct_amplitude.py`` from the standard template.

        The file reads ``nefluct.dat`` from the same directory it lives in, so
        no path needs to be hard-coded.

        Parameters
        ----------
        ballooning : float
            Poloidal ballooning parameter.
            ``1`` → isotropic; ``0`` → fully ballooned (amplitude ∝ cos²(θ/2)).
        """
        src  = os.path.join(STANDARD_MODELS_DIR, 'fluct_amplitude.py')
        dest = os.path.join(self.sim_dir, 'input', 'fluct_amplitude.py')
        make_config(src, dest, {'ballooning': ballooning})
        return self

    def make_lperp_model(self, mu=2.0, Z=1.0, epsilon=0.3, factor=7.4):
        """Generate ``input/Lperp_rhos_model.py`` from the standard template.

        Parameters
        ----------
        mu : float      Ion mass / proton mass  (2 for deuterium, 1 for hydrogen).
        Z  : float      Ion charge state.
        epsilon : float Minimum Lperp in cm  (floor to avoid Lperp → 0).
        factor  : float Proportionality constant Lperp = factor * rho_s
                        (default 7.4, calibrated against GBS).
        """
        src  = os.path.join(STANDARD_MODELS_DIR, 'Lperp_rhos_model.py')
        dest = os.path.join(self.sim_dir, 'input', 'Lperp_rhos_model.py')
        make_config(src, dest, {'mu': mu, 'Z': Z, 'epsilon': epsilon, 'factor': factor})
        return self

    # -----------------------------------------------------------------------
    # Config file generators
    # -----------------------------------------------------------------------

    def make_raytracing_config(self,
                                freq, sigma,
                                beamwidth1, beamwidth2,
                                curvatureradius1, curvatureradius2,
                                rayStartX, rayStartY, rayStartZ,
                                antennatordeg, antennapoldeg,
                                nmbrRays=1000,
                                anglespecification='ASDEX',
                                vesselfile=None,
                                **extra):
        """Generate ``RayTracing.txt``.

        Parameters
        ----------
        freq : float            Frequency in GHz.
        sigma : float           Mode index (±1).
        beamwidth1/2 : float    Beam widths in cm.
        curvatureradius1/2 : float  Curvature radii in cm.
        rayStartX/Y/Z : float   Antenna position in cm (lab frame).
        antennatordeg : float   Toroidal injection angle in degrees.
        antennapoldeg : float   Poloidal injection angle in degrees.
        nmbrRays : int          Number of rays to trace.
        anglespecification : str  ``'ASDEX'`` (default) or ``'ITER'``.
        vesselfile : str, optional
            Absolute path to the ``*_vessel.mat`` file for this shot.
            If omitted, the value already in the standard template is kept.
        **extra
            Any other RayTracing parameter to override (e.g. ``scattering=False``).
        """
        overrides = {
            'output_dir':           self.output_dir,
            'output_filename':      self.sim_name,
            'equilibriumdirectory': self.equil_dir,
            'freq':                 freq,
            'sigma':                sigma,
            'beamwidth1':           beamwidth1,
            'beamwidth2':           beamwidth2,
            'curvatureradius1':     curvatureradius1,
            'curvatureradius2':     curvatureradius2,
            'rayStartX':            rayStartX,
            'rayStartY':            rayStartY,
            'rayStartZ':            rayStartZ,
            'antennatordeg':        antennatordeg,
            'antennapoldeg':        antennapoldeg,
            'nmbrRays':             nmbrRays,
            'anglespecification':   anglespecification,
        }
        if vesselfile is not None:
            overrides['vesselfile'] = vesselfile
        overrides.update(extra)
        print('Generating RayTracing.txt ...')
        make_config(self._std('RayTracing.txt'), self._out('RayTracing.txt'), overrides)
        return self

    def make_absorption_config(self,
                                nmbr, bin_min, bin_max,
                                uniform_bins=True,
                                grids_file=None, grids_key=None,
                                outputfilename='Absorption_binned',
                                **extra):
        """Generate ``Absorption.txt`` (1-D rho absorption profile binning).

        Parameters
        ----------
        nmbr : list[int]        Number of bins per dimension.
        bin_min, bin_max : list[float]
            Lower and upper bin boundaries.  Used when ``uniform_bins=True``.
        uniform_bins : bool
            ``True``  → uniform spacing defined by nmbr/bin_min/bin_max.
            ``False`` → non-uniform; provide *grids_file* + *grids_key* to
                        load bin edges from a MATLAB file, or pass ``bins``
                        directly via *extra*.
        grids_file : str, optional
            Name of the ``.mat`` file with grid edges (e.g. ``'WKBacca_grids.mat'``).
        grids_key : str, optional
            Key inside the grids file (e.g. ``'rho_S'``).
        outputfilename : str
            Output HDF5 base name (without extension).
        **extra
            Any other binning parameter to override.
        """
        overrides = self._shared_binning()
        overrides.update({
            'outputfilename': [outputfilename],
            'uniform_bins':   uniform_bins,
            'nmbr':           nmbr,
            'min':            bin_min,
            'max':            bin_max,
            'label':          self.tag,
        })
        if grids_file is not None:
            overrides['bins'] = [[grids_file, grids_key]]
        overrides.update(extra)
        print('Generating Absorption.txt ...')
        make_config(self._std('Absorption.txt'), self._out('Absorption.txt'), overrides)
        return self

    def make_xz_config(self,
                        nmbr, bin_min, bin_max,
                        uniform_bins=True,
                        bins=None,
                        outputfilename='XZ_binned',
                        **extra):
        """Generate ``XZ.txt`` (2-D poloidal-plane R-Z binning).

        Parameters
        ----------
        nmbr : list[int]        Number of bins for [R, Z].
        bin_min, bin_max : list[float]
            Boundaries for [R, Z] in cm.
        uniform_bins : bool     See :meth:`make_absorption_config`.
        bins : list, optional
            Non-uniform bin edges.  Each element is either a ``np.ndarray``
            or a plain list.  Pass ``None`` for dimensions with uniform bins.
        outputfilename : str    Output HDF5 base name.
        **extra                 Any other parameter to override.
        """
        overrides = self._shared_binning()
        overrides.update({
            'outputfilename': [outputfilename],
            'uniform_bins':   uniform_bins,
            'nmbr':           nmbr,
            'min':            bin_min,
            'max':            bin_max,
        })
        if bins is not None:
            overrides['bins'] = [b.tolist() if isinstance(b, np.ndarray) else b
                                  for b in bins]
        overrides.update(extra)
        print('Generating XZ.txt ...')
        make_config(self._std('XZ.txt'), self._out('XZ.txt'), overrides)
        return self

    def make_angular_config(self,
                             nmbr, bin_min, bin_max,
                             outputfilename='Angular_binned',
                             **extra):
        """Generate ``Angular.txt`` (2-D R-phiN angular spectrum binning).

        Parameters
        ----------
        nmbr : list[int]        Number of bins for [R, phiN].
        bin_min, bin_max : list[float]
            Boundaries for [R, phiN].
        outputfilename : str    Output HDF5 base name.
        **extra                 Any other parameter to override.
        """
        overrides = self._shared_binning()
        overrides.update({
            'outputfilename': [outputfilename],
            'nmbr':           nmbr,
            'min':            bin_min,
            'max':            bin_max,
        })
        overrides.update(extra)
        print('Generating Angular.txt ...')
        make_config(self._std('Angular.txt'), self._out('Angular.txt'), overrides)
        return self

    def make_rhothetaN_config(self,
                               nmbr, bin_min, bin_max,
                               uniform_bins=True,
                               grids_file=None, grids_key=None,
                               outputfilename='RhoThetaN_binned',
                               **extra):
        """Generate ``RhoThetaN.txt`` (4-D rho-Theta-Nparallel-Nperp binning).

        Parameters
        ----------
        nmbr : list[int]        Number of bins for [rho, Theta, Nparallel, Nperp].
        bin_min, bin_max : list[float]
            Boundaries for each dimension.
        uniform_bins : bool     See :meth:`make_absorption_config`.
        grids_file, grids_key : str, optional
            Load non-uniform rho bin edges from a MATLAB file.
            The remaining three dimensions use uniform bins.
        outputfilename : str    Output HDF5 base name.
        **extra                 Any other parameter to override.
        """
        overrides = self._shared_binning()
        overrides.update({
            'outputfilename': [outputfilename],
            'uniform_bins':   uniform_bins,
            'nmbr':           nmbr,
            'min':            bin_min,
            'max':            bin_max,
        })
        if grids_file is not None:
            # Non-uniform rho from file; remaining dims use uniform (empty list)
            overrides['bins'] = [[grids_file, grids_key], [], [], []]
        overrides.update(extra)
        print('Generating RhoThetaN.txt ...')
        make_config(self._std('RhoThetaN.txt'), self._out('RhoThetaN.txt'), overrides)
        return self

    def make_qldiff_config(self,
                            harmonics=None,
                            manual_grids=False,
                            gridfile='WKBacca_grids.mat',
                            outputfilename='QLdiff_binned',
                            absorption_file='Absorption.txt',
                            absorption_data_file='Absorption_binned.hdf5',
                            **extra):
        """Generate ``QLdiff.txt`` (quasi-linear diffusion coefficient).

        Parameters
        ----------
        harmonics : list[int]   Cyclotron harmonics to include (default ``[2]``).
        manual_grids : bool     Set ``True`` to specify p/ksi grids manually.
        gridfile : str          MATLAB grid file for momentum-space grid.
        outputfilename : str    Output HDF5 base name.
        absorption_file : str   Config file for the absorption binning step.
        absorption_data_file : str  Binned absorption HDF5 filename.
        **extra                 Any other parameter to override (e.g. pmin, pmax, np, nksi).
        """
        overrides = {
            'outputdirectory':      self.output_dir,
            'outputfilename':       outputfilename,
            'harmonics':            harmonics if harmonics is not None else [2],
            'manual_grids':         manual_grids,
            'gridfile':             gridfile,
            'absorption_file':      absorption_file,
            'absorption_data_file': absorption_data_file,
        }
        overrides.update(extra)
        print('Generating QLdiff.txt ...')
        make_config(self._std('QLdiff.txt'), self._out('QLdiff.txt'), overrides)
        return self

    # -----------------------------------------------------------------------
    # Script generation
    # -----------------------------------------------------------------------

    def generate_run_script(self,
                             n_trace, n_bin,
                             bin_configs=None,
                             script_path=None):
        """Generate an executable bash script that traces and bins.

        Parameters
        ----------
        n_trace : int
            Number of MPI ranks for ray tracing.
        n_bin : int
            Number of MPI ranks for each binning step.
        bin_configs : list[str], optional
            Config filenames (basename only) to include in the binning step.
            Defaults to all ``.txt`` files in the simulation directory except
            ``RayTracing.txt`` and ``QLdiff.txt``.
        script_path : str, optional
            Where to write the script.  Defaults to ``<sim_dir>/run.sh``.
        """
        wkbeam_py = os.path.join(self.wkbeam_dir, 'WKBeam.py')
        rt_config  = self._out('RayTracing.txt')

        if bin_configs is None:
            bin_configs = sorted(
                f for f in os.listdir(self.sim_dir)
                if f.endswith('.txt') and f not in ('RayTracing.txt', 'QLdiff.txt')
            )

        bin_lines = '\n'.join(
            f'mpiexec -np {n_bin} python3 {wkbeam_py} bin {self._out(cfg)}'
            for cfg in bin_configs
        )

        script = (
            f'#!/bin/bash\n'
            f'# Auto-generated by SimulationSetup.generate_run_script\n\n'
            f'mpiexec -np {n_trace} python3 {wkbeam_py} trace {rt_config}\n'
            f'wait\n\n'
            f'{bin_lines}\n'
        )

        path = script_path or self._out('run.sh')
        with open(path, 'w') as f:
            f.write(script)
        os.chmod(path, 0o755)
        print(f'Run script: {path}')
        return path

    def generate_plot_script(self, xz_outputfilename='XZ_binned', script_path=None):
        """Generate an executable bash script that runs all plotting steps.

        Plotting commands are launched in parallel (``&``) and the script
        waits for all of them to finish.  Commands are included based on
        which config ``.txt`` files exist in the simulation directory (checked
        at generation time, not at run time).

        ``plotbin`` and ``beamFluct`` both receive the XZ-binned HDF5 path and
        the RayTracing config:
          - ``plotbin``   : 2-D beam overlaid on equilibrium flux surfaces
          - ``beamFluct`` : 2-D beam overlaid with fluctuation amplitude profile

        Parameters
        ----------
        xz_outputfilename : str
            Base name of the XZ-binned HDF5 file (without extension).
            Must match the ``outputfilename`` used in :meth:`make_xz_config`
            (default ``'XZ_binned'``).
        script_path : str, optional
            Where to write the script.  Defaults to ``<sim_dir>/plot.sh``.
        """
        wkbeam_py = os.path.join(self.wkbeam_dir, 'WKBeam.py')
        rt_config = self._out('RayTracing.txt')
        xz_binned = self._out(os.path.join('output', xz_outputfilename + '.hdf5'))

        lines = ['#!/bin/bash', '# Auto-generated by SimulationSetup.generate_plot_script', '']

        if os.path.exists(self._out('Angular.txt')):
            lines.append(f'python3 {wkbeam_py} plot2d {self._out("Angular.txt")} &')
        if os.path.exists(self._out('Absorption.txt')):
            lines.append(f'python3 {wkbeam_py} plotabs {self._out("Absorption.txt")} &')
        if os.path.exists(self._out('XZ.txt')):
            # Both commands take: <xz_binned_hdf5> <raytracing_config>
            #lines.append(f'python3 {wkbeam_py} plotbin   {xz_binned} {rt_config} &')
            lines.append(f'python3 {wkbeam_py} beamFluct {xz_binned} {rt_config} &')

        lines += ['wait', 'echo "All done!"', '']

        path = script_path or self._out('plot.sh')
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        os.chmod(path, 0o755)
        print(f'Plot script: {path}')
        return path
