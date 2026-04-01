"""Binning interface: helpers for serial and parallel binning.

Exported helpers used by both the serial entry point (binning_pyinterface)
and the parallel org/worker cores (binning_org, binning_worker):

  _setup_bins(idata)                       -- parse bin configuration
  _allocate_accum(idata, setup)            -- allocate zero accumulator arrays
  _find_indices(idata)                     -- discover HDF5 file indices
  _bin_one_file(file_idx, idata, setup)    -- bin a single file, return partial dict
  _accumulate(accum, partial)              -- add partial result to accumulator
  _postprocess_and_write(idata, accum, setup, outputfilename)
"""

# Load standard modules
import sys
import os
import re
import glob
import h5py
import numpy as np
# Load local modules
from Binning.modules.binning import binning
from Binning.modules.binning_nonuni import binning as binning_nonuni
from Binning.modules.compute_normalisationfactor import compute_norm_factor

from scipy.io import loadmat


def _select_weight(component, data_sim_NxNyNz=None, data_sim_VxVyVz=None,
                   data_sim_Nparallel=None, data_sim_phiN=None, data_sim_Nperp=None):
    """Return the weight array slice for a named velocity/refractive-index component."""
    if component == "Nx":          return data_sim_NxNyNz[:,0,:]
    elif component == "Ny":        return data_sim_NxNyNz[:,1,:]
    elif component == "Nz":        return data_sim_NxNyNz[:,2,:]
    elif component == "Vx":        return data_sim_VxVyVz[:,0,:]
    elif component == "Vy":        return data_sim_VxVyVz[:,1,:]
    elif component == "Vz":        return data_sim_VxVyVz[:,2,:]
    elif component == "Nparallel": return data_sim_Nparallel
    elif component == "phiN":      return data_sim_phiN
    elif component == "Nperp":     return data_sim_Nperp
    raise ValueError(f"Unknown velocity component: {component}")


############################################################################
# SHARED HELPERS
############################################################################

def _setup_bins(idata):
    """Parse idata to set up bin arrays.

    Returns a dict with keys:
      nmbr, uniform_bins,
      bin_min / bin_max  (uniform case, else None),
      bins               (non-uniform case, else None),
      VelocityComponentsToStore, all_vars
    """
    VelocityComponentsToStore = idata.VelocityComponentsToStore if idata.storeVelocityField else []
    all_vars = set(idata.WhatToResolve) | set(VelocityComponentsToStore)
    uniform_bins = getattr(idata, 'uniform_bins', True)

    nmbr = np.empty([4], dtype=int)
    bin_min = bin_max = bins = None

    if uniform_bins:
        bin_min = np.empty([4])
        bin_max = np.empty([4])
        for i in range(4):
            if i < len(idata.nmbr):
                nmbr[i] = idata.nmbr[i]
                bin_min[i] = idata.min[i]
                bin_max[i] = idata.max[i]
                if bin_min[i] > bin_max[i]:
                    print('ERROR: lower boundary larger than the upper one for %s\n' % idata.WhatToResolve[i])
                    raise ValueError('Invalid bin boundaries')
            else:
                nmbr[i] = 1
                bin_min[i] = -1.
                bin_max[i] = +1.
    else:
        bins = np.empty([4], dtype=np.ndarray)
        for i in range(4):
            if i < len(idata.bins):
                if len(idata.bins[i]) == 0:
                    # Fall back to uniform spacing for this dimension
                    nmbr[i] = idata.nmbr[i]
                    bins[i] = np.linspace(idata.min[i], idata.max[i], idata.nmbr[i]+1)
                else:
                    if isinstance(idata.bins[i][0], str):
                        grids = loadmat(idata.outputdirectory + idata.bins[i][0])['WKBacca_grids']
                        bins[i] = grids[idata.bins[i][1]][0,0][0]
                    else:
                        bins[i] = idata.bins[i]
                    nmbr[i] = len(bins[i]) - 1
            else:
                bins[i] = np.linspace(-1., 1., 2)
                nmbr[i] = 1

    return dict(nmbr=nmbr, uniform_bins=uniform_bins,
                bin_min=bin_min, bin_max=bin_max, bins=bins,
                VelocityComponentsToStore=VelocityComponentsToStore,
                all_vars=all_vars)


def _allocate_accum(idata, setup):
    """Allocate zero accumulator arrays for all outputs required by idata.

    Returns a dict whose array keys match those returned by _bin_one_file.
    """
    nmbr = setup['nmbr']
    VelocityComponentsToStore = setup['VelocityComponentsToStore']

    accum = {'nmbrRaysUnscattered': 0, 'nmbrRaysScattered': 0, 'file_params': None}

    if idata.storeWfct:
        accum['WfctUnscattered'] = np.zeros(np.append(nmbr, 2))
    if idata.storeVelocityField:
        accum['VelocityFieldUnscattered'] = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
    if idata.storeAbsorption:
        accum['AbsorptionUnscattered'] = np.zeros(np.append(nmbr, 2))
    if idata.computeAmplitude or idata.computeScatteringEffect:
        if idata.storeWfct:
            accum['WfctScattered'] = np.zeros(np.append(nmbr, 2))
        if idata.storeVelocityField:
            accum['VelocityFieldScattered'] = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
        if idata.storeAbsorption:
            accum['AbsorptionScattered'] = np.zeros(np.append(nmbr, 2))

    return accum


def _find_indices(idata):
    """Return a sorted list of integer file indices to process."""
    if idata.nmbrFiles == 'all':
        pattern = os.path.join(idata.inputdirectory, f"{idata.inputfilename}_file*.hdf5")
        existing_files = glob.glob(pattern)
        pattern_re = re.compile(f"{re.escape(idata.inputfilename)}_file(\\d+)\\.hdf5")
        return sorted(
            int(m.group(1))
            for f in existing_files
            for m in [pattern_re.search(os.path.basename(f))]
            if m
        )
    else:
        return list(idata.nmbrFiles)


def _bin_one_file(file_idx, idata, setup):
    """Load one HDF5 ray-tracing file, bin its contents, and return partial sums.

    Returns a dict with the same array keys as _allocate_accum produces, plus:
      'nmbrRaysUnscattered', 'nmbrRaysScattered', 'nmbrRaysToUse', 'file_params'
    """
    nmbr                      = setup['nmbr']
    uniform_bins              = setup['uniform_bins']
    bin_min                   = setup['bin_min']
    bin_max                   = setup['bin_max']
    bins                      = setup['bins']
    VelocityComponentsToStore = setup['VelocityComponentsToStore']
    all_vars                  = setup['all_vars']

    # Allocate fresh partial arrays (zeros) for this file
    partial = {'nmbrRaysUnscattered': 0, 'nmbrRaysScattered': 0}
    if idata.storeWfct:
        partial['WfctUnscattered'] = np.zeros(np.append(nmbr, 2))
    if idata.storeVelocityField:
        partial['VelocityFieldUnscattered'] = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
    if idata.storeAbsorption:
        partial['AbsorptionUnscattered'] = np.zeros(np.append(nmbr, 2))
    if idata.computeAmplitude or idata.computeScatteringEffect:
        if idata.storeWfct:
            partial['WfctScattered'] = np.zeros(np.append(nmbr, 2))
        if idata.storeVelocityField:
            partial['VelocityFieldScattered'] = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
        if idata.storeAbsorption:
            partial['AbsorptionScattered'] = np.zeros(np.append(nmbr, 2))

    filename = idata.inputdirectory + idata.inputfilename + '_file%i.hdf5' % file_idx
    print("loading file %s ...\n" % filename)
    sys.stdout.flush()

    with h5py.File(filename, 'r') as fid:
        data_sim_CorrectionFactor = fid.get('TracesCorrectionFactor')[()] if idata.correctionfactor else np.empty([0,0])
        data_sim_XYZ             = fid.get('TracesXYZ')[()]
        data_sim_Wfct            = fid.get('TracesWfct')[()]
        data_sim_nmbrscattevents = fid.get('TracesNumberScattEvents')[()]

        nmbrRaysToUse    = data_sim_XYZ.shape[0]
        nmbrPointsPerRay = data_sim_XYZ.shape[2]

        # Load only the arrays that are actually needed
        if all_vars & {'Nx', 'Ny', 'Nz'}:
            data_sim_NxNyNz = fid.get('TracesNxNyNz')[()]
        if all_vars & {'Vx', 'Vy', 'Vz'}:
            data_sim_VxVyVz = fid.get('TracesGroupVelocity')[()]
        if all_vars & {'Psi', 'rho'}:
            data_sim_Psi = fid.get('TracesPsi')[()]
        if 'Theta' in all_vars:
            data_sim_Theta = fid.get('TracesTheta')[()]
        if 'Nparallel' in all_vars:
            data_sim_Nparallel = fid.get('TracesNparallel')[()]
        if 'phiN' in all_vars:
            data_sim_phiN = fid.get('TracesphiN')[()]
        if 'Nperp' in all_vars:
            data_sim_Nperp = fid.get('TracesNperpendicular')[()]

        # Use time stored in file if available, otherwise reconstruct from timestep
        if "TracesTime" in fid:
            data_sim_time = fid.get("TracesTime")[()]
            print('time along the rays was found and is used.\n')
        else:
            timestep = fid.get("timestep")[()]
            data_sim_time = np.empty([nmbrRaysToUse, nmbrPointsPerRay])
            for k in range(nmbrRaysToUse):
                data_sim_time[k,:] = np.linspace(0., (nmbrPointsPerRay-1)*timestep, nmbrPointsPerRay)

        # Read antenna / beam parameters
        _antenna_keys = [
            "Mode", "FreqGHz", "antennapolangle", "antennatorangle",
            "rayStartX", "rayStartY", "rayStartZ",
            "beamwidth1", "beamwidth2",
            "curvatureradius1", "curvatureradius2",
            "centraleta1", "centraleta2",
        ]
        file_params = {key: fid.get(key)[()] for key in _antenna_keys}

    # Compute normalisation factor from this file's antenna parameters
    InputPower = getattr(idata, 'InputPower', None)
    if InputPower is not None:
        NormFactor = InputPower / compute_norm_factor(
            file_params["FreqGHz"],
            file_params["beamwidth1"], file_params["beamwidth2"],
            file_params["curvatureradius1"], file_params["curvatureradius2"],
            file_params["centraleta1"], file_params["centraleta2"])
        print("Input power is %.3fMW\n" % InputPower)
    else:
        NormFactor = 1.
        print("Normalisation such that central electric field on antenna is 1.\n")

    # Compose data_sim from WhatToResolve
    data_sim = np.zeros([nmbrRaysToUse, 4, nmbrPointsPerRay])
    for dim, var in enumerate(idata.WhatToResolve):
        if var == 'X':           data_sim[:,dim,:] = data_sim_XYZ[:,0,:]
        elif var == 'Y':         data_sim[:,dim,:] = data_sim_XYZ[:,1,:]
        elif var == 'Z':         data_sim[:,dim,:] = data_sim_XYZ[:,2,:]
        elif var == 'Nx':        data_sim[:,dim,:] = data_sim_NxNyNz[:,0,:]
        elif var == 'Ny':        data_sim[:,dim,:] = data_sim_NxNyNz[:,1,:]
        elif var == 'Nz':        data_sim[:,dim,:] = data_sim_NxNyNz[:,2,:]
        elif var == 'Nparallel': data_sim[:,dim,:] = data_sim_Nparallel
        elif var == 'phiN':      data_sim[:,dim,:] = data_sim_phiN
        elif var == 'Nperp':     data_sim[:,dim,:] = data_sim_Nperp
        elif var == 'Vx':        data_sim[:,dim,:] = data_sim_VxVyVz[:,0,:]
        elif var == 'Vy':        data_sim[:,dim,:] = data_sim_VxVyVz[:,1,:]
        elif var == 'Vz':        data_sim[:,dim,:] = data_sim_VxVyVz[:,2,:]
        elif var == 'Psi':       data_sim[:,dim,:] = data_sim_Psi
        elif var == 'rho':       data_sim[:,dim,:] = np.sqrt(data_sim_Psi)
        elif var == 'Theta':     data_sim[:,dim,:] = data_sim_Theta
        elif var == 'R':         data_sim[:,dim,:] = np.sqrt(data_sim_XYZ[:,0,:]**2 + data_sim_XYZ[:,1,:]**2)
        else:
            print('IN THE WhatToResolve LIST IN THE INPUT FILE IS A NON-SUPPORTED ELEMENT. FIX THAT.\n')
            raise ValueError(f'Unsupported WhatToResolve element: {var}')

    data_sim_Wfct      = data_sim_Wfct * NormFactor
    data_sim_scattered = data_sim_nmbrscattevents.copy()

    # Build the weight arrays dict for _select_weight calls
    _weight_data = {}
    if all_vars & {'Nx', 'Ny', 'Nz'}:  _weight_data['data_sim_NxNyNz']   = data_sim_NxNyNz
    if all_vars & {'Vx', 'Vy', 'Vz'}:  _weight_data['data_sim_VxVyVz']   = data_sim_VxVyVz
    if 'Nparallel' in all_vars:         _weight_data['data_sim_Nparallel'] = data_sim_Nparallel
    if 'phiN' in all_vars:              _weight_data['data_sim_phiN']      = data_sim_phiN
    if 'Nperp' in all_vars:             _weight_data['data_sim_Nperp']     = data_sim_Nperp

    # Dispatch closure — hides uniform/nonuniform choice for the rest of this file
    if uniform_bins:
        def _run_binning(weight, result, scatter_param, absorb):
            return binning(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                           weight, result,
                           bin_min[0], bin_max[0], bin_min[1], bin_max[1],
                           bin_min[2], bin_max[2], bin_min[3], bin_max[3],
                           nmbr[0], nmbr[1], nmbr[2], nmbr[3],
                           data_sim_time, scatter_param, data_sim_scattered, absorb)
    else:
        def _run_binning(weight, result, scatter_param, absorb):
            return binning_nonuni(data_sim, data_sim_Wfct, data_sim_CorrectionFactor,
                                  weight, result,
                                  bins[0], bins[1], bins[2], bins[3],
                                  data_sim_time, scatter_param, data_sim_scattered, absorb)

    # BINNING UNSCATTERED RAYS
    print("BINNING UNSCATTERED RAYS.\n")
    if idata.storeWfct:
        print("BINNING WITH WEIGHT 1.\n")
        sys.stdout.flush()
        tmpnmbrrays = _run_binning(np.empty([0,0]), partial['WfctUnscattered'], 1, 0)

    if idata.storeVelocityField:
        for k, comp in enumerate(VelocityComponentsToStore):
            print("BINNING WITH WEIGHT %s.\n" % comp)
            sys.stdout.flush()
            tmpnmbrrays = _run_binning(_select_weight(comp, **_weight_data),
                                       partial['VelocityFieldUnscattered'][:,:,:,:,k,:], 1, 0)

    if idata.storeAbsorption:
        print("BINNING WITH ABSORPTION WITH WEIGHT 1.\n")
        sys.stdout.flush()
        tmpnmbrrays = _run_binning(np.empty([0,0]), partial['AbsorptionUnscattered'], 1, 1)

    print('%i unscattered rays have been binned, %i available in total.\n' % (tmpnmbrrays, nmbrRaysToUse))
    partial['nmbrRaysUnscattered'] = tmpnmbrrays

    # BINNING SCATTERED RAYS
    if idata.computeAmplitude or idata.computeScatteringEffect:
        print("BINNING SCATTERED RAYS.\n")

        if idata.storeWfct:
            print("BINNING WITH WEIGHT 1.\n")
            sys.stdout.flush()
            tmpnmbrrays = _run_binning(np.empty([0,0]), partial['WfctScattered'], 2, 0)

        if idata.storeVelocityField:
            for k, comp in enumerate(VelocityComponentsToStore):
                print("BINNING WITH WEIGHT %s.\n" % comp)
                sys.stdout.flush()
                tmpnmbrrays = _run_binning(_select_weight(comp, **_weight_data),
                                           partial['VelocityFieldScattered'][:,:,:,:,k,:], 2, 0)

        if idata.storeAbsorption:
            print("BINNING WITH ABSORPTION WITH WEIGHT 1.\n")
            sys.stdout.flush()
            tmpnmbrrays = _run_binning(np.empty([0,0]), partial['AbsorptionScattered'], 2, 1)

        print('%i scattered rays have been binned, %i available in total.\n' % (tmpnmbrrays, nmbrRaysToUse))
        partial['nmbrRaysScattered'] = tmpnmbrrays

    partial['nmbrRaysToUse'] = nmbrRaysToUse
    partial['file_params']   = file_params
    return partial


def _accumulate(accum, partial):
    """Add a partial result (from one file) into the accumulator in-place.

    Checks antenna parameter consistency across files: the first call sets
    accum['file_params']; subsequent calls raise ValueError on mismatch.
    """
    accum['nmbrRaysUnscattered'] += partial['nmbrRaysUnscattered']
    accum['nmbrRaysScattered']   += partial['nmbrRaysScattered']

    for key in ('WfctUnscattered', 'WfctScattered',
                'AbsorptionUnscattered', 'AbsorptionScattered',
                'VelocityFieldUnscattered', 'VelocityFieldScattered'):
        if key in partial:
            accum[key] += partial[key]

    if accum['file_params'] is None:
        accum['file_params'] = partial['file_params']
    else:
        ref = accum['file_params']
        if any(partial['file_params'][k] != ref[k] for k in ref):
            print("ATTENTION: THERE IS A PROBLEM HERE, NOT ALL INPUT FILES AGREE IN ALL PARAMETERS.\n")
            sys.stdout.flush()
            raise ValueError('Inconsistent antenna parameters across input files')


def _postprocess_and_write(idata, accum, setup, outputfilename):
    """Normalise accumulated arrays, reduce dimensions, and write HDF5 output."""

    nmbr                      = setup['nmbr']
    uniform_bins              = setup['uniform_bins']
    bin_min                   = setup['bin_min']
    bin_max                   = setup['bin_max']
    bins                      = setup['bins']
    VelocityComponentsToStore = setup['VelocityComponentsToStore']

    nmbrRaysUnscattered = accum['nmbrRaysUnscattered']
    nmbrRaysScattered   = accum['nmbrRaysScattered']
    nmbrRays            = nmbrRaysUnscattered + nmbrRaysScattered
    file_params         = accum['file_params']

    print("COMPOSING AND NORMALISING THE FINAL QUANTITIES.\n")
    sys.stdout.flush()

    # IF CHOSEN IN INPUT FILE COMPUTE THE EFFECT OF SCATTERING
    ############################################################################
    if idata.computeScatteringEffect:

        if idata.storeWfct:
            WfctScatteringEffect = np.zeros(np.append(nmbr, 2))
            WfctScatteringEffect[:,:,:,:,0] = (accum['WfctScattered'][:,:,:,:,0] \
                                               - nmbrRaysScattered/nmbrRaysUnscattered \
                                               * accum['WfctUnscattered'][:,:,:,:,0]) / nmbrRays
            WfctScatteringEffect[:,:,:,:,1] = (accum['WfctScattered'][:,:,:,:,1] \
                                               + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                               * accum['WfctUnscattered'][:,:,:,:,1]) / nmbrRays**2
            WfctScatteringEffect[:,:,:,:,1] = np.sqrt(WfctScatteringEffect[:,:,:,:,1])

        if idata.storeVelocityField:
            VelocityFieldScatteringEffect = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
            VelocityFieldScatteringEffect[:,:,:,:,:,0] = (accum['VelocityFieldScattered'][:,:,:,:,:,0] \
                                                          - nmbrRaysScattered/nmbrRaysUnscattered \
                                                          * accum['VelocityFieldUnscattered'][:,:,:,:,:,0]) / nmbrRays
            VelocityFieldScatteringEffect[:,:,:,:,:,1] = (accum['VelocityFieldScattered'][:,:,:,:,:,1] \
                                                          + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                                          * accum['VelocityFieldUnscattered'][:,:,:,:,:,1]) / nmbrRays**2
            VelocityFieldScatteringEffect[:,:,:,:,:,1] = np.sqrt(VelocityFieldScatteringEffect[:,:,:,:,:,1])

        if idata.storeAbsorption:
            AbsorptionScatteringEffect = np.zeros(np.append(nmbr, 2))
            AbsorptionScatteringEffect[:,:,:,:,0] = (accum['AbsorptionScattered'][:,:,:,:,0] \
                                                     - nmbrRaysScattered/nmbrRaysUnscattered \
                                                     * accum['AbsorptionUnscattered'][:,:,:,:,0]) / nmbrRays
            AbsorptionScatteringEffect[:,:,:,:,1] = (accum['AbsorptionScattered'][:,:,:,:,1] \
                                                     + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                                     * accum['AbsorptionUnscattered'][:,:,:,:,1]) / nmbrRays**2
            AbsorptionScatteringEffect[:,:,:,:,1] = np.sqrt(AbsorptionScatteringEffect[:,:,:,:,1])

    # IF CHOSEN IN INPUT FILE ESTIMATE THE CONTRIBUTION OF SCATTERED RAYS
    ############################################################################
    if idata.computeScatteredContribution:
        if idata.storeWfct:
            WfctScatteredContribution = np.zeros(np.append(nmbr, 2))
            WfctScatteredContribution[:,:,:,:,0] = accum['WfctScattered'][:,:,:,:,0] / nmbrRays
            WfctScatteredContribution[:,:,:,:,1] = np.sqrt(accum['WfctScattered'][:,:,:,:,1]) / nmbrRays

        if idata.storeVelocityField:
            VelocityFieldScatteredContribution = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
            VelocityFieldScatteredContribution[:,:,:,:,:,0] = accum['VelocityFieldScattered'][:,:,:,:,:,0] / nmbrRays
            VelocityFieldScatteredContribution[:,:,:,:,:,1] = np.sqrt(accum['VelocityFieldScattered'][:,:,:,:,:,1]) / nmbrRays

        if idata.storeAbsorption:
            AbsorptionScatteredContribution = np.zeros(np.append(nmbr, 2))
            AbsorptionScatteredContribution[:,:,:,:,0] = accum['AbsorptionScattered'][:,:,:,:,0] / nmbrRays
            AbsorptionScatteredContribution[:,:,:,:,1] = np.sqrt(accum['AbsorptionScattered'][:,:,:,:,1]) / nmbrRays

    # IF CHOSEN IN INPUT FILE COMPUTE THE TOTAL AMPLITUDE
    ############################################################################
    if idata.computeAmplitude:
        if idata.storeWfct:
            Wfct = np.zeros(np.append(nmbr, 2))
            Wfct[:,:,:,:,0] = (accum['WfctScattered'][:,:,:,:,0] + accum['WfctUnscattered'][:,:,:,:,0]) / nmbrRays
            Wfct[:,:,:,:,1] = np.sqrt(accum['WfctScattered'][:,:,:,:,1] + accum['WfctUnscattered'][:,:,:,:,1]) / nmbrRays

        if idata.storeVelocityField:
            VelocityField = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
            VelocityField[:,:,:,:,:,0] = (accum['VelocityFieldScattered'][:,:,:,:,:,0] + accum['VelocityFieldUnscattered'][:,:,:,:,:,0]) / nmbrRays
            VelocityField[:,:,:,:,:,1] = np.sqrt(accum['VelocityFieldScattered'][:,:,:,:,:,1] \
                                                  + accum['VelocityFieldUnscattered'][:,:,:,:,:,1]) / nmbrRays

        if idata.storeAbsorption:
            Absorption = np.zeros(np.append(nmbr, 2))
            Absorption[:,:,:,:,0] = (accum['AbsorptionScattered'][:,:,:,:,0] + accum['AbsorptionUnscattered'][:,:,:,:,0]) / nmbrRays
            Absorption[:,:,:,:,1] = np.sqrt(accum['AbsorptionScattered'][:,:,:,:,1] + accum['AbsorptionUnscattered'][:,:,:,:,1]) / nmbrRays

    # IF CHOSEN IN INPUT FILE NORMALISE THE UNSCATTERED AMPLITUDE
    ############################################################################
    if idata.computeAmplitudeUnscattered:
        if idata.storeWfct:
            accum['WfctUnscattered'][:,:,:,:,0] = accum['WfctUnscattered'][:,:,:,:,0] / nmbrRaysUnscattered
            accum['WfctUnscattered'][:,:,:,:,1] = np.sqrt(accum['WfctUnscattered'][:,:,:,:,1]) / nmbrRaysUnscattered

        if idata.storeVelocityField:
            accum['VelocityFieldUnscattered'][:,:,:,:,:,0] = VelocityField[:,:,:,:,:,0] / nmbrRaysUnscattered
            accum['VelocityFieldUnscattered'][:,:,:,:,:,1] = np.sqrt(VelocityField[:,:,:,:,:,1]) / nmbrRaysUnscattered

        if idata.storeAbsorption:
            accum['AbsorptionUnscattered'][:,:,:,:,0] = accum['AbsorptionUnscattered'][:,:,:,:,0] / nmbrRaysUnscattered
            accum['AbsorptionUnscattered'][:,:,:,:,1] = np.sqrt(accum['AbsorptionUnscattered'][:,:,:,:,1]) / nmbrRaysUnscattered

    # REDUCE THE MATRICES TO THE MEANINGFUL DIMENSIONS
    ############################################################################
    firstaxistosum = len(idata.WhatToResolve)
    for i in range(firstaxistosum, 4):
        if idata.computeAmplitude:
            if idata.storeWfct:           Wfct            = np.sum(Wfct,            axis=firstaxistosum)
            if idata.storeVelocityField:  VelocityField   = np.sum(VelocityField,   axis=firstaxistosum)
            if idata.storeAbsorption:     Absorption      = np.sum(Absorption,      axis=firstaxistosum)
        if idata.computeAmplitudeUnscattered:
            if idata.storeWfct:           accum['WfctUnscattered']          = np.sum(accum['WfctUnscattered'],          axis=firstaxistosum)
            if idata.storeVelocityField:  accum['VelocityFieldUnscattered'] = np.sum(accum['VelocityFieldUnscattered'], axis=firstaxistosum)
            if idata.storeAbsorption:     accum['AbsorptionUnscattered']    = np.sum(accum['AbsorptionUnscattered'],    axis=firstaxistosum)
        if idata.computeScatteringEffect:
            if idata.storeWfct:           WfctScatteringEffect          = np.sum(WfctScatteringEffect,          axis=firstaxistosum)
            if idata.storeVelocityField:  VelocityFieldScatteringEffect = np.sum(VelocityFieldScatteringEffect, axis=firstaxistosum)
            if idata.storeAbsorption:     AbsorptionScatteringEffect    = np.sum(AbsorptionScatteringEffect,    axis=firstaxistosum)
        if idata.computeScatteredContribution:
            if idata.storeWfct:           WfctScatteredContribution          = np.sum(WfctScatteredContribution,          axis=firstaxistosum)
            if idata.storeVelocityField:  VelocityFieldScatteredContribution = np.sum(VelocityFieldScatteredContribution, axis=firstaxistosum)
            if idata.storeAbsorption:     AbsorptionScatteredContribution    = np.sum(AbsorptionScatteredContribution,    axis=firstaxistosum)

    # WRITE THE RESULTS TO FILE
    ############################################################################
    outputfilename = idata.outputdirectory + outputfilename + '.hdf5'
    print('write results to file %s \n' % outputfilename)
    sys.stdout.flush()

    with h5py.File(outputfilename, 'w') as fid:
        if idata.computeAmplitude:
            if idata.storeWfct:          fid.create_dataset("BinnedTraces",  data=Wfct)
            if idata.storeVelocityField: fid.create_dataset("VelocityField", data=VelocityField)
            if idata.storeAbsorption:    fid.create_dataset("Absorption",    data=Absorption)
        if idata.computeAmplitudeUnscattered:
            if idata.storeWfct:          fid.create_dataset("BinnedTracesUnscattered",  data=accum['WfctUnscattered'])
            if idata.storeVelocityField: fid.create_dataset("VelocityFieldUnscattered", data=accum['VelocityFieldUnscattered'])
            if idata.storeAbsorption:    fid.create_dataset("AbsorptionUnscattered",    data=accum['AbsorptionUnscattered'])
        if idata.computeScatteringEffect:
            if idata.storeWfct:          fid.create_dataset("BinnedTracesScatteringEffect",  data=WfctScatteringEffect)
            if idata.storeVelocityField: fid.create_dataset("VelocityFieldScatteringEffect", data=VelocityFieldScatteringEffect)
            if idata.storeAbsorption:    fid.create_dataset("AbsorptionScatteringEffect",    data=AbsorptionScatteringEffect)
        if idata.computeScatteredContribution:
            if idata.storeWfct:          fid.create_dataset("BinnedTracesScatteredContribution",   data=WfctScatteredContribution)
            if idata.storeVelocityField: fid.create_dataset("VelocityFieldScattererContribution",  data=VelocityFieldScatteredContribution)
            if idata.storeAbsorption:    fid.create_dataset("AbsorptionScatteredContribution",     data=AbsorptionScatteredContribution)

        if idata.storeVelocityField:
            fid.create_dataset("VelocityFieldStored", data=','.join(VelocityComponentsToStore) + ',')

        fid.create_dataset("WhatToResolve", data=','.join(idata.WhatToResolve) + ',')

        if uniform_bins:
            for i, var in enumerate(idata.WhatToResolve):
                fid.create_dataset(f"{var}min",  data=bin_min[i])
                fid.create_dataset(f"{var}max",  data=bin_max[i])
                fid.create_dataset(f"nmbr{var}", data=idata.nmbr[i])
        else:
            for i, var in enumerate(idata.WhatToResolve):
                fid.create_dataset(f"{var}bins", data=bins[i])

        fid.create_dataset("nmbrRays",           data=nmbrRays)
        fid.create_dataset("nmbrRaysUnscattered", data=nmbrRaysUnscattered)
        fid.create_dataset("nmbrRaysScattered",   data=nmbrRaysScattered)
        fid.create_dataset("uniform_bins",        data=uniform_bins)
        fid.create_dataset("Mode",               data=file_params["Mode"])
        fid.create_dataset("FreqGHz",            data=file_params["FreqGHz"])
        fid.create_dataset("antennapolangle",    data=file_params["antennapolangle"])
        fid.create_dataset("antennatorangle",    data=file_params["antennatorangle"])
        fid.create_dataset("rayStartX",          data=file_params["rayStartX"])
        fid.create_dataset("rayStartY",          data=file_params["rayStartY"])
        fid.create_dataset("rayStartZ",          data=file_params["rayStartZ"])
        fid.create_dataset("beamwidth1",         data=file_params["beamwidth1"])
        fid.create_dataset("beamwidth2",         data=file_params["beamwidth2"])
        fid.create_dataset("curvatureradius1",   data=file_params["curvatureradius1"])
        fid.create_dataset("curvatureradius2",   data=file_params["curvatureradius2"])
        fid.create_dataset("centraleta1",        data=file_params["centraleta1"])
        fid.create_dataset("centraleta2",        data=file_params["centraleta2"])


############################################################################
# SERIAL ENTRY POINT
############################################################################
def binning_pyinterface(idata):
    """Serial binning entry point. Processes all files sequentially."""

    outputfilename = getattr(idata, 'outputfilename', idata.inputfilename + '_binned')

    if len(idata.WhatToResolve) > 4:
        print('THE MAXIMUM NUMBER OF DIMENSIONS 4 IS EXCEEDED.\n')
        raise ValueError('Too many dimensions in WhatToResolve')

    setup  = _setup_bins(idata)
    accum  = _allocate_accum(idata, setup)
    indices = _find_indices(idata)
    print("NUMBER OF FILES TO BE PROCESSED: %i\n" % len(indices))

    for file_idx in indices:
        partial = _bin_one_file(file_idx, idata, setup)
        _accumulate(accum, partial)

    _postprocess_and_write(idata, accum, setup, outputfilename)

# END OF FILE
