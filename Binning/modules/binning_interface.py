"""THIS FILE CAN BE USED TO CALL THE CYTHON BINNING CODE DIRECTLY FROM TERMINAL.
ALL RELEVANT PARAMETERS CAN BE CHOSEN BELOW. THE DATA IS LOADED AND SAVED FROM
AND TO THE FILES CHOSEN AMONG THE PARAMETERS.
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
    if component == "Nx":       return data_sim_NxNyNz[:,0,:]
    elif component == "Ny":     return data_sim_NxNyNz[:,1,:]
    elif component == "Nz":     return data_sim_NxNyNz[:,2,:]
    elif component == "Vx":     return data_sim_VxVyVz[:,0,:]
    elif component == "Vy":     return data_sim_VxVyVz[:,1,:]
    elif component == "Vz":     return data_sim_VxVyVz[:,2,:]
    elif component == "Nparallel": return data_sim_Nparallel
    elif component == "phiN":   return data_sim_phiN
    elif component == "Nperp":  return data_sim_Nperp
    raise ValueError(f"Unknown velocity component: {component}")


############################################################################
# BELOW: A FUNCTION WHICH DOES THE BINNING IS PROVIDED.
# IT IS CALLED WHEN THIS FILE IS EXECUTED AND ALSO CAN BE CALLED
# FROM OTHER PIECES OF PYTHON CODE. AS PARAMETER, IT TAKES AN idata
# INSTANCE WITH ALL RELEVANT PARAMETERS.
############################################################################
def binning_pyinterface(idata):

    # MODIFY INPUT PARAMETERS IF NEEDED
    ############################################################################

    # see if outputfilename is defined.
    # if not, just attach _binned to inputfilename
    outputfilename = getattr(idata, 'outputfilename', idata.inputfilename + '_binned')

    VelocityComponentsToStore = idata.VelocityComponentsToStore if idata.storeVelocityField else []

    # ALLOCATE MEMORY WHERE THE RESULTS CAN BE WRITTEN AND CALL THE
    # CYTHON BINNING FUNCTION
    ############################################################################

    # see how many directions are needed in the input file.
    if len(idata.WhatToResolve) > 4:
        print('THE MAXIMUM NUMBER OF DIMENSIONS 4 IS EXCEEDED.\n')
        raise ValueError('Too many dimensions in WhatToResolve')

    uniform_bins = getattr(idata, 'uniform_bins', True)

    # All variables needed for either WhatToResolve or VelocityComponentsToStore
    all_vars = set(idata.WhatToResolve) | set(VelocityComponentsToStore)

    if uniform_bins:
        # put one bin in directions not in use
        # and the boundaries around zero (because the data_sim-values will be 0 for those dimensions)
        nmbr = np.empty([4], dtype=int)
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
        nmbr = np.empty([4], dtype=int)
        for i in range(4):
            if i < len(idata.bins):
                if len(idata.bins[i]) == 0:
                    # We want to use the simple description of uniform bins anyway, nothing was provided for this dimension
                    nmbr[i] = idata.nmbr[i]
                    bins[i] = np.linspace(idata.min[i], idata.max[i], idata.nmbr[i]+1)
                else:
                    if isinstance(idata.bins[i][0], str):
                        # Then we take the bins from a file, and the location and name are provided
                        grids = loadmat(idata.outputdirectory + idata.bins[i][0])['WKBacca_grids']
                        bins[i] = grids[idata.bins[i][1]][0,0][0]
                    else:
                        bins[i] = idata.bins[i]
                    nmbr[i] = len(bins[i]) - 1
            else:
                bins[i] = np.linspace(-1., 1., 2)
                nmbr[i] = 1

    # Allocate output arrays
    if idata.storeWfct:
        WfctUnscattered = np.zeros(np.append(nmbr, 2))
    if idata.storeVelocityField:
        VelocityFieldUnscattered = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
    if idata.storeAbsorption:
        AbsorptionUnscattered = np.zeros(np.append(nmbr, 2))

    if idata.computeAmplitude or idata.computeScatteringEffect:
        if idata.storeWfct:
            WfctScattered = np.zeros(np.append(nmbr, 2))
        if idata.storeVelocityField:
            VelocityFieldScattered = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
        if idata.storeAbsorption:
            AbsorptionScattered = np.zeros(np.append(nmbr, 2))

    nmbrRaysScattered = 0
    nmbrRaysUnscattered = 0


    # READ BEAM PARAMETERS FROM THE INPUT FILE
    ############################################################################
    if idata.nmbrFiles == 'all':
        # Pattern: <input_dir>/<input_filename>_file#.hdf5
        pattern = os.path.join(idata.inputdirectory, f"{idata.inputfilename}_file*.hdf5")
        existing_files = glob.glob(pattern)
        pattern_re = re.compile(f"{re.escape(idata.inputfilename)}_file(\\d+)\\.hdf5")
        indices = sorted(
            int(m.group(1))
            for f in existing_files
            for m in [pattern_re.search(os.path.basename(f))]
            if m
        )
    else:
        indices = idata.nmbrFiles
    print("NUMBER OF FILES TO BE PROCESSED: %i\n" % len(indices))

    for file_idx in indices:
        filename = idata.inputdirectory + idata.inputfilename + '_file%i.hdf5' % file_idx
        print("loading file %s ...\n" % filename)
        sys.stdout.flush()

        with h5py.File(filename, 'r') as fid:
            data_sim_CorrectionFactor = fid.get('TracesCorrectionFactor')[()] if idata.correctionfactor else np.empty([0,0])

            data_sim_XYZ              = fid.get('TracesXYZ')[()]
            data_sim_Wfct             = fid.get('TracesWfct')[()]
            data_sim_nmbrscattevents  = fid.get('TracesNumberScattEvents')[()]

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

        if file_idx == indices[0]:
            sigma            = file_params["Mode"]
            freq             = file_params["FreqGHz"]
            antennapolangle  = file_params["antennapolangle"]
            antennatorangle  = file_params["antennatorangle"]
            rayStartX        = file_params["rayStartX"]
            rayStartY        = file_params["rayStartY"]
            rayStartZ        = file_params["rayStartZ"]
            beamwidth1       = file_params["beamwidth1"]
            beamwidth2       = file_params["beamwidth2"]
            curvatureradius1 = file_params["curvatureradius1"]
            curvatureradius2 = file_params["curvatureradius2"]
            centraleta1      = file_params["centraleta1"]
            centraleta2      = file_params["centraleta2"]

            # see if the normalisation factor for energy flow is needed or not
            InputPower = getattr(idata, 'InputPower', None)
            if InputPower is not None:
                NormFactor = InputPower / compute_norm_factor(freq,
                                                              beamwidth1, beamwidth2,
                                                              curvatureradius1, curvatureradius2,
                                                              centraleta1, centraleta2)
                print("Input power is %.3fMW\n" % InputPower)
            else:
                NormFactor = 1.
                print("Normalisation such that central electric field on antenna is 1.\n")
        else:
            ref = dict(Mode=sigma, FreqGHz=freq,
                       antennapolangle=antennapolangle, antennatorangle=antennatorangle,
                       rayStartX=rayStartX, rayStartY=rayStartY, rayStartZ=rayStartZ,
                       beamwidth1=beamwidth1, beamwidth2=beamwidth2,
                       curvatureradius1=curvatureradius1, curvatureradius2=curvatureradius2,
                       centraleta1=centraleta1, centraleta2=centraleta2)
            if any(file_params[k] != ref[k] for k in ref):
                print("ATTENTION: THERE IS A PROBLEM HERE, NOT ALL INPUT FILES AGREE IN ALL PARAMETERS.\n")
                sys.stdout.flush()
                raise ValueError('Inconsistent antenna parameters across input files')

        # Compose data_sim from WhatToResolve
        data_sim = np.zeros([nmbrRaysToUse, 4, nmbrPointsPerRay])
        for dim, var in enumerate(idata.WhatToResolve):
            if var == 'X':          data_sim[:,dim,:] = data_sim_XYZ[:,0,:]
            elif var == 'Y':        data_sim[:,dim,:] = data_sim_XYZ[:,1,:]
            elif var == 'Z':        data_sim[:,dim,:] = data_sim_XYZ[:,2,:]
            elif var == 'Nx':       data_sim[:,dim,:] = data_sim_NxNyNz[:,0,:]
            elif var == 'Ny':       data_sim[:,dim,:] = data_sim_NxNyNz[:,1,:]
            elif var == 'Nz':       data_sim[:,dim,:] = data_sim_NxNyNz[:,2,:]
            elif var == 'Nparallel':data_sim[:,dim,:] = data_sim_Nparallel
            elif var == 'phiN':     data_sim[:,dim,:] = data_sim_phiN
            elif var == 'Nperp':    data_sim[:,dim,:] = data_sim_Nperp
            elif var == 'Vx':       data_sim[:,dim,:] = data_sim_VxVyVz[:,0,:]
            elif var == 'Vy':       data_sim[:,dim,:] = data_sim_VxVyVz[:,1,:]
            elif var == 'Vz':       data_sim[:,dim,:] = data_sim_VxVyVz[:,2,:]
            elif var == 'Psi':      data_sim[:,dim,:] = data_sim_Psi
            elif var == 'rho':      data_sim[:,dim,:] = np.sqrt(data_sim_Psi)
            elif var == 'Theta':    data_sim[:,dim,:] = data_sim_Theta
            elif var == 'R':        data_sim[:,dim,:] = np.sqrt(data_sim_XYZ[:,0,:]**2 + data_sim_XYZ[:,1,:]**2)
            else:
                print('IN THE WhatToResolve LIST IN THE INPUT FILE IS A NON-SUPPORTED ELEMENT. FIX THAT.\n')
                raise ValueError(f'Unsupported WhatToResolve element: {var}')

        data_sim_Wfct     = data_sim_Wfct * NormFactor
        data_sim_scattered = data_sim_nmbrscattevents.copy()

        # Build the weight arrays dict for _select_weight calls this iteration
        _weight_data = {}
        if all_vars & {'Nx', 'Ny', 'Nz'}:       _weight_data['data_sim_NxNyNz']   = data_sim_NxNyNz
        if all_vars & {'Vx', 'Vy', 'Vz'}:       _weight_data['data_sim_VxVyVz']   = data_sim_VxVyVz
        if 'Nparallel' in all_vars:              _weight_data['data_sim_Nparallel'] = data_sim_Nparallel
        if 'phiN' in all_vars:                   _weight_data['data_sim_phiN']      = data_sim_phiN
        if 'Nperp' in all_vars:                  _weight_data['data_sim_Nperp']     = data_sim_Nperp

        # Define a single dispatch function that hides the uniform/nonuniform choice
        # for the rest of this file's processing.
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
        ########################################################################
        print("BINNING UNSCATTERED RAYS.\n")
        if idata.storeWfct:
            print("BINNING WITH WEIGHT 1.\n")
            sys.stdout.flush()
            tmpnmbrrays = _run_binning(np.empty([0,0]), WfctUnscattered, 1, 0)

        if idata.storeVelocityField:
            for k, comp in enumerate(VelocityComponentsToStore):
                print("BINNING WITH WEIGHT %s.\n" % comp)
                sys.stdout.flush()
                tmpnmbrrays = _run_binning(_select_weight(comp, **_weight_data),
                                           VelocityFieldUnscattered[:,:,:,:,k,:], 1, 0)

        if idata.storeAbsorption:
            print("BINNING WITH ABSORPTION WITH WEIGHT 1.\n")
            sys.stdout.flush()
            tmpnmbrrays = _run_binning(np.empty([0,0]), AbsorptionUnscattered, 1, 1)

        print('%i unscattered rays have been binned, %i available in total.\n' % (tmpnmbrrays, nmbrRaysToUse))
        nmbrRaysUnscattered += tmpnmbrrays

        # BINNING SCATTERED RAYS
        ########################################################################
        if idata.computeAmplitude or idata.computeScatteringEffect:
            print("BINNING SCATTERED RAYS.\n")

            if idata.storeWfct:
                print("BINNING WITH WEIGHT 1.\n")
                sys.stdout.flush()
                tmpnmbrrays = _run_binning(np.empty([0,0]), WfctScattered, 2, 0)

            if idata.storeVelocityField:
                for k, comp in enumerate(VelocityComponentsToStore):
                    print("BINNING WITH WEIGHT %s.\n" % comp)
                    sys.stdout.flush()
                    tmpnmbrrays = _run_binning(_select_weight(comp, **_weight_data),
                                               VelocityFieldScattered[:,:,:,:,k,:], 2, 0)

            if idata.storeAbsorption:
                print("BINNING WITH ABSORPTION WITH WEIGHT 1.\n")
                sys.stdout.flush()
                tmpnmbrrays = _run_binning(np.empty([0,0]), AbsorptionScattered, 2, 1)

            print('%i scattered rays have been binned, %i available in total.\n' % (tmpnmbrrays, nmbrRaysToUse))
            nmbrRaysScattered += tmpnmbrrays


    print("COMPOSING AND NORMALISING THE FINAL QUANTITIES.\n")
    sys.stdout.flush()
    nmbrRays = nmbrRaysScattered + nmbrRaysUnscattered

    # IF CHOSEN IN INPUT FILE COMPUTE THE EFFECT OF SCATTERING
    ############################################################################
    if idata.computeScatteringEffect:

        if idata.storeWfct:
            WfctScatteringEffect = np.zeros(np.append(nmbr, 2))
            WfctScatteringEffect[:,:,:,:,0] = (WfctScattered[:,:,:,:,0] \
                                               - nmbrRaysScattered/nmbrRaysUnscattered \
                                               * WfctUnscattered[:,:,:,:,0]) / nmbrRays
            WfctScatteringEffect[:,:,:,:,1] = (WfctScattered[:,:,:,:,1] \
                                               + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                               * WfctUnscattered[:,:,:,:,1]) / nmbrRays**2
            WfctScatteringEffect[:,:,:,:,1] = np.sqrt(WfctScatteringEffect[:,:,:,:,1])

        if idata.storeVelocityField:
            VelocityFieldScatteringEffect = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
            VelocityFieldScatteringEffect[:,:,:,:,:,0] = (VelocityFieldScattered[:,:,:,:,:,0] \
                                                          - nmbrRaysScattered/nmbrRaysUnscattered \
                                                          * VelocityFieldUnscattered[:,:,:,:,:,0]) / nmbrRays
            VelocityFieldScatteringEffect[:,:,:,:,:,1] = (VelocityFieldScattered[:,:,:,:,:,1] \
                                                          + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                                          * VelocityFieldUnscattered[:,:,:,:,:,1]) / nmbrRays**2
            VelocityFieldScatteringEffect[:,:,:,:,:,1] = np.sqrt(VelocityFieldScatteringEffect[:,:,:,:,:,1])

        if idata.storeAbsorption:
            AbsorptionScatteringEffect = np.zeros(np.append(nmbr, 2))
            AbsorptionScatteringEffect[:,:,:,:,0] = (AbsorptionScattered[:,:,:,:,0] \
                                                     - nmbrRaysScattered/nmbrRaysUnscattered \
                                                     * AbsorptionUnscattered[:,:,:,:,0]) / nmbrRays
            AbsorptionScatteringEffect[:,:,:,:,1] = (AbsorptionScattered[:,:,:,:,1] \
                                                     + nmbrRaysScattered**2/nmbrRaysUnscattered**2 \
                                                     * AbsorptionUnscattered[:,:,:,:,1]) / nmbrRays**2
            AbsorptionScatteringEffect[:,:,:,:,1] = np.sqrt(AbsorptionScatteringEffect[:,:,:,:,1])


    # IF CHOSEN IN INPUT FILE ESTIMATE THE CONTRIBUTION OF SCATTERED RAYS
    ############################################################################
    if idata.computeScatteredContribution:
        if idata.storeWfct:
            WfctScatteredContribution = np.zeros(np.append(nmbr, 2))
            WfctScatteredContribution[:,:,:,:,0] = WfctScattered[:,:,:,:,0] / nmbrRays
            WfctScatteredContribution[:,:,:,:,1] = np.sqrt(WfctScattered[:,:,:,:,1]) / nmbrRays

        if idata.storeVelocityField:
            VelocityFieldScatteredContribution = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
            VelocityFieldScatteredContribution[:,:,:,:,:,0] = VelocityFieldScattered[:,:,:,:,:,0] / nmbrRays
            VelocityFieldScatteredContribution[:,:,:,:,:,1] = np.sqrt(VelocityFieldScattered[:,:,:,:,:,1]) / nmbrRays

        if idata.storeAbsorption:
            AbsorptionScatteredContribution = np.zeros(np.append(nmbr, 2))
            AbsorptionScatteredContribution[:,:,:,:,0] = AbsorptionScattered[:,:,:,:,0] / nmbrRays
            AbsorptionScatteredContribution[:,:,:,:,1] = np.sqrt(AbsorptionScattered[:,:,:,:,1]) / nmbrRays


    # IF CHOSEN IN INPUT FILE COMPUTE THE TOTAL AMPLITUDE
    ############################################################################
    if idata.computeAmplitude:
        if idata.storeWfct:
            Wfct = np.zeros(np.append(nmbr, 2))
            Wfct[:,:,:,:,0] = (WfctScattered[:,:,:,:,0] + WfctUnscattered[:,:,:,:,0]) / nmbrRays
            Wfct[:,:,:,:,1] = np.sqrt(WfctScattered[:,:,:,:,1] + WfctUnscattered[:,:,:,:,1]) / nmbrRays

        if idata.storeVelocityField:
            VelocityField = np.zeros(np.append(nmbr, [len(VelocityComponentsToStore), 2]))
            VelocityField[:,:,:,:,:,0] = (VelocityFieldScattered[:,:,:,:,:,0] + VelocityFieldUnscattered[:,:,:,:,:,0]) / nmbrRays
            VelocityField[:,:,:,:,:,1] = np.sqrt(VelocityFieldScattered[:,:,:,:,:,1] \
                                                  + VelocityFieldUnscattered[:,:,:,:,:,1]) / nmbrRays

        if idata.storeAbsorption:
            Absorption = np.zeros(np.append(nmbr, 2))
            Absorption[:,:,:,:,0] = (AbsorptionScattered[:,:,:,:,0] + AbsorptionUnscattered[:,:,:,:,0]) / nmbrRays
            Absorption[:,:,:,:,1] = np.sqrt(AbsorptionScattered[:,:,:,:,1] + AbsorptionUnscattered[:,:,:,:,1]) / nmbrRays


    # IF CHOSEN IN INPUT FILE NORMALISE THE UNSCATTERED AMPLITUDE
    ############################################################################
    if idata.computeAmplitudeUnscattered:
        if idata.storeWfct:
            WfctUnscattered[:,:,:,:,0] = WfctUnscattered[:,:,:,:,0] / nmbrRaysUnscattered
            WfctUnscattered[:,:,:,:,1] = np.sqrt(WfctUnscattered[:,:,:,:,1]) / nmbrRaysUnscattered

        if idata.storeVelocityField:
            VelocityFieldUnscattered[:,:,:,:,:,0] = VelocityField[:,:,:,:,:,0] / nmbrRaysUnscattered
            VelocityFieldUnscattered[:,:,:,:,:,1] = np.sqrt(VelocityField[:,:,:,:,:,1]) / nmbrRaysUnscattered

        if idata.storeAbsorption:
            AbsorptionUnscattered[:,:,:,:,0] = AbsorptionUnscattered[:,:,:,:,0] / nmbrRaysUnscattered
            AbsorptionUnscattered[:,:,:,:,1] = np.sqrt(AbsorptionUnscattered[:,:,:,:,1]) / nmbrRaysUnscattered


    # REDUCE THE MATRICES TO THE MEANINGFUL DIMENSIONS
    ############################################################################
    firstaxistosum = len(idata.WhatToResolve)
    for i in range(firstaxistosum, 4):
        if idata.computeAmplitude:
            if idata.storeWfct:           Wfct            = np.sum(Wfct,            axis=firstaxistosum)
            if idata.storeVelocityField:  VelocityField   = np.sum(VelocityField,   axis=firstaxistosum)
            if idata.storeAbsorption:     Absorption      = np.sum(Absorption,      axis=firstaxistosum)
        if idata.computeAmplitudeUnscattered:
            if idata.storeWfct:           WfctUnscattered          = np.sum(WfctUnscattered,          axis=firstaxistosum)
            if idata.storeVelocityField:  VelocityFieldUnscattered = np.sum(VelocityFieldUnscattered, axis=firstaxistosum)
            if idata.storeAbsorption:     AbsorptionUnscattered    = np.sum(AbsorptionUnscattered,    axis=firstaxistosum)
        if idata.computeScatteringEffect:
            if idata.storeWfct:           WfctScatteringEffect          = np.sum(WfctScatteringEffect,          axis=firstaxistosum)
            if idata.storeVelocityField:  VelocityFieldScatteringEffect = np.sum(VelocityFieldScatteringEffect, axis=firstaxistosum)
            if idata.storeAbsorption:     AbsorptionScatteringEffect    = np.sum(AbsorptionScatteringEffect,    axis=firstaxistosum)
        if idata.computeScatteredContribution:
            if idata.storeWfct:           WfctScatteredContribution          = np.sum(WfctScatteredContribution,          axis=firstaxistosum)
            if idata.storeVelocityField:  VelocityFieldScatteredContribution = np.sum(VelocityFieldScatteredContribution, axis=firstaxistosum)
            if idata.storeAbsorption:     AbsorptionScatteredContribution    = np.sum(AbsorptionScatteredContribution,    axis=firstaxistosum)


    # WRITE THE RESULTS TO FILE outputfilename
    ############################################################################
    outputfilename = idata.outputdirectory + outputfilename + '.hdf5'
    print('write results to file %s \n' % outputfilename)
    sys.stdout.flush()

    with h5py.File(outputfilename, 'w') as fid:
        # Store computed datasets
        if idata.computeAmplitude:
            if idata.storeWfct:          fid.create_dataset("BinnedTraces",  data=Wfct)
            if idata.storeVelocityField: fid.create_dataset("VelocityField", data=VelocityField)
            if idata.storeAbsorption:    fid.create_dataset("Absorption",    data=Absorption)
        if idata.computeAmplitudeUnscattered:
            if idata.storeWfct:          fid.create_dataset("BinnedTracesUnscattered",  data=WfctUnscattered)
            if idata.storeVelocityField: fid.create_dataset("VelocityFieldUnscattered", data=VelocityFieldUnscattered)
            if idata.storeAbsorption:    fid.create_dataset("AbsorptionUnscattered",    data=AbsorptionUnscattered)
        if idata.computeScatteringEffect:
            if idata.storeWfct:          fid.create_dataset("BinnedTracesScatteringEffect",  data=WfctScatteringEffect)
            if idata.storeVelocityField: fid.create_dataset("VelocityFieldScatteringEffect", data=VelocityFieldScatteringEffect)
            if idata.storeAbsorption:    fid.create_dataset("AbsorptionScatteringEffect",    data=AbsorptionScatteringEffect)
        if idata.computeScatteredContribution:
            if idata.storeWfct:          fid.create_dataset("BinnedTracesScatteredContribution",    data=WfctScatteredContribution)
            if idata.storeVelocityField: fid.create_dataset("VelocityFieldScattererContribution",   data=VelocityFieldScatteredContribution)
            if idata.storeAbsorption:    fid.create_dataset("AbsorptionScatteredContribution",      data=AbsorptionScatteredContribution)

        if idata.storeVelocityField:
            fid.create_dataset("VelocityFieldStored", data=','.join(VelocityComponentsToStore) + ',')

        fid.create_dataset("WhatToResolve", data=','.join(idata.WhatToResolve) + ',')

        # Store grid metadata — each component name maps directly to its dataset prefix
        if uniform_bins:
            for i, var in enumerate(idata.WhatToResolve):
                fid.create_dataset(f"{var}min",  data=bin_min[i])
                fid.create_dataset(f"{var}max",  data=bin_max[i])
                fid.create_dataset(f"nmbr{var}", data=idata.nmbr[i])
        else:
            for i, var in enumerate(idata.WhatToResolve):
                fid.create_dataset(f"{var}bins", data=bins[i])

        fid.create_dataset("nmbrRays",            data=nmbrRays)
        fid.create_dataset("nmbrRaysUnscattered",  data=nmbrRaysUnscattered)
        fid.create_dataset("nmbrRaysScattered",    data=nmbrRaysScattered)
        fid.create_dataset("uniform_bins",         data=uniform_bins)
        fid.create_dataset("Mode",               data=sigma)
        fid.create_dataset("FreqGHz",            data=freq)
        fid.create_dataset("antennapolangle",    data=antennapolangle)
        fid.create_dataset("antennatorangle",    data=antennatorangle)
        fid.create_dataset("rayStartX",          data=rayStartX)
        fid.create_dataset("rayStartY",          data=rayStartY)
        fid.create_dataset("rayStartZ",          data=rayStartZ)
        fid.create_dataset("beamwidth1",         data=beamwidth1)
        fid.create_dataset("beamwidth2",         data=beamwidth2)
        fid.create_dataset("curvatureradius1",   data=curvatureradius1)
        fid.create_dataset("curvatureradius2",   data=curvatureradius2)
        fid.create_dataset("centraleta1",        data=centraleta1)
        fid.create_dataset("centraleta2",        data=centraleta2)

# END OF FILE
