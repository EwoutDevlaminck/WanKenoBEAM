"""This python script reads data from the binning code, 
assuming that only rho is resolved and as a weight absorption
is chosen. 
A torbeam-volume file volumes.dat may be read or the volumes
in the tokamak can be estimated with the equilibrium file as 
a starting point. 
"""


# Load standard modules
import h5py
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# Load local modules
from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import MagneticSurfaces

_BEAM_COLOURS = [ '#007480','r', '#F39869', '#413C3A',  '#00A79F']
_BEAM_LINESTYLES = ['-', '--']



# This read the data from a binned file and return
# the profile of dP/drho, dV/drho, and dP/dV
def compute_deposition_profiles(idata, inputfilename=None):
    """Compute the power deposition profile for the give configuration file.
    """

    # READ DATA
    ############################################################################
    if inputfilename is None:
        inputfilename = idata.outputdirectory+idata.outputfilename[0]+'.hdf5'
    print('\n') 
    print('Reading data file %s\n' %(inputfilename))
    fid = h5py.File(inputfilename,'r')
    try:
        rhomin = fid.get('rhomin')[()]
        rhomax = fid.get('rhomax')[()]
        nmbrrho = fid.get('nmbrrho')[()]
        Deltarho = (rhomax - rhomin) / nmbrrho
        dP_drho = fid.get('Absorption')[()] / Deltarho
        uniform_bins = True
    except:
        rhobins = fid.get('rhobins')[()]
        rho = 0.5 * (rhobins[1:] + rhobins[:-1])
        Deltarho = np.diff(rhobins)
        dP_drho = np.array(fid.get('Absorption')[()]) / Deltarho[:,np.newaxis]
        nmbrrho = rho.size
        rhomin = rho[0]
        rhomax = rho[-1]
        uniform_bins = False
    mode = fid.get('Mode')[()]
    freq = fid.get('FreqGHz')[()]
    centraleta1 = fid.get('centraleta1')[()]
    centraleta2 = fid.get('centraleta2')[()]
    beamwidth1 = fid.get('beamwidth1')[()]
    beamwidth2 = fid.get('beamwidth2')[()]
    curvatureradius1 = fid.get('curvatureradius1')[()]
    curvatureradius2 = fid.get('curvatureradius2')[()]
    fid.close()
    #############################################################################

    # print some messages
    print('... processing data from: ' + inputfilename)
    if uniform_bins:
        print('    total absorbed power is %.3fMW' %(np.sum(dP_drho*Deltarho)))
    else:
        print('    total absorbed power is %.3fMW' %(np.sum(dP_drho*Deltarho[:, np.newaxis], axis=0)[0]))
    #print('    grid size drho is %.3f' %(Deltarho))
    print('    volume calculation flag: ' + idata.VolumeSource)
    try:
        print('    label of the data set: ' + idata.label + '\n')
    except AttributeError:
        print('    label of the data set: not given \n')

    # CALCULATION OF dV/drho
    #############################################################################
    # Compute the derivative of the volume with respect to rho
    # (recomputing the volumes for every single profile might appear redundant
    #  but do not try to be fancy here: The calculation takes some time, but
    #  we want to have the flexibility of plotting on the same graph power 
    #  deposition profile with different range in rho, and thus, with different
    #  volumes.)
    # ... read volumes from TORBEAM data files ...
    if idata.VolumeSource == 'TORBEAM':
        # read torbeam volumes
        torbeamvolumefilename = idata.torbeam_dir + 'volumes.dat'
        data = np.loadtxt(torbeamvolumefilename)
        rhoTemp = data[:,0]
        volTemp = data[:,1]
        # numerical gradient 
        # (the function ediff1d is used because of the non uniform grid)
        dV_drhoTemp = np.ediff1d(volTemp, to_begin=0.) / \
                      np.ediff1d(rhoTemp, to_begin=1.)
        # interpolation object
        dV_drho_int = interp1d(rhoTemp, dV_drhoTemp, kind='linear')

    # ... compute volume ...
    elif idata.VolumeSource == 'COMPUTE':
        # compute the volumes, based on the topfile
        Eq = MagneticSurfaces(idata)
        rhoTemp = np.linspace(rhomin, rhomax, idata.nmbrVolumes)          
        # derivatives of the volume (dV_drho = dV_dpsi * dpsi_drho
        print("Computing dV_dpsi ... \n")
        dV_dpsi = np.array(list(map(Eq.compute_dvolume_dpsi, rhoTemp**2)))
        dV_drhoTemp = 2. * rhoTemp * dV_dpsi
        # interpolation object for dV_drho
        dV_drho_int = interp1d(rhoTemp, dV_drhoTemp, kind='quadratic')
        
    # ... do not use volumes at all: plot dP/drho without normalization ...
    elif idata.VolumeSource == 'NONE':
        # identically equal to one
        dV_drho_int = lambda rho: np.ones([rho.size])
        
    # ... none of the above ...
    else:
        print('ERROR: which source for the volumes do you mean ?\n')
        raise

    # Grid in rho and derivative of the volume with respect to rho
    try :
        dV_drho = dV_drho_int(rho)
    except:
        rho = np.linspace(rhomin, rhomax, nmbrrho) 
        dV_drho = dV_drho_int(rho)
    #############################################################################


    # COMPUTE THE PROFILES
    #############################################################################
    dP_dV = np.empty([nmbrrho,2])
    for j in range(0,2):
        dP_dV[:,j] = dP_drho[:,j] / dV_drho[:] 
    #############################################################################

    # COLLECT THE DATA INTO A DICTIONARY
    #############################################################################
#    f = open("dPdV.out","w")
#    for i in range(nmbrrho):
#        f.write('{0:12.4e}{1:12.4e}'.format(rho[i],dP_dV[i,0])+"\n")

    results = {}
    # load profiles
    results['rho'] = rho
    results['dV_drho'] = dV_drho
    results['dP_drho'] = dP_drho
    results['dP_dV'] = dP_dV
    # load the label for the legend
    try:
        results['label'] = idata.label
    except AttributeError:
        results['label'] = ' '
    
    # return the relevant information
    return results


# Driver called by WKBeam
def plot_abs(listofconfigfiles):

    """Given a list of configuration files for the binning of the 
    power deposition profiles of a number of runs, this computes
    the corresponding profiles of dP/drho, dV/drho, and dP/dV and
    plot them on the same axes for comparison.
    """

    # INITIAL MESSAGE
    #############################################################################
    print('\n')
    print('Plotting power deposition profiles')
    print('Considered configuration files:')
    for configfile in listofconfigfiles:
        print('    ' + configfile)
    print('\n')


    # PLOT THE RESULTS
    #############################################################################
    # matplotlib graphics parameters
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)

    # INITIALIZE THE FIGURES AND AXES
    #############################################################################
    fig_dP_drho = plt.figure(1, figsize=(10,7))
    fig_dV_drho = plt.figure(2, figsize=(10,7))
    fig_dP_dV = plt.figure(3, figsize=(11,7))
    ax_dP_drho = fig_dP_drho.add_subplot(111)
    ax_dV_drho = fig_dV_drho.add_subplot(111)
    ax_dP_dV = fig_dP_dV.add_subplot(111)
    
    # LOOP OVER THE LIST OF CONFIGURATION FILES AND COMPUTE THE PROFILES
    #############################################################################
    for i, configfile in enumerate(listofconfigfiles):

        # LOAD INPUT PARAMETERS
        idata = InputData(configfile)

        # Compute data that are returned as a dictionary ...
        # (it follows that the object "profiles" is a dictionary of dictionaries)
        profiles = compute_deposition_profiles(idata) 
        label = profiles['label']
        rho = profiles['rho']
        dP_drho = profiles['dP_drho']
        dV_drho = profiles['dV_drho']
        dP_dV = profiles['dP_dV']
        
        # Save the profile
        do_something_list = ['yes', 'Yes', True]
        if (hasattr(idata, 'save_profile') and
            idata.save_profile in do_something_list):
            
            workdir = idata.outputdirectory
            datafile_name = idata.outputfilename[0]
            
            if hasattr(idata, 'profile_filename'):
                outputfile_name = workdir + idata.profile_filename
            else:
                
                outputfile_name = workdir + 'power_dep_profile_from_' + datafile_name + '.txt'
                
            print('\nSaving profile to   {}\n'.format(outputfile_name))
            np.savetxt(outputfile_name, (rho, dP_drho[:,0], dV_drho[:], dP_dV[:,0]))
            
        else:
            # Do nothing and continue
            pass

        color = _BEAM_COLOURS[i // 2]
        linestyle = _BEAM_LINESTYLES[i % 2]

        # add deposited power density in rho
        ax_dP_drho.errorbar(rho, dP_drho[:,0], yerr=dP_drho[:,1], label=label, c=color, linestyle=linestyle)

        # add the volume and the power deposition
        ax_dV_drho.plot(rho, dV_drho, c='#007480', label=label)        

        # add power deposition profile
        ax_dP_dV.errorbar(rho, dP_dV[:,0], yerr=dP_dV[:,1], label=label, c=color, linestyle=linestyle)

        # if requested add the corresponding TORBEAM result for the
        # power deposition profile
        try:
            if idata.compare_to_TORBEAM == 'yes':
                # ... load TORBEAM data ...
                t2 = np.loadtxt(idata.torbeam_dir+'t2_new_LIB.dat')
                ax_dP_dV.plot(t2[:,0], t2[:,1], label='TORBEAM '+label)
        except AttributeError:
            pass

    # COMPLETE FIGURES WITH LABELS, LEGENDS AND GRIDS
    #############################################################################  
    # plot of dP/drho
    ax_dP_drho.set_xlim(0)
    ax_dP_drho.set_ylim(0)
    ax_dP_drho.set_xlabel(r'$\rho$',fontsize=20)
    ax_dP_drho.set_ylabel(r'$dP(\rho)/d\rho$ (MW)',fontsize=20)
    ax_dP_drho.legend(fontsize=15)
    ax_dP_drho.grid()

    # plot of dV/drho
    ax_dV_drho.set_xlim(0)
    ax_dV_drho.set_ylim(0)
    ax_dV_drho.set_xlabel(r'$\rho$',fontsize=20)
    ax_dV_drho.set_ylabel(r'$dV(\rho) / d\rho$ (m$^3$)',fontsize=20)
    ax_dV_drho.legend(loc='upper left', fontsize=15)
    ax_dV_drho.grid()

    # plot of dP/dV
    ax_dP_dV.set_xlim(0)
    ax_dP_dV.set_ylim(0)
    ax_dP_dV.set_xlabel(r'$\rho$',fontsize=20)
    ax_dP_dV.set_ylabel(r'$dP / dV$ (MW / m$^3$)',fontsize=20)    
    ax_dP_dV.set_title('power deposition profile',fontsize=20)
    ax_dP_dV.legend(fontsize=15)
    ax_dP_dV.grid()

    plt.show()

    # return
    pass
#
# END OF FILE
#################################################################################
