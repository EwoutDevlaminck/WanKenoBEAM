"""This file reads binned data file, where R (or alternatively X) and Z should 
be resolved and does a 2D-plot of it.
"""

import numpy as np
import h5py

from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import IntSample, StixParamSample
from CommonModules.PlasmaEquilibrium import ModelEquilibrium
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium, TokamakEquilibrium2
from Tools.PlotData.CommonPlotting import plotting_functions
def plot_binned(inputdata):
	#changed by Ewout, get the flux surface info from the configfile
    inputfilenames, configfile = inputdata[0:-1], inputdata[-1]
    # read the data from data file given in inputfilename
    FreqGHz = []
    mode = []
    Wfct = []
    Absorption = []
    Velocity = []

    for i,file in enumerate(inputfilenames):
        print('Reading data file...\n')
        fid = h5py.File(file,'r')
        FreqGHz.append(fid.get('FreqGHz')[()])
        mode.append(fid.get('Mode')[()])
        Wfct.append(fid.get('BinnedTraces')[()])
        try:
            Absorption.append(id.get('Absorption')[()])
            Abs_recorded = True
        except:
            Abs_recorded = False
        try:
            Velocity.append(fid.get('VelocityField')[()])
            # Stored as a tuple of size (N_1stdim, N_2nddim, Component, 0)
            Vel_recorded = True
        except:
            Vel_recorded = False
            
        try:
            Xmin = fid.get('Xmin')[()]
            Xmax = fid.get('Xmax')[()]
            nmbrX = fid.get('nmbrX')[()]
            resolve = "X"
        except:
            Xmin = fid.get('Rmin')[()]
            Xmax = fid.get('Rmax')[()]
            nmbrX = fid.get('nmbrR')[()]
            resolve = "R"

        try:
            resolveY = True
            Zmin = fid.get('Ymin')[()]
            Zmax = fid.get('Ymax')[()]
            nmbrZ = fid.get('nmbrY')[()]
        except:
            resolveY = False

        try:
            resolveZ = True
            Zmin = fid.get('Zmin')[()]
            Zmax = fid.get('Zmax')[()]
            nmbrZ = fid.get('nmbrZ')[()]
        except:
            resolveZ = False


        try:
            resolveNpar = True
            Nparallelmin = fid.get('Nparallelmin')[()]
            Nparallelmax = fid.get('Nparallelmax')[()]
            nmbrNparallel = fid.get('nmbrNparallel')[()]
        except:
            resolveNpar = False



        fid.close()

	

        # calculate the corresponding cube-edgelength
        DeltaX = (Xmax-Xmin)/nmbrX
        DeltaZ = (Zmax-Zmin)/nmbrZ

        #if resolveY == True:
        #    Wfct[i] = np.sum(Wfct[i],axis=1)
        if resolveNpar == True:
            Wfct[i] = np.sum(Wfct[i],axis=2)



        Wfct[i] = Wfct[i]/DeltaX/DeltaZ
        
        if Abs_recorded:
            Absorption[i] /= DeltaX*DeltaZ


    """Part underneath added by Ewout"""
    # Load the input data fro the ray tracing configuration file
    idata = InputData(configfile)

    # Load the equilibrium, depending on the type of device
    # and extract the appropriate function to visualize the equilibrium:
    # either the psi coordinate for tokamaks or the density
    # for generic axisymmetric devices (TORPEX).
    if idata.equilibrium == 'Tokamak' and resolveZ:

        Eq = TokamakEquilibrium(idata)


        # Define the grid on the poloidal plane of the device
        #
        print('Using resolution nptR = {}, nptZ = {}'.format(nmbrX, nmbrZ))
        #
        R1d = np.linspace(Xmin, Xmax, nmbrX)
        Z1d = np.linspace(Zmin, Zmax, nmbrZ)

        # Position of the magnetic axis
        axis = Eq.magn_axis_coord_Rz

        StixX, StixY, field_and_density = StixParamSample(R1d, Z1d, Eq, idata.freq)


        # Define the quantity for the visualization of the equilibrium
        psi = IntSample(R1d, Z1d, Eq.PsiInt.eval)
        equilibrium = psi 
        Ne = IntSample(R1d, Z1d, Eq.NeInt.eval)
            
    if idata.equilibrium == 'Tokamak2D' and resolveZ:

        Eq = TokamakEquilibrium2(idata)


        # Define the grid on the poloidal plane of the device
        #
        print('Using resolution nptR = {}, nptZ = {}'.format(nmbrX, nmbrZ))
        #
        R1d = np.linspace(Xmin, Xmax, nmbrX)
        Z1d = np.linspace(Zmin, Zmax, nmbrZ)

        # Position of the magnetic axis
        axis = Eq.magn_axis_coord_Rz

        StixX, StixY, field_and_density = StixParamSample(R1d, Z1d, Eq, idata.freq)


        # Define the quantity for the visualization of the equilibrium
        psi = IntSample(R1d, Z1d, Eq.PsiInt.eval)
        equilibrium = psi 
        Ne = IntSample(R1d, Z1d, Eq.NeInt.eval)
        
    """Until here"""


    # plot the resulting 2D-array over X,Z indices
    # therefore: first generate X and Z vectors
    #Xlist = np.linspace(0.,1.,plotnmbrS)
    Xlist = np.linspace(Xmin,Xmax,nmbrX)
    Zlist = np.linspace(Zmin,Zmax,nmbrZ)
    Zgrid, Xgrid = np.meshgrid(Zlist, Xlist)


    # and plot
    print('Plotting data...\n')

    import matplotlib.pyplot as plt
    import matplotlib.colors as clrs

    plt.figure(1, figsize=(8, 6))
    #adjustprops = dict(left=0.14, bottom=0.1, right=0.95, top=0.85, wspace=0.2, hspace=0.48)
    #plt.subplots_adjust(**adjustprops)

    plt.rcParams.update({'font.size': 16})
    plt.rc('xtick', labelsize=16) 
    plt.rc('ytick', labelsize=16) 
	
    ax = plt.subplot(121, aspect='equal')
    if idata.equilibrium == 'Tokamak' and resolveZ:
        plt.contour(R1d, Z1d, equilibrium, np.linspace(0, 2, 19), colors='grey', linestyles='dashed', linewidths=1)
        plt.contour(R1d, Z1d, equilibrium, [1.], colors='r', linewidths=1)
    Wfct = np.sum(Wfct, axis=0)
    if Vel_recorded:
        Vel_amplitude = [ np.sqrt(Velocity[i][:, :, 0, 0]**2 + Velocity[i][:, :, 1, 0]**2) for i in range(len(Velocity))]
        Vel_amplitude = np.sum(Vel_amplitude, axis=0)
        E_density = plt.contourf(Xgrid, Zgrid, Vel_amplitude/Wfct[:,:,0], 100, cmap='cividis')
        #E_density = plt.contourf(Xgrid,Zgrid,Wfct[:,:,0],100, cmap='cividis')
        plt.colorbar(E_density, shrink=0.6)
        #plt.xlim(95, 115)
        #plt.ylim(-10, 10)
        plt.title('Group velocity \n f = '+str(FreqGHz)+' GHz')
    else:
        E_density = plt.contourf(Xgrid,Zgrid,Wfct[:,:,0],100, cmap='cividis')
        plt.colorbar(E_density, shrink=0.6)
        plt.title('E-field energy density \n f = '+str(FreqGHz)+' GHz')
    if Abs_recorded:
        Absorption = np.sum(Absorption, axis=0)
        levels = np.linspace(np.amax(Absorption)/100, np.amax(Absorption), 10)
        abs_coeff = np.where(Wfct[:, :, 0]>0., Absorption[:, :, 0] / Wfct[:, :, 0], 0.)
        plt.contour(Xgrid,Zgrid,Absorption[:,:0],levels=100, cmap='afmhot')
        plt.colorbar(E_density, shrink=0.6)
    if resolveZ:
        h1, h2, h3 = plotting_functions.add_cyclotron_resonances(R1d, Z1d, StixY, ax)
    #plot the flux surfaces to see where the plasma is situated  


    

    if resolve == "X":
        plt.xlabel('X (cm)')
    else:
        plt.xlabel('R (cm)')
    if resolveZ:
        plt.ylabel('Z (cm)')
    else:
        plt.ylabel('Y (cm)')



    plt.subplot(122, aspect='equal')
    plt.contourf(Xgrid,Zgrid,Wfct[:,:,1],100, cmap='cividis')
    plt.colorbar(shrink=0.6)
    if idata.equilibrium == 'Tokamak' and resolveZ:    
        #plot the flux surfaces to see where the plasma is situated
        plt.contour(R1d, Z1d, equilibrium, np.linspace(0, 2, 19), colors='grey', linestyles='dashed', linewidths=1)
        #LCFS
        plt.contour(R1d, Z1d, equilibrium, [1.], colors='r', linewidths=1)

    plt.title('Statistical uncertainty \n on |E²|')

    if resolve == "X":
        plt.xlabel('X (cm)')
    else:
        plt.xlabel('R (cm)')
    plt.ylabel('Z (cm)')



    plt.show()

    # return
    return
#
# END OF FILE
