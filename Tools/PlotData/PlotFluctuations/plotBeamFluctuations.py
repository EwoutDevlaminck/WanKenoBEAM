"""This module collects plotting routines for the disgnostics of the fluctuation.
The calculations are done with the exact same functions as the main code and
with the same ray-tracing configuration file, so that the resulting plot gives 
a reliable visualization of the actual fluctuation that have been seen by the 
ray tracing procedures.
"""

# Load standard modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import h5py
# Load local modules
from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import IntSample, StixParamSample
from CommonModules.PlasmaEquilibrium import ModelEquilibrium
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium, TokamakEquilibrium2
from CommonModules.PlasmaEquilibrium import AxisymmetricEquilibrium
from RayTracing.modules.atanrightbranch import atanRightBranch
from RayTracing.modules.scattering.GaussianModel import GaussianModel_base
from RayTracing.modules.scattering.ShaferModel import ShaferModel_base
from Tools.PlotData.CommonPlotting import plotting_functions
from Tools.PlotData.PlotVessel.plotVessel import plotVessel

import CommonModules.physics_constants as PhysConst

# Sample the envelope of the fluctuations as used by the code
def sample_fluct_envelop(R1d, Z1d, axis, envelope, Lpp, radial_coord, Eq):
    
    """Sample the envelope of the scattering cross-section.
    The result is normalized to its maximum.
    """

    Raxis, Zaxis = axis
    nptR = np.size(R1d)
    nptZ = np.size(Z1d)
    fluct_sample = np.empty([nptR, nptZ])
    length_sample = np.empty([nptR, nptZ])
    for iR in range(0, nptR):
        Rloc = R1d[iR]
        deltaR = Rloc - Raxis
        for jZ in range(0, nptZ):
            Zloc = Z1d[jZ]
            deltaZ = Zloc - Zaxis
            Ne = Eq.NeInt.eval(Rloc, Zloc)
            Te = Eq.TeInt.eval(Rloc, Zloc)
            Bt = Eq.BtInt.eval(Rloc, Zloc)
            BR = Eq.BRInt.eval(Rloc, Zloc)
            Bz = Eq.BzInt.eval(Rloc, Zloc)
            Bnorm = np.sqrt(Bt**2+BR**2+Bz**2)
            rho = radial_coord(Rloc, Zloc)
            theta = atanRightBranch(deltaZ, deltaR)
            fluct_sample[iR, jZ] = envelope(Ne, rho, theta)
            length_sample[iR, jZ] = Lpp(rho, theta, Ne, Te, Bnorm)

    # OBSOLETE BEHAVIOR (Normalization to max)
    # max_amplitude = np.max(sample.flatten())
    # sample = sample / max_amplitude

    return fluct_sample, length_sample


# Main plotting function
def plot_beam_fluct(inputdata):
    
    """Plot the fluctuation envelope using the parameters in the 
    ray-tracing configuration file configfile passed as the only
    argument. """
	
    # Load the input data from the ray tracing file and configuration file
    #both are needed for the double plotting intended here.
    
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

        c = PhysConst.SpeedOfLight               # speed of light in cm / s
        omega = PhysConst.AngularFrequency(FreqGHz[-1]) # 2.*math.pi*freq*1e9
        k0 = PhysConst.WaveNumber(omega)         # omega / c  wave vector in free space
        mode.append(fid.get('Mode')[()])
        Wfct.append(fid.get('BinnedTraces')[()]) # This is the integrated v from your notes, so Normfac * (k0/2pi)³ * w_unit
        try:
            Absorption.append(fid.get('Absorption')[()])
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
            uniform_bins = fid.get('uniform_bins')[()]
        except: 
            uniform_bins = True
        
        if uniform_bins:
            try:
                Xmin_beam = fid.get('Xmin')[()]
                Xmax_beam = fid.get('Xmax')[()]
                nmbrX_beam = fid.get('nmbrX')[()]
                resolve = "X"
            except:
                Xmin_beam = fid.get('Rmin')[()]
                Xmax_beam = fid.get('Rmax')[()]
                nmbrX_beam = fid.get('nmbrR')[()]
                resolve = "R"

            try:
                resolveY = True
                Ymin_beam = fid.get('Ymin')[()]
                Ymax_beam = fid.get('Ymax')[()]
                nmbrY_beam = fid.get('nmbrY')[()]
            except:
                resolveY = False


            try:
                resolveNpar = True
                Nparallelmin_beam = fid.get('Nparallelmin')[()]
                Nparallelmax_beam = fid.get('Nparallelmax')[()]
                nmbrNparallel_beam = fid.get('nmbrNparallel')[()]
            except:
                resolveNpar = False


            Zmin_beam = fid.get('Zmin')[()]
            Zmax_beam = fid.get('Zmax')[()]
            nmbrZ_beam= fid.get('nmbrZ')[()]

        else:
            try:
                Xbins = fid.get('Xbins')[()]
                resolve = "X"
            except:
                Xbins = fid.get('Rbins')[()]
                resolve = "R"

            try:
                resolveY = True
                Ybins = fid.get('Ybins')[()]
            except:
                resolveY = False

            try:
                resolveNpar = True
                Nparallelbins = fid.get('Nparallelbins')[()]
            except:
                resolveNpar = False

            Zbins = fid.get('Zbins')[()]

        fid.close()

        if resolveY == True:
            Wfct[i] = np.sum(Wfct[i],axis=1)
        if resolveNpar == True:
            Wfct[i] = np.sum(Wfct[i],axis=2)

        # calculate the corresponding cube-edgelength
        if uniform_bins:
            DeltaX = (Xmax_beam-Xmin_beam)/nmbrX_beam
            DeltaZ = (Zmax_beam-Zmin_beam)/nmbrZ_beam

            #Wfct[i] = Wfct[i]/DeltaX/DeltaZ
            R_beam = np.linspace(Xmin_beam, Xmax_beam, nmbrX_beam) / 100 # m
            Z_beam = np.linspace(Zmin_beam, Zmax_beam, nmbrZ_beam) / 100 # m
            ZZ_beam, RR_beam = np.meshgrid(Z_beam, R_beam)

            # Calculate the energy density in a gridcell [J/m³]
            # E_dens = 4pi/c * BinnedTraces /dV

            Wfct[i] *= 1e6  * 4 * np.pi / (c/100) # Convert to J in the full volume

            print(f'Total field energy = {np.sum(Wfct[i])}J')

            Wfct[i] /= DeltaX*DeltaZ * 1e-4 * 2 * np.pi * np.expand_dims(RR_beam, axis=-1) # Devided by toroidal volume element in m³

            if Abs_recorded:
                P_abs = np.sum(Absorption[i], axis=(0,1))
                # And convert to MW/m³ by dividing by the toroidal length
                Absorption[i] /= DeltaX*DeltaZ * 1e-4 * 2 * np.pi * np.expand_dims(RR_beam, axis=-1) # MW/m³
                Absorption[i] /= 2 * np.pi * np.expand_dims(RR_beam, axis=-1) # Averaged over the toroidal angle
            
            if Vel_recorded:
                # We have to convert to MW/m², I'm not entirely sure how yet...
                for j in range(Velocity[i].shape[2]):
    
                    Velocity[i][:,:,j, :] /= DeltaX*DeltaZ * 1e-4 * 2 * np.pi * np.expand_dims(RR_beam, axis=-1) # MW/m³

        else:
            DeltaX = np.diff(Xbins)
            DeltaZ = np.diff(Zbins)
            DeltaArea = np.outer(DeltaX, DeltaZ)
            R_beam = (Xbins[:-1] + Xbins[1:]) / 2 / 100 # m
            Z_beam = (Zbins[:-1] + Zbins[1:]) / 2 / 100 # m
            ZZ_beam, RR_beam = np.meshgrid(Z_beam, R_beam)

            # Calculate the energy density in a gridcell [J/m³]
            # E_dens = 4pi/c * BinnedTraces /dV

            Wfct[i] *= 1e6  * 4 * np.pi / (c/100) # Convert to J in the full volume

            print(f'Total field energy = {np.sum(Wfct[i])}J')

            Wfct[i] /= np.expand_dims(DeltaArea, axis=-1) * 1e-4 * 2 * np.pi * np.expand_dims(RR_beam, axis=-1) # Devided by toroidal volume element in m³

            if Abs_recorded:
                P_abs = np.sum(Absorption[i], axis=(0,1))
                # And convert to MW/m³ by dividing by the toroidal length
                Absorption[i] /= np.expand_dims(DeltaArea, axis=-1) * 1e-4 * 2 * np.pi * np.expand_dims(RR_beam, axis=-1)
                Absorption[i] /= 2 * np.pi * np.expand_dims(RR_beam, axis=-1) # Averaged over the toroidal angle

            if Vel_recorded:
                # We have to convert to MW/m², I'm not entirely sure how yet...
                for j in range(Velocity[i].shape[2]):
    
                    Velocity[i][:,:,j, :] /= np.expand_dims(DeltaArea, axis=-1) * 1e-4 * 2 * np.pi * np.expand_dims(RR_beam, axis=-1)
    
    ##############################################
    
    #Now move on to equilibrium and fluctuation level calculations
    
    idata = InputData(configfile)

    # Load the equilibrium, depending on the type of device
    # and extract the appropriate function to visualize the equilibrium:
    # either the psi coordinate for tokamaks or the density
    # for generic axisymmetric devices (TORPEX).
    if idata.equilibrium == 'Tokamak':

        Eq = TokamakEquilibrium(idata)

        # Figure size
        figsize = (6,8)

        # Define the grid on the poloidal plane of the device
        Rmin = Eq.Rgrid[0, 0]
        Rmax = Eq.Rgrid[-1, 0]
        Zmin = Eq.zgrid[0, 0]
        Zmax = Eq.zgrid[0, -1]
        nptR = int((Rmax - Rmin) / (idata.rmin / 100.)) # dR = a/100 
        nptZ = int((Zmax - Zmin) / (idata.rmin / 100.)) # dZ = a/100
        #
        print('Using resolution nptR = {}, nptZ = {}'.format(nptR, nptZ))
        #
        R1d = np.linspace(Rmin, Rmax, nptR)
        Z1d = np.linspace(Zmin, Zmax, nptZ)

        # Position of the magnetic axis
        axis = Eq.magn_axis_coord_Rz

        StixX, StixY, field_and_density = StixParamSample(R1d, Z1d, Eq, idata.freq)

        # Define the quantity for the visualization of the equilibrium
        psi = IntSample(R1d, Z1d, Eq.PsiInt.eval)
        equilibrium = psi 

    elif idata.equilibrium == 'Tokamak2D':

        Eq = TokamakEquilibrium2(idata)

        # Figure size
        figsize = (6,8)

        # Define the grid on the poloidal plane of the device
        Rmin = Eq.Rgrid[0, 0]
        Rmax = Eq.Rgrid[-1, 0]
        Zmin = Eq.zgrid[0, 0]
        Zmax = Eq.zgrid[0, -1]
        nptR = int((Rmax - Rmin) / (idata.rmin / 100.)) # dR = a/100 
        nptZ = int((Zmax - Zmin) / (idata.rmin / 100.)) # dZ = a/100
        #
        print('Using resolution nptR = {}, nptZ = {}'.format(nptR, nptZ))
        #
        R1d = np.linspace(Rmin, Rmax, nptR)
        Z1d = np.linspace(Zmin, Zmax, nptZ)

        # Position of the magnetic axis
        axis = Eq.magn_axis_coord_Rz

        StixX, StixY, field_and_density = StixParamSample(R1d, Z1d, Eq, idata.freq)

        # Define the quantity for the visualization of the equilibrium
        psi = IntSample(R1d, Z1d, Eq.PsiInt.eval)
        equilibrium = psi 

    elif idata.equilibrium == 'Axisymmetric':

        Eq = AxisymmetricEquilibrium(idata)

        # Figure size
        figsize = (8,8)

        # Define the grid on the poloidal plane of the device
        Rmin = idata.rmaj - idata.rmin
        Rmax = idata.rmaj + idata.rmin
        Zmin = -idata.rmin
        Zmax = +idata.rmin
        nptR = int((Rmax - Rmin) / (idata.rmin / 100.)) # dR = a/100 
        nptZ = int((Zmax - Zmin) / (idata.rmin / 100.)) # dZ = a/100
        #
        print('Using resolution nptR = {}, nptZ = {}'.format(nptR, nptZ))
        #
        R1d = np.linspace(Rmin, Rmax, nptR)
        Z1d = np.linspace(Zmin, Zmax, nptZ)

        # Position of the effective center of the machine
        axis = [idata.rmaj, 0.]

        # Define the quantity for the visualization of the equilibrium
        equilibrium = IntSample(R1d, Z1d, Eq.NeInt.eval) # this is set to Ne

    else:
        msg = "Keyword 'equilibrium' must be either 'Tomakak' or 'Axisymmetric'"
        raise ValueError(msg)

    # Construct the object for the fluctuations depending on the model
    rank = 0 # dummy
    if idata.scatteringGaussian == True:
        Fluct = GaussianModel_base(idata,rank)
        envelope = lambda Ne, rho, theta: \
                   Fluct.scatteringDeltaneOverne(Ne,rho,theta)**2
        
    elif idata.scatteringGaussian == False:
        Fluct = ShaferModel_base(idata,rank)
        envelope = lambda Ne, rho, theta: Fluct.ShapeModel(rho, theta)
        
    # In both cases the function which evaluates the perpendicular correlation
    # length is set as an attribute of the fluctuation model.
    # This coincides with the function given as input if given.
    Lpp = Fluct.scatteringLengthPerp
        
    # Define the relevant radial coordinate in presence of flux surfaces
    radial_coord = lambda R, Z: np.sqrt(Eq.PsiInt.eval(R, Z))

    # Sample the envelope of the given poloidal grid
    fluct, _ = sample_fluct_envelop(R1d, Z1d, axis,
                                         envelope, Lpp, radial_coord, Eq)
    	
    # Plotting directives
    fig1 = plt.figure(1, figsize=figsize)
    ax1 = fig1.add_subplot(111, aspect='equal')

    #Plot the vessel
    try:
        print('Plotting vessel')
        Rv_in, Rv_out, Zv_in, Zv_out, Zt, Rt = plotVessel(idata)
        print('Plotting vessel')
        R_fill = np.concatenate([Rv_in, np.flipud(Rv_out)])
        Z_fill = np.concatenate([Zv_in, np.flipud(Zv_out)])
        Rt_fill = np.concatenate([Rt, Rv_in])
        Zt_fill = np.concatenate([Zt, Zv_in])
        ax1.fill(R_fill, Z_fill, color=[0.4, 0.4, 0.4], edgecolor='none', zorder=13)
        ax1.fill(Rt_fill, Zt_fill, color=[0.5, 0.5, 0.5], edgecolor='none', zorder=13)
        vessel_plotted = True
    except:
        print('No vessel data found')
        vessel_plotted = False
        pass

    # ... fluctuation envelope ...
    if not idata.scattering:
        reds_map = matplotlib.colormaps.get_cmap('Reds')
        #ax1.set_facecolor(reds_map(0.))
        Ne = IntSample(R1d, Z1d, Eq.NeInt.eval)
        c1 = ax1.pcolormesh(R1d, Z1d, Ne, cmap='Reds', vmin=0., alpha=.9, zorder=0)
        colorbarNe = plt.colorbar(c1, orientation='vertical', pad=.05, shrink=.7)
        colorbarNe.set_label(label=r'$n_e [1e19 m^{-3}]$', size=13)
    else:
        Ne = IntSample(R1d, Z1d, Eq.NeInt.eval)
        deltaNe = np.where(equilibrium<1.6, fluct.T*Ne, 0)
        c1 = ax1.pcolormesh(R1d, Z1d, deltaNe, cmap='Reds', vmin=0., vmax=1, alpha=.9, zorder=0)
        colorbarFluct = plt.colorbar(c1, orientation='vertical', pad=.05, shrink=.7)
        #colorbarFluct.set_label(label=r'$\langle \delta n_e\rangle /n_e$', size=16)
        colorbarFluct.set_label(label=r'$RMS\ \delta n_e [1e19 m^{-3}]$', size=13)
    ### colorbarFluct.set_label(r'')
    # ... flux surfaces ...
    lines = np.arange(0, np.amax(equilibrium), 0.1)
    ax1.contour(R1d, Z1d, np.sqrt(equilibrium), lines, colors='grey', linestyles='dashed', linewidths=1, zorder=3)
    ax1.contour(R1d, Z1d, np.sqrt(equilibrium), [1.], colors='black', linestyles='solid', linewidths=1, zorder=6)
    ax1.set_xlabel('$R$ [cm]') 
    ax1.set_ylabel('$Z$ [cm]')
    ax1.set_title('Wave propagation \n through fluctuations', fontsize=20)
    
    # If vessel is plotted, set everything that is outside the vessel to white
    if vessel_plotted:

        from matplotlib.path import Path
        # Create a Path object from the contour
        contour_path = Path(np.column_stack((Rt, Zt)))
        RR = np.tile(R1d, (nptZ, 1))
        ZZ = np.tile(Z1d, (nptR, 1)).T

        # Flatten grid arrays and check which points are inside the contour
        points = np.column_stack((RR.ravel(), ZZ.ravel()))
        mask = ~contour_path.contains_points(points)  # Invert mask: True outside

        # Reshape mask back to 2D
        mask = mask.reshape(RR.shape)

        # Plot a white rectangle over the points outside the contour
        c_white_trans = clrs.colorConverter.to_rgba('white', alpha=0.)
        c_white = clrs.colorConverter.to_rgba('white', alpha=1.)
        transMap_help = clrs.LinearSegmentedColormap.from_list('transMap_help',  
            [c_white_trans, c_white], 512)

        ax1.pcolormesh(RR, ZZ, mask, cmap=transMap_help, zorder=8)
        # And turn the axis borders transparent
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)

    #Then plot the propagation of the wave on top.
    
    
    # Create a colormap that is nice and blends easily into the background,
    # while not sacrificing visibility of having it being completely see-through for low values 
    
    
    #This color matches my slides, cheers ED
    clrs_Ewout = ['#007480', '#413C3A', '#00A79F']
    c_white_trans = clrs.colorConverter.to_rgba('white', alpha=0.8)
    Vel_recorded = False
    for beam in range(len(Wfct)):
        if Vel_recorded:
            Quantity = Velocity[beam]
            #Q = c/100 * Quantity[:,:,0,0]
            Q = np.sqrt(Quantity[:, :, 0, 0]**2 + Quantity[:, :, 1, 0]**2)
            #Q = np.where(Wfct[beam][:,:,0]>1e-3, Q/Wfct[beam][:,:,0], 0)
        else:
            Quantity = Wfct[beam]
            Q = Quantity[:,:,0]

    #The transmap is such that the lower vlues become increasingly transparent,
    # so we can overlap colormaps. But going all the way to white& transparent is too much
    # therefore, we cut it off somewhere
    
        transMap_help = clrs.LinearSegmentedColormap.from_list('transMap_help',  
            [c_white_trans, clrs_Ewout[beam]], 512)
        halfway = transMap_help(.25) #Get the value somwhere along the colormap
        transMap = clrs.LinearSegmentedColormap.from_list('transMap',
            [halfway, clrs_Ewout[beam]], 512)
    
        #Set the lower bound to be transparent, so we see the background contourf
        transMap.set_under(color='b', alpha=0.)
        upperBound = np.amax(Q)
        lowerBound = upperBound*1e-2 #What's the lowest value we still display?

        #beamfig = ax1.contourf(100*RR_beam, 100*ZZ_beam, Q ,100, vmin=lowerBound, cmap=transMap, zorder=9) 
        beamfig = ax1.pcolormesh(100*RR_beam, 100*ZZ_beam, Q, vmin=lowerBound, vmax=upperBound, cmap=transMap, zorder=9)
    
        #Make it so that the colourmap only starts at lowerBound
        displayedMap = plt.cm.ScalarMappable(norm=clrs.Normalize(lowerBound, upperBound), cmap=transMap)  
        colorbarBeam = plt.colorbar(beamfig, orientation='vertical', pad=.1, shrink=.7)
        if Vel_recorded:
            colorbarBeam.set_label(label=r'$|s| (MW/m^2s)$', size=10, labelpad=-30, y=1.08, rotation=0)
        else:
            colorbarBeam.set_label(label=f'|E|² (J/m³)\n f={FreqGHz[beam]}GHz', size=10, labelpad=-30, y=1.08, rotation=0)

    #Plot the absorption on top of the beam
    if Abs_recorded:
        Absorption = np.sum(Absorption, axis=0)
        cmap_abs = plt.cm.afmhot
        cmap_abs.set_under(color='b', alpha=0.)
        abs_fig = ax1.contourf(100*RR_beam,100*ZZ_beam,Absorption[:,:,0], levels=100,vmin=np.amax(Absorption)/20, cmap=cmap_abs, zorder=12)
        colorbarAbs = plt.colorbar(abs_fig, orientation='vertical', pad=.1, shrink=.7)
        colorbarAbs.set_label(label=r'$P_{abs} (MW/m³)$', size=10, labelpad=-30, y=1.08, rotation=0)
        print(f'Total P_abs={P_abs[0]} +- {P_abs[1]}MW')
    #Plot the cyclotron resonances
    h1, h2, h3 = plotting_functions.add_cyclotron_resonances(R1d, Z1d, StixY, ax1)
    
    #With the beamView parameter, we can specify if we want to see the full equilibrium
    #or zoomed in on only the beam calculated part
    try:
        beamView = idata.beamView
    except AttributeError:
        beamView = False


    if uniform_bins:
        if beamView == True:
            ax1.set_xlim(Xmin_beam, Xmax_beam)
            ax1.set_ylim(Zmin_beam, Zmax_beam)
        else:
            ax1.set_xlim(Rmin, Rmax)
            ax1.set_ylim(Zmin, Zmax)
    else:
        if beamView == True:
            ax1.set_xlim(Xbins[0], Xbins[-1])
            ax1.set_ylim(Zbins[0], Zbins[-1])
        else:
            ax1.set_xlim(Rmin, Rmax)
            ax1.set_ylim(Zmin, Zmax)

    if vessel_plotted:
        ax1.set_xlim(np.amin(Rv_out), np.max([np.amax(Rv_out), np.amax(100*RR_beam)]))
        ax1.set_ylim(np.amin(Zv_out), np.max([np.amax(Zv_out), np.amax(100*ZZ_beam)]))
    plt.show()

    # return
    pass
#
# END OF FILE
