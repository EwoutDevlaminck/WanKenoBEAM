"""This module collects plotting routines for the disgnostics of the fluctuation.
The calculations are done with the exact same functions as the main code and
with the same ray-tracing configuration file, so that the resulting plot gives 
a reliable visualization of the actual fluctuation that have been seen by the 
ray tracing procedures.
"""

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt
# Load local modules
from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import IntSample
from CommonModules.PlasmaEquilibrium import ModelEquilibrium
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium, TokamakEquilibrium2
from CommonModules.PlasmaEquilibrium import AxisymmetricEquilibrium
from RayTracing.modules.atanrightbranch import atanRightBranch
from RayTracing.modules.scattering.GaussianModel import GaussianModel_base
from RayTracing.modules.scattering.ShaferModel import ShaferModel_base

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
def plot_fluct(configfile):
    
    """Plot the fluctuation envelope using the parameters in the 
    ray-tracing configuration file configfile passed as the only
    argument. """

    # Load the input data fro the ray tracing configuration file
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
        axis = Eq. magn_axis_coord_Rz

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
        axis = Eq. magn_axis_coord_Rz

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
    fluct, length = sample_fluct_envelop(R1d, Z1d, axis,
                                         envelope, Lpp, radial_coord, Eq)
    
    
    # Plotting directives
    fig1 = plt.figure(1, figsize=figsize)
    ax1 = fig1.add_subplot(111, aspect='equal')
    # ... fluctuation envelope ...
    c1 = ax1.pcolormesh(R1d, Z1d,np.sqrt(fluct.T), cmap='Reds', vmin=0.0, vmax=1)
    colorbarFluct = plt.colorbar(c1, orientation='vertical')
    ### colorbarFluct.set_label(r'')
    # ... flux surfaces ...
    ax1.contour(R1d, Z1d, equilibrium, 20, colors='grey', linestyles='dashed')
    ax1.contour(R1d, Z1d, equilibrium, [1.], colors='black')
    ax1.set_xlabel('$R$ [cm]') 
    ax1.set_ylabel('$Z$ [cm]')
    ax1.set_title(r'$RMS\ \delta n_e /n_e$', fontsize=20)

    
    fig2 = plt.figure(2, figsize=figsize)
    ax2 = fig2.add_subplot(111, aspect='equal')
    # ... perpendicular correlation length ...
    c2 = ax2.pcolormesh(R1d, Z1d, length.T, cmap='hot')
    colorbarLpp = plt.colorbar(c2, orientation='vertical')
    ### colorbarFluct.set_label(r'')
    # ... flux surfaces ...
    ax2.contour(R1d, Z1d, equilibrium, 20, colors='grey', linestyles='dashed')
    ax2.contour(R1d, Z1d, equilibrium, [1.], colors='black')
    ax2.set_xlabel('$R$ [cm]') 
    ax2.set_ylabel('$Z$ [cm]')
    ax2.set_title(r'$L_\perp$', fontsize=20)
    
    plt.show()

    # return
    pass
#
# END OF FILE
