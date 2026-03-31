"""This module provide a function for quick visualization of the equilibrium
of the plasma actually seen by the code, for a given configuration file.
"""

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt
# Load local modules
from CommonModules.PlasmaEquilibrium import IntSample, StixParamSample
from Tools.PlotData.CommonPlotting import plotting_functions


# Auxiliary plotting functions
def __setup_axes(nrows, ncols):
    
    # One big figure
    f, axs = plt.subplots(nrows, ncols, sharex='col', sharey='row', 
                            figsize=(12,8))

    # Make room for the legend
    plt.subplots_adjust(top=0.8, wspace=0.5)

    # Set equal aspect ration for all axes
    for ax in axs.flatten():
        ax.set_aspect('equal')

    return f, axs


def __plot_B(axs, R, Z, Bcomponents):


    # Extract the components
    BR, Bz, Bp, Bt = Bcomponents

    # Radial field
    axBR = axs[0, 0]
    c_BR = axBR.contourf(R, Z, BR, 100, cmap='RdBu_r')
    plt.colorbar(c_BR, ax=axBR, format="%1.2f")
    axBR.set_ylabel(r'$z$ [m]', fontsize=15)
    axBR.set_title(r'$B_R$ [Tesla]')

    # Vertical field 
    axBz = axs[0, 1]
    c_Bz = axBz.contourf(R, Z, Bz, 100, cmap='RdBu_r')
    plt.colorbar(c_Bz, ax=axBz, format="%1.2f")
    axBz.set_title(r'$B_z$ [Tesla]')

    # Poloidal field
    axBp = axs[1, 0]
    c_Bp = axBp.contourf(R, Z, Bp, 100, cmap='OrRd')
    plt.colorbar(c_Bp, ax=axBp, format="%1.2f")
    axBp.set_xlabel(r'$R$ [m]', fontsize=15)
    axBp.set_ylabel(r'$z$ [m]', fontsize=15)
    axBp.set_title(r'$B_p$ [Tesla]')

    # Toroidal field 
    axBt = axs[1, 1]
    c_Bt = axBt.contourf(R, Z, Bt, 100, cmap='OrRd')
    plt.colorbar(c_Bt, ax=axBt, format="%1.2f")
    axBt.set_xlabel(r'$R$ [m]', fontsize=15)
    axBt.set_title(r'$B_t$ [Tesla]')

    return


def __plot_Stix_parameters(axs, R, Z, StixX, StixY):

    # Cyclotron frequency omega_ce / omega
    axY = axs[1, 2]
    c_Y = axY.contourf(R, Z, StixY, 100, cmap='OrRd')
    plt.colorbar(c_Y, ax=axY, format="%1.2f")
    plotting_functions.add_cyclotron_resonances(R, Z, StixY, axY)
    axY.set_xlabel(r'$R$ [m]', fontsize=15)
    axY.set_title(r'$\omega_{c \mathrm{e}}/\omega$')

    # Plasma frequency omega_pe^2 / omega^2
    axX = axs[0, 2]
    c_X = axX.contourf(R, Z, StixX, 100, cmap='OrRd')
    plt.colorbar(c_X, ax=axX, format="%1.2f")
    plotting_functions.add_Omode_cutoff(R, Z, StixX, axX)
    axX.set_title(r'$\omega_{p \mathrm{e}}^2/\omega^2$') 

    h1, h2, h3 = plotting_functions.add_cyclotron_resonances(R, Z, StixY, axX)
    O_cutoff = plotting_functions.add_Omode_cutoff(R, Z, StixX, axX)
    X_cutoff = plotting_functions.add_Xmode_cutoff(R, Z, StixX, StixY, axX)
    UH_res = plotting_functions.add_UHresonance(R, Z, StixX, StixY, axX)

    return h1, h2, h3, O_cutoff, X_cutoff, UH_res


def __plot_electron_density(axs, iaxs, jaxs, R, Z, Ne, StixX, StixY):

    axNe = axs[iaxs, jaxs]
    c_Ne = axNe.contourf(R, Z, Ne, 100, cmap='OrRd')
    plt.colorbar(c_Ne, ax=axNe, format="%1.2f")
    axNe.set_title(r'$n_{\mathrm{e}}$ [$10^{13}$ cm$^{-3}$]')    
            
    return


def __plot_surfaces_and_psi(axs, R, Z, psi, StixX, StixY):
    
    axsurf = axs[0, 4]
    c_surf = axsurf.contour(R, Z, psi, 20, cmap='bone')
    plt.colorbar(c_surf, ax=axsurf)  
    axsurf.contour(R, Z, psi, [1.], colors='k', linewidths=1.5)
    axsurf.set_title(r'flux surfaces')    
 
    axpsi2d = axs[1, 4]
    c_psi2d = axpsi2d.contourf(R, Z, psi, 100, cmap='OrRd')
    plt.colorbar(c_psi2d, ax=axpsi2d)    
    axpsi2d.set_xlabel(r'$R$ [m]', fontsize=15)
    axpsi2d.set_title(r'$\psi$')
    
    plotting_functions.add_cyclotron_resonances(R, Z, StixY, axpsi2d)
    plotting_functions.add_Omode_cutoff(R, Z, StixX, axpsi2d)
    plotting_functions.add_Xmode_cutoff(R, Z, StixX, StixY, axpsi2d)
    plotting_functions.add_UHresonance(R, Z, StixX, StixY, axpsi2d)

    return


def __plot_electron_temperature(axs, iaxs, jaxs, R, Z, Te, StixX, StixY):

    axTe = axs[iaxs, jaxs]
    c_Te = axTe.contourf(R, Z, Te, 100, cmap='OrRd')
    plt.colorbar(c_Te, ax=axTe)
    axTe.set_xlabel(r'$R$ [m]', fontsize=15)
    axTe.set_title(r'$T_{\mathrm{e}}$ [keV]')

    return


###############
# Main function
def plot_eq(configfile, nptR=500, nptZ=500, nptPsi=50):
    
    """
    Plot all the equilibrium quantities, namely,
    the three components of the magnetic field, the poloidal field, as 
    well as the flux function psi, density and temperature profiles.
    
    For the case of tokamak equilibria, the density and temperature profiles
    are plotted in both the two-dimensional poloidal section and versus 
    the flux function psi.

    All plots are obtained with a grid different from the equilibrium grid
    in order to test the interpolation procedures.
    """
    
    # Load the input data fro the ray tracing configuration file
    from CommonModules.input_data import InputData
    idata = InputData(configfile)

    # Load the equilibrium, depending on the type of device
    if idata.equilibrium == 'Tokamak':
        from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
        Eq = TokamakEquilibrium(idata)
    elif idata.equilibrium == 'Tokamak2D':
        from CommonModules.PlasmaEquilibrium import TokamakEquilibrium2
        Eq = TokamakEquilibrium2(idata)
    elif idata.equilibrium == 'Axisymmetric':
        from CommonModules.PlasmaEquilibrium import AxisymmetricEquilibrium
        Eq = AxisymmetricEquilibrium(idata)
    elif idata.equilibrium == 'Model':
        from CommonModules.PlasmaEquilibrium import ModelEquilibrium
        Eq = ModelEquilibrium(idata)
    else:
        msg = "Keyword 'equilibrium' not recognized."
        raise ValueError(msg)    

    # Build the grid in the two-dimensional poloidal plane
    Rmin = Eq.Rgrid[0,0] 
    Rmax = Eq.Rgrid[-1,0] 
    Zmin = Eq.zgrid[0,0] 
    Zmax = Eq.zgrid[0,-1]

    R = np.linspace(Rmin, Rmax, nptR)
    Z = np.linspace(Zmin, Zmax, nptZ)

    Rm = R / 100. # converted from cm to m
    Zm = Z / 100. # converted from cm to m
    
    # Cyclotron and plasma frequency
    StixX, StixY, field_and_density = StixParamSample(R, Z, Eq, idata.freq)
    Bt2d, BR2d, Bz2d, Ne2d = field_and_density

    # Poloidal field
    Bp2d = np.sqrt(BR2d**2 + Bz2d**2)

    # The plots depend on the considered case
    if idata.equilibrium == 'Tokamak' or idata.equilibrium == 'Tokamak2D':

        # Print some relevant numbers
        Raxis, Zaxis = Eq.magn_axis_coord_Rz
        BtOnAxis = Eq.BtInt.eval(Raxis, Zaxis)
        print('\nMagnetic axis at R = {} cm, Z = {} cm'.format(Raxis, Zaxis))
        print('Toroidal B field on axis = {} Tesla\n'.format(round(BtOnAxis,2)))
        
        # Setup the figure
        nrows = 2; ncols = 5
        fig, axs = __setup_axes(nrows, ncols)
                
        # Magnetic field components
        __plot_B(axs, Rm, Zm, [BR2d, Bz2d, Bp2d, Bt2d])

        # Stix parameters (labels for the legends are handled here!)
        contours = __plot_Stix_parameters(axs, Rm, Zm, StixX, StixY)
        h1, h2, h3, O_cutoff, X_cutoff, UH_res = contours
        
        # Density
        iaxs = 0; jaxs = 3
        __plot_electron_density(axs, iaxs, jaxs, Rm, Zm, Ne2d, StixX, StixY)
        
        # Temperature
        iaxs = 1; jaxs = 3
        Te2d = IntSample(R, Z, Eq.TeInt.eval)  
        __plot_electron_temperature(axs, iaxs, jaxs, Rm, Zm, Te2d, StixX, StixY)

        # Surfaces and psi
        psi2d = IntSample(R, Z, Eq.PsiInt.eval) 
        __plot_surfaces_and_psi(axs, Rm, Zm, psi2d, StixX, StixY)
        
    elif idata.equilibrium == 'Axisymmetric':
        
        nrows = 2; ncols = 3
        fig, axs = __setup_axes(nrows, ncols)
                
        # Magnetic field components
        __plot_B(axs, Rm, Zm, [BR2d, Bz2d, Bp2d, Bt2d])

        # Stix parameters (labels for the legends are handled here!)
        contours = __plot_Stix_parameters(axs, Rm, Zm, StixX, StixY)
        h1, h2, h3, O_cutoff, X_cutoff, UH_res = contours

    else:
        raise RuntimeError('The equilibrium flag does not seem correct.')

    # Adding the legends for resonances and cutoffs
    h1_handles, h1_labels = h1.legend_elements()
    h2_handles, h2_labels = h2.legend_elements()
    h3_handles, h3_labels = h3.legend_elements()
    
    O_cutoff_handles, O_cutoff_labels = O_cutoff.legend_elements()
    X_cutoff_handles, X_cutoff_labels = X_cutoff.legend_elements()
    UH_res_handles, UH_res_labels = UH_res.legend_elements()

    handles = h1_handles
    handles += h2_handles
    handles += h3_handles
    handles += O_cutoff_handles
    handles += X_cutoff_handles
    handles += UH_res_handles

    labels = ['first harm.', 'second harm.', 'third harm.',
              'O-mode cutoff', 'X-mode cutoff',
              'perp. UH res.']

    legend = fig.legend(handles, labels,
                        loc=3, bbox_to_anchor=(0.05, 0.9, 0.9, 0.1),
                        ncol=3, mode='expand', borderaxespad=0.0)


    # Plot horizontal and vertical cuts of the 2d density map through either
    #  the magnetic axis (Tomakaks) or the geometric center of the device
    # (generic axysymmetric devices)
    Ra, Za = Eq.magn_axis_coord_Rz
    Rcut = np.linspace(Ra, Ra + 2.0*Eq.rmin, 2000) 
    Zcut = np.linspace(Za, Za + 2.0*Eq.rmin, 2000)  
    #
    ne_Rcut = IntSample(Rcut, np.array([Za]), Eq.NeInt.eval)
    Te_Rcut = IntSample(Rcut, np.array([Za]), Eq.TeInt.eval)
    #    
    ne_Zcut = IntSample(np.array([Ra]), Zcut, Eq.NeInt.eval)
    Te_Zcut = IntSample(np.array([Ra]), Zcut, Eq.TeInt.eval)
    #
    fig2 = plt.figure(2, figsize=(8.0, 4.8)) 
    ax21 = fig2.add_subplot(121)
    ax21.plot(Rcut, ne_Rcut[0,:], 'b-')
    ax21.set_title(r'$Z=Z_{\mathrm{axis}}$ outer-equatorial cut')
    ax21.set_xlabel(r'$R$ [cm]', fontsize=15)
    ax21.set_ylabel(r'$n_{\mathrm{e}}$ [$10^{13}$ cm$^{-3}$]',   
                    color='b', fontsize=15)
    for tl in ax21.get_yticklabels():
        tl.set_color('b')
    ax22 = ax21.twinx()
    ax22.plot(Rcut, Te_Rcut[0,:], 'r-')
    ax22.set_ylabel(r'$T_{\mathrm{e}}$ [keV]', 
                    color='r', fontsize=15)
    for tl in ax22.get_yticklabels():
        tl.set_color('r')
    #
    ax23 = fig2.add_subplot(122)
    ax23.plot(Zcut, ne_Zcut[:,0], 'b-')
    ax23.set_title(r'$R=R_{\mathrm{axis}}$ upward-vertical cut')
    ax23.set_xlabel(r'$Z$ [cm]', fontsize=15)
    ax23.set_ylabel(r'$n_{\mathrm{e}}$ [$10^{13}$ cm$^{-3}$]',   
                    color='b', fontsize=15)
    for tl in ax23.get_yticklabels():
        tl.set_color('b')
    ax24 = ax23.twinx()
    ax24.plot(Zcut, Te_Zcut[:,0], 'r-')
    ax24.set_ylabel(r'$T_{\mathrm{e}}$ [keV]', 
                    color='r', fontsize=15)
    for tl in ax24.get_yticklabels():
        tl.set_color('r')
    #
    fig2.tight_layout()
        
    
    # For tokamaks also plot the original density and temperature profiles
    if idata.equilibrium == 'Tokamak' and idata.analytical_tokamak == 'No':

        # Original data
        psi_ne = Eq.psi_profile_ne
        psi_Te = Eq.psi_profile_Te
        rho_ne = np.sqrt(psi_ne)
        rho_Te = np.sqrt(psi_Te)
        #
        ne_data = Eq.ne_profile
        Te_data = Eq.Te_profile

        # Interpolated
        psi_max = max(np.max(psi_ne), np.max(psi_Te))
        psi_sample = np.linspace(0.0, psi_max, 5000)
        #
        rho_sample = np.sqrt(psi_sample)
        #
        Ne_sample = Eq.NeInt.__profile__(psi_sample, nu=0)
        Te_sample = Eq.TeInt.__profile__(psi_sample, nu=0)

        # Derivatives of the profile
        derNe = Eq.NeInt.__profile__(psi_sample, nu=1)
        derTe = Eq.TeInt.__profile__(psi_sample, nu=1)

        # Plotting
        # ... prifiles and original data ...
        fig3 = plt.figure(3)
        ax31 = fig3.add_subplot(111)
        ax31.plot(rho_sample, Ne_sample, 'b-')
        ax31.plot(rho_ne, ne_data, 'bx')
        ax31.set_xlabel(r'$\rho$', fontsize=15)
        ax31.set_ylabel(r'$n_{\mathrm{e}}$ [$10^{13}$ cm$^{-3}$]',   
                        color='b', fontsize=15)
        for tl in ax31.get_yticklabels():
            tl.set_color('b')
        ax32 = ax31.twinx()
        ax32.plot(rho_sample, Te_sample, 'r-')
        ax32.plot(rho_Te, Te_data, 'rx')
        ax32.set_ylabel(r'$T_{\mathrm{e}}$ [keV]', 
                       color='r', fontsize=15)
        for tl in ax32.get_yticklabels():
            tl.set_color('r')
        fig3.tight_layout()            
        # ... derivative of profiles with respect to rho ...
        fig4 = plt.figure(4)
        ax41 = fig4.add_subplot(111)
        ax41.plot(rho_sample, derNe, 'b-')
        ax41.set_xlabel(r'$\rho$', fontsize=15)
        ax41.set_ylabel(r'$dn_{\mathrm{e}}/d\psi$ [$10^{13}$ cm$^{-3}$]',   
                        color='b', fontsize=15)
        for tl in ax41.get_yticklabels():
            tl.set_color('b')
        ax42 = ax41.twinx()
        ax42.plot(rho_sample, derTe, 'r-')
        ax42.set_ylabel(r'$dT_{\mathrm{e}}/d\psi$ [keV]', 
                        color='r', fontsize=15)
        for tl in ax42.get_yticklabels():
            tl.set_color('r')
        fig4.tight_layout()
        
    plt.show()

    # return
    pass

#
# END OF FILE    
