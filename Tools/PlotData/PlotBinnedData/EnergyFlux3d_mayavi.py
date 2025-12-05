"""
This module provides functions to visualize the beam amplitude and the 
average wave energy flux in three-dimensional plot using mayavi as a 
graphic backend. For the calculations it relies on the procedures defined in
EnergyFlux3d.py.
"""

# Standard modules
import numpy as np

# Local modules
import Tools.PlotData.PlotBinnedData.grids as grids
import Tools.PlotData.PlotBinnedData.EnergyFlux3d_computations as EFcomp
from Tools.PlotData.PlotBinnedData.Beam3D import *


# This is the main procedure called by WKBeam
def Data_for_flux_and_beam_in_3d(idata, hdf5data, surface_model):

    """
    Takes data from the WKBeam run (input data 'idata', hdf5 dataset 'hdf5data',
    and the surface model 'surface') and construct an instance of the 
    named tuple FluxVector3D for the energy flux and Surface3D for the surface in 
    three dimentions. The named tuples are then passed to the relevant procedure
    to compute the energy flux through the surface and the normal component of 
    the energy flux vector on the surface.
    """

    # Number of points to jump (t avoid dense quiver plots
    if hasattr(idata, 'skip'):
        skip = idata.skip
    else:
        skip = 3

    # Prepare the data
    fluxData = EFcomp.load_energy_flux_and_surface(hdf5data, surface_model)

    # Prepare the energy flux
    field, flux, fdata = fluxData
    u, v, X, Y, Z, FnJ, Fn = fdata
    Xgrid, Ffield = grids.build_grid_and_vector_field(field, skip)

    # Build a representation of the antenna plane
    # ... launching point ...
    x0 = np.array([
        hdf5data.get('rayStartX')[()],
        hdf5data.get('rayStartY')[()],
        hdf5data.get('rayStartZ')[()],
    ])
    # ... antenna orientation (angles are stored in radiants and not 
    # need to specify the convention for the angle definitions) ...
    polangle = hdf5data.get('antennapolangle')[()]
    torangle = hdf5data.get('antennatorangle')[()] 
    # ... beam widths
    w1 = hdf5data.get('beamwidth1')[()]
    w2 = hdf5data.get('beamwidth1')[()]
    Xant, Yant, Zant = grids.build_antenna_plane(x0, polangle, torangle, w1, w2)

    # Pack the data and return
    flux_field = (Xgrid[0], Xgrid[1], Xgrid[2], Ffield[0], Ffield[1], Ffield[2])
    antenna =  (Xant, Yant, Zant)
    return flux, fdata, flux_field, antenna


# This is the main procedure called by WKBeam
def flux_and_beam_in_3d(idata, hdf5data, surface_model):

    """
    Takes data from the procedure Data_for_flux_and_beam_in_3d(...) in this
    module and visualize with mayavi.
    """

    # This procedure requeres mayavi
    from mayavi import mlab

    # Prepare the data
    data = Data_for_flux_and_beam_in_3d(idata, hdf5data, surface_model)
        
    # Colormap for surfaces (not every installation have all colormaps)
    if hasattr(idata, 'colormap'):
        colormap = idata.colormap
    else:
        colormap = 'coolwarm'
        
    # Equilibrium flag
    if hasattr(idata, 'plotequilibrium'):
        plotequilibrium = idata.plotequilibrium
    else:
        plotequilibrium = True

    # Get the relevant raytracing input file
    rt_idata = InputData(idata.raytracing_input)

    # Extract data
    flux, fdata, flux_field, antenna = data
    X0, X1, X2, Ffield0, Ffield1, Ffield2 = flux_field
    Xant, Yant, Zant = antenna
    X = fdata[2]
    Y = fdata[3]
    Z = fdata[4]
    Fn = fdata[6]

    # Initialize mayavi figure
    fig = mlab.figure(size=(1300,1300))  

    # Plot the energy flux vector field
    mlab.quiver3d(X0, X1, X2, Ffield0, Ffield1, Ffield2, mode='arrow')
    
    # Plot the surface
    surf = mlab.mesh(X, Y, Z, scalars=Fn, colormap=colormap)
    mlab.colorbar(surf)

    # Set symmetric color range
    vmax = np.abs(Fn).max()
    surf.module_manager.scalar_lut_manager.data_range = (-vmax, vmax)


    # plot the antenna plane
    mlab.mesh(Xant, Yant, Zant, color=(0.5,0.5,0.5))

    # Extract the magnetic field equilibrium and build data    
    if plotequilibrium:

        # ... load equilibrium ...
        Eq = TokamakEquilibrium(rt_idata)
        Req = Eq.Rgrid
        Zeq = Eq.zgrid
        psi = IntSample(Req[:,0], Zeq[0,:], Eq.PsiInt.eval).T
        psisep = 1.0 # psi is always normalized in the euilibrium object


        nxy = 50
        nz = 100
        psicontours = [psisep]

        xEq, yEq, zEq = equilibrium_3Dgrid(Req, Zeq, nxy, nz, 
                                           section='aroundzero')

        plot_equilibrium3D(Req, Zeq, xEq, yEq, zEq, psi, psicontours, mlab,
                           colormap='RdYlBu', opacity = 0.3)    

    
    # Display the figure
    mlab.show()

    return flux, fdata, flux_field, antenna

