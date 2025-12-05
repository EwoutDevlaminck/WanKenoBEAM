"""
Three-dimensional visualization of the beam and, if available the associated
energy flux, via mayavi2.

The calculation can either be run from WKBeam directly

 $ python WkBeam.py beam3d <binning_file>

where the binning file must correspond to a binning  in the three spatial
coordinates ['X', 'Y', 'Z'], or equivalently ['R', Y', 'Z'].

If the information of the wave energy flux is available in the data set,
streamlines of the energy flux and normal component on a surface (if given)
are also displayed and in the command 'WKBeam.py flux'
"""


# Import statements
import numpy as np
import scipy.interpolate as spl
import mayavi.mlab as mlab

# Local modules
from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium, IntSample
from Tools.PlotData.PlotBinnedData.EnergyFlux import get_binned_data
import Tools.PlotData.PlotBinnedData.grids as grids
import Tools.PlotData.PlotBinnedData.EnergyFlux3d_computations as EFcomp

# Main function called by WKBeam.py
def plot_beam_with_mayavi(filename):
    
    """
    Representation of the beam in three-dimensions with mayavi.
    This function is meant to be called directly by WKBeam.py.
    
    The function just load the data and call a lower level function
    for the processing of the data and the actual plotting directives.
    """
    
    # Initial message
    print("\n Three-dimensional representation of the beam ...\n")
    
    # Load the input data in the usual way with the idata object
    idata = InputData(filename)
    hdf5data, wtr, EnergyFluxAvailable, EnergyFluxValid = get_binned_data(idata)

    # Equilibrium flag
    if hasattr(idata, 'plotequilibrium'):
        plotequilibrium = idata.plotequilibrium
    else:
        plotequilibrium = True
    
    # Get the relevant raytracing input file
    rt_idata = InputData(idata.raytracing_input)

    # ... skip a few data to avoid loading a busy plot (for the energy flux) ...
    try:
        skip = idata.skip
    except AttributeError:
        skip = None

    # ... now plot ...
    process_data_and_plot(rt_idata, hdf5data, plotequilibrium=plotequilibrium,
                          skip=skip)

    return None

def process_data_and_plot (rt_idata, hdf5data, plotequilibrium=False, skip=None):
    
    """
    This function processes the input data as appropriate for the 
    required data visualization and generates the plot within a
    mayavi.mlab scene. 
    
    The function can be called by WKBeam, or directly by __main__    
    when the module is executed as a script.
    
    Call sequence: 
    """
    
    # Extract parameters
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

    # Get the coordinates from the dataset
    x1, dx1 = grids.get_grid(hdf5data, 'X')
    x2, dx2 = grids.get_grid(hdf5data, 'Y')
    x3, dx3 = grids.get_grid(hdf5data, 'Z')
    Xgrid = np.array(np.meshgrid(x1, x2, x3, indexing='ij'))

    # Get the wave field energy density
    field_energy = hdf5data.get('BinnedTraces')[...,0]
    beamcontours = [0.1*np.max(field_energy), 0.01*np.max(field_energy)]

    # Build a representation of the antenna
    Xant, Yant, Zant = grids.build_antenna_plane(x0, polangle, torangle, w1, w2)

    # Plotting function
    fig = mlab.figure(size=(1300,1600))
    mlab.contour3d(Xgrid[0], Xgrid[1], Xgrid[2], field_energy, 
                   contours=beamcontours, colormap='autumn', 
                   opacity=0.2, transparent=True)
    mlab.axes()
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

    # If it is available in the dataset, add the fieldlines of the energy flux
    if skip is None:
        skip = 1

    try:
        fluxVector = EFcomp.get_flux(hdf5data)
        Xgrid, Ffield = grids.build_grid_and_vector_field(fluxVector, skip)
        streamlines = mlab.flow(Xgrid[0], Xgrid[1], Xgrid[2], 
                                Ffield[0], Ffield[1], Ffield[2],
                                seedtype='plane', linetype='line',
                                color=(0.0,0.0,0.0))

        streamlines.stream_tracer.integrator_type = 'runge_kutta45'
        streamlines.stream_tracer.maximum_error = 1.e-10
        streamlines.stream_tracer.maximum_propagation = 1000.0
        streamlines.stream_tracer.maximum_number_of_steps = 100000
        streamlines.stream_tracer.integration_direction = 'forward'
        streamlines.actor.property.line_width = 6.0
        seed_origin = np.array([Xant[-1,-1], Yant[-1,-1], Zant[-1,-1]])
        seed_point1 = np.array([Xant[0,-1], Yant[0,-1], Zant[0,-1]])
        seed_point2 = np.array([Xant[-1,0], Yant[-1,0], Zant[-1,0]])

        streamlines.seed.widget.resolution = 10
        streamlines.seed.widget.origin = seed_origin
        streamlines.seed.widget.point1 = seed_point1
        streamlines.seed.widget.point2 = seed_point2
#        streamlines.seed.widget.enabled = False

    except RuntimeError:
        pass

    mlab.show()

    return None


# -------------------------------------------------------------------------------
# The following functions are adapted from the plotting module of bt_visual,
# a set of visualizations tools for TORBEAM

# A quick and dirty interpolation of the poloidal magnetic flux
def interp_poloidal2lab(xgrid, ygrid, zgrid, R1d, Z1d, data):

    """
    This interpolates an axisymmetric scalar quantity given on a 3d mesh 
    constituted by poloidal surfaces, to a 3d rectangular mesh.
    
    Usage:
          X, Y, Z, int_data = interp_poloidal2lab(xgrid, ygrid, zgrid, 
                                                  R1d, Z1d, data)
          
    Input:
          > xgrid, ygrid, zgrid, 1d arrays with grid points in x, y, and z,
            respectively. The 3d grid is obtained by Cartesian product of 
            these three 1d grids.
          > R1d, Z1d, 1d arrays with grid points in R and z on a poloidal 
            section. Since the scalar is assumed to be axisymmetric, there
            is no need to provide the grid in the toroidal angle.
          > data, list of arrays of the form F[j,k] for F evaluated
            at the grid point (R_j, Z_k).
          
    Output:
          > X, Y, Z 3d arrays with grid points on the 3d grid. 
          > int_data, list of arrays of the form F[i,j,k] of interpolated 
            values on the 3d grid (x_i, y_j, z_k).

    Important remark: the 3D Cartesian mesh might cover a larger volume 
    than the original equilibrium grid in cylindrical coodinates. For
    instance, if one wants to cover a toroidal region of space with a 
    Cartesian grid, the corners of the Cartesian grid exceed the given
    toroidal region. In order to cope with this case, the equilibrium 
    data are extended outside the equilibrium grid by selecting the
    value at the nearest neighbor point in the equilibrium grid. 
    """

    # Extract the number of data to interpolate
    Ndata = len(data)

    # Extract the number of points in R and z and steps
    nptR = np.size(R1d)
    nptZ = np.size(Z1d)
    dR = R1d[1] - R1d[0]
    dZ = Z1d[1] - Z1d[0]
    
    # Extract number of grid points in x, y, and z
    nx = np.size(xgrid)
    ny = np.size(ygrid)
    nz = np.size(zgrid)

    # Define the array for the 3D grid
    X = np.empty([nx, ny, nz]) 
    Y = np.empty([nx, ny, nz]) 
    Z = np.empty([nx, ny, nz])

    # Find by how many points the equilibrium grid should be enlarged
    Rgrid = np.sqrt(xgrid**2 + ygrid**2)
    Rmin = Rgrid.min()
    Rmax = Rgrid.max()
    Zmin = zgrid[0]
    Zmax = zgrid[-1]
    # ... radial points on the right (large R) ...
    NRr = max(int((Rmax - R1d[-1])/dR) + 1, 0)
    # ... radial points on the left (small R) ...
    NRl =  max(int((R1d[0] - Rmin)/dR) + 1, 0)
    # ... vertical points on the top of the tokamak (large z) ...
    NZt =  max(int((Zmax - Z1d[-1])/dZ) + 1, 0)
    # ... vertical points on the bottom of the tokamak (small z) ...
    NZb =  max(int((Z1d[0] - Zmin)/dZ) + 1, 0)
    # ... final grid size ...
    NRtot = NRl + nptR + NRr
    NZtot = NZb + nptZ + NZt

    # Extend the grid
    Rx = np.empty([NRtot])
    Zx = np.empty([NZtot])
    # ... load the bulk of the grid ...
    Rx[NRl:(NRl+nptR)] = R1d[:]
    Zx[NZb:(NZb+nptZ)] = Z1d[:]
    # ... load the left block ...
    for iR in range(0, NRl):
        index = NRl - iR - 1
        Rx[index] = R1d[0] - (iR + 1) * dR
    # ... load the right block ...
    for iR in range(0, NRr):
        index = NRl + nptR + iR
        Rx[index] = R1d[nptR-1] + (iR + 1) * dR        
    # ... load the bottom block ...
    for iZ in range(0, NZb):
        index = NZb - iZ - 1
        Zx[index] = Z1d[0] - (iZ + 1) * dZ
    # ... load the top block ...
    for iZ in range(0, NZt):
        index = NZb + nptZ + iZ
        Zx[index] = Z1d[nptZ-1] + (iZ + 1) * dZ

    # Extend data to a larger grid and interpolate
    n1 = NRl + nptR
    n2 = NZb + nptZ
    int_data = []
    for i in range(0, Ndata):
        item = data[i]
        # ... create a temporary array for extended data set ...
        xd = np.empty([NRtot, NZtot])
        # ... load the bulk of the grid ...
        xd[NRl:n1, NZb:n2] = item[:,:]
        # ... load the corners of the extended rectangular grid ...
        xd[0:NRl, 0:NZb] = item[0, 0]
        xd[n1:NRtot, 0:NZb] = item[nptR-1, 0]
        xd[n1:NRtot, n2:NZtot] = item[nptR-1, nptZ-1]
        xd[0:NRl, n2:NZtot] = item[0, nptZ-1] 
        # ... load the edges of the extended rectangular grid ...
        for iz in range(0, nptZ):
            index = NZb + iz
            xd[0:NRl, index] = item[0, iz]
            xd[n1:NRtot, index] = item[nptR-1, iz]
        for iR in range(0, nptR):
            index = NRl + iR
            xd[index, 0:NZb] = item[iR, 0]
            xd[index, n2:NZtot] = item[iR, nptZ-1]
        # ... interpolation object ...
        int_xd = spl.RectBivariateSpline(Rx, Zx, xd)
        # ... create a temporary array for interpolated data ...
        F = np.empty([nx, ny, nz])        
        # ... load the array ...
        for ixgrid in range(0, nx):
            # ... current x of the 3d grid ...
            xloc = xgrid[ixgrid]
            for iygrid in range(0, ny):
                # ... current y of the 3d grid ...
                yloc = ygrid[iygrid]
                # ... find the major radius coordinate ...
                Rloc = np.sqrt(xloc*xloc + yloc*yloc)
                for izgrid in range(0, nz):
                    # ... current z of the 3d grid ...
                    zloc = zgrid[izgrid]
                    # ... call the interpolation objects ...
                    X[ixgrid,iygrid,izgrid] = xloc
                    Y[ixgrid,iygrid,izgrid] = yloc
                    Z[ixgrid,iygrid,izgrid] = zloc
                    F[ixgrid,iygrid,izgrid] = int_xd(Rloc, zloc)
                # ... end for izgrid ...
            # ... end for iygrid ...
        # ... end for ixgrid ...
        # ... add the result to the final list ...
        int_data.append(F)

    # Return data
    return X, Y, Z, int_data

# Create standard Cartesian 3D grids covering the tokamak volume
def equilibrium_3Dgrid(R, z, nptxy, nptz, section='all'):

    """
    Given the 2D array R and z for the radial and vertical
    coordinates of the equilibrium grid, this defines 1D arrays
    xgrid, ygrid, zgrid covering standard section of the torus
    depending on the flag section.

    Usage:
      xgrid, ygrid, zgrid = equilibrium_3Dgrid(R, z, nptxy, nptz, section)

    where
       > R and z are 2d arrays obtained from the function read_topfile of 
         the module bt_def_equilibrium.
       > nptxy, number of points in both x and y, used to build a regular
         rectangular grid in the x-y plane.
       > nptz, number of grid points in the vertical coordinate.
       > section must be one of the following: 'all' (full torus), 
         'half' (half torus), 'quarter' (a quarter of the torus), or 
         'rotatedquarter' (another quarter of the torus).
    """
    
    # Define the Cartesian mesh
    Rmax = R.max()
    zmax = z.max()
    if section == 'all':
        xgrid = np.linspace(- Rmax, + Rmax, nptxy)
        ygrid = np.linspace(- Rmax, + Rmax, nptxy)
    elif section == 'half':
        xgrid = np.linspace(- Rmax, + Rmax, nptxy)
        ygrid = np.linspace(- Rmax, - 0.1, nptxy)
    elif section == 'quarter':
        xgrid = np.linspace(- Rmax, - 0.1, nptxy)
        ygrid = np.linspace(- Rmax, - 0.1, nptxy)
    elif section == 'rotatedquarter':
        xgrid = np.linspace(0.1, Rmax, nptxy)
        ygrid = np.linspace(0.1, Rmax, nptxy)
    elif section == 'aroundzero':
        xgrid = np.linspace(0.1, + Rmax, nptxy)
        ygrid = np.linspace(- Rmax*0.71, + Rmax*0.71, nptxy)
    else:
        raise ValueError('section flag not recognized.')
    zgrid = np.linspace(- zmax, + zmax, nptz)

    # Return the grid
    return xgrid, ygrid, zgrid


# Plot 3d contours of the poloidal magnetic flux in a tokamak
def plot_equilibrium3D(R, z, xgrid, ygrid, zgrid, Psi, contours, ml,
                       Bfield=None, seed=None,
                       colormap='black-white', opacity=0.5, psimax=1.):

    """ 
    Contour plot 3D for the poloidal magnetic flux in a 
    tokamak. This should represent the magnetic equilibrium of a tokamak.
    
    Usage:
       plot_equilibrium3D(R, z, xgrid, ygrid, zgrid, Psi, contours, ml, 
                          Bfield=None, seed=None, opacity=0.5)
    where
       > R and z are 2d arrays obtained from the function read_topfile of 
         the module bt_def_equilibrium.    
       > xgrid, ygrid, zgrid are 1d arrays specifying the x, y, and z
         coordinates of the nodes of a regular Cartesian grid in 3D, 
         covering a subdomain of the equilibrium domain.
       > Psi is a 2d array with the flux function psi on the grid (R, z),
         as obtained from the function read_topfile of the module
         bt_def_equilibrium.
       > contours, 1d arrays of contours to be plotted.
       > ml, import name of the mlab module.
       > Bfield, optional magnetic field components coolocated on the 
         numerical grid as obtained by the read_topfile function.
       > seed, optional array of the form [x, y, z] with the 
         Cartesian coordinates of an initial point for field line tracing.
       > opacity, float in the interval [0., 1.] that controls the opacity
         of contours (default = 0.3).
       > psimax, float, maximum value of psi to be considered (default=1.0).

    When Bfield is passed, the behavior of the procedure is overridden: 
    instead of the magnetic surfaces, a magnetic field line is plotted.
    """

    # Extract 1d grid in R and z
    R1d = R[:, 0]
    z1d = z[0, :]

    # Evaluate the poloidal flux at grid points
    try:
        data = [Psi[:,:], Bfield[0,:,:], Bfield[1,:,:], Bfield[2,:,:]]    
    except:
        data = [Psi]
    X, Y, Z, int_data = interp_poloidal2lab(xgrid, ygrid, zgrid, R1d, z1d, data)
    # Graphics instructions
    # (Note that the presente of B overrides the behaviour of the function.)
    if Bfield != None:
        PSI, BR, Bz, Bt = int_data
        R = np.sqrt(X**2 + Y**2)
        Bx = (BR * X - Bt * Y) / R
        By = (BR * Y + Bt * X) / R

        # Copied from the magnetic field line example on
        # http://docs.enthought.com
        field = ml.pipeline.vector_field(X, Y, Z, Bx, By, Bz)
        # (The above call makes a copy of the arrays, so we delete
        # this copy from free memory.)
        del Bx, By, Bz
        magnitude = ml.pipeline.extract_vector_norm(field)
        if seed != None:
            field_line = ml.pipeline.streamline(magnitude, 
                                                seedtype='point',
                                                colormap=colormap,
                                                integration_direction='both')
            field_line.seed.widget.position = seed
            field_line.seed.widget.enabled = False
            field_line.stream_tracer.integrator_type = 'runge_kutta45'
            field_line.stream_tracer.maximum_propagation = 50000.
            field_line.streamline_type = 'tube'
            field_line.tube_filter.radius = 3. ###1.5
    else:
        PSI = int_data[0]
        ml.contour3d(X, Y, Z, PSI, 
                     contours=contours, 
                     transparent=True,
                     colormap=colormap,
                     opacity=opacity, vmax=psimax)

    return None


# If the module is called as a script plot the arguments
if __name__ == '__main__':
    
    import sys
    import h5py

    help_message = \
    """ Usage:
      $ python beam3d.py 3D_binnedfile, raytracing_input_file, eq_flag
    
    where 3D_binnedfile is the WKBeam output fo binning in the variables
    X,Y,Z (full three dimensional physical domain) and raytracing_input_file
    is the corresponding input file for the ray tracing calculation.
    At last, eq_flag is one of the following:
     1. ploteq - plot the magnetic surface of the equilibrium.
     2. noeq - do not plot the magnetic surfaces.
        """    


    try:
        binnedfile = sys.argv[1]
        raytracing_input = sys.argv[2]
        eq_flag = sys.argv[3]
        if eq_flag == 'ploteq':
            ploteq = True
        elif eq_flag == 'noeq':
            ploteq = False
        else: 
            print('Equilibrium flag not understood\n')
            print(help_message)
            raise
    
        hdf5data = h5py.File(binnedfile, 'r')          
        rt_idata = InputData(raytracing_input) 
        process_data_and_plot(rt_idata, hdf5data, plotequilibrium=ploteq)

    except:
        print(help_message)
        raise
    
# end of file
