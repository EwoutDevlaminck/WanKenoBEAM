"""
Computational engines for EnergyFlux3d_mpl and EnergyFlux3d_mayavi.
"""

# Import statements
import collections
import numpy as np
import scipy.interpolate as Interp
import scipy.integrate as Integrate

# Local modules
from Tools.PlotData.PlotBinnedData.grids import get_grid



# -------------------------------------------------------------------------------
# Data structures



# Definition of a namedtuple spacifying parametric surfaces and fluxes
Surface3D = collections.namedtuple('Surface3D', ['phi', 'phi_u', 'phi_v', 
                                                 'umin', 'umax', 'nptu',
                                                 'vmin', 'vmax', 'nptv'])
FluxVector3D = collections.namedtuple('FluxVector3D', ['x1', 'x2', 'x3', 'F'])



# -------------------------------------------------------------------------------
# Procedure to load data from WKBeam results


# Preprocess the bema

# Preprocess the data for the energy flux
def load_energy_flux_and_surface(hdf5data, surface_model):
    
    """
    Extract the information on the energy flux and compute
    the total power through the considered surface in the 
    three-dimensional physical space.
    """
    
    field = get_flux(hdf5data)
    surf = get_surface(surface_model)

    # Compute the normal compenent of F.n J
    flux, data = compute_flux(field, surf)
    Fn = np.ma.masked_invalid(data[-1])

    # Compute the flux 
    print('\n Computed power through the surface = {}'.format(flux))
    print('\n Maximum of the normal flux = {}\n'.format(np.max(np.abs(Fn))))

    return field, flux, data


# Construct data needed to compute and plot the energy flux
def get_flux(hdf5data):
    
    """
    Build an instance of the named tuple for the energy flux field and
    the considered surface model.
    """

    # Check the velocity field stored (cut the last element which is empty)
    # (The first exception is for backward compatibility)
    try:
        Vfield = hdf5data.get('VelocityFieldStored').asstr()[()].split(',')[0:3]
    except AttributeError:
        Vfield = hdf5data.get('VelocityFieldStored')[()].split(',')[0:3]
    except:
        msg = """Dataset does not appear to have velocity field stored."""
        raise RuntimeError(msg)

    # Physical domain
    # (try loop for backward compatibility)
    try:
        Coordinates = hdf5data.get('WhatToResolve').asstr()[()].split(',')
    except AttributeError:
        Coordinates = hdf5data.get('WhatToResolve')[()].split(',')
        
    x1, dx1 = get_grid(hdf5data, Coordinates[0])
    x2, dx2 = get_grid(hdf5data, Coordinates[1])
    x3, dx3 = get_grid(hdf5data, Coordinates[2])

    # Energy flux Field
    F1 = hdf5data.get('VelocityField')[...,0,0] / dx1 / dx2 / dx3
    F2 = hdf5data.get('VelocityField')[...,1,0] / dx1 / dx2 / dx3
    F3 = hdf5data.get('VelocityField')[...,2,0] / dx1 / dx2 / dx3
    F = np.array([F1,F2,F3])

    # Define the flux vector    
    field = FluxVector3D(x1, x2, x3, F)

    return field

def get_surface(surface_model):

    """
    Build a Surface3D object from the module surface_model.
    """

    # Check the defaults of the surface model
    if hasattr(surface_model, 'umin'):
        umin = surface_model.umin
    else:
        umin=0.0
    if hasattr(surface_model, 'umax'):
        umax = surface_model.umax
    else:
        umax=1.0
    if hasattr(surface_model, 'nptu'):
        nptu = surface_model.nptu
        # Simpson quadrature rule requires an odd number of points (even number
        # of intervals) 
        if nptu % 2 == 0: nptu += 1
    else:
        nptu=101
    if hasattr(surface_model, 'vmin'):
        vmin = surface_model.vmin
    else:
        vmin=0.0
    if hasattr(surface_model, 'vmax'):
        vmax = surface_model.vmax
    else:
        vmax=1.0
    if hasattr(surface_model, 'nptv'):
        nptv = surface_model.nptv
        # Simpson quadrature rule requires an odd number of points (even number
        # of intervals) 
        if nptv % 2 == 0: nptv += 1
    else:
        nptv=101        

    # Define the surface_model
    surf = Surface3D(surface_model.phi, surface_model.phi_u, surface_model.phi_v,
                     surface_model.umin, surface_model.umax, surface_model.nptu, 
                     surface_model.vmin, surface_model.vmax, surface_model.nptv)
    
    return surf
    


# -------------------------------------------------------------------------------
# Energy flux computations



# Auxiliary function which perform most of the field and surface 
# reconstruction needed to compute the energy flux
# Main function which computes fluxes through surfaces
def build_FnJ(field, surface, interpolate_field=True):

    """
    Compute the scalar product of the vector field with the unit normal times
    the Jacobian of the area element on the surface, namely
    
       F .ndS = F.n J du dv = F \cdot (e_u x e_v) dudv,
    
    where (u,v) are the parameter of the surface, n is the unit normal given by
    
       n = (e_u x e_v) / |e_u x e_v|,
    
    and 
    
       dS = J du dv = |e_u x e_v| du dv,
    
    is the area element of the surface.
 
    This function has the same input argument as the procedure compute_flux from 
    which it is called, and returns the tuple
    
       (u, v, X, Y, Z, FnJ, Fn)
    
    where u, v are the one-dimensional grids, X, Y, Z are two-dimenaional arrays
    for the Cartesian coordinates of the points of the surface, and FnJ is the 
    value of the scalar product F.nJ on the point of the surface discretized by 
    a uniform grid in the parameter space. At last, Fn = F.nJ / J is the normal
    component of the flux without the Jacobian.
    """

    # Sample the points on the surface. The surface is given parametrically
    # as a map from the unit square with coordinates (u,v) onto R^3.
    # It is expected that the surface is contained in the domain where the
    # vector field is defined.
    # ... grid generated including points on the boundary ...
    u = np.linspace(surface.umin, surface.umax, surface.nptu)
    v = np.linspace(surface.vmin, surface.vmax, surface.nptv)
    U, V = np.meshgrid(u, v, indexing='ij')
    # ... the three-dimensional interpolation requires flattened array ...
    npt = np.size(U)
    uv = [(U.flatten()[i], V.flatten()[i]) for i in range(npt)]
    xyz = [surface.phi(*uv[i]) for i in range(npt)]

    # Contruction of the interpolant for the vector field
    if interpolate_field:

        grid_points = (field.x1, field.x2, field.x3)
        data = np.rollaxis(field.F, 0, start=4)

        if np.min(np.array(xyz)[:,0]) < field.x1.min():
            print("\nWarning: Surface: x1 lower then minimum.\n")
            print("x1 = {}, min = {}".format(np.min(np.array(xyz)[:,0]),
                                             field.x1.min()))
        if np.max(np.array(xyz)[:,0]) > field.x1.max():
            print("\nWarning: Surface: x1 larger then maximum.")
            print("x1 = {}, max = {}\n".format(np.max(np.array(xyz)[:,0]),
                                               field.x1.max()))
        if np.min(np.array(xyz)[:,1]) < field.x2.min():
            print("\nWarning: Surface: x2 lower then minimum.")
            print("x2 = {}, min = {}\n".format(np.min(np.array(xyz)[:,1]),
                                               field.x2.min()))
        if np.max(np.array(xyz)[:,1]) > field.x2.max():
            print("\nWarning: Surface: x2 larger then maximum")
            print("x2 = {}, max = {}\n".format(np.max(np.array(xyz)[:,1]),
                                               field.x2.max()))
        if np.min(np.array(xyz)[:,2]) < field.x3.min():
            print("\nWarning: Surface: x3 lower then minimum.")
            print("x3 = {}, min = {}\n".format(np.min(np.array(xyz)[:,2]),
                                               field.x3.min()))
        if np.max(np.array(xyz)[:,2]) > field.x3.max():
            print("\nWarning: Surface: x3 larger then maximum.")
            print("x3 = {}, max = {}\n".format(np.max(np.array(xyz)[:,2]),
                                               field.x3.max()))

        fluxV = Interp.RegularGridInterpolator(grid_points, data)
        fluxV_on_surface = fluxV(xyz)

    else:

        fluxV = field
        fluxV_on_surface = [field(xyz[i]) for i in range(npt)]

    # Unit normals and area element:
    # nJ is the normal vector n times the Jacobian J defined so that
    # the element of area is dS = Jdudv
    phi_u = [surface.phi_u(*uv[i]) for i in range(npt)]
    phi_v = [surface.phi_v(*uv[i]) for i in range(npt)]
    nJ = [np.cross(phi_u[i], phi_v[i]) for i in range(npt)]
    J = [np.linalg.norm(nJ[i]) for i in range(npt)]
    integrand = [np.dot(fluxV_on_surface[i], nJ[i]) for i in range(npt)]

    # Reshaping to get the integrand on the surface
    xyz = np.array(xyz).T
    x, y, z  = xyz
    X = np.array(x).reshape(surface.nptu, surface.nptv)
    Y = np.array(y).reshape(surface.nptu, surface.nptv)
    Z = np.array(z).reshape(surface.nptu, surface.nptv)
    FnJ = np.array(integrand).reshape(surface.nptu, surface.nptv)
    J = np.array(J).reshape(surface.nptu, surface.nptv)
    # J = np.ma.masked_equal(np.array(J).reshape(surface.nptu, surface.nptv), 0.0)
    # np.ma.set_fill_value(J, np.nan)

    return u, v, X, Y, Z, FnJ, FnJ/J
    


# Main function which computes fluxes through surfaces
def compute_flux(field, surface, interpolate_field=True):

    """
    Compute the flux of a vector field defined on a regular tri dimensional
    grid through a surface given parametrically.

    USAGE:
    
        r = compute_flux(field, surface, interpolate_field=True)
    
    where for the intended use:
      - field represent the vector field and is an instance on the 
        nemedtuple FluxVector3D,
      - surface represent the surface through which the flux is computed 
        and it is an instance of the namedtuple Surface3D,
    
    Optional arguments:
      - interpolate_field, is a boolean variable. When True one triggers the 
        intended behaviour described above. When False, field is assumed to be
        a callable object which evaluated on the point give the exact vector
        with no need for interpolation. The callable object shoulf be of the 
        form lambda (x,y,z): flux_vector(*(x,y,z))
        This is used for testing only.
    Return argument:
      - r = (computed_flux, list_of_other_data), where list_of_other_data
        is the output of build_FnJ.
    """

    # Compute the vector field F on the surface and the scalar product F.n J
    # where n is the unit normal and J the Jacobian of the area element, i.e., 
    # dS = J dudv.
    u, v, X, Y, Z, FnJ, Fn = build_FnJ(field, surface,
                                       interpolate_field=interpolate_field)

    # Simpson quadrature applied to each dimension
    computed_flux = Integrate.simpson(Integrate.simpson(FnJ, x=v), x=u)

    return computed_flux, [u, v, X, Y, Z, FnJ, Fn]
