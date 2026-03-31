"""This module defines classes for the varius equilibria supported by the code. 
In this classes, basic functions for the magnetic field, density, and 
temperature profiles are defined.

Equilibrium objects are used in the ray tracing code as follows:
(cf. RayTracing.modules.trace_one_ray)

        # define interpolation object for the plasma equilibrium
        if idata.equilibrium == 'Tokamak':

            # for the physical tokamak geometry based on magnetic surfaces
            # (but with the possibility of analytical profiles)
            self.Eq = TokamakEquilibrium(idata)

        elif idata.equilibrium == 'Model':

            # for model Hamiltonians (Helmholtz equation, etc ...)
            self.Eq = ModelEquilibrium(idata)

        elif idata.equilibrium == 'Axisymmetric':

            # for the physical TORPEX-like equilibrium
            self.Eq = AxisymmetricEquilibrium(idata)            

"""

###############################################################################
# IMPORT STATEMENTS
###############################################################################
# standard modules
import math
import numpy as np
# scipy modules
import scipy.interpolate as spl
import scipy.optimize as opt
import scipy.integrate as intgr
# local modules
import CommonModules.BiSplineDer as bispl
import CommonModules.LoadMagneticField as topfile
import CommonModules.LoadDensityField as nefile
import CommonModules.AxisymmetricEquilibrium as axeq
from CommonModules.grid_extender import extend_regular_grid_data as gext


###############################################################################
# SOME CLASSES WHICH ARE SIMILAR TO INTERPOLATION OBJECTS, BUT RETURN
# VALUES USED IN TEST MODELS. THEY CAN BE CHOSEN TO BE USED IN THE INPUT FILE.
###############################################################################

# class for defining density models for analytical tokamaks
# Tokamak-like model with a pedestal controlled by deltarho
class NeModelTokamakLike(object):

    """
    Density model of the form
    
       Ne(psi) =  NE / (1. + exp((rho-1.)/self.deltarho)), rho^2 = psi

    Constructor:
      
       model_object = NeModelTokamakLike(rmaj,rmin,parameters)
    
    where
    
       rmaj = tokamak major radius in cm
       rmin = tokamak minor radius in cm
       parameters = [NE, deltarho, cuteldensAtPsi]
       NE = peak (pedestal) density in 10^13 cm^-3
       deltarho = width of the pedestal in normalized rho = sqrt(psi)
       cueldensAtPsi = cu the profile to a constant for psi < this value
    """

    def __init__(self, rmaj, rmin, parameters):
        self.NE = parameters[0]
        self.deltarho = parameters[1]
        self.cuteldensAtPsi = parameters[2]
        self.rmaj = rmaj
        self.rmin = rmin

    def value(self, PSI):
        psi = max(PSI, self.cuteldensAtPsi)
        Ne = self.NE / (1. + math.exp((math.sqrt(psi)-1.)/self.deltarho))
        return Ne

    def derivative(self, PSI):
        if PSI > self.cuteldensAtPsi:
            rhoNorm = (math.sqrt(PSI) - 1.)/self.deltarho
            dNe_dpsi = -0.5 * self.NE * \
                       math.exp(rhoNorm)/(1.+math.exp(rhoNorm))**2 / \
                       self.deltarho / math.sqrt(PSI)
            return dNe_dpsi
        else:
            return 0.

# class for defining density models for analytical tokamaks
# Linear density ramp-up
class NeModelLinear(object):

    """
    Linear density model of the form
    
            Ne = NE * (X2 - X) / (X2 - X1)

    for X1 < X = rmaj + rmin * rho < X2, extended continuously to a constant.

    Constructor:
      
       model_object = NeModelLinear(rmaj,rmin,parameters)
    
    where
    
       rmaj = tokamak major radius in cm
       rmin = tokamak minor radius in cm
       parameters = [NE, X1, X2]
       NE = peak density in 10^13 cm^-3
       X1 = Cartesian X position of the end of the linear ramp-up
       X2 = Cartesian X position of the beginning of the linear ramp-up
    """

    def __init__(self, rmaj, rmin, parameters):
        self.NE = parameters[0]
        self.X1 = parameters[1] 
        self.X2 = parameters[2]
        self.rmaj = rmaj
        self.rmin = rmin
        # consistency check
        consistent = self.rmaj <= self.X1 < self.X2
        if not consistent:
            raise ValueError('Density model should have rmaj <= X1 < X2.')

    def value(self,PSI):
        X = self.rmaj + self.rmin * math.sqrt(PSI)
        if X < self.X1:
            Ne = self.NE
        elif self.X1 <= X < self.X2:
            Ne = self.NE * (self.X2 - X) / (self.X2 - self.X1)
        else:
            Ne = 0.
        return Ne

    def derivative(self,PSI):
        X = self.rmaj + self.rmin * math.sqrt(PSI)
        if X < self.X1:
            dNe_dpsi = 0.
        elif self.X1 <= X < self.X2:
            dpsi_dX = 2. * (X - self.rmaj) / self.rmin**2
            dNe_dpsi = (-self.NE / (self.X2 - self.X1)) / dpsi_dX
        else:
            dNe_dpsi = 0.
        return dNe_dpsi

# class for defining trivial (i.e., zero) density of temperature profiles
class ZeroProfile(object):

    """
    Return a zero profile 

    Constructor:
      
       model_object = ZeroProfile()
    
    where the arguments are for compatibility and can be replaced by dummies.
    """

    def __init__(self):
        pass

    def value(self,PSI):
        return 0.

    def derivative(self,PSI):
        return 0.

# basic class for both density and temperature models
class UniBiSplineTestModels(object):

    def __init__(self, rmaj, rmin, modelflag, parameters):
        # Select the model for the poloidal flux psi
        self.__psi__ = BiSplineTestModelPsi(rmaj, rmin)
        # Initialize a dummy model that will be overwitten by setup
        self.model = ZeroProfile()
        # setup method for derived classes
        self.__setup__(rmaj, rmin, modelflag, parameters)

    def __setup__(self, rmaj, rmin, modelflag, parameters):
        pass

    def eval(self, x, y):
        psiloc = self.__psi__.eval(x, y)
        value = self.model.value(psiloc)
        return value   
    
    def derx(self, x, y):
        psiloc = self.__psi__.eval(x, y)
        derivative = self.model.derivative(psiloc)
        dpsi_dx = self.__psi__.derx(x,y)
        derivative_x = derivative * dpsi_dx
        return derivative_x

    def dery(self, x, y):
        psiloc = self.__psi__.eval(x, y)
        derivative = self.model.derivative(psiloc)
        dpsi_dy = self.__psi__.dery(x,y)
        derivative_y = derivative * dpsi_dy
        return derivative_y

# wrapper class for density models, derived from UniBiSplineTestModels
class UniBiSplineTestModelNe(UniBiSplineTestModels):

    def __setup__(self, rmaj, rmin, Ne_modelflag, parameters):
        # select the density profile depending on the model
        if Ne_modelflag == 'tokamak-like':
            self.model = NeModelTokamakLike(rmaj, rmin, parameters)
        elif Ne_modelflag == 'linear':
            self.model = NeModelLinear(rmaj, rmin, parameters)
        elif Ne_modelflag == 'zero profile':
            self.model = ZeroProfile()
        else:
            raise ValueError('Flag modelflag not understood.')

# wrapper class for temperature models, derived from UniBiSplineTestModels
class UniBiSplineTestModelTe(UniBiSplineTestModels):

    def __setup__(self, rmaj, rmin, Te_modelflag, parameters):
        # So far all analytical models have zero temperature
        # It might be needed in the future to implement an if statement
        # selecting different profiles depending on the flag Te_modelflag.
        self.model = ZeroProfile()

# class, that provides a constant magnetic field
class BiSplineTestModelB(object):
    def __init__(self,val):
        self.value = val
        
    def eval(self,x,y):
        return self.value  
    
    def derx(self,x,y):
        return 0. 

    def dery(self,x,y):
        return 0.

# class, that provides an analytical model for psi.
class BiSplineTestModelPsi(object):
    def __init__(self, rmaj, rmin):
        self.rmaj = rmaj
        self.rmin = rmin

    def eval(self,x,y):
        return ((x-self.rmaj)/self.rmin)**2
    
    def derx(self,x,y):
        return 2.*(x-self.rmaj)/self.rmin**2

    def dery(self,x,y):
        return 0.



###############################################################################
# MagneticSurfaces CLASS. 
# READS THE INPUT FILES AND INTERPOLATES THE MAGNETIC SURFACES
# IN TERMS OF THE POLOIDAL FLUX FUCNTION PSI AND BUILT THE
# INTERPOLATION OBJECT.
#
# THIS IS THE BASE CLAS UPON WHICH THE FULL PLASMA EQUILIBRIUM 
# OBJECT IS CONSTRUCTED, CF. THE CLASS PlasmaEquilibrium
###############################################################################
class MagneticSurfaces(object):

    """This class reads the topfile and provides the interpolation of the 
    normalized poloidal flux.

    LIST OF ATTRIBUTES:

     <> rmaj: 
        float scalar, nominal major radius of the tokamak;

     <> rmin: 
        float scalar, nominal minor radius of the tokamak;

     <> Rgrid: 
        float ndarray shape = (nptR, nptz), major radius coordinate 
        on the equilinrium grid (in cm); the grid is rectangular with
        nptR point in the major radius coordinate and nptz points in
        the vertical coordinate;

     <> zgrid: 
        float ndarray shape = (nptR, nptz), vertical coordinate on 
        the equilibrium grid (in cm), cf. the definition of Rgird;

     <> psigrid: 
        float ndarray shape = (nptR, nptz), poloidal flux function 
        on the equilibrium grid (normalized), cf. definition of Rgrid;

     <> Bgrid
        float ndarray shape = (3, nptR, nptz), magnetic field components 
        on the equilibrium grid in Tesla; specifically, Bgrid[ib, iR, iz]
        is the value at the grid point labeled by (iR, iz) of the radial
        (ib = 0), vertical (ib = 1), and toroidal (ib = 2) component of
        the equilibrium magnetic field;

     <> PsiInt:
        BiSpline object for the interpolation
        of the normalized poloidal flux; 

     <> magn_axis_coord_Rz:
        float ndarray shape = (2), coordinates in the R-z (poloidal) plane
        of the magnetic axis; coordinate are extracted by
            R, z = magn_axis_coord_Rz
        with R and z being the major radius and the vertical coordinate of
        the magnetic axis in cm, respectively.

    LIST OF METHODS:

      - flux_to_grid_coord(psi, theta)
        transform flux coordinates (psi, theta) into (R, z) coordinates; 

      - volume_element_J(theta, psi):
        Volume elemente in flux coordinates.

      - compute_dvolume_dpsi(psi):
        Derivative of the volumes enclosed by flux surfaces with
        respect to psi.
    """
	
    ########################################################################
    # INITIALIZATION OF THE CLASS.
    ########################################################################
    def __init__(self, idata):
        
        """Inizialization procedure. Given the input data, this initialises 
        the interpolation objects for plasma equilibrium field and
        profiles (Electron density and temperature).
        """
		
        # Nominal major and minor radii of the tokamak
        self.rmaj = idata.rmaj
        self.rmin = idata.rmin

        # Control flag for analytical density and temperature profiles
        try:
            assert idata.analytical_tokamak in ['Yes', 'No']
        except (AttributeError, AssertionError) as error:
            print("WARNING: analytical_tokamak set to default value = 'No'")
            idata.analytical_tokamak = 'No'
        


        # Load the flux surfaces either by interpolation of a topfile
        # or by analytical expressions
        if idata.analytical_tokamak == 'No':

            # Read the equilibrium (grid and poloidal flux)
            # (Here, psiloc is the poloidal flux NOT necessarily normalized)
            topfile_data = topfile.read(idata.equilibriumdirectory)
            Rloc, zloc, Bloc, psiloc, psi_sep = topfile_data               

            # One dimensional arrays with the values of both the major-radius 
            # and the vertical coordinates on the equilibrium grid
            Rloc1D = Rloc[:,0]
            zloc1D = zloc[0,:]

            if hasattr(idata, 'axis_guess_Rz'):
                R0_guess, z0_guess = idata.axis_guess_Rz
                psi_center = psiloc[np.abs(Rloc1D - R0_guess).argmin(), 
                                    np.abs(zloc1D - z0_guess).argmin()]
            else:

                # Check if the flux function is convex or concave by comparing
                # the value in the guessed c of the grid with the value at the 
                # separatrix, and reverse sign when it is concave
                nptR, nptz = np.shape(psiloc)
                iR = int(nptR / 2)
                iz = int(nptz / 2)
                psi_center = psiloc[iR, iz]
            if psi_center > psi_sep:
                psiloc = - psiloc
                psi_sep = - psi_sep


            # Normalize the poloidal flux and compute the magnetic axis
            # R-z coordinates
            npsiloc, axis = self.__normalize_psi__(psiloc, psi_sep, Rloc1D, zloc1D)
            self.magn_axis_coord_Rz = axis
            
            # If needed extend the equilibrium grid.
            if idata.extend_grid_by == None:
                # ... nothiing to extend, store the data ...
                self.Rgrid = Rloc
                self.zgrid = zloc
                self.psigrid = npsiloc
                self.Bgrid = Bloc

            else:
                # ... extend psi by quadratic extrapolation ...
                npoints = idata.extend_grid_by
                eR, ez, epsi = gext(Rloc1D, zloc1D, npsiloc, extend_by=npoints)
                # ... the magnetic field data must be reformatted in order to
                # pass them to the grid extender (the component index must be
                # the last axis of the array) ...
                Bloc = np.moveaxis(Bloc, 0, 2)
                eR, ez, eB = gext(Rloc1D, zloc1D, Bloc, extend_by=npoints)
                # ... store the data on the extended grid ...
                self.Rgrid, self.zgrid = np.meshgrid(eR, ez, indexing='ij')
                self.psigrid = epsi
                self.Bgrid = np.moveaxis(eB, 2, 0)

            # Grig geometry depending on the position of the magnetic axis
            Raxis, zaxis = self.magn_axis_coord_Rz

            Rmin = self.Rgrid.min()
            Rmax = self.Rgrid.max()

            zmin = self.zgrid.min()
            zmax = self.zgrid.max()

            self.Deast = Rmax - Raxis
            self.Dwest = Raxis - Rmin
            self.Dnorth = zmax - zaxis
            self.Dsouth = zaxis - zmin            

            # Interpolation object for the poloidal flux
            Rgrid1D = self.Rgrid[:,0]
            zgrid1D = self.zgrid[0,:]
            self.PsiInt = bispl.BiSpline(Rgrid1D, zgrid1D, self.psigrid)

        elif idata.analytical_tokamak == 'Yes':

            # Define conventional grid parameters for consistency with
            # the case idata.analytical_tokamak == 'no'
            # These are not used in the actual ray tracing calculations.
            # We take 100 points in each direction and extend the grid
            # to twice the minor radius.
            Rgrid1D = np.linspace(idata.rmaj - 2. * idata.rmin, 
                                  idata.rmaj + 2. * idata.rmin, 100)
            zgrid1D = np.linspace(- 2. * idata.rmin, 
                                  + 2. * idata.rmin, 100)
            Rgrid, zgrid = np.meshgrid(Rgrid1D, zgrid1D)
            # ... transposition is needed for consistency with the 
            #     usual numerical tokamak equilibrium ...
            self.Rgrid = Rgrid.T
            self.zgrid = zgrid.T

            # For analytical models, the magnetic axis coincides with the 
            # point (rmaj, 0) in the poloidal plane.
            self.magn_axis_coord_Rz = np.array([idata.rmaj, 0.])

            # Define the model of flux function
            self.PsiInt = BiSplineTestModelPsi(idata.rmaj,idata.rmin) 

            # Load a sample of psi which plays the same role as psigrid
            # for real tokamak equilibria 
            self.psigrid = IntSample(self.Rgrid[:,0], self.zgrid[0,:], 
                                     self.PsiInt.eval).T
            
        # This should be used only for testing.
        # # Echo the position of the magnetic axis
        # print('Magnetic axis at R=%f, z=%f\n' 
        #       %(self.magn_axis_coord_Rz[0],self.magn_axis_coord_Rz[1]))

        # setup method for derived classes
        self.__setup__(idata)

        # return from constructor	
        return

    # setup constructor for derived classes 
    def __setup__(self, idata):
        pass
        
    #######################################################################
    # METHODS OF THE CLASS 
    #######################################################################
    # R-z coordinates of the magnetic axis and normalization of psi
    def __normalize_psi__(self, psiloc, psi_sep, Rgrid1D, zgrid1D):

        """Given the raw grid psiloc of the flux funtions, this
        produces an interpolation object which is then used to 
        localize the magnetic axis. The value of the flux function at
        the axis and at the separatrix are used to normalize the 
        flux flunction so that it attains the values zero and one at the
        magnetic axis and at the separatrix, respectively.

        USAGE:
          psigrid, magn_axis = 
            self.__normalize_psi__(psiloc, psi_sep, Rgrid1D, zgrid1D)

        INPUT:
          - psiloc:
            float ndarray shape = (nptR, nptz), raw values of the flux
            funtion on the equilibrium grid;
          - psi_sep:
            float scalar, value of the flux function at the separatrix;
          - Rgrid1D:
            float ndarray shape = (nptR), values of the major-radius 
            coordinate at grid nodes:
          - zgrid1D:
            float ndarray shape = (nptz), values of the vertical coordinate
            at grid nodes;
        OUTPUT:
          - psigrid:
            float ndarray shape = (nptR, nptz), normalized poloidal flux
            at grid nodes.
          - magn_axis:
            float ndarray shape = (2), magn_axis = [Raxis, zaxis] are the
            grid coordinates of the magnetic axis;
        """

        # Interpolation of the raw array of poloidal flux
        raw_psi_interpol = spl.RectBivariateSpline(Rgrid1D, 
                                                   zgrid1D, psiloc, 
                                                   kx=3, ky=3, s=0)

        
        # Some geometric quantity of the grid
        Rmax = Rgrid1D[-1]
        Rmin = Rgrid1D[0]
        zmax = zgrid1D[-1]
        zmin = zgrid1D[0]
        midR = 0.5 * (Rmin + Rmax)
        midz = 0.5 * (zmin + zmax)

        # Find the magnetic axis and the value of the raw poloidal
        # flux on the axis
        magn_axis, psi_axis = self.__magn_axis__(raw_psi_interpol, midR, midz)

        # Normalizazion of the poloidal flux funtion
        psiloc = (psiloc - psi_axis) / (psi_sep - psi_axis)

        # Exit
        return psiloc, magn_axis

    # Calculation of the magnetic axis coordinates
    def __magn_axis__(self, raw_psi_interpol, midR, midz):
        
        """Compute the position of the magnetic axis by searching for the
        minimum of psi.
        """

        # Define the callable function of x = (R, z)
        # (Note that the PsiInt.ev method requires arrays as input and 
        #  returns and array)
        psif = lambda x: raw_psi_interpol.ev(np.array([x[0]]), 
                                             np.array([x[1]]))[0]

        # Define the guess for the minimization
        # (The point R = 0, z = 0 should be close enought to the minimum)
        x0 = np.array([midR, midz])

        # Optimization procedure from scipy.optimize (Nelder-Mead simplex
        # method. Set the option disp=True for convergence diagnostics.)
        magn_axis = opt.fmin_powell(psif, x0, disp=False)

        # Value of psi at the magnetic axis
        psi_axis = psif(magn_axis)

        # Exit
        return magn_axis, psi_axis

    # This is needed in the flux-to-grid coordinate transformation
    def __maximum_r__(self, theta):
        
        """
        Compute the maximum radial distance from the axis to the
        boundary of the computational box, along a given poloidal
        angle theta.
        """
        
        # Copy data for convenience
        Deast = self.Deast
        Dwest = self.Dwest
        Dnorth = self.Dnorth
        Dsouth = self.Dsouth

        # Preliminary calculation
        theta = theta % (2. * np.pi)
        cst = np.cos(theta)
        snt = np.sin(theta)

        # I found no better way than dinstinguish among the four
        # quadrants in theta in [0, 2*pi)
        if theta == 0.:
            return Deast
        elif 0 < theta < 0.5 * np.pi:
            r1 = Deast / cst
            r2 = Dnorth / snt
            return min(r1, r2)
        elif theta == 0.5 * np.pi:
            return Dnorth
        elif 0.5 * np.pi < theta < np.pi:
            r1 = -Dwest / cst
            r2 = Dnorth / snt
            return min(r1, r2)
        elif theta == np.pi:
            return Dwest
        elif np.pi < theta < 1.5 * np.pi:
            r1 = -Dwest / cst
            r2 = -Dsouth / snt
            return min(r1, r2)
        elif theta == 1.5 * np.pi:
            return Dsouth
        elif 1.5 * np.pi < theta < 2. * np.pi:
            r1 = Deast / cst
            r2 = -Dsouth / snt
            return min(r1, r2)

    # Method for remapping coordinates (psi, theta) to (R, z)
    # Here theta is the poloidal angle centered on the magnetic axis.
    def flux_to_grid_coord(self, psi, theta, npt=100):

        """Calculate the (R,z) coordinates corresponding to (psi, theta) 
        using the interpolated magnetic flux PsiInt as a matrix of R, z.
        """

        # Extract magnetic axis R-z coordinates
        Raxis, zaxis = self.magn_axis_coord_Rz

        # Evaluate cosine and sine to the poloidal angle
        cst = np.cos(theta)
        snt = np.sin(theta)

        # Line search for a good guess of the root 
        # (maybe slow, but fail safe near the X point)
        # ... fist sample psi along the relevant radial ...
        rmax = self.__maximum_r__(theta)
        r_sample = np.linspace(0., rmax, npt)
        R_radial = Raxis + r_sample * cst  
        z_radial = zaxis + r_sample * snt
        psi_Rz_radial = np.array([self.PsiInt.eval(R_radial[i], z_radial[i])
                                  for i in range(0, npt)])
        # ... cut the possibly non-monotonic tail out of the separatrix ...
        imax = np.argmax(psi_Rz_radial)
        psi_Rz_cut = psi_Rz_radial[0:imax+1]
        # ... now search the intervals starting from the axis ...
        try:
            index = np.digitize(np.array([psi]), psi_Rz_cut)[0]
        except ValueError:
            # ... just in case of bad equilibria ...
            psi_Rz_sorted = np.sort(psi_Rz_cut) 
            index = np.digitize(np.array([psi]), psi_Rz_sorted)[0]
        # ... guess interval ...     
        r1 = r_sample[index-1]
        r2 = r_sample[index]     

        # Define the callable function: normalized poloidal flux restricted
        # to the theta = const. line. (The input value r is assumed to be 
        # an array and so is the output of PsiInt.ev, even though they are 
        # both scalar qualtities.)
        f = lambda r: psi - self.PsiInt.eval(np.array([Raxis + r * cst]),
                                             np.array([zaxis + r * snt]))

        # Compute the minor-radius coordinate solving psi - psif(r) = 0
        if psi > 0.:
            # Compute the root
            r_sol = opt.brentq(f, r1, r2)
        else:
            r_sol = 0.

        # Extract grid coordinates
        R = Raxis + r_sol * cst
        z = zaxis + r_sol * snt

        # Exit
        return np.array([R, z])

    # Method for computing the volume element in flux coordinates
    # (psi, theta) where psi is the normalized poloidal flu and theta 
    # is the poloidal angle centered on the magnetic axis.
    # DO NOT change the order of the variable or the integration over theta
    # by scipy.integrate.quad will be broken.
    def volume_element_J(self, theta, psi):        
        
        """Determinant of the coordinate transformation
             (psi, theta) |---> (R, z),
        over the poloidal plane as a function of (psi, theta). 
        Here, theta is defined from the magnetic axis, cf. the
        function self.flux_to_grid_coord.
        """

        # Evaluate grid coordinates
        R, z = self.flux_to_grid_coord(psi, theta)
        
        # read axis coordinates
        Rshift = R - self.magn_axis_coord_Rz[0]
        zshift = z - self.magn_axis_coord_Rz[1]

        # Deterninant J as in R*dR*dz = J*dpsi*dtheta
        determinant = R * (Rshift**2 + zshift**2)  
        determinant /= abs(Rshift*self.PsiInt.derx(R,z) 
                           + zshift*self.PsiInt.dery(R,z))

        # Exit 
        return determinant     


    # EXPERIMENTAL: DO NOT USE!!!!
    # Method for computing the toroidal volume enclosed by a magnetic
    # surface of given normalized poloidal flux
    def compute_volume(self, psimin,psimax):
        
        """ EXPERIMENTAL: returns the volume in between the flux surfaces
        psimin and psimax.
        Therefore integrates the volume_element_J function"""

        volume = 2.*math.pi * intgr.dblquad(self.volume_element_J,
                                            psimin, psimax,
                                            lambda x: 0., lambda x: 2.*math.pi,
                                            args = ())[0]

       
        # Exit: convert volume from cm^3 to m^3
        return volume * 1.e-6
       
    # Method for computing the toroidal volume enclosed by a magnetic
    # surface of given normalized poloidal flux
    def compute_dvolume_dpsi(self, psi):
        
        """ returns the derivative of the function volume = volume(psi)
        defined above."""

        integral = intgr.quad(self.volume_element_J, 0., 2.*math.pi,
                              args = (psi), 
                              epsabs=1.e-03, epsrel=1.49e-07, limit=5000)
        
        dvolume_dpsi =  2.*math.pi * integral[0]

        # Exit: convert volume from cm^3 to m^3
        return dvolume_dpsi * 1.e-6

    def __compute_dP_dpsi_help__(self,theta,psi):
        
        """Comment when done!"""

        # Evaluate grid coordinates
        R, z = self.flux_to_grid_coord(psi, theta)
        
        # read axis coordinates
        Rshift = R - self.magn_axis_coord_Rz[0]
        zshift = z - self.magn_axis_coord_Rz[1]

        determinant = (Rshift**2 + zshift**2) \
            / abs(Rshift*self.PsiInt.derx(R,z) \
                              + zshift*self.PsiInt.dery(R,z))
                            

        # Exit 
        return R * determinant * self.P_Rz.eva

    def compute_dP_dpsi(self,psi):
        
        """ returns the integral on P_Rz derived by psi"""

        volume = 2.*math.pi * intgr.quad(self.__compute_dP_dpsi_help__,
                                         0.,2.*math.pi,
                                         args = (psi), limit=5000)[0]

       
        # Exit: convert volume from cm^3 to m^3
        return volume * 1.e-6


    # EXPERIMENTAL
    def compute_volume2(self,psi):
        """ EXPERIMENTAL"""

        # read axis coordinates
        Raxis = self.magn_axis_coord_Rz[0]
        zaxis = self.magn_axis_coord_Rz[1]

        nmbrTheta = 200
        theta = np.empty([nmbrTheta])
        R = np.empty([nmbrTheta])
        z = np.empty([nmbrTheta])

        for i in range(0,nmbrTheta):
            theta[i] = 2.*math.pi * i / nmbrTheta
            R[i], z[i] = self.flux_to_grid_coord(psi, theta[i])
            R[i] -= Raxis
            z[i] -= zaxis

        # compute the areas
        area = 0.
        for i in range(0,nmbrTheta):
            j = i+1
            if j >= nmbrTheta:
                j = 0
            A = 0.5 * abs(R[i]*z[j] - R[j]*z[i])
            
            area += A
      
        return 2.*math.pi * self.rmaj * area / 1.e6

#
# End of class MagneticSurfaces



###############################################################################
# TokamakEquilibrium CLASS. 
# READS THE INPUT FILES AND INTERPOLATES THE MAGNETIC FIELD,
# ELECTRON DENSITY, AND ELECTRON TEMPERATURE 
# THIS APPLIES STRICTLY TO TOKAMAK GEOMETRY ONLY! 
# ANALYTICAL MODELS AND GENERIC AXISYMMETRIC DEVICES HAVE THEIR OWN CLASS,
# WHICH DO NOT INVOLVE MAGNETIC SURFACES.
###############################################################################
class TokamakEquilibrium(MagneticSurfaces):

    """This class reads the topfile and provides the interpolation of the 
    magnetic equilibrium parameters on the equilibrium grid, namely, 
    the normalized poloidal flux and the components of the magnetic 
    field, more specifically, the toroidal component, the mjor-radius 
    component and the vertical component all expressend in Tesla.

    The input files Te.dat and ne.dat for the electron temperature 
    and density, respectively, are also loaded and an interpolation 
    object is provided for such quantities.

    LIST OF ATTRIBUTES:

     <> rmaj: 
        float scalar, nominal major radius of the tokamak;

     <> rmin: 
        float scalar, nominal minor radius of the tokamak;

     <> Rgrid: 
        float ndarray shape = (nptR, nptz), major radius coordinate 
        on the equilinrium grid (in cm); the grid is rectangular with
        nptR point in the major radius coordinate and nptz points in
        the vertical coordinate;

     <> zgrid: 
        float ndarray shape = (nptR, nptz), vertical coordinate on 
        the equilibrium grid (in cm), cf. the definition of Rgird;

     <> psigrid: 
        float ndarray shape = (nptR, nptz), poloidal flux function 
        on the equilibrium grid (normalized), cf. definition of Rgrid;

     <> Bgrid:
        float ndarray shape = (3, nptR, nptz), magnetic field components 
        on the equilibrium grid in Tesla; specifically, Bgrid[ib, iR, iz]
        is the value at the grid point labeled by (iR, iz) of the radial
        (ib = 0), vertical (ib = 1), and toroidal (ib = 2) component of
        the equilibrium magnetic field;

     <> psi_profile: (_ne for el. density, _Te for the one gained when 
		reading the temperature file)
        float ndarray shape = (nsurf_profile), normalized poloidal flux
        array of the density and temperature profiles;

     <> ne_profile: 
        float ndarray shape = (nsurf_profiles), electron number density in 
        units of 1.e13 cm^-3 on magnetic surfaces used for profiles;

     <> Te_profile:
        float ndarray shape = (nsurf_profiles), electron temperature in 
        keV on magnetic surfaces used for profiles;

     <> PsiInt:
        BiSpline object for the interpolation
        of the normalized poloidal flux 

     <> BtInt, BRInt, BzInt:
        BiSpline object for the interpolation
        of the toroidal, major-radius and vertical component 
        of the magnetic field respectively in Tesla 

     <> magn_axis_coord_Rz:
        float ndarray shape = (2), coordinates in the R-z (poloidal) plane
        of the magnetic axis; coordinate are extracted by
            R, z = magn_axis_coord_Rz
        with R and z being the major radius and the vertical coordinate of
        the magnetic axis in cm, respectively.

     <> NeInt:
        instance of the class UniBiSpline for the interpolation
        of the electron density profile (in units of 1.e13 cm^-3);

     <> TeInt:
        instance of the class UnivariateSpline for the interpolation
        of the electron temperature profiles (in keV);

     		
    LIST OF METHODS:

      - flux_to_grid_coord(psi, theta)
        transform flux coordinates (psi, theta) into (R, z) coordinates; 

      - volume_element_J(theta, psi):
        Volume elemente in flux coordinates.

      - compute_dvolume_dpsi(psi):
        Derivative of the volumes enclosed by flux surfaces with
        respect to psi.
    """
	
    ########################################################################
    # INITIALIZATION OF THE CLASS.
    ########################################################################
    def __setup__(self, idata):
        
        """Complete the initialidation of the object.
        """

        # if not the test models are considered
        if idata.analytical_tokamak == 'No':

            # if defined, multiply the magnetic field with a specified 
            # factor to shift absorption for test purposes
            try:
                self.Bgrid = self.Bgrid * idata.factormagneticfield
            except AttributeError:
                pass

            # One dimensional arrays with the values of both the major-radius 
            # and the vertical coordinates on the equilibrium grid
            Rgrid1D = self.Rgrid[:,0]
            zgrid1D = self.zgrid[0,:]

            # Interpolation object for the toroidal field 
            self.BtInt = bispl.BiSpline(Rgrid1D, zgrid1D, self.Bgrid[2,:,:])
            
            # Interpolation object for the radial field 
            self.BRInt = bispl.BiSpline(Rgrid1D, zgrid1D, self.Bgrid[0,:,:])
            
            # Interpolation object for the vertical field
            self.BzInt = bispl.BiSpline(Rgrid1D, zgrid1D, self.Bgrid[1,:,:])

            # Load the electron density profile
            input_dir = idata.equilibriumdirectory
            psi_prf, ne_prf = self.__plasma_profile__(input_dir, 'ne.dat')
            self.psi_profile_ne = psi_prf
            self.ne_profile = ne_prf
            self.NeInt = bispl.UniBiSpline(psi_prf, ne_prf, self.PsiInt)

            # Load the electron temperature profile
            input_dir = idata.equilibriumdirectory
            psi_prf, Te_prf = self.__plasma_profile__(input_dir, 'Te.dat')
            self.psi_profile_Te = psi_prf  
            self.Te_profile = Te_prf
            self.TeInt =  bispl.UniBiSpline(psi_prf, Te_prf, self.PsiInt)

        # if the test model for the TOKAMAK geometry is considered
        elif idata.analytical_tokamak == 'Yes':
            # use testmodel
            # use constant value for psi smaller than some treshhold, 
            # which is defined in the inputfile
            try:
                self.cuteldensAtPsi = \
                            ((idata.cuteldensAt - self.rmaj)/self.rmin)**2
            except:
                self.cuteldensAtPsi = 0.
                
            # Analytical density profile
            Ne_modelflag = idata.analytical_tokamak_ne_model
            self.NeInt = UniBiSplineTestModelNe(idata.rmaj, idata.rmin,
                                                Ne_modelflag,
                                                idata.ne_model_parameters)

            # Analytical temperature profile
            # (the flag for the temperature profile for the moment is not
            #  needed and might not appear in the configuration files. The
            #  try loop is added for backward compatibility.)
            try:
                Te_modelflag = idata.analytical_tokamak_Te_model
            except AttributeError:
                Te_modelflag = 'dummy flag'
            self.TeInt = UniBiSplineTestModelTe(idata.rmaj, idata.rmin,
                                                Te_modelflag,
                                                idata.ne_model_parameters)

            # Magnetic field components set by input file or, in case it isn't,
            # set by default to a purely toroidal magnetic field
            if not hasattr(idata, 'BRInt'):
                self.BRInt = BiSplineTestModelB(0.)
            else:
                self.BRInt = BiSplineTestModelB(idata.BRInt)
            if not hasattr(idata, 'BzInt'):                
                self.BzInt = BiSplineTestModelB(0.)
            else:
                self.BzInt = BiSplineTestModelB(idata.BzInt)
            if not hasattr(idata, 'BtInt'):            
                self.BtInt = BiSplineTestModelB(-1.)
            else:
                self.BtInt = BiSplineTestModelB(idata.BtInt)
                
        # anything else
        else:
            msg = "Check input keyword 'analytical_tokamak'."
            raise ValueError(msg)

        # return from constructor	
        return

        
    # This function loads TORBEAM profile files
    def __plasma_profile__(self, input_dir, fname):

        """Open the data file specified in the argument string fname in
        the TORBEAM data directory and read the profile therein.
        """

        # Account for differences in data formatting
        if fname == 'volumes.dat':
            skiprows = ' '
        else:
            skiprows = 1

        # Load data
        path_to_file = input_dir + '/' + fname
        data = np.loadtxt(path_to_file, skiprows=skiprows)

        # Extract profiles
        rho_prf = data[:,0]
        psi_prf = rho_prf**2
        y_prf = data[:,1]
            
        # Close file and return
        return psi_prf, y_prf

#
# End of class TokamakEquilibrium

###############################################################################
# TokamakEquilibrium CLASS. 
# READS THE INPUT FILES AND INTERPOLATES THE MAGNETIC FIELD,
# ELECTRON DENSITY, AND ELECTRON TEMPERATURE 
# THIS APPLIES STRICTLY TO TOKAMAK GEOMETRY ONLY! 
# ANALYTICAL MODELS AND GENERIC AXISYMMETRIC DEVICES HAVE THEIR OWN CLASS,
# WHICH DO NOT INVOLVE MAGNETIC SURFACES.
###############################################################################
class TokamakEquilibrium2(MagneticSurfaces):

    """This class reads the topfile and provides the interpolation of the 
    magnetic equilibrium parameters on the equilibrium grid, namely, 
    the normalized poloidal flux and the components of the magnetic 
    field, more specifically, the toroidal component, the mjor-radius 
    component and the vertical component all expressend in Tesla.

    The input files Te.dat and ne.dat are different now, and consist of 2D data already. 
    Specifically used for the FIR diagnostic trial on TCV

    LIST OF ATTRIBUTES:

     <> rmaj: 
        float scalar, nominal major radius of the tokamak;

     <> rmin: 
        float scalar, nominal minor radius of the tokamak;

     <> Rgrid: 
        float ndarray shape = (nptR, nptz), major radius coordinate 
        on the equilinrium grid (in cm); the grid is rectangular with
        nptR point in the major radius coordinate and nptz points in
        the vertical coordinate;

     <> zgrid: 
        float ndarray shape = (nptR, nptz), vertical coordinate on 
        the equilibrium grid (in cm), cf. the definition of Rgird;

     <> psigrid: 
        float ndarray shape = (nptR, nptz), poloidal flux function 
        on the equilibrium grid (normalized), cf. definition of Rgrid;

     <> Bgrid:
        float ndarray shape = (3, nptR, nptz), magnetic field components 
        on the equilibrium grid in Tesla; specifically, Bgrid[ib, iR, iz]
        is the value at the grid point labeled by (iR, iz) of the radial
        (ib = 0), vertical (ib = 1), and toroidal (ib = 2) component of
        the equilibrium magnetic field;

     <> psi_profile: (_ne for el. density, _Te for the one gained when 
		reading the temperature file)
        float ndarray shape = (nsurf_profile), normalized poloidal flux
        array of the density and temperature profiles;

     <> ne_profile: 
        float ndarray shape = (nsurf_profiles), electron number density in 
        units of 1.e13 cm^-3 on magnetic surfaces used for profiles;

     <> Te_profile:
        float ndarray shape = (nsurf_profiles), electron temperature in 
        keV on magnetic surfaces used for profiles;

     <> PsiInt:
        BiSpline object for the interpolation
        of the normalized poloidal flux 

     <> BtInt, BRInt, BzInt:
        BiSpline object for the interpolation
        of the toroidal, major-radius and vertical component 
        of the magnetic field respectively in Tesla 

     <> magn_axis_coord_Rz:
        float ndarray shape = (2), coordinates in the R-z (poloidal) plane
        of the magnetic axis; coordinate are extracted by
            R, z = magn_axis_coord_Rz
        with R and z being the major radius and the vertical coordinate of
        the magnetic axis in cm, respectively.

     <> NeInt:
        instance of the class BiSpline for the interpolation
        of the electron density profile (in units of 1.e13 cm^-3);

     <> TeInt:
        instance of the class BiSpline for the interpolation
        of the electron temperature profiles (in keV);

     		
    LIST OF METHODS:

      - flux_to_grid_coord(psi, theta)
        transform flux coordinates (psi, theta) into (R, z) coordinates; 

      - volume_element_J(theta, psi):
        Volume elemente in flux coordinates.

      - compute_dvolume_dpsi(psi):
        Derivative of the volumes enclosed by flux surfaces with
        respect to psi.
    """
	
    ########################################################################
    # INITIALIZATION OF THE CLASS.
    ########################################################################
    def __setup__(self, idata):
        
        """Complete the initialidation of the object.
        """

        # if not the test models are considered
        if idata.analytical_tokamak == 'No':

            # if defined, multiply the magnetic field with a specified 
            # factor to shift absorption for test purposes
            try:
                self.Bgrid = self.Bgrid * idata.factormagneticfield
            except AttributeError:
                pass

            # One dimensional arrays with the values of both the major-radius 
            # and the vertical coordinates on the equilibrium grid
            Rgrid1D = self.Rgrid[:,0]
            zgrid1D = self.zgrid[0,:]

            # Interpolation object for the toroidal field 
            self.BtInt = bispl.BiSpline(Rgrid1D, zgrid1D, self.Bgrid[2,:,:])
            
            # Interpolation object for the radial field 
            self.BRInt = bispl.BiSpline(Rgrid1D, zgrid1D, self.Bgrid[0,:,:])
            
            # Interpolation object for the vertical field
            self.BzInt = bispl.BiSpline(Rgrid1D, zgrid1D, self.Bgrid[1,:,:])

            # Load the electron density and temperature profile
            input_dir = idata.equilibriumdirectory
            nedata = nefile.read(input_dir)
            Rloc, zloc, neloc = nedata
            Rgrid1D = Rloc[:,0]
            zgrid1D = zloc[0,:]
            self.NeInt = bispl.BiSpline(Rgrid1D, zgrid1D, neloc)

            # Load the electron temperature profile
            input_dir = idata.equilibriumdirectory
            psi_prf, Te_prf = self.__plasma_profile__(input_dir, 'Te.dat')
            self.psi_profile_Te = psi_prf  
            self.Te_profile = Te_prf
            self.TeInt =  bispl.UniBiSpline(psi_prf, Te_prf, self.PsiInt)
  

        # if the test model for the TOKAMAK geometry is considered
        elif idata.analytical_tokamak == 'Yes':
            # use testmodel
            # use constant value for psi smaller than some treshhold, 
            # which is defined in the inputfile
            try:
                self.cuteldensAtPsi = \
                            ((idata.cuteldensAt - self.rmaj)/self.rmin)**2
            except:
                self.cuteldensAtPsi = 0.
                
            # Analytical density profile
            Ne_modelflag = idata.analytical_tokamak_ne_model
            self.NeInt = UniBiSplineTestModelNe(idata.rmaj, idata.rmin,
                                                Ne_modelflag,
                                                idata.ne_model_parameters)

            # Analytical temperature profile
            # (the flag for the temperature profile for the moment is not
            #  needed and might not appear in the configuration files. The
            #  try loop is added for backward compatibility.)
            try:
                Te_modelflag = idata.analytical_tokamak_Te_model
            except AttributeError:
                Te_modelflag = 'dummy flag'
            self.TeInt = UniBiSplineTestModelTe(idata.rmaj, idata.rmin,
                                                Te_modelflag,
                                                idata.ne_model_parameters)

            # Magnetic field components set by input file or, in case it isn't,
            # set by default to a purely toroidal magnetic field
            if not hasattr(idata, 'BRInt'):
                self.BRInt = BiSplineTestModelB(0.)
            else:
                self.BRInt = BiSplineTestModelB(idata.BRInt)
            if not hasattr(idata, 'BzInt'):                
                self.BzInt = BiSplineTestModelB(0.)
            else:
                self.BzInt = BiSplineTestModelB(idata.BzInt)
            if not hasattr(idata, 'BtInt'):            
                self.BtInt = BiSplineTestModelB(-1.)
            else:
                self.BtInt = BiSplineTestModelB(idata.BtInt)
                
        # anything else
        else:
            msg = "Check input keyword 'analytical_tokamak'."
            raise ValueError(msg)

        # return from constructor	
        return
    
        # This function loads TORBEAM profile files
    def __plasma_profile__(self, input_dir, fname):

        """Open the data file specified in the argument string fname in
        the TORBEAM data directory and read the profile therein.
        """

        # Account for differences in data formatting
        if fname == 'volumes.dat':
            skiprows = ' '
        else:
            skiprows = 1

        # Load data
        path_to_file = input_dir + '/' + fname
        data = np.loadtxt(path_to_file, skiprows=skiprows)

        # Extract profiles
        rho_prf = data[:,0]
        psi_prf = rho_prf**2
        y_prf = data[:,1]
            
        # Close file and return
        return psi_prf, y_prf

        

#
# End of class TokamakEquilibrium




###############################################################################
# AxisymmetricEquilibrium CLASS. 
# READS THE INPUT FILES AND INTERPOLATES THE MAGNETIC FIELD,
# ELECTRON DENSITY, AND ELECTRON TEMPERATURE FOR AN AXISYMMETRIC EQUILIBRIUM. 
# DIFFERENTLY FROM TokamakEquilibrium, THIS APPLIES TO GENERIY AXISYMMETRIC
# DEVICES AND DOES NOT RELY ON THE MAGNETIC SURFACES, I.E., BOTH ELECTRON
# DENSITY AND TEMPERATURE ARE FULL 2D PROFILES AND ARE INTERPOLATED IN THE 
# SAME WAY AS THE MAGNETIC FIELD COMPONENTS. 
###############################################################################
class AxisymmetricEquilibrium(object):

    """This reads the input file for a generic axisymmetric device. 
    (The appropriate data format is explained in the doc string of the module
    AxisymmetricEquilibrium.py which provides the data-reading functions.)

    Differently from the class TokamakEquilibrium, this generic axisymmetric
    equilibrium does not rely on the existence of magnetic surfaces. As a 
    consequences the electron density is treated as fully 2D profile and 
    interpolated in the same way as for the components of the equilibrium 
    magnetic field. The electron temperature is not meant to be used for this
    cases and it is set to zero as in the class ModelEquilibrium.

    The resulting interpolation objects are stores as attributes of the class.

    List of attributes:

    <> rmaj, scalar float, major radius in cm.

    <> rmin, scalar float, minor radius in cm.

    <> PsiInt, interpolation object of the poloidal flux as in ModelEquilibrium.
       It is defined by a dummy model (and not used in the main calculation).

    <> BtInt, BRInt, BzInt as in TokamakEquilibrium.

    <> magn_axis_coord_Rz, are defined as the geometric center of the device.

    <> NeInt and TeInt are 2D interpolation objects of the same class as the
       components of the magnetic field.

    <> TeInt as in ModelEquilibrium.
    """

    ########################################################################
    # INITIALIZATION OF THE CLASS.
    ########################################################################
    def __init__(self, idata):
        
        """Constructor of the object.
        """
        
        # Read rmaj and rmin from the input data
        # (those are required for the psi model only)
        self.rmaj = idata.rmaj
        self.rmin = idata.rmin        

        # Interpolation object for the poloidal flux (a model)
        self.PsiInt = BiSplineTestModelPsi(idata.rmin,idata.rmaj)

        # Read the actual dada
        # (This allows the grid to be different for each quantity)
        input_dir = idata.equilibriumdirectory
        radial_field, vertical_field, toroidal_field, density = \
                            axeq.read_axisymmetric_equilibrium(input_dir)

        # Extract and interpolate the radial field
        R_BR, z_BR, BR = radial_field        
        self.BRInt = bispl.BiSpline(R_BR, z_BR, BR)

        # Extract and interpolate the vertical field
        R_Bz, z_Bz, Bz = vertical_field
        self.BzInt = bispl.BiSpline(R_Bz, z_Bz, Bz)

        # Extract and interpolate the toroidal field
        R_Bt, z_Bt, Bt = toroidal_field
        self.BtInt = bispl.BiSpline(R_Bt, z_Bt, Bt)

        # Extract and interpolate the electron density
        R_Ne, z_Ne, Ne = density
        # ... convert the density from m^-3 to 10^13 cm^-3
        Ne = 1.e-19 * Ne
        # ... construction of the interpolation object ...
        self.NeInt = bispl.BiSpline(R_Ne, z_Ne, Ne) 

        # Pick one of the grids as a reference
        Rg, zg = np.meshgrid(R_BR, z_BR)
        self.Rgrid = Rg.T
        self.zgrid = zg.T

        # There is no magnetic axeis here, but for consistency define
        # the coordinates of the magnetic axes and the mid point of the grid
        R_mid = 0.5*(R_BR.min() + R_BR.max())
        z_mid = 0.5*(z_BR.min() + z_BR.max())
        self.magn_axis_coord_Rz = np.array([R_mid, z_mid])


        # The temperature is set to zero for the moment
        # (Not used. Defined only for compatibility.)
        self.TeInt = UniBiSplineTestModelTe(idata.rmaj, idata.rmin,
                                            'zero profile', 
                                            [])

#
# End of class AxisymmetricEquilibrium



###############################################################################
# ModelEquilibrium CLASS. 
# THIS EMULATE THE EQUILIBRIUM QUANTITIES WHEN ANALYTICAL HAMILTONIANS
# ARE USED, E.G., FREE SPACE, LINEAR LAYER AND LENS-LIKE.
###############################################################################
class ModelEquilibrium(object):

    """This class mirror the TokamakEquilibrium class, put load models
    for the physical quantities.
    
    This is called as equilibrium configuration when running with analytical
    Hamiltonians and not input file is required in the initialization.

    This model equilibrium is defined as follows:
      - The electron density is zero;
      - The electron temperature is zero.
      - The magnetic field is set to the constant vector (0., 0., -1.),
        in the laboratory Cartesian frame. 
      - The normalized poloidal flux psi is given by the expression
             psi(x,y) = ((x - rmaj) / rmin)**2,
        with rmaj and rmin read from the input file.
    """

    ########################################################################
    # INITIALIZATION OF THE CLASS.
    ########################################################################
    def __init__(self, idata):
        
        """Constructor of the object.
        """
        
        # Read rmaj and rmin from the input data
        # (those are required for the psi model only)
        self.rmaj = idata.rmaj
        self.rmin = idata.rmin 
        self.magn_axis_coord_Rz = np.array([idata.rmaj, idata.rmin])

        # Interpolation object for the magnetic field components
        self.BRInt = BiSplineTestModelB(0.)
        self.BzInt = BiSplineTestModelB(0.)
        self.BtInt = BiSplineTestModelB(-1.)

        # Interpolation object for the poloidal flux
        self.PsiInt = BiSplineTestModelPsi(idata.rmaj, idata.rmin)

        # Interpolation object for the density and temperature
        # (both based on the density model, but thy define a zero profile)
        self.NeInt = UniBiSplineTestModelNe(idata.rmaj, idata.rmin, 
                                            'zero profile', [])
        self.TeInt = UniBiSplineTestModelTe(idata.rmaj, idata.rmin, 
                                            'zero profile', [])
        # return from constructor	
        return    

#
# End of class ModelEquilibrium


###############################################################################
# UTILITY FUNCTIOMS
###############################################################################
# Produce a sample of a 2D interpolated profile
def IntSample(R, z, IntObjFunct):
    jmax = np.size(R)
    imax = np.size(z)
    s = np.empty([imax, jmax])
    s[:,:] = np.nan
    for i in range(imax):
        zloc = z[i]
        for j in range(jmax):
            Rloc = R[j]
            s[i,j] = IntObjFunct(Rloc, zloc)
    return s


# Produce a 2D sample of Stix Parameters X, Y 
def StixParamSample(R, z, EqObj, FreqGHz):

    # Sampling the relevant interpolated quantities
    Bt2d = IntSample(R, z, EqObj.BtInt.eval)   
    BR2d = IntSample(R, z, EqObj.BRInt.eval)  
    Bz2d = IntSample(R, z, EqObj.BzInt.eval)  
    Ne2d = IntSample(R, z, EqObj.NeInt.eval)

    # Round-off in the interpolation routine can produce
    # negative density of order mush smaller than e^-16
    Ne2d = np.abs(Ne2d)

    # Computing using the same procedures as in the code
    import RayTracing.modules.dispersion_matrix_cfunctions as disp
    import CommonModules.physics_constants as phys
    omega = phys.AngularFrequency(FreqGHz)
    Bnorm = np.sqrt(Bt2d**2 + BR2d**2 + Bz2d**2)
    OmegaP = np.array(list(map(disp.disParamomegaP, Ne2d.flatten())))
    OmegaP = OmegaP.reshape(Ne2d.shape)
    OmegaC = np.array(list(map(disp.disParamOmega, Bnorm.flatten())))
    OmegaC = OmegaC.reshape(Bnorm.shape)
    StixX = OmegaP**2 / omega**2
    StixY = OmegaC / omega

    return StixX, StixY, [Bt2d, BR2d, Bz2d, Ne2d]


################################################################################
# TESTING

if __name__=='__main__':

    from pylab import *
    from CommonModules.input_data import InputData
   
    # Use some of the standard cases for this test
    idata_torpex = InputData('StandardCases/TORPEX/TORPEX_raytracing.txt')
    idata_iter = InputData('StandardCases/ITER/ITER0fluct_raytracing.txt')
    idata_focus = InputData('StandardCases/Cutoff/Cutoff_raytracing.txt')

    Eq_torpex = AxisymmetricEquilibrium(idata_torpex)
    Eq_iter = TokamakEquilibrium(idata_iter)
    Eq_focus = TokamakEquilibrium(idata_focus)

    f_torpex = 1.0
    f_iter = 2.0
    f_focus = 2.0

    nptR = 200
    nptz = 150
    
    R_torpex = linspace(idata_torpex.rmaj - f_torpex * idata_torpex.rmin, 
                        idata_torpex.rmaj + f_torpex * idata_torpex.rmin, nptR)
    z_torpex = linspace(- f_torpex * idata_torpex.rmin, 
                        + f_torpex * idata_torpex.rmin, nptz) 

    R_iter = linspace(idata_iter.rmaj - f_iter * idata_iter.rmin, 
                        idata_iter.rmaj + f_iter * idata_iter.rmin, nptR)
    z_iter = linspace(- f_iter * idata_iter.rmin, 
                      + f_iter * idata_iter.rmin, nptz)

    R_focus = linspace(idata_focus.rmaj - f_focus * idata_focus.rmin, 
                       idata_focus.rmaj + f_focus * idata_focus.rmin, nptR)
    z_focus = linspace(- idata_focus.rmin, + idata_focus.rmin, nptz) 
    
    BtSample_torpex = IntSample(R_torpex, z_torpex, Eq_torpex.BtInt.eval)
    BRSample_torpex = IntSample(R_torpex, z_torpex, Eq_torpex.BRInt.eval)
    BzSample_torpex = IntSample(R_torpex, z_torpex, Eq_torpex.BzInt.eval)
    NeSample_torpex = IntSample(R_torpex, z_torpex, Eq_torpex.NeInt.eval)
    NeRSample_torpex = IntSample(R_torpex, z_torpex, Eq_torpex.NeInt.derx)
    NezSample_torpex = IntSample(R_torpex, z_torpex, Eq_torpex.NeInt.dery)

    BtSample_iter = IntSample(R_iter, z_iter, Eq_iter.BtInt.eval)
    BRSample_iter = IntSample(R_iter, z_iter, Eq_iter.BRInt.eval)
    BzSample_iter = IntSample(R_iter, z_iter, Eq_iter.BzInt.eval)
    NeSample_iter = IntSample(R_iter, z_iter, Eq_iter.NeInt.eval)
    NeRSample_iter = IntSample(R_iter, z_iter, Eq_iter.NeInt.derx)
    NezSample_iter = IntSample(R_iter, z_iter, Eq_iter.NeInt.dery)

    NeSample_focus = IntSample(R_focus, z_focus, Eq_focus.NeInt.eval)
    NeRSample_focus = IntSample(R_focus, z_focus, Eq_focus.NeInt.derx)
    NezSample_focus = IntSample(R_focus, z_focus, Eq_focus.NeInt.dery)

    figure(1, figsize=(10,10))
    #
    subplot(321, aspect='equal')
    pcolor(R_torpex, z_torpex, BtSample_torpex)
    colorbar()
    title('Bt - TORPEX')
    #
    subplot(322, aspect='equal')
    pcolor(R_torpex, z_torpex, BRSample_torpex)
    colorbar()
    title('BR - TORPEX')
    #
    subplot(323, aspect ='equal')
    pcolor(R_torpex, z_torpex, BzSample_torpex)
    colorbar()
    title('Bz - TORPEX')
    #
    subplot(324, aspect='equal')
    pcolor(R_torpex, z_torpex, NeSample_torpex)
    colorbar()
    title('Ne - TORPEX')
    #
    subplot(325, aspect='equal')
    pcolor(R_torpex, z_torpex, NeRSample_torpex)
    colorbar()
    title('dNe/dR - TORPEX')
    #
    subplot(326, aspect='equal')
    pcolor(R_torpex, z_torpex, NezSample_torpex)
    colorbar()
    title('dNe/dz - TORPEX')


    figure(2, figsize=(10,10))
    #
    subplot(321, aspect='equal')
    pcolor(R_iter, z_iter, BtSample_iter)
    colorbar()
    title('Bt - ITER')
    #
    subplot(322, aspect='equal')
    pcolor(R_iter, z_iter, BRSample_iter)
    colorbar()
    title('BR - ITER')
    #
    subplot(323, aspect ='equal')
    pcolor(R_iter, z_iter, BzSample_iter)
    colorbar()
    title('Bz - ITER')
    #
    subplot(324, aspect='equal')
    pcolor(R_iter, z_iter, NeSample_iter)
    colorbar()
    title('Ne - ITER')
    #
    subplot(325, aspect='equal')
    pcolor(R_iter, z_iter, NeRSample_iter)
    colorbar()
    title('dNe/dR - ITER')
    #
    subplot(326, aspect='equal')
    pcolor(R_iter, z_iter, NezSample_iter)
    colorbar()
    title('dNe/dz - ITER')


    figure(3, figsize=(10,10))
    #
    subplot(221, aspect='equal')
    pcolor(R_focus, z_focus, NeSample_focus)
    title('Ne - Focus')
    #
    subplot(222, aspect='equal')
    pcolor(R_focus, z_focus, NeRSample_focus)
    colorbar()
    title('dNe/dR - Focus')
    #
    subplot(223)
    plot(R_focus, NeSample_focus[100,:])
    #
    subplot(224, aspect='equal')
    pcolor(R_focus, z_focus, NezSample_focus)
    title('dNe/dz = '+str(np.max(np.abs(NezSample_focus.flatten()))))

    show()
