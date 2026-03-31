"""Provides a class which allows to trace
one single ray given the initial values.
"""

############################################################################
# IMPORT STATEMENTS
############################################################################

# load standard modules
import numpy as np
import math
import h5py
from scipy.integrate import ode
from scipy.optimize import fsolve
# load local modules
import CommonModules.physics_constants as phys
from CommonModules.PlasmaEquilibrium import ModelEquilibrium
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium, TokamakEquilibrium2
from CommonModules.PlasmaEquilibrium import AxisymmetricEquilibrium
from RayTracing.modules.scattering.GaussianModel import GaussianModel_SingleMode
from RayTracing.modules.scattering.GaussianModel import GaussianModel_MultiMode
from RayTracing.modules.scattering.ShaferModel import ShaferModel_SingleMode
from RayTracing.modules.scattering.ShaferModel import ShaferModel_MultiMode
from RayTracing.modules.atanrightbranch import atanRightBranch
from RayTracing.modules.dispersion_matrix_cfunctions import *
from RayTracing.lib.ecdisp.farinaECabsorption import warmdamp
from RayTracing.lib.westerino.westerinoECabsorption import dampbq


############################################################################
# DEFINE THE RAY-TRACE CLASS
############################################################################
class TraceOneRay(object):
    
    """Class which traces one single ray given the initial parameters.
    As Hamiltonian, the toy models or the physical plasma Hamiltonian can
    be chosen. 
    """

    ############################################################################
    # INITIALIZATION OF THE CLASS
    # DEFINITION OF BASIC PROPERTIES AND FUNCTIONS.
    ############################################################################
    # takes: idata instance an the max. Wfct as parameters
    def __init__(self, idata, rank):
        
        """Initialisation procedure."""
        
        # define interpolation object for the plasma equilibrium
        if idata.equilibrium == 'Tokamak':
            self.Eq = TokamakEquilibrium(idata)
        elif idata.equilibrium == 'Tokamak2D':
            self.Eq = TokamakEquilibrium2(idata)
        elif idata.equilibrium == 'Model':
            self.Eq = ModelEquilibrium(idata)
        elif idata.equilibrium == 'Axisymmetric':
            self.Eq = AxisymmetricEquilibrium(idata)            
        else:
            msg = "Input keyword 'equilibrium' not understood."
            raise ValueError(msg)            

        # copy some parameters which are given in the input file:
        # integrator parameters
        self.integratorMaxNSteps = idata.integratormaxnmbrsteps
        self.integratorRelTol = idata.integratorreltol
        self.integratorAbsTol = idata.integratorabstol
       
        self.npt = idata.npt                      # (max.) number of ray points
        self.timestep = idata.timestep            # timestep for the integration

        self.c = phys.SpeedOfLight                      # speed of light in cm/s
        self.omega = phys.AngularFrequency(idata.freq)  # beam omega
        self.k0 = phys.WaveNumber(self.omega)           # wavevector in free space in cm^-1
        self.epsilonRegS = idata.epsilonRegS      # regularisation parameter for S^-1
        
        # absorption parameters
        self.absorption = idata.absorption        # absorption  turned on or off
        
        if self.absorption == True:
            # see which absorption routine to use
            self.absorptionModule = idata.absorptionModule
            self.absorptionLayerX = idata.absorptionLayerX
            self.absorptionSmallerTimestepsFactor = idata.absorptionSmallerTimestepsFactor
            self.absorptionConsiderAsNonZeroTreshhold = idata.absorptionConsiderAsNonZeroTreshhold

            self.startAbsorptionComputation = 0.   # by default set to 0
           
        # treshhold, when ray tracing is stopped because ray is absorbed
        self.absorptionWfctTreshhold = idata.absorptionWfctTreshhold 

       
        # reflektometrie parameters
        # (when the reflectometry mode is on, the ray tracing is stopped
        #  as soon as the plasma density is below a geiven treshold, when
        #  the ray exit the plasma)
        self.reflektometrie = idata.reflektometrie 
        self.reflektometrierhoTreshhold = idata.reflektometrierhoTreshhold

        # scattering parameters
        if not hasattr(idata, 'CrossPolarizationScatt'):
            idata.CrossPolarizationScatt = False
            print('WARNING - Cross-polarization scattering set to False by default')
            
        self.scattering = idata.scattering    # scattering turned on or off
        # if scattering is turned on, create a scattering object 
        # and initialise it
        # ! scattering may only be used in case the plasma is considered.
        if self.scattering == True:
            self.Lperp = idata.scatteringLengthPerp  
            self.Lparallel = idata.scatteringLengthParallel
            if idata.scatteringGaussian == True:
                # choose Gaussian scattering model
                if idata.CrossPolarizationScatt == True:
                    self.ScatteringDistr = GaussianModel_MultiMode(idata, rank)
                else:
                    self.ScatteringDistr = GaussianModel_SingleMode(idata, rank)
            else:
                # choose M. W. Shaefer (2012) model
                if idata.CrossPolarizationScatt == True:
                    self.ScatteringDistr = ShaferModel_MultiMode(idata, rank)
                else:
                    self.ScatteringDistr = ShaferModel_SingleMode(idata, rank)

        # properties for models
        try:
            self.linearlayer = idata.linearlayer
        except:
            self.linearlayer = False
        try:
            self.valley = idata.valley
        except:
            self.valley = False
        try:
            self.paraxialapprox = idata.paraxialapprox
        except:
            self.paraxialapprox = False

        # set the parameters for analytical models
        if self.linearlayer == True or self.valley == True:
            self.linearlayervalleyL = idata.linearlayervalleyL

        self.equilibrium = idata.equilibrium

        # number of rays (if only one, some more information might be printed)
        self.nmbrRays = idata.nmbrRays

        # if possible, read filename for diagnostics output
        try:
            self.dispersionSurfacesOnCentralRay = idata.dispersionSurfacesOnCentralRay
            self.output_dir = idata.output_dir
        except:
            pass


    ######################################################################
    # PROVIDE A FUNCTION WHICH RETURNS THE ABSORPTION COEFFICIENT
    ######################################################################
    def __absorption_coefficient__(self,X,Y,Z,Nx,Ny,Nz,sigma):

        """returns the absorption coefficient.
        The absorption routine of D. Farina or Westerhof is used
        as chosen in the input file.
        """

        R = disROutOf(X,Y)
        
        Bt = self.Eq.BtInt.eval(R,Z)
        BR = self.Eq.BRInt.eval(R,Z)
        Bz = self.Eq.BzInt.eval(R,Z)
        Bnorm = math.sqrt(Bt**2+BR**2+Bz**2)
        dBt_dR = self.Eq.BtInt.derx(R,Z)
        dBR_dR = self.Eq.BRInt.derx(R,Z)
        dBz_dR = self.Eq.BzInt.derx(R,Z)
        dBt_dz = self.Eq.BtInt.dery(R,Z)
        dBR_dz = self.Eq.BRInt.dery(R,Z)
        dBz_dz = self.Eq.BzInt.dery(R,Z)
        psi = self.Eq.PsiInt.eval(R,Z)
        dpsi_dR = self.Eq.PsiInt.derx(R,Z)
        dpsi_dz = self.Eq.PsiInt.dery(R,Z)
        
        Ne = self.Eq.NeInt.eval(R,Z)
        dNe_dR = self.Eq.NeInt.derx(R,Z)
        dNe_dz = self.Eq.NeInt.dery(R,Z)
        
        # in case the interpolation produces negative electron density.
        if Ne < 0.: 
            Ne = 0.

        derX,derY,derZ,derNx,derNy,derNz = disHamiltonianDerivatives(self.omega, 
                                                                     Bt, dBt_dR, dBt_dz,
                                                                     BR, dBR_dR, dBR_dz,
                                                                     Bz, dBz_dR, dBz_dz,
                                                                     Ne, dNe_dR, dNe_dz,
                                                                     X,Y,Z,
                                                                     Nx,Ny,Nz,
                                                                     sigma,
                                                                     self.epsilonRegS)
        # compute correction factor
        alpha, beta = disrotMatrixAngles(Bt, BR, Bz, X, Y, Z)
        Nparallel,Nperp,phiN = disNparallelNperpphiNOutOfNxNyNz(alpha, beta, Nx, Ny, Nz)
        f = abs(4.*disTrDispersionMatrixDivBySHamiltonian(self.omega,
                                                          Bnorm, Ne,
                                                          Nparallel,
                                                          sigma)) 

 
        # ... and absorption coefficient    
        Omega = disParamOmega(Bnorm)
        omegaP = disParamomegaP(Ne)
        parAlpha = (omegaP / self.omega)**2
        parBeta = (Omega / self.omega)**2
        Nnorm = math.sqrt(Nperp**2 + Nparallel**2)
        Te = self.Eq.TeInt.eval(R,Z)

        if Te <= 0.:   # could be < 0 due to oscillataion in the interpolation
            return 0.
        
        # if parAlpha == 0 or parBeta == 0 do not compute absorption coefficient but assume that it is 0,
        # this is achieved by setting temperature to 0
        if parAlpha == 0. or parBeta == 0.:
            return 0.
        
        if self.absorptionModule == 0:  # Westerhof absorption module
            # thermal speed
            me = 9.1e-31
            VTe = math.sqrt(3.2e-16 * Te/me) / 3.0e8
               
            # initialise dampbq routine and then call it
            dampbq(math.acos(Nparallel / Nnorm), Nnorm, #refractive index 
                   parAlpha, parBeta,                   #plasma parameterse
                   VTe,                                 #thermal electron speed
                   sigma,                               #wave mode
                   0)                                   #initialise
                    

            refractdamp = dampbq(math.acos(Nparallel / Nnorm),   #refractive index
                                 Nnorm,                           
                                 parAlpha, parBeta,              #parameters defined above
                                 VTe,                            #thermal electron speed
                                 sigma,                          #wave mode
                                 1)                              #do not initialise

            # correct the result as in TORBEAM: Assume that refractdamp is the
            # projection of Im(Nperp) onto the refractive index as well as
            # that Im(Nparallel) = 0 and Im(Nperp) is complanar to N. 
            # Those assumptions give
            #
            #   refractdamp = Im(Nperp).Nperp/|Nperp| = |Im(Nperp)| * N_perp^2/|Nperp|
            #               = |Im(N)_perpendicular| sin(thetaN)
            #               = |Im(N)_perpendicular| sqrt(1. - Nparallel^2/|N|^2)
            #
            # from which we obtain absImN = |Im(N)| = |Im(N)_perpendicular|.
            Nnorm2 = Nparallel**2 + Nperp**2
            sinthetaN2 = max(0., 1. - Nparallel**2 / Nnorm2)
            sinthetaN = math.sqrt(sinthetaN2)
            absImN = refractdamp / sinthetaN

        if self.absorptionModule == 1:   # Farina absorption module
            refractdamp = warmdamp(parAlpha, parBeta,                    # plasma parameters
                                   Nnorm, math.acos(Nparallel / Nnorm),  # refractive index vector
                                   Te, sigma)                            # temperature and mode
            
            # correct the result as in TORBEAM:
            # in this case the output is directly equal to absImN defined above
            absImN = refractdamp

        # THE FOLLOWING IS NOT RIGOROUS, BUT IT IS THE WAY THE TWO ABSORPTION
        # ROUTINES (FROM EXTERNAL LIBRARIES) ARE DESIGNED TO WORK AND IT IS
        # THE EXACT SAME WAY IMPLEMENTED IN THE TORBEAM CODE:
        #
        # In theory, given the imaginary part of the refractive index vector
        # ImN, this extracts the absorption coefficient gamma from
        #
        #   gamma = k0 * Im(N).dD/dN = f Im(N).V,
        #
        # where D=fH is the dispersion function of the considered
        # wave mode, f is the scale factor and H is the ray tracing
        # Hamiltonian and V = dH/dN. Since, according to the assumption above,
        # 
        #   Im(N) = absImN * Nperp / |Nperp|
        # 
        # where Nperp = N - Nparallel b, one finds
        #
        #   gamma = k0 * f * absImN * (Nperp.V) / |Nperp|
        #         = k0 * f * absImN * (Nperp.V_perp) / |Nperp|.
        #
        # The remaining scalar product reads
        #
        #   Nperp.Vperp = |Nperp| |Vperp| cos(Nperp^Vperp),
        #
        # where Nperp^Vperp is the angle between Nperp and Vperp.
        # In TORBEAM this is replaced by the angle N^V between the 
        # full refractive index and the full velocity, leading to
        #
        #   gamma = k0 * f * absImN * |V_perp)| cos(N_perp^V_perp),
        #         = k0 * f * absImN * |V| * sin(V^b) * cos(N_perp^V_perp).
        #
        # In the usual implementation (cf. TORBEAM source code) one
        # set cos(N_perp^V_perp) = 1., with the result that
        #
        #   gamma = k0 * absImN * f * Vnorm * sinVb 
        #
        # which is implemented here.
        #
        # the relevant part.
        
        # Part common to both methods (local Cartesian components of the
        # equilibrium magnetic field)
        Vx = derNx
        Vy = derNy
        Vz = derNz   
        Vnorm = math.sqrt(Vx**2 + Vy**2 + Vz**2)

        sinphi = Y / R
        cosphi = X / R
        Bx = BR * cosphi - Bt * sinphi
        By = BR * sinphi + Bt * cosphi

        # gamma as in TORBEAM
        cosVb = abs(Vx * Bx + Vy * By + Vz * Bz) / (Bnorm * Vnorm)
        sinVb = math.sqrt(1. - cosVb**2) 
        gamma = self.k0 * absImN * f * Vnorm * sinVb 

# test - NON-STANDARD VERSION 
        # # The scalar product Nperp.Vperp can however be computed exacly,
        # #
        # #   N.V = Nparallel*Vparallel + Nperp.Vperp,
        # #
        # # leading to the expression
        # #
        # VdotN = Vx * Nx + Vy * Ny + Vz * Nz
        # Vparallel = abs(Vx * Bx + Vy * By + Vz * Bz) / Bnorm
        # VperpNperp = VdotN - Vparallel * Nparallel
        # gamma = self.k0 * absImN * f * VperpNperp / Nperp
# end test -

        # and return
        return gamma





    ######################################################################
    # PROVIDE A RAY-TRACING FUNCTION
    # THAT PROVIDES TO THE INTEGRATOR THE DERIVATIVE OF THE 
    # OBSERVED VARIABLES
    ######################################################################
    def __ray_trace_function__(self,t,variables,sigma):
        """function which provides the derivatives of (x,y,z,Nx,Ny,Nz,Wfct)
        needed for the Runge-Kutta solver. The mode index is passed as a 
		parameter."""

        X = variables[0]
        Y = variables[1]
        Z = variables[2]
        Nx = variables[3]
        Ny = variables[4]
        Nz = variables[5]
        Wfct = variables[6]

        if math.isnan(variables[0]):
            print('WKBeam WARNING')
            print('Nan detected in r.h.s. of ray equations at '.format(variables))
            print('Returning r.h.s. = 0.0')
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        R = disROutOf(X,Y)
        
        Bt = self.Eq.BtInt.eval(R,Z)
        BR = self.Eq.BRInt.eval(R,Z)
        Bz = self.Eq.BzInt.eval(R,Z)
        Bnorm = math.sqrt(Bt**2+BR**2+Bz**2)
        dBt_dR = self.Eq.BtInt.derx(R,Z)
        dBR_dR = self.Eq.BRInt.derx(R,Z)
        dBz_dR = self.Eq.BzInt.derx(R,Z)
        dBt_dz = self.Eq.BtInt.dery(R,Z)
        dBR_dz = self.Eq.BRInt.dery(R,Z)
        dBz_dz = self.Eq.BzInt.dery(R,Z)
        psi = self.Eq.PsiInt.eval(R,Z)
        dpsi_dR = self.Eq.PsiInt.derx(R,Z)
        dpsi_dz = self.Eq.PsiInt.dery(R,Z)
        
        Ne = self.Eq.NeInt.eval(R,Z)
        dNe_dR = self.Eq.NeInt.derx(R,Z)
        dNe_dz = self.Eq.NeInt.dery(R,Z)

        # in case the interpolation produces negative electron density.
        if Ne < 0: 
            Ne = 0.

        derX,derY,derZ,derNx,derNy,derNz = disHamiltonianDerivatives(self.omega, 
                                                                     Bt, dBt_dR, dBt_dz,
                                                                     BR, dBR_dR, dBR_dz,
                                                                     Bz, dBz_dR, dBz_dz,
                                                                     Ne, dNe_dR, dNe_dz,
                                                                     X,Y,Z,
                                                                     Nx,Ny,Nz,
                                                                     sigma,
                                                                     self.epsilonRegS)

        # compute absorption coefficient if needed
        if self.computeAbsorption == True:
            dWfct_dt = -2. * self.absorptionCoefficient * Wfct
        else:   # if absorption coefficient is assumed to vanish anyway
            dWfct_dt = 0.
       
        return [derNx,derNy,derNz,-derX,-derY,-derZ,dWfct_dt]


    ######################################################################
    # PROVIDE A RAY-TRACING FUNCTION
    # THAT PROVIDES TO THE INTEGRATOR THE DERIVATIVE OF THE 
    # OBSERVED VARIABLES FOR THE CASE OF FREE SPACE AND OTHER GENERIC
    # HAMILTONIANS
    ######################################################################
    def __ray_trace_function_vac__(self,t,variables,sigma):
        """function which provides the derivatives for the Runge-Kutta
        solver using some toy model Hamiltonian."""
        X = variables[0]
        Y = variables[1]
        Z = variables[2]
        Nx = variables[3]
        Ny = variables[4]
        Nz = variables[5]
        Wfct = variables[6]

        if self.linearlayer == True:
            derX = - 2./self.linearlayervalleyL
            derY = 0.
            derZ = 0.
            derNx = 4.*Nx
            derNy = 4.*Ny
            derNz = 4.*Nz   
        elif self.valley == True:
            derX = 0.
            derY = 4.*Y/self.linearlayervalleyL**2
            derZ = 4.*Z/self.linearlayervalleyL**2
            derNx = 4.*Nx
            derNy = 4.*Ny
            derNz = 4.*Nz   
        elif self.paraxialapprox == True:
            derX = 0.
            derY = 0.
            derZ = 0.
            derNx = -4.
            derNy = 4.*Ny
            derNz = 4.*Nz   
        else:
            derX = 0.
            derY = 0.
            derZ = 0.
            derNx = 4.*Nx
            derNy = 4.*Ny
            derNz = 4.*Nz   
                                                                     
        # in case absorption is desired in idata, also calculate the derivative of the Wfct with respect to time.
        # therefore, use the model chosen
        if self.absorption == True:
            if X > self.absorptionLayerX:
                dWfct_dt = 0.
            else:
                dWfct_dt = -3. * Wfct
        else:
            dWfct_dt = 0.  

        return [derNx,derNy,derNz,-derX,-derY,-derZ,dWfct_dt]





    ######################################################################
    # Initialice the ray
    ######################################################################
    def initializeRay(self,t,X0,Y0,Z0,Nx0,Ny0,Nz0,Wfct,sigma0,stretchtorightlength,equilibrium,turnoffscattering):
        
        """Initialices the ode integrator with the given initial values."""

        # create an integrator object (call the right derivatives function corresponding on
        # if vacuum or plasma
	
        if equilibrium == 'Tokamak' or equilibrium == 'Tokamak2D' or equilibrium == 'Axisymmetric':
            self.r = ode(self.__ray_trace_function__).set_integrator('dopri5',
                                                                     rtol=self.integratorRelTol,
                                                                     atol=self.integratorAbsTol,
                                                                     nsteps=self.integratorMaxNSteps)  
        elif equilibrium == 'Model':
            self.r = ode(self.__ray_trace_function_vac__).set_integrator('dopri5',
                                                                         rtol=self.integratorRelTol,
                                                                         atol=self.integratorAbsTol,
                                                                         nsteps=self.integratorMaxNSteps)  
            
        else:
            msg = "Input keyword 'equilibrium' not understood in ray initialization."
            raise ValueError(msg)        

        # turn scattering off if needed
        self.turnoffscattering = turnoffscattering

        # set initial values, t=0
        self.r.set_initial_value([X0,Y0,Z0,Nx0,Ny0,Nz0,Wfct],0)

        # set parameters the ray trace function needs.
        self.r.set_f_params(sigma0)   

        # create an array where the results can be stored and use the first entry for the initial values
        # index 0-2: X,Y,Z
        # index 3-5: Nx, Ny, Nz
        # index 6: power
        self.rayPoints = np.empty([7,self.npt])
        self.rayPoints[0,0] = X0
        self.rayPoints[1,0] = Y0
        self.rayPoints[2,0] = Z0
        
        # stretch the wavevector to length such that the dispersion relation is fullfilled
        # this is needed in case around the antenna plane, there is a small electron density
        # (don't do that if vanishing electron density is concidered)
        if stretchtorightlength == True:
            R = disROutOf(X0,Y0)
            Bt = self.Eq.BtInt.eval(R,Z0)
            BR = self.Eq.BRInt.eval(R,Z0)
            Bz = self.Eq.BzInt.eval(R,Z0)
            Bnorm = math.sqrt(Bt**2 + BR**2 + Bz**2)
            Ne = self.Eq.NeInt.eval(R,Z0)
            if Ne < 0.: 
                Ne = 0.
            alpha, beta = disrotMatrixAngles(Bt, BR, Bz, X0, Y0, Z0)
            
            Nparallel,Nperp,phiN = disNparallelNperpphiNOutOfNxNyNz(alpha, beta, Nx0, Ny0, Nz0)
        	
            f = fsolve(lambda x: x*disNperp(self.omega, Bnorm, Ne, Nparallel/x, sigma0, self.epsilonRegS)-Nperp,  #fct to solve
                       1.) #starting estimate for length factor
            

            Nparallel /= f
            Nperp /= f
            f = 1. 
    
            self.rayPoints[3,0], self.rayPoints[4,0], self.rayPoints[5,0] = \
                disNxNyNzOutOfNparallelNperpphiN(alpha, beta, Nparallel, Nperp, phiN)
        else:
            self.rayPoints[3,0] = Nx0
            self.rayPoints[4,0] = Ny0
            self.rayPoints[5,0] = Nz0
        

        # Wigner function for the ray
        self.rayPoints[6,0] = Wfct

        # Wave mode
        self.rayMode = np.empty([self.npt])
        self.rayMode[0] = sigma0

        # create an array, where the time for the ray points is stored.
        self.time = np.empty([self.npt])
        self.time[0] = 0.    # rays are started at t=0

        # and note that the ray has not been scattered (if it has, this is changed when it is)
        self.numberofscatteringevents = 0
  
        # initialise absorption started
        self.absorptionStarted = -1.
        
        # only start computing the absorption when it is around the absorption region
        self.computeAbsorption = False  #(is set to true when needed)

        return None


    
    ######################################################################
    # Print information along the ray when needed
    # This function is called only if one single ray is used (info=True).
    ######################################################################
    def estimate_scattering_and_print_diagnostics(self, index, time, timestep):

        """
        Compute and print on stdout a few diagnostic quantities along
        the ray.
        """

        X = self.rayPoints[0,index]
        Y = self.rayPoints[1,index]
        Z = self.rayPoints[2,index]
        Nx = self.rayPoints[3,index]
        Ny = self.rayPoints[4,index]
        Nz = self.rayPoints[5,index]
        Wfct = self.rayPoints[6,index]

        sigma = self.rayMode[index]

        R = math.sqrt(X**2+Y**2)
        psi = self.Eq.PsiInt.eval(R,Z)
        rho = math.sqrt(psi)
    
        Bt = self.Eq.BtInt.eval(R,Z)
        BR = self.Eq.BRInt.eval(R,Z)
        Bz = self.Eq.BzInt.eval(R,Z)
        Bnorm = math.sqrt(Bt**2+BR**2+Bz**2)

        Ne = self.Eq.NeInt.eval(R,Z)
        if Ne < 0.: Ne = 0.
        Te = self.Eq.TeInt.eval(R,Z)
        if Te < 0.: Te = 0.

        alpha, beta = disrotMatrixAngles(Bt, BR, Bz, X, Y, Z)
        Nparallel,Nperp,phiN = disNparallelNperpphiNOutOfNxNyNz(alpha, beta, Nx, Ny, Nz)
        f = abs(4.*disTrDispersionMatrixDivBySHamiltonian(self.omega, Bnorm, Ne, Nparallel, sigma))
                                        
        print('ray Point %i: t=%f; X,Y,Z = %f, %f, %f; Nx,Ny,Nz=%f, %f, %f, sigma = %f, rho=%f, Wfct = %f, factor f=%f\n' %(index, time,
                                                                                                                            X,Y,Z, Nx, Ny, Nz, 
                                                                                                                            sigma, rho, Wfct, f))
        print('Ne = {}'.format(Ne), 'omega_p/omega = {}'.format(disParamomegaP(Ne) / self.omega))
        print('absorption coefficient = {}'.format(self.absorptionCoefficient))
        print('phiN=%f\n' %(phiN))


        # estimate the mean number of scattering events
        if self.scattering == True and self.equilibrium != 'Model':

            Raxis, zaxis = self.Eq.magn_axis_coord_Rz
            theta = atanRightBranch(Z-zaxis,R-Raxis)
            self.ScatteringDistr.timestep = timestep 
            increment_meannumberofscatteringevents = self.ScatteringDistr.EstimateMeanNumberOfScatteringEvents(Bnorm, Ne, Te,
                                                                                                               rho,theta,
                                                                                                               f, 
                                                                                                               Nparallel, 
                                                                                                               Nperp, phiN, sigma)
            increment_meannumberofmodetomodescatteringevents = self.ScatteringDistr.EstimateMeanNumberOfModeToModeScatteringEvents(Bnorm, Ne, Te,
                                                                                                                                   rho,theta,
                                                                                                                                   f, 
                                                                                                                                   Nparallel, 
                                                                                                                                   Nperp, phiN, sigma)
        else:
            increment_meannumberofscatteringevents = 0
            increment_meannumberofmodetomodescatteringevents = 0            

        return increment_meannumberofscatteringevents, increment_meannumberofmodetomodescatteringevents
        

    ######################################################################
    # Print diagnostics on the whole ray and write hdf5 file for the
    # test on the dispersion relation
    # This function is called only if one single ray is used (info=True).
    ######################################################################
    def print_ray_and_dispersion_diagnostics(self, loctimesteps, n_events, n_events_modetomode):
        
        """
        Print diagnostics information on the whole ray after the tracing 
        procedure has ended and write the hdf5 file with the information on
        the dispersion relation.
        """

        # Mean numbers of scattering events
        if self.scattering == True:
            print('mean number of scattering events for this ray: %f \n' %(n_events))
            print('mean number of mode-to-mode scattering events for this ray: %f \n' %(n_events_modetomode))

        # store diagnostics output file
        # store hdf5 file with electron density, dispersion surfaces and scattering intensities
        if hasattr(self, 'dispersionSurfacesOnCentralRay'):

            filename = self.output_dir + self.dispersionSurfacesOnCentralRay
            fid = h5py.File(filename,'w') 

            timeline = self.time[0:self.npt]
            
            psi = np.zeros([self.npt])
            eldens = np.zeros([self.npt])
            eltemp = np.zeros([self.npt])
            rayXYZ = np.zeros([3,self.npt])
            scattRay = np.zeros([3,self.npt])
            Nparallel = np.zeros([self.npt])
            Nperp = np.zeros([self.npt])
            Mode = np.zeros([self.npt])
            StixParamX = np.zeros([self.npt])
            StixParamY = np.zeros([self.npt])
            absorption = np.zeros([self.npt])
            NperpOtherMode = np.zeros([self.npt])
            Lperp_on_ray = np.zeros([self.npt])
            Lparallel_on_ray = np.zeros([self.npt])
            IntensityOfScatteringEvents = np.zeros([self.npt])
            IntensityOfScatteringEventsOffDiagonal = np.zeros([self.npt])
                
            sigma0 = self.rayMode[0]
            for i in range(0,self.npt):

                timestep = loctimesteps[i]
                
                X = self.rayPoints[0,i]
                Y = self.rayPoints[1,i]
                Z = self.rayPoints[2,i]
                Nx = self.rayPoints[3,i]
                Ny = self.rayPoints[4,i]
                Nz = self.rayPoints[5,i]
                sigma = self.rayMode[i]
    
                # define poloidal coordinates
                R = math.sqrt(X**2+Y**2)
                z = Z
                rayXYZ[:,i] = np.array([X, Y, Z])

                # read magnetic field and electron density
                Bt = self.Eq.BtInt.eval(R,z)
                BR = self.Eq.BRInt.eval(R,z)
                Bz = self.Eq.BzInt.eval(R,z)
                Bnorm = math.sqrt(Bt**2+BR**2+Bz**2)
                psi[i] = self.Eq.PsiInt.eval(R,z)
                eldens[i] = max(0., self.Eq.NeInt.eval(R,z))
                eltemp[i] = max(0., self.Eq.TeInt.eval(R,z))
                StixParamX[i] = disParamomegaP(eldens[i])**2 / self.omega**2
                StixParamY[i] = disParamOmega(Bnorm) / self.omega
                f = abs(4.*disTrDispersionMatrixDivBySHamiltonian(self.omega,
                                                                  Bnorm, eldens[i],
                                                                  Nparallel[i],
                                                                  sigma)) 

                # polar coordinates
                Raxis, zaxis = self.Eq.magn_axis_coord_Rz
                rho = math.sqrt(psi[i])
                theta = atanRightBranch(Z-zaxis,R-Raxis)
                
                # compute absorption
                if self.time[i] > self.startAbsorptionComputation and self.absorption:
                    absorption[i] = self.__absorption_coefficient__(X, Y, Z, Nx, Ny, Nz, sigma)                    

                # compute refractive index vector components alined to the magnetic field
                alpha, beta = disrotMatrixAngles(Bt, BR, Bz, X, Y, Z)
                Nparallel[i],Nperp[i],phiN = disNparallelNperpphiNOutOfNxNyNz(alpha, beta, Nx, Ny, Nz)
                    
                # compute the refractive index Nperp-component for the other mode
                NperpOtherMode[i] = disNperp(self.omega, Bnorm, eldens[i], Nparallel[i], -sigma, self.epsilonRegS)
                    
                # compute scattering probability
                if self.scattering == True and Nperp[i]**2 + Nparallel[i]**2 > 0.0:
                    Lparallel_on_ray[i] = self.Lparallel(rho, theta, eldens[i], eltemp[i], Bnorm)
                    Lperp_on_ray[i] = self.Lperp(rho, theta, eldens[i], eltemp[i], Bnorm)
                    sigma_perp = 1.0 / (self.k0 * Lperp_on_ray[i])
                    self.ScatteringDistr.timestep = timestep 
                    IntensityOfScatteringEvents[i] = self.ScatteringDistr.EstimateMeanNumberOfScatteringEvents(Bnorm, eldens[i], eltemp[i],
                                                                                                               math.sqrt(psi[i]),theta,
                                                                                                               f, 
                                                                                                               Nparallel[i], 
                                                                                                               Nperp[i], phiN, sigma) / timestep
                    IntensityOfScatteringEventsOffDiagonal[i] = self.ScatteringDistr.EstimateMeanNumberOfModeToModeScatteringEvents(Bnorm, eldens[i], eltemp[i],
                                                                                                                                    math.sqrt(psi[i]),theta,
                                                                                                                                    f, 
                                                                                                                                    Nparallel[i], 
                                                                                                                                    Nperp[i], phiN, sigma) / timestep
            # Indentation: out of the loop over time!!!                    
            fid.create_dataset("timeline", data=timeline)
            fid.create_dataset("rayXYZ", data=rayXYZ)
            fid.create_dataset("psi", data=psi)
            fid.create_dataset("eldens", data=eldens)
            fid.create_dataset("eltemp", data=eltemp)
            fid.create_dataset("Nparallel", data=Nparallel)
            fid.create_dataset("Nperp", data=Nperp)
            fid.create_dataset("StixParamX", data=StixParamX)
            fid.create_dataset("StixParamY", data=StixParamY)
            fid.create_dataset("absorption", data=absorption)
            fid.create_dataset("Mode",data=self.rayMode)
            fid.create_dataset("NperpOtherMode", data=NperpOtherMode)
            fid.create_dataset("k0",data=self.k0)
            
            if self.scattering == True:
                fid.create_dataset("Lperp",data=Lperp_on_ray)
                fid.create_dataset("Lparallel",data=Lparallel_on_ray)
                fid.create_dataset("sigma_perp", data=sigma_perp)
                fid.create_dataset("IntensityOfScatteringEvents", data=IntensityOfScatteringEvents)
                fid.create_dataset("IntensityOfScatteringEventsOffDiagonal", data=IntensityOfScatteringEventsOffDiagonal)
                
            fid.close()

        else:
            pass


        return None
        

    ######################################################################
    # perform the integration steps
    ######################################################################
    def traceRay(self,findabsorption=False,info=False):

        """Function that performs the integration steps.
        The result is stored in the self.rayTrace array."""
        
        PowerAbsorbed = False    # if power is absorbed, this will be set true and all the following points are set to 0.
        WasInsidePlasma = False  # if Ne was already above the treshhold
        OutsidePlasma = False    # if Ne is small enough that one considers the ray to be in plasma. Is stored to 
                                 # abord in reflectometrie case.

        
        time = 0.
        timestep = self.timestep

        # for diagnostic purposes: local time step an mean number of scattering events
        loctimesteps = np.ones([self.npt])*self.timestep
        meannumberofscatteringevents = 0.
        meannumberofmodetomodescatteringevents = 0.

        # Perform timesteps
        for i in range(1,self.npt):

            # change timestep, when it comes to absorption
            if findabsorption == False:
                if self.absorption == True and self.equilibrium != 'Model':
                    if time >= self.startAbsorptionComputation:
                        timestep = self.timestep / self.absorptionSmallerTimestepsFactor
                        loctimesteps[i] = timestep
                        
            # if there is no more power, write zeros to the result array.
            if (PowerAbsorbed == True) \
                    or (self.reflektometrie == True and WasInsidePlasma == True and OutsidePlasma == True):
                for j in range(0,7):
                    self.rayPoints[j,i] = 0.
                self.rayMode[i]=self.rayMode[i-1]
                self.time[i] = 0.0
               
            # if there is still some power trace the ray
            else:
                # compute the absorption coefficient which will be used for the whole timestep
                if self.absorption == True and self.equilibrium != 'Model' \
                        and time >= self.startAbsorptionComputation:
                    self.absorptionCoefficient = self.__absorption_coefficient__(self.rayPoints[0,i-1],
                                                                                 self.rayPoints[1,i-1], 
                                                                                 self.rayPoints[2,i-1],
                                                                                 self.rayPoints[3,i-1],
                                                                                 self.rayPoints[4,i-1], 
                                                                                 self.rayPoints[5,i-1],
                                                                                 self.rayMode[i-1])
                    
                    # note, that now also the absorption coefficients must be computed.
                    # in derivatives.
                    # do not compute them when the absorption position is found
                    if findabsorption == False:
                        self.computeAbsorption = True
                    
                    # if absorption coefficient is non-negligible
                    # save time when it started.
                    if self.absorptionCoefficient >= self.absorptionConsiderAsNonZeroTreshhold:
                        if self.absorptionStarted == -1.:
                            self.absorptionStarted = time
                    


                else:
                    self.absorptionCoefficient = 0.

                # now integrate using this stepsize
                self.r.integrate(self.r.t+timestep)  # integrate   
          

                self.rayPoints[:,i] = self.r.y            # and save the result in the array
                time = self.r.t
                self.time[i] = time                       # and save the corresponding time

                self.rayMode[i] = self.rayMode[i-1]       # mode is constant during integration

                # print information, if only one ray is traced
                if info==True:
                    dn, dn_modetomode = self.estimate_scattering_and_print_diagnostics(i, time, timestep)
                    meannumberofscatteringevents += dn
                    meannumberofmodetomodescatteringevents += dn_modetomode                    
                    
                # if scattering is turned on, perform the corresponding MC-step here
                if self.scattering == True and self.equilibrium != 'Model' and self.turnoffscattering == False:
                    
                    self.ScatteringDistr.timestep = timestep
                    # plasma scattering
                    
                    # read recent ray points
                    X = self.rayPoints[0,i]
                    Y = self.rayPoints[1,i]
                    Z = self.rayPoints[2,i]
                    Nx = self.rayPoints[3,i]
                    Ny = self.rayPoints[4,i]
                    Nz = self.rayPoints[5,i]

                    # read recent wave mode
                    sigma = self.rayMode[i]

                    # define poloidal coordinates
                    R = math.sqrt(X**2+Y**2)
                    z = Z
                        
                    # read magnetic field and electron density
                    Bt = self.Eq.BtInt.eval(R,z)
                    BR = self.Eq.BRInt.eval(R,z)
                    Bz = self.Eq.BzInt.eval(R,z)
                    Bnorm = math.sqrt(Bt**2+BR**2+Bz**2)
                    psi = self.Eq.PsiInt.eval(R,z)
                    Ne = self.Eq.NeInt.eval(R,z)
                    if Ne < 0.: Ne = 0.
                    Te = self.Eq.TeInt.eval(R,z)                    
                    if Te < 0.: Te = 0.
                            
                    # compute refractive index vector components alined to the magnetic field
                    alpha, beta = disrotMatrixAngles(Bt, BR, Bz, X, Y, Z)
                    Nparallel,Nperp,phiN = disNparallelNperpphiNOutOfNxNyNz(alpha, beta, Nx, Ny, Nz)

                    # magnetic axis coordinates 
                    rho = math.sqrt(psi)
                    Raxis, zaxis = self.Eq.magn_axis_coord_Rz
                    theta = atanRightBranch(z-zaxis,R-Raxis)

                    # group velocity correction factor
                    f = abs(4.*disTrDispersionMatrixDivBySHamiltonian(self.omega,
                                                                      Bnorm,Ne,
                                                                      Nparallel,
                                                                      sigma))


                    # perform scattering step
                    scatter = self.ScatteringDistr.DecideScattering(Bnorm,Ne,Te,rho,theta,f,Nparallel,Nperp,phiN,sigma)
                    # see if scattering step has taken place
                    if scatter == True:
                        # if yes, compute the new cartesian refractive index components and reinitialise solver object
                        newNx, newNy, newNz = disNxNyNzOutOfNparallelNperpphiN(alpha, beta,
                                                                                   self.ScatteringDistr.newNparallel,
                                                                                   self.ScatteringDistr.newNperp,
                                                                                   self.ScatteringDistr.newphiN)
                        newMode = self.ScatteringDistr.newMode
                        self.r.set_initial_value([X,Y,Z,newNx,newNy,newNz,self.rayPoints[6,i]],time)
                        self.r.set_f_params(newMode)
                        self.rayMode[i] = newMode
                        # note, that the ray has been scatterer
                        self.numberofscatteringevents += 1

                # see if the Wfct is below the treshhold value 
                if self.rayPoints[6,i] < self.rayPoints[6,0] * self.absorptionWfctTreshhold:
                    if self.absorption == True:
                        PowerAbsorbed = True    # if it is, in the next steps 
                                                # write zeros to the data array


                # see if one is outside the plasma in refl. case
                if self.reflektometrie == True:
                     R = disROutOf(self.rayPoints[0,i],self.rayPoints[1,i])
                     psi = self.Eq.PsiInt.eval(R,self.rayPoints[2,i])
                 
                     # if inside, set the corresponding flag
                     if math.sqrt(psi) < self.reflektometrierhoTreshhold:
                         WasInsidePlasma = True
                     # if outside, set the corresponding flag
                     if WasInsidePlasma == True and math.sqrt(psi) > self.reflektometrierhoTreshhold:
                         OutsidePlasma = True

        if info==True:
            self.print_ray_and_dispersion_diagnostics(loctimesteps,
                                                      meannumberofscatteringevents,
                                                      meannumberofmodetomodescatteringevents)
               

#END OF FILE
#
