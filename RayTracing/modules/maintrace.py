"""In this file, the main-function for the ray-tracing cores is given.
It waits for the initial parameters sent by the org-core and does the ray tracing. 
After each ray, the trajectory is sent back to the organisation core via MPI."""


#
# MAIN FILE FOR RAY TRACING CORES
#


###########################################################################
# MAIN ROUTINE FOR RAY TRACING CORES
###########################################################################

# Load standard modules
import sys
import numpy as np
import math
from mpi4py import MPI
# Load local modules
import CommonModules.physics_constants as phys
from CommonModules.input_data import InputData
from RayTracing.modules.trace_one_ray import TraceOneRay
from RayTracing.modules.dispersion_matrix_cfunctions import *   
from RayTracing.modules.atanrightbranch import atanRightBranch

###########################################################################
# MAIN ROUTINE FOR RAY TRACING CORES
###########################################################################
def mainTrace(idata, comm):
    
    #######################################################################
    # ORGANISE MPI WORLD
    #######################################################################
    rank=comm.rank      # core index
    size = comm.size    # number of cores in total

    print("rank %i waiting for ray parameters ...\n" %(rank))
    sys.stdout.flush()

    # Number of CPUS per group
    nmbrCPUperGroup = idata.nmbrCPUperGroup

    # wait for initial ray parameters to trace
    # ... find the rank of the control CPU for this process ...
    sourceCPU = int(rank / nmbrCPUperGroup) * nmbrCPUperGroup
    # ... receive data from the control CPU ...
    ray_parameters_to_trace = comm.recv(source=sourceCPU, tag=1)

    print("rank %i ray parameters received\n" %(rank))
    sys.stdout.flush()
   

    #######################################################################
    # DEFINE SOME CONSTANTS
    #######################################################################

    # Beam frequency omega in rad/s
    omega = phys.AngularFrequency(idata.freq)  

    #######################################################################
    # DO THE RAY TRACING
    #######################################################################
    # create TraceRay-object
    TraceRay = TraceOneRay(idata, rank)

    # Load qlabs EDF once per MPI worker before the ray loop.
    # qlabs_init stores the distribution function in Fortran module-level
    # storage; all subsequent qlabs_absorption calls within this worker use it.
    if getattr(idata, 'absorptionModule', 1) == 2:
        from RayTracing.lib.qlabs.qlabsECabsorption import qlabs_init
        from RayTracing.lib.qlabs.qlabs_loader import load_luke_edf
        # Check if the flag for a Maxwellian distribution is set in the input file. If so, use the flag

        maxwellian = getattr(idata, 'qlabs_maxwellian', False)
        _p_mc, _xi_g, _psi_g, _f0 = load_luke_edf(idata.qlabs_edf_file, maxwellian=maxwellian)
        qlabs_init(_p_mc, _xi_g, _psi_g, _f0)

    # see, if stretch to right length should be turned off.
    # This stretches the refractive index vector to a length
    # such that it fullfills exactly the dispersion relation.
    # Should be done if there is a small electron density around the
    # antenna plane.
    # (is not used, when analytical toy models are considered)
    if idata.equilibrium == 'Model':
        stretchtorightlength = False
    else:
        stretchtorightlength = True

    # number of physical rays that must be traced
    # (ray_parameters_to_trace is a list of dictionaries: One dictionary for each
    #  ray to be traced, plus one pilot ray used to probe absorption along the beam
    # path. Therefore, the length of the list gives the total number of rays plus one.)
    nmbrrays = len(ray_parameters_to_trace) - 1

    # assume for the first ray, that absorption has started at t=0
    # (this means that the absorption coefficient is computed all along the ray.
    # the first ray is used to give a rough estimate of where absorption takes
    # place. For the next rays, then, absorption is turned on only a little before)
    absorptionStarted = 0.

    # Prepare the dataset 
    # Data on this ray are store in the form of a dictionary
    RayData = {'initial mode index': 0,                # initialized to zero, valid values are +1 or -1
               'orbit': np.zeros([6,TraceRay.npt]),    # phase space orbit X,Y,Z,Nx,Ny,Nz
               'Wfct':  np.zeros([TraceRay.npt]),      # Wigner factor Wfct
               'V Group': np.zeros([3,TraceRay.npt]),  # vector V = (dX/dtau, dY/dtau, dZ/dtau) 
               'N parallel': np.zeros([TraceRay.npt]), # N parallel
               'N perp': np.zeros([TraceRay.npt]),     # N perpendicular
               'phi_N': np.zeros([TraceRay.npt]),      # polar angle phi of the vector N 
               'scaling': np.zeros([TraceRay.npt]),    # HAmiltonian scaling factor
               'Psi': np.zeros([TraceRay.npt]),        # magnetic equilibrium psi on the ray
               'Theta': np.zeros([TraceRay.npt]),      # magnetic equilibrium theta on the ray
               'time': np.zeros([TraceRay.npt]),        # time-like variable tau of the ray
               'mode index': np.zeros([TraceRay.npt]), # mode index of the ray (which can change due to cross-polarization scattering)
               'probfunction': 0.0,                    # initialized to zero.
               'n. scatt. events': 0,                  # number of scattering events, initialized to zero
    }

    # Addition by Ewout. Need the magnetic axis at least to keep track of theta
    Raxis, Zaxis = TraceRay.Eq.magn_axis_coord_Rz
    
    
    # trace one ray after the other
    for i in range(-1,nmbrrays):        
  
        # Scattering is turned off in the following two conditions.
        # 1) When required in the input file (takecentralrayfirst == True)
        # the first ray tracing CPU (rank == 1) computes the central ray
        # (indexed i == 0).
        # 2) All ray tracing CPUs compute a "pilot ray" (indexed i == -1) 
        # in order to locate the absorption layer.
        TraceCentralRay = idata.takecentralrayfirst and  \
                          rank == 1 and                  \
                          i == 0
        if TraceCentralRay or i == -1: 
            turnoffscattering = True
        else:
            turnoffscattering = False

        # Initial conditions for the current ray
        # ... initial value of the time-like parameter ...
        t0 = 0.
        # ... initial position in the Cartesian system ...
        X0 = ray_parameters_to_trace[i]['X']
        Y0 = ray_parameters_to_trace[i]['Y']
        Z0 = ray_parameters_to_trace[i]['Z']
        # ... initial refractive index in the Cartesian frame ... 
        Nx0 = ray_parameters_to_trace[i]['Nx']
        Ny0 = ray_parameters_to_trace[i]['Ny']
        Nz0 = ray_parameters_to_trace[i]['Nz']
        # ... initial weight of the Wigner fucntion ...
        Wfct = ray_parameters_to_trace[i]['Wfct']
        # ... initial mode
        sigma0 = ray_parameters_to_trace[i]['initial mode index']

        # ... initialize ray tracing object ...
        TraceRay.initializeRay(t0, X0, Y0, Z0, Nx0, Ny0, Nz0, Wfct, sigma0,
                               stretchtorightlength, idata.equilibrium, turnoffscattering)

            
        # if i == -1: this is the first ray to determine the absorption position,
        # do not save the result and store, where absorption has started
        # This procedure is performed because the absorption routine is slow, 
        # which makes it beneficial to evaluate the absorption coefficient only
        # where absorption is expected, an information which is obtained from this
        # first ray, along which the absorption coefficient is computed.
        # This procedure allows further to reduce the timestep in the absorption
        # region (cf. input file parameters)
        if i == -1:
            # trace the ray to find absorption
            TraceRay.traceRay(findabsorption=True, info=False)
            absorptionStarted = TraceRay.absorptionStarted

            # set time, when absorption is considered
            # (i.e. when absorption was found in the first ray minus a shift 
            #  indicated in the input file, at lowest 0)
            if idata.absorptionStartEarlierThenCentralRay != 0.:
                TraceRay.startAbsorptionComputation = max(absorptionStarted-idata.absorptionStartEarlierThenCentralRay, 0.)
            else:
                TraceRay.startAbsorptionComputation = 0.            
            continue
        
        # Perform timesteps
        # if only one ray is traced, print out a list with the steps (diagnostics)
        if idata.nmbrRays == 1:
            TraceRay.traceRay(info=True)
        else:
            TraceRay.traceRay(info=False)

        # re-set the data arrays
        RayData['orbit'].fill(0.0)
        RayData['Wfct'].fill(0.0)
        RayData['V Group'].fill(0.0)
        RayData['N parallel'].fill(0.0) 
        RayData['N perp'].fill(0.0)
        RayData['phi_N'].fill(0.0)
        RayData['scaling'].fill(0.0)
        RayData['Psi'].fill(0.0)
        RayData['Theta'].fill(0.0)
        RayData['time'].fill(0.0)
        RayData['mode index'].fill(0.0)

        # store time-independent quantities
        RayData['initial mode index'] = TraceRay.rayMode[0]
        RayData['probfunction'] = ray_parameters_to_trace[i]['probfunction']
        RayData['n. scatt. events'] = TraceRay.numberofscatteringevents


        # Loop over the time steps and store time-dependent data
        for j in range(0,TraceRay.npt): 
        
            # if ray-tracing has been stopped before (indicated by 0 everywhere) 
            # also don't calculate further quantities and write 
            # ray tracing is stopped due to absorption or due to 
            # reflectometry when the plasma is left.
            if np.sum(TraceRay.rayPoints[0:7,j]) == 0.:
                break
        
            # store the information already available from TraceRay object.
            RayData['orbit'][0:6,j] = TraceRay.rayPoints[0:6,j]
            RayData['Wfct'][j] = TraceRay.rayPoints[6,j]
         
            # if ray tracing has not been stopped, just go on:                
            # compute Nparallel and phiN, and other quantities of interest
            # therefore: read recent ray point
            X = TraceRay.rayPoints[0,j]
            Y = TraceRay.rayPoints[1,j]
            Z = TraceRay.rayPoints[2,j]
            Nx = TraceRay.rayPoints[3,j]
            Ny = TraceRay.rayPoints[4,j]
            Nz = TraceRay.rayPoints[5,j]
            sigma = TraceRay.rayMode[j]
                        
            # compute the corresponding plasma parameters
            R = disROutOf(X,Y)
            Theta = disThetaOutOf(R-Raxis, Z-Zaxis) # This is based on the relative coordinates w.r.t. the magnetic axis
            Bt = TraceRay.Eq.BtInt.eval(R,Z)
            BR = TraceRay.Eq.BRInt.eval(R,Z)
            Bz = TraceRay.Eq.BzInt.eval(R,Z)
            Bnorm = math.sqrt(Bt**2 + BR**2 + Bz**2)
            Psi = TraceRay.Eq.PsiInt.eval(R,Z)
            Ne = TraceRay.Eq.NeInt.eval(R,Z)
            if Ne < 0.: 
                Ne = 0.
            alpha, beta = disrotMatrixAngles(Bt, BR, Bz, X, Y, Z)
            # Nparallel and phiN
            Nparallel,Nperp,phiN = disNparallelNperpphiNOutOfNxNyNz(alpha, beta, Nx, Ny, Nz)
            RayData['N parallel'][j] = Nparallel
            RayData['N perp'][j] = Nperp
            RayData['phi_N'][j] = phiN
        
            RayData['Psi'][j] = Psi
            RayData['Theta'][j] = Theta
            RayData['time'][j] = TraceRay.time[j]
            RayData['mode index'][j] = TraceRay.rayMode[j]
                        
            # Scaling factor 
            if idata.equilibrium == 'Model':
                # in case of analytical models
                RayData['scaling'][j] = 2.        
            else:
                # in case of the plasma
                RayData['scaling'][j] = abs(4.*disTrDispersionMatrixDivBySHamiltonian(omega, Bnorm, Ne, Nparallel, sigma))
        
            # Full dimensionless group velocity (grad_x H(x,N) scaled back the the 'physical' Hamiltonian by the scaling factor)
            dX_dt, dY_dt, dZ_dt, dNx_dt, dNy_dt, dNz_dt, dW_dt = TraceRay.__ray_trace_function__(0,TraceRay.rayPoints[:,j], sigma)
            RayData['V Group'][0,j] = dX_dt / RayData['scaling'][j]
            RayData['V Group'][1,j] = dY_dt / RayData['scaling'][j]
            RayData['V Group'][2,j] = dZ_dt / RayData['scaling'][j]

        # End loop over time. 
        # Send the results via MPI
        comm.send([rank,i,RayData],dest=int(rank/nmbrCPUperGroup)*nmbrCPUperGroup,tag=2)
        
# END OF FILE
