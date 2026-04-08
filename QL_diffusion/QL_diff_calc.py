# General imports
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # suppress sqrt-of-negative in ksi calculations

# Imports for python
import numpy as np
import h5py
from scipy.io import loadmat

# MPI import
from mpi4py import MPI

# WKBeam-specific imports
from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
import CommonModules.physics_constants as phys
from Tools.PlotData.PlotAbsorptionProfile.plotabsprofile import compute_deposition_profiles

# WKBacca functions import
from QL_diffusion.QL_diff_aux import *
from QL_diffusion.QL_diff_aux import _compute_Edens  # underscore names excluded from import *


# ---------------------------------------------------------------------------
# Helper: momentum / pitch-angle grids
# ---------------------------------------------------------------------------

def _load_momentum_grids(idata):
    """Load or construct the momentum and pitch-angle grids.

    Returns
    -------
    p_norm_w, p_norm_h, ksi0_w, ksi0_h : 1-D ndarrays
        Whole-grid and half-grid arrays for normalised momentum and pitch angle.
    """
    if idata.manual_grids:
        p_norm_w = np.linspace(idata.pmin, idata.pmax, idata.np)
        anglegrid = np.linspace(-np.pi, 0, idata.nksi)
        ksi0_w   = np.cos(anglegrid)

        # Half-grid: midpoints between consecutive whole-grid values
        p_norm_h = 0.5 * (p_norm_w[1:] + p_norm_w[:-1])
        ksi0_h   = 0.5 * (ksi0_w[1:]   + ksi0_w[:-1])

    else:
        try:
            grids    = loadmat(idata.gridfile)['WKBacca_grids']
            ksi0_h   = grids['ksi0_h'][0, 0][0]
            ksi0_w   = grids['ksi0_w'][0, 0][0]
            p_norm_h = grids['p_norm_h'][0, 0][0]
            p_norm_w = grids['p_norm_w'][0, 0][0]
        except Exception as e:
            print(f"Error: Could not load the grids from the provided file: {e}")
            sys.exit(1)

    return p_norm_w, p_norm_h, ksi0_w, ksi0_h


# ---------------------------------------------------------------------------
# Helper: sparse storage
# ---------------------------------------------------------------------------

def _sparsify(arr):
    """Flatten arr in Fortran order and return (values, indices) above threshold.

    LUKE reads the diffusion tensor as a sparse vector stored in Fortran
    (column-major) order so that MATLAB reshape() reconstructs the correct
    multi-dimensional layout.

    Returns
    -------
    sparse : 1-D ndarray  — non-zero values (> 1e-5)
    mask   : tuple        — flat indices of those values (output of np.where)
    """
    flat = arr.flatten(order='F')
    mask = np.where(flat > 1e-5)
    return flat[mask], mask


# ---------------------------------------------------------------------------
# Helper: save results to HDF5
# ---------------------------------------------------------------------------

def _save_results(idata, psi, ksi0_w, ksi0_h, p_norm_w, p_norm_h,
                  FreqGHz, mode, harmonics,
                  Trapksi0_h, Trapksi0_w,
                  DRF0_wh, DRF0_hw, DRF0_hh,
                  DRF0D_wh, DRF0D_hw, DRF0D_hh,
                  DRF0F_wh, DRF0F_hw,
                  DKE_calc):
    """Transpose, sparsify, and write all D_RF arrays to an HDF5 file.

    The arrays are transposed from (n_psi, np, nksi, n_harm) to
    (np, nksi, n_psi, n_harm) to match the LUKE convention.
    Sparse storage (Fortran-order flatten + threshold mask) keeps the file
    compact and avoids large zero-padded blocks.
    """
    outputname = idata.outputdirectory + idata.outputfilename + '.h5'

    # Transpose to LUKE axis ordering: [np, nksi, n_psi, n_harm]
    DRF0_wh = np.transpose(DRF0_wh, (1, 2, 0, 3))
    DRF0_hw = np.transpose(DRF0_hw, (1, 2, 0, 3))
    DRF0_hh = np.transpose(DRF0_hh, (1, 2, 0, 3))

    if DKE_calc:
        DRF0D_wh = np.transpose(DRF0D_wh, (1, 2, 0, 3))
        DRF0D_hw = np.transpose(DRF0D_hw, (1, 2, 0, 3))
        DRF0D_hh = np.transpose(DRF0D_hh, (1, 2, 0, 3))
        DRF0F_wh = np.transpose(DRF0F_wh, (1, 2, 0, 3))
        DRF0F_hw = np.transpose(DRF0F_hw, (1, 2, 0, 3))

    with h5py.File(outputname, 'w') as file:
        # --- Grid and metadata ---
        file.create_dataset('harmonics',   data=harmonics)
        file.create_dataset('psi',         data=psi)
        file.create_dataset('ksi0_w',      data=ksi0_w)
        file.create_dataset('ksi0_h',      data=ksi0_h)
        file.create_dataset('p_norm_w',    data=p_norm_w)
        file.create_dataset('p_norm_h',    data=p_norm_h)
        file.create_dataset('FreqGHz',     data=FreqGHz)
        file.create_dataset('mode',        data=mode)
        file.create_dataset('Trapksi0_h',  data=Trapksi0_h)
        file.create_dataset('Trapksi0_w',  data=Trapksi0_w)

        # --- Absorption profile from the WKBeam absorption step ---
        filename_abs     = idata.absorption_file
        filename_abs_dat = idata.absorption_data_file
        absorption_data  = compute_deposition_profiles(InputData(filename_abs), filename_abs_dat)
        file.create_dataset('rho_abs',  data=absorption_data['rho'])
        file.create_dataset('dP_dV',    data=absorption_data['dP_dV'])
        file.create_dataset('dP_drho',  data=absorption_data['dP_drho'])
        file.create_dataset('dV_drho',  data=absorption_data['dV_drho'])

        # --- Sparse D_RF matrices (main bounce integral) ---
        DRF0_wh_sp, mask_wh = _sparsify(DRF0_wh)
        DRF0_hw_sp, mask_hw = _sparsify(DRF0_hw)
        DRF0_hh_sp, mask_hh = _sparsify(DRF0_hh)

        file.create_dataset('DRF0_wh_sparse', data=DRF0_wh_sp)
        file.create_dataset('mask_DRF0_wh',   data=mask_wh)
        file.create_dataset('DRF0_hw_sparse', data=DRF0_hw_sp)
        file.create_dataset('mask_DRF0_hw',   data=mask_hw)
        file.create_dataset('DRF0_hh_sparse', data=DRF0_hh_sp)
        file.create_dataset('mask_DRF0_hh',   data=mask_hh)

        # --- Sparse DKE first-order components (only meaningful when DKE_calc=True) ---
        if DKE_calc:
            DRF0D_wh_sp, mask_D_wh = _sparsify(DRF0D_wh)
            DRF0D_hw_sp, mask_D_hw = _sparsify(DRF0D_hw)
            DRF0D_hh_sp, mask_D_hh = _sparsify(DRF0D_hh)
            DRF0F_wh_sp, mask_F_wh = _sparsify(DRF0F_wh)
            DRF0F_hw_sp, mask_F_hw = _sparsify(DRF0F_hw)
        else:
            # Store placeholder zeros so LUKE always finds these datasets
            DRF0D_wh_sp = DRF0D_hw_sp = DRF0D_hh_sp = np.zeros(1)
            mask_D_wh   = mask_D_hw   = mask_D_hh   = np.zeros(1)
            DRF0F_wh_sp = DRF0F_hw_sp = np.zeros(1)
            mask_F_wh   = mask_F_hw   = np.zeros(1)

        file.create_dataset('DRF0D_wh_sparse', data=DRF0D_wh_sp)
        file.create_dataset('mask_DRF0D_wh',   data=mask_D_wh)
        file.create_dataset('DRF0D_hw_sparse', data=DRF0D_hw_sp)
        file.create_dataset('mask_DRF0D_hw',   data=mask_D_hw)
        file.create_dataset('DRF0D_hh_sparse', data=DRF0D_hh_sp)
        file.create_dataset('mask_DRF0D_hh',   data=mask_D_hh)

        file.create_dataset('DRF0F_wh_sparse', data=DRF0F_wh_sp)
        file.create_dataset('mask_DRF0F_wh',   data=mask_F_wh)
        file.create_dataset('DRF0F_hw_sparse', data=DRF0F_hw_sp)
        file.create_dataset('mask_DRF0F_hw',   data=mask_F_hw)

    print(f'\nResults saved to {outputname}')


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def call_QLdiff(input_file):
    """Compute and save the quasi-linear RF diffusion tensor for LUKE.

    Orchestrates an MPI master/worker calculation:
      - Rank 0 (master) loads input data, constructs the task queue, distributes
        psi slices to workers, collects results, and writes the HDF5 output.
      - Ranks 1..N-1 (workers) receive individual psi slices, call D_RF to
        perform the bounce-averaged integration, and return results to master.

    The task queue is sorted in descending psi order so that the most
    expensive (deeply trapped) psi values are dispatched first.

    Parameters
    ----------
    input_file : str
        Path to the QLdiff configuration file (e.g. QLdiff.txt).

    Returns
    -------
    0 on success.
    """
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # -----------------------------------------------------------------------
    # Rank 0: load input data and build task queue
    # -----------------------------------------------------------------------
    if rank == 0:
        idata      = InputData(input_file)
        configfile = idata.configfile
        eqdata     = InputData(configfile)
        Eq         = TokamakEquilibrium(eqdata)
        harmonics  = np.array(idata.harmonics)
        DKE_calc   = idata.DKE_calc

        # Momentum / pitch-angle grids
        p_norm_w, p_norm_h, ksi0_w, ksi0_h = _load_momentum_grids(idata)

        # Load 4-D (rho, Theta, Nparallel, Nperp) binned ray-tracing data
        try:
            filename_RhoThetaN = idata.outputdirectory + idata.binned_data_filename
            WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, \
                rhobins, thetabins, Nparallelbins, Nperpbins = read_h5file(filename_RhoThetaN)
        except Exception as e:
            print(f"Error: Could not find the binned data for the quasilinear calculation: {e}")
            sys.exit(1)

        # Bin centres and widths
        d_psi   = np.diff(rhobins**2)
        psi     = 0.5 * (rhobins[1:]**2  + rhobins[:-1]**2)
        d_theta = np.diff(thetabins)[0]
        theta   = 0.5 * (thetabins[1:]  + thetabins[:-1])
        d_npar  = np.diff(Nparallelbins)[0]
        Npar    = 0.5 * (Nparallelbins[1:] + Nparallelbins[:-1])
        d_nperp = np.diff(Nperpbins)[0]
        Nperp   = 0.5 * (Nperpbins[1:]  + Nperpbins[:-1])

        # Convert WKBeam Wfct to k-space energy density (J/N^2)
        Edens = _compute_Edens(Wfct, psi, d_psi, theta, d_theta, Nperp, d_nperp, d_npar, Eq)

        # Reference plasma quantities for normalisation
        omega = phys.AngularFrequency(FreqGHz)
        _, _, _, _, _, _, ptNe, ptTe, _, _, _, _, _ = config_quantities(psi, theta, omega, Eq)
        Ne_ref = np.amax(ptNe)
        Te_ref = np.amax(ptTe)

        # Allocate result arrays: shape (n_psi, np, nksi, n_harm)
        DRF0_wh  = np.zeros((len(psi), len(p_norm_w), len(ksi0_h), len(harmonics)))
        DRF0_hw  = np.zeros((len(psi), len(p_norm_h), len(ksi0_w), len(harmonics)))
        DRF0_hh  = np.zeros((len(psi), len(p_norm_h), len(ksi0_h), len(harmonics)))
        if DKE_calc:
            DRF0D_wh = np.zeros_like(DRF0_wh)
            DRF0D_hw = np.zeros_like(DRF0_hw)
            DRF0D_hh = np.zeros_like(DRF0_hh)
            DRF0F_wh = np.zeros_like(DRF0_wh)
            DRF0F_hw = np.zeros_like(DRF0_hw)
        else:
            DRF0D_wh = DRF0D_hw = DRF0D_hh = np.zeros_like(psi)
            DRF0F_wh = DRF0F_hw = np.zeros_like(psi)

        Trapksi0_h = np.zeros(len(psi))
        Trapksi0_w = np.zeros(len(psi))

        # Task queue: (psi_index, psi_value, Edens_slice)
        # Sorted by descending psi so that deeply trapped (expensive) slices go first
        task_queue = [(i, psi_val, Edens[i]) for i, psi_val in enumerate(psi)]
        task_queue.sort(key=lambda x: x[1], reverse=True)

    else:
        # Workers only need the broadcast variables; initialise to None for bcast
        mode      = None
        FreqGHz   = None
        harmonics = None
        thetabins = None
        p_norm_w  = p_norm_h = None
        ksi0_w    = ksi0_h   = None
        Npar      = Nperp    = None
        Eq        = None
        Ne_ref    = Te_ref   = None
        DKE_calc  = None

    # -----------------------------------------------------------------------
    # Broadcast shared data to all ranks
    # -----------------------------------------------------------------------
    mode      = comm.bcast(mode,      root=0)
    FreqGHz   = comm.bcast(FreqGHz,   root=0)
    harmonics = comm.bcast(harmonics, root=0)
    thetabins = comm.bcast(thetabins, root=0)
    p_norm_w  = comm.bcast(p_norm_w,  root=0)
    p_norm_h  = comm.bcast(p_norm_h,  root=0)
    ksi0_w    = comm.bcast(ksi0_w,    root=0)
    ksi0_h    = comm.bcast(ksi0_h,    root=0)
    Npar      = comm.bcast(Npar,      root=0)
    Nperp     = comm.bcast(Nperp,     root=0)
    Eq        = comm.bcast(Eq,        root=0)
    Ne_ref    = comm.bcast(Ne_ref,    root=0)
    Te_ref    = comm.bcast(Te_ref,    root=0)
    DKE_calc  = comm.bcast(DKE_calc,  root=0)

    # -----------------------------------------------------------------------
    # Master process: distribute tasks and collect results
    # -----------------------------------------------------------------------
    # Tag convention: 0 = termination, 1 = new task (master → worker),
    #                 2 = result       (worker → master)
    if rank == 0:
        num_tasks     = len(task_queue)
        num_workers   = size - 1
        task_idx      = 0
        active_workers = 0
        finished_tasks = 0

        # Seed each worker with its first task
        for i in range(1, min(num_workers + 1, num_tasks + 1)):
            comm.send(task_queue[task_idx], dest=i, tag=1)
            task_idx      += 1
            active_workers += 1
            print(f'\rActive workers: {active_workers}', end='', flush=True)

        while task_idx < num_tasks or active_workers > 0:
            result = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            worker_id, idx, result_data = result
            active_workers -= 1
            finished_tasks += 1
            print(f'\rActive workers: {active_workers}, Progress: {finished_tasks}/{num_tasks}',
                  end='', flush=True)

            # Unpack worker result into the global arrays
            DRF0_wh[idx],  DRF0D_wh[idx],  DRF0F_wh[idx], \
            DRF0_hw[idx],  DRF0D_hw[idx],  DRF0F_hw[idx], \
            DRF0_hh[idx],  DRF0D_hh[idx],  \
            Trapksi0_h[idx], Trapksi0_w[idx] = result_data

            if task_idx < num_tasks:
                comm.send(task_queue[task_idx], dest=worker_id, tag=1)
                task_idx      += 1
                active_workers += 1
                print(f'\rActive workers: {active_workers}, Progress: {finished_tasks}/{num_tasks}',
                      end='', flush=True)

        # Signal all workers to exit
        for i in range(1, size):
            comm.send(None, dest=i, tag=0)

    # -----------------------------------------------------------------------
    # Worker processes: compute D_RF for each assigned psi slice
    # -----------------------------------------------------------------------
    else:
        while True:
            task = comm.recv(source=0, tag=MPI.ANY_TAG)
            if task is None:
                # Termination signal received
                break

            idx, psi_value, Edens_slice = task

            # D_RF expects a leading psi dimension of size 1
            Edens_slice = np.expand_dims(Edens_slice, axis=0)

            DRF0_wh_loc,  DRF0D_wh_loc,  DRF0F_wh_loc, \
            DRF0_hw_loc,  DRF0D_hw_loc,  DRF0F_hw_loc, \
            DRF0_hh_loc,  DRF0D_hh_loc,  \
            Trapksi0_h_loc, Trapksi0_w_loc = \
                D_RF([psi_value], thetabins, p_norm_w, p_norm_h, ksi0_w, ksi0_h,
                     Npar, Nperp, Edens_slice, Eq, Ne_ref, Te_ref,
                     n=harmonics, FreqGHz=FreqGHz, DKE_calc=DKE_calc, gaussian_smooth=True)

            result_data = (
                DRF0_wh_loc[0],  DRF0D_wh_loc[0],  DRF0F_wh_loc[0],
                DRF0_hw_loc[0],  DRF0D_hw_loc[0],  DRF0F_hw_loc[0],
                DRF0_hh_loc[0],  DRF0D_hh_loc[0],
                Trapksi0_h_loc,  Trapksi0_w_loc,
            )
            comm.send((rank, idx, result_data), dest=0, tag=2)

    # -----------------------------------------------------------------------
    # Rank 0: write HDF5 output
    # -----------------------------------------------------------------------
    if rank == 0:
        _save_results(
            idata, psi, ksi0_w, ksi0_h, p_norm_w, p_norm_h,
            FreqGHz, mode, harmonics,
            Trapksi0_h, Trapksi0_w,
            DRF0_wh, DRF0_hw, DRF0_hh,
            DRF0D_wh, DRF0D_hw, DRF0D_hh,
            DRF0F_wh, DRF0F_hw,
            DKE_calc,
        )
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    MPI.Finalize()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python QL_diff_calc.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    call_QLdiff(input_file)
    sys.exit(0)
