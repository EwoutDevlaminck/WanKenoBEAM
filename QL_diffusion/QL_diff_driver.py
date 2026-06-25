"""MPI driver for the quasi-linear RF diffusion tensor.

Orchestrates a master/worker computation:
  - Rank 0 (master): loads input data, builds task queue, seeds workers,
    collects results, writes the HDF5 output file.
  - Ranks 1..N-1 (workers): receive individual psi slices, call D_RF, send
    results back.

Task ordering is by descending number of occupied theta bins (most expensive
tasks first), which prevents one slow task from stalling the master's
collection loop near the end of the run.

The TokamakEquilibrium object is NOT broadcast (too large to pickle
efficiently).  Instead, only the config-file path is broadcast and each
worker reconstructs its own Eq from disk.
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import h5py
from scipy.io import loadmat
from mpi4py import MPI

from CommonModules.input_data import InputData
from CommonModules.PlasmaEquilibrium import TokamakEquilibrium
import CommonModules.physics_constants as phys
from scipy.spatial import KDTree

from Tools.PlotData.PlotAbsorptionProfile.plotabsprofile import compute_deposition_profiles

from QL_diffusion.QL_diff_aux import (
    read_h5file,
    config_quantities,
    _compute_Edens,
    _load_sparse_edens,
)
from QL_diffusion.QL_diff_coefficient import D_RF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_momentum_grids(idata):
    """Load momentum and pitch-angle grids from config or WKBacca_grids.mat.
    Also the reference electron density, temperature, and Coulomb logarithm if present in the .mat file.

    Parameters
    ----------
    idata : InputData

    Returns
    -------
    p_norm_w, p_norm_h, ksi0_w, ksi0_h : 1-D ndarrays
        Whole-grid and half-grid momentum and pitch-angle arrays.
    Ne_ref : float or None   Reference electron density [1e19 m⁻³].
    Te_ref : float or None   Reference electron temperature [keV].
    lnc_e_ref : float or None   LUKE Coulomb logarithm.
    """
    Ne_ref = Te_ref = lnc_e_ref = None

    if idata.manual_grids:
        p_norm_w = np.linspace(idata.pmin, idata.pmax, idata.np)
        ksi0_w   = np.cos(np.linspace(-np.pi, 0, idata.nksi))
        p_norm_h = 0.5 * (p_norm_w[1:] + p_norm_w[:-1])
        ksi0_h   = 0.5 * (ksi0_w[1:]   + ksi0_w[:-1])
    else:
        try:
            grids    = loadmat(idata.gridfile)['WKBacca_grids']
            ksi0_h   = grids['ksi0_h'][0, 0][0]
            ksi0_w   = grids['ksi0_w'][0, 0][0]
            p_norm_h = grids['p_norm_h'][0, 0][0]
            p_norm_w = grids['p_norm_w'][0, 0][0]
            if 'ne_ref' in grids.dtype.names:
                Ne_ref    = float(grids['ne_ref'][0, 0]) * 1e-19   # m⁻³ → 1e19 m⁻³
                Te_ref    = float(grids['Te_ref'][0, 0])            # keV
                lnc_e_ref = float(grids['lnc_e_ref'][0, 0])
        except Exception as exc:
            print(f"Error: Could not load grids from {idata.gridfile}: {exc}")
            sys.exit(1)

    return p_norm_w, p_norm_h, ksi0_w, ksi0_h, Ne_ref, Te_ref, lnc_e_ref


def _sparsify(arr):
    """Flatten arr in Fortran order and return (values, indices) above threshold.

    LUKE reads the diffusion tensor as a sparse vector in column-major order so
    that MATLAB reshape() reconstructs the correct multi-dimensional layout.

    Parameters
    ----------
    arr : ndarray   Any shape.

    Returns
    -------
    sparse : 1-D ndarray   Non-zero values (> 1e-5).
    mask   : tuple         Flat indices of those values.
    """
    flat = arr.flatten(order='F')
    mask = np.where(flat > 1e-5)
    return flat[mask], mask


def _save_results(idata, psi, ksi0_w, ksi0_h, p_norm_w, p_norm_h,
                  FreqGHz, mode, harmonics,
                  Trapksi0_h, Trapksi0_w,
                  DRF0_wh, DRF0_hw, DRF0_hh,
                  DRF0D_wh, DRF0D_hw, DRF0D_hh,
                  DRF0F_wh, DRF0F_hw,
                  DKE_calc):
    """Transpose, sparsify, and write all D_RF arrays to an HDF5 output file.

    Arrays are transposed from (n_psi, n_p, n_ksi, n_harm) to
    (n_p, n_ksi, n_psi, n_harm) to match the LUKE convention before
    Fortran-order sparse storage.

    Parameters
    ----------
    idata      : InputData
    psi        : 1-D ndarray   Radial grid.
    ksi0_w/h   : 1-D ndarrays  Pitch-angle grids.
    p_norm_w/h : 1-D ndarrays  Momentum grids.
    FreqGHz    : float
    mode       : int
    harmonics  : 1-D ndarray
    Trapksi0_h/w : 1-D ndarrays  Trapping boundaries.
    DRF0_*, DRF0D_*, DRF0F_* : ndarrays  Diffusion and drag/convection tensors.
    DKE_calc   : bool   When False, DRF0D/F arrays are placeholder zeros.
    """
    outputname = idata.outputdirectory + idata.outputfilename + '.h5'

    # Transpose to LUKE axis ordering: (n_p, n_ksi, n_psi, n_harm)
    DRF0_wh = np.transpose(DRF0_wh, (1, 2, 0, 3))
    DRF0_hw = np.transpose(DRF0_hw, (1, 2, 0, 3))
    DRF0_hh = np.transpose(DRF0_hh, (1, 2, 0, 3))

    if DKE_calc:
        DRF0D_wh = np.transpose(DRF0D_wh, (1, 2, 0, 3))
        DRF0D_hw = np.transpose(DRF0D_hw, (1, 2, 0, 3))
        DRF0D_hh = np.transpose(DRF0D_hh, (1, 2, 0, 3))
        DRF0F_wh = np.transpose(DRF0F_wh, (1, 2, 0, 3))
        DRF0F_hw = np.transpose(DRF0F_hw, (1, 2, 0, 3))

    with h5py.File(outputname, 'w') as fid:
        # Grid and metadata
        fid.create_dataset('harmonics',  data=harmonics)
        fid.create_dataset('psi',        data=psi)
        fid.create_dataset('ksi0_w',     data=ksi0_w)
        fid.create_dataset('ksi0_h',     data=ksi0_h)
        fid.create_dataset('p_norm_w',   data=p_norm_w)
        fid.create_dataset('p_norm_h',   data=p_norm_h)
        fid.create_dataset('FreqGHz',    data=FreqGHz)
        fid.create_dataset('mode',       data=mode)
        fid.create_dataset('Trapksi0_h', data=Trapksi0_h)
        fid.create_dataset('Trapksi0_w', data=Trapksi0_w)

        # Absorption profile from the WKBeam absorption step
        abs_data = compute_deposition_profiles(
            InputData(idata.absorption_file), idata.absorption_data_file)
        fid.create_dataset('rho_abs',  data=abs_data['rho'])
        fid.create_dataset('dP_dV',    data=abs_data['dP_dV'])
        fid.create_dataset('dP_drho',  data=abs_data['dP_drho'])
        fid.create_dataset('dV_drho',  data=abs_data['dV_drho'])

        # Sparse D_RF (main Fokker-Planck diffusion tensor)
        for name, arr in [('DRF0_wh', DRF0_wh), ('DRF0_hw', DRF0_hw), ('DRF0_hh', DRF0_hh)]:
            sp, mask = _sparsify(arr)
            fid.create_dataset(f'{name}_sparse', data=sp)
            fid.create_dataset(f'mask_{name}',   data=mask)

        # DKE first-order components (placeholder zeros when DKE_calc=False)
        if DKE_calc:
            dke_pairs = [
                ('DRF0D_wh', DRF0D_wh), ('DRF0D_hw', DRF0D_hw), ('DRF0D_hh', DRF0D_hh),
                ('DRF0F_wh', DRF0F_wh), ('DRF0F_hw', DRF0F_hw),
            ]
        else:
            zeros1 = np.zeros(1)
            dke_pairs = [
                ('DRF0D_wh', zeros1), ('DRF0D_hw', zeros1), ('DRF0D_hh', zeros1),
                ('DRF0F_wh', zeros1), ('DRF0F_hw', zeros1),
            ]

        for name, arr in dke_pairs:
            sp, mask = _sparsify(arr) if DKE_calc else (arr, np.zeros(1))
            fid.create_dataset(f'{name}_sparse', data=sp)
            fid.create_dataset(f'mask_{name}',   data=mask)

    print(f'\nResults saved to {outputname}')


def _theta_cost(data_slice, use_sparse):
    """Estimate the computational cost of a psi slice by counting occupied theta bins.

    Parameters
    ----------
    data_slice : dict (sparse) or ndarray (dense)
    use_sparse : bool

    Returns
    -------
    int   Number of occupied theta bins.
    """
    if use_sparse:
        return len(data_slice)
    else:
        return int(np.any(data_slice > 0, axis=(1, 2)).sum())


# ---------------------------------------------------------------------------
# Main MPI driver
# ---------------------------------------------------------------------------

def call_QLdiff(input_file):
    """Compute and save the quasi-linear RF diffusion tensor for LUKE.

    MPI master/worker orchestration:
      - Rank 0: loads data, builds and sorts the task queue by descending
        number of occupied theta bins (most expensive first), seeds workers,
        collects results, writes HDF5 output.
      - Ranks 1..N-1: reconstruct the equilibrium from disk, receive psi slices,
        call D_RF, return results.

    Parameters
    ----------
    input_file : str   Path to the QLdiff configuration file.

    Returns
    -------
    0 on success.
    """
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # -----------------------------------------------------------------------
    # Rank 0: load data, build task queue
    # -----------------------------------------------------------------------
    if rank == 0:
        idata      = InputData(input_file)
        configfile = idata.configfile
        eqdata     = InputData(configfile)
        Eq         = TokamakEquilibrium(eqdata)
        harmonics  = np.array(idata.harmonics)
        DKE_calc   = idata.DKE_calc

        p_norm_w, p_norm_h, ksi0_w, ksi0_h, \
            Ne_ref_mat, Te_ref_mat, lnc_e_ref = _load_momentum_grids(idata)

        try:
            filename_binned = idata.outputdirectory + idata.binned_data_filename
            WhatToResolve, FreqGHz, mode, Wfct, Absorption, EnergyFlux, \
                rhobins, thetabins, Nparallelbins, Nperpbins = read_h5file(filename_binned)
        except Exception as exc:
            print(f"Error: Could not load binned data: {exc}")
            sys.exit(1)

        d_psi   = np.diff(rhobins**2)
        psi     = 0.5 * (rhobins[1:]**2  + rhobins[:-1]**2)
        d_theta = np.diff(thetabins)[0]
        theta   = 0.5 * (thetabins[1:]  + thetabins[:-1])
        d_npar  = np.diff(Nparallelbins)[0]
        Npar    = 0.5 * (Nparallelbins[1:] + Nparallelbins[:-1])
        d_nperp = np.diff(Nperpbins)[0]
        Nperp   = 0.5 * (Nperpbins[1:]  + Nperpbins[:-1])

        omega = phys.AngularFrequency(FreqGHz)

        if Ne_ref_mat is not None:
            Ne_ref = Ne_ref_mat
            Te_ref = Te_ref_mat
        else:
            _, _, _, _, _, _, ptNe, ptTe, _, _, _, _, _ = config_quantities(
                psi, theta, omega, Eq)
            Ne_ref    = np.amax(ptNe)
            Te_ref    = np.amax(ptTe)
            lnc_e_ref = None

        use_sparse = bool(getattr(idata, 'use_sparse', False))

        if use_sparse:
            edens_sparse_full = _load_sparse_edens(
                filename_binned, psi, d_psi, theta, d_theta, Npar, d_npar, Nperp, d_nperp, Eq)
            Edens = None
        else:
            Edens = _compute_Edens(Wfct, psi, d_psi, theta, d_theta, Nperp, d_nperp, d_npar, Eq)

        # Result arrays: shape (n_psi, n_p, n_ksi, n_harm)
        n_psi, n_harm = len(psi), len(harmonics)
        DRF0_wh  = np.zeros((n_psi, len(p_norm_w), len(ksi0_h), n_harm))
        DRF0_hw  = np.zeros((n_psi, len(p_norm_h), len(ksi0_w), n_harm))
        DRF0_hh  = np.zeros((n_psi, len(p_norm_h), len(ksi0_h), n_harm))
        if DKE_calc:
            DRF0D_wh = np.zeros_like(DRF0_wh)
            DRF0D_hw = np.zeros_like(DRF0_hw)
            DRF0D_hh = np.zeros_like(DRF0_hh)
            DRF0F_wh = np.zeros_like(DRF0_wh)
            DRF0F_hw = np.zeros_like(DRF0_hw)
        else:
            DRF0D_wh = DRF0D_hw = DRF0D_hh = np.zeros(n_psi)
            DRF0F_wh = DRF0F_hw = np.zeros(n_psi)

        Trapksi0_h = np.zeros(n_psi)
        Trapksi0_w = np.zeros(n_psi)

        # Task queue: (psi_index, psi_value, data_slice)
        # Sorted by descending occupied-theta-bin count so the most expensive
        # psi surfaces are dispatched first (prevents late-stage bottlenecks).
        if use_sparse:
            task_queue = [(i, psi_val, edens_sparse_full.get(i, {}))
                          for i, psi_val in enumerate(psi)]
        else:
            task_queue = [(i, psi_val, Edens[i])
                          for i, psi_val in enumerate(psi)]

        task_queue.sort(key=lambda x: _theta_cost(x[2], use_sparse), reverse=True)

    else:
        # Workers initialise broadcast variables to None
        mode = FreqGHz = harmonics = thetabins = None
        p_norm_w = p_norm_h = ksi0_w = ksi0_h = None
        Npar = Nperp = None
        use_sparse = None
        Ne_ref = Te_ref = lnc_e_ref = None
        DKE_calc = None
        configfile = None

    # -----------------------------------------------------------------------
    # Broadcast only lightweight shared data; equilibrium is rebuilt per worker
    # -----------------------------------------------------------------------
    mode       = comm.bcast(mode,       root=0)
    FreqGHz    = comm.bcast(FreqGHz,    root=0)
    harmonics  = comm.bcast(harmonics,  root=0)
    thetabins  = comm.bcast(thetabins,  root=0)
    p_norm_w   = comm.bcast(p_norm_w,   root=0)
    p_norm_h   = comm.bcast(p_norm_h,   root=0)
    ksi0_w     = comm.bcast(ksi0_w,     root=0)
    ksi0_h     = comm.bcast(ksi0_h,     root=0)
    Npar       = comm.bcast(Npar,       root=0)
    Nperp      = comm.bcast(Nperp,      root=0)
    Ne_ref     = comm.bcast(Ne_ref,     root=0)
    Te_ref     = comm.bcast(Te_ref,     root=0)
    lnc_e_ref  = comm.bcast(lnc_e_ref,  root=0)
    DKE_calc   = comm.bcast(DKE_calc,   root=0)
    use_sparse = comm.bcast(use_sparse, root=0)
    configfile = comm.bcast(configfile, root=0)

    # Each worker rebuilds its own equilibrium from the shared config path.
    # This avoids pickling a large TokamakEquilibrium object over MPI.
    if rank != 0:
        eqdata = InputData(configfile)
        Eq = TokamakEquilibrium(eqdata)

    # Pre-build the KDTree once per worker (shared across all received psi slices)
    npar_tree = KDTree(Npar.reshape(-1, 1))

    # -----------------------------------------------------------------------
    # Master: distribute tasks and collect results
    # Tag convention: 0 = termination, 1 = new task, 2 = result
    # -----------------------------------------------------------------------
    if rank == 0:
        num_tasks      = len(task_queue)
        num_workers    = size - 1
        task_idx       = 0
        active_workers = 0
        finished_tasks = 0

        for worker in range(1, min(num_workers + 1, num_tasks + 1)):
            comm.send(task_queue[task_idx], dest=worker, tag=1)
            task_idx      += 1
            active_workers += 1

        while task_idx < num_tasks or active_workers > 0:
            result = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            worker_id, idx, result_data = result
            active_workers -= 1
            finished_tasks += 1
            print(f'\rProgress: {finished_tasks}/{num_tasks}  active: {active_workers}',
                  end='', flush=True)

            (DRF0_wh[idx],  DRF0D_wh[idx],  DRF0F_wh[idx],
             DRF0_hw[idx],  DRF0D_hw[idx],  DRF0F_hw[idx],
             DRF0_hh[idx],  DRF0D_hh[idx],
             Trapksi0_h[idx], Trapksi0_w[idx]) = result_data

            if task_idx < num_tasks:
                comm.send(task_queue[task_idx], dest=worker_id, tag=1)
                task_idx      += 1
                active_workers += 1

        for worker in range(1, size):
            comm.send(None, dest=worker, tag=0)

    # -----------------------------------------------------------------------
    # Workers: receive psi slices and compute D_RF
    # -----------------------------------------------------------------------
    else:
        while True:
            task = comm.recv(source=0, tag=MPI.ANY_TAG)
            if task is None:
                break

            idx, psi_value, data_slice = task

            if use_sparse:
                edens_sparse_w = {0: data_slice}
                Edens_w = None
            else:
                Edens_w = np.expand_dims(data_slice, axis=0)
                edens_sparse_w = None

            (DRF0_wh_loc, DRF0D_wh_loc, DRF0F_wh_loc,
             DRF0_hw_loc, DRF0D_hw_loc, DRF0F_hw_loc,
             DRF0_hh_loc, DRF0D_hh_loc,
             Trapksi0_h_loc, Trapksi0_w_loc) = D_RF(
                [psi_value], thetabins, p_norm_w, p_norm_h, ksi0_w, ksi0_h,
                Npar, Nperp, Edens_w, Eq, Ne_ref, Te_ref,
                n=harmonics.tolist(), FreqGHz=FreqGHz, DKE_calc=DKE_calc,
                gaussian_smooth=True,
                edens_sparse=edens_sparse_w,
                lnc_e_ref=lnc_e_ref,
                npar_tree=npar_tree,
            )

            result_data = (
                DRF0_wh_loc[0],  DRF0D_wh_loc[0],  DRF0F_wh_loc[0],
                DRF0_hw_loc[0],  DRF0D_hw_loc[0],  DRF0F_hw_loc[0],
                DRF0_hh_loc[0],  DRF0D_hh_loc[0],
                float(Trapksi0_h_loc[0]),  float(Trapksi0_w_loc[0]),
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
        print(f"\nElapsed time: {time.time() - start_time:.2f} s")

    MPI.Finalize()
    return 0


# ---------------------------------------------------------------------------
# Entry point (used when the module is invoked directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python QL_diff_driver.py <input_file>")
        sys.exit(1)
    call_QLdiff(sys.argv[1])
    sys.exit(0)
