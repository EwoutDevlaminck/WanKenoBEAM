"""Organisation core for parallel binning.

Rank 0 calls mainOrgBinning.  It discovers the file list, distributes files
one-at-a-time to free worker ranks (dynamic / work-stealing schedule), collects
the partial results and accumulates them, then performs the final post-processing
and writes the output HDF5.

Protocol
--------
  org  → worker   comm.send(file_idx, dest=worker, tag=1)   # file index to bin
  worker → org    comm.send(partial,  dest=0,      tag=2)   # partial result dict
  org  → worker   comm.send(-1,       dest=worker, tag=1)   # termination signal
"""

import sys
from mpi4py import MPI

from Binning.modules.binning_interface import (
    _setup_bins,
    _allocate_accum,
    _find_indices,
    _accumulate,
    _postprocess_and_write,
)


def mainOrgBinning(idata, comm):
    """Organisation core: distribute files, collect partials, write output."""

    rank = comm.rank          # should be 0
    size = comm.size          # total number of MPI ranks
    n_workers = size - 1      # ranks 1 .. size-1 are workers

    outputfilename = getattr(idata, 'outputfilename', idata.inputfilename + '_binned')

    if len(idata.WhatToResolve) > 4:
        print('THE MAXIMUM NUMBER OF DIMENSIONS 4 IS EXCEEDED.\n')
        raise ValueError('Too many dimensions in WhatToResolve')

    setup  = _setup_bins(idata)
    accum  = _allocate_accum(idata, setup)
    indices = _find_indices(idata)

    print("NUMBER OF FILES TO BE PROCESSED: %i\n" % len(indices))
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    # DYNAMIC WORK DISTRIBUTION
    # -------------------------------------------------------------------------
    # Queue of file indices still to be sent.
    queue = list(indices)

    # Phase 1: seed every worker with its first job (or a termination signal
    # if there are fewer files than workers).
    active_workers = 0
    for worker in range(1, size):
        if queue:
            file_idx = queue.pop(0)
            print("rank %i sending file index %i to worker rank %i\n"
                  % (rank, file_idx, worker))
            sys.stdout.flush()
            comm.send(file_idx, dest=worker, tag=1)
            active_workers += 1
        else:
            # No work for this worker at all — terminate it immediately.
            comm.send(-1, dest=worker, tag=1)

    # Phase 2: as workers report back, accumulate their result and send them
    # the next job (or terminate them when the queue is empty).
    status = MPI.Status()
    while active_workers > 0:
        # Block until *any* worker sends a result back.
        partial = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        worker  = status.Get_source()

        _accumulate(accum, partial)
        active_workers -= 1

        if queue:
            file_idx = queue.pop(0)
            print("rank %i sending file index %i to worker rank %i\n"
                  % (rank, file_idx, worker))
            sys.stdout.flush()
            comm.send(file_idx, dest=worker, tag=1)
            active_workers += 1
        else:
            # No more work — terminate this worker.
            comm.send(-1, dest=worker, tag=1)

    # -------------------------------------------------------------------------
    # POST-PROCESS AND WRITE
    # -------------------------------------------------------------------------
    _postprocess_and_write(idata, accum, setup, outputfilename)

# END OF FILE
