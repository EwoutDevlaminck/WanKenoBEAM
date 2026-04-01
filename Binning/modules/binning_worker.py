"""Worker core for parallel binning.

Each rank > 0 calls mainWorkerBinning.  It waits for a file index from the
organisation core (tag=1), bins that file, and sends the partial result back
(tag=2).  A file index of -1 is the termination signal.

Protocol
--------
  org  → worker   comm.send(file_idx, dest=worker, tag=1)   # file index to bin
  worker → org    comm.send(partial,  dest=0,      tag=2)   # partial result dict
  org  → worker   comm.send(-1,       dest=worker, tag=1)   # termination signal
"""

import sys

from Binning.modules.binning_interface import (
    _setup_bins,
    _bin_one_file,
)


def mainWorkerBinning(idata, comm):
    """Worker core: receive file indices, bin them, return partial results."""

    rank = comm.rank

    # Set up bins once (all workers have the same idata).
    setup = _setup_bins(idata)

    while True:
        # Wait for next assignment from organisation core.
        file_idx = comm.recv(source=0, tag=1)

        # -1 is the termination signal.
        if file_idx == -1:
            print("worker rank %i received termination signal.\n" % rank)
            sys.stdout.flush()
            break

        print("worker rank %i binning file index %i\n" % (rank, file_idx))
        sys.stdout.flush()

        partial = _bin_one_file(file_idx, idata, setup)

        # Send partial result back to organisation core.
        comm.send(partial, dest=0, tag=2)

# END OF FILE
