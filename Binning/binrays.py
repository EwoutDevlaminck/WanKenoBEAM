"""Driver for the binning procedures.

If mpi4py is available and more than one MPI rank is present, the binning
is parallelised:
  - rank 0  runs the organisation core (mainOrgBinning)
  - rank > 0 runs the worker core     (mainWorkerBinning)

With a single rank (or when mpi4py is not installed) the original serial
path is used.
"""

############################################################################
# IMPORT
############################################################################

from CommonModules.input_data import InputData
from Binning.modules.binning_interface import binning_pyinterface

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False


############################################################################
# DRIVER
############################################################################
def call_binning(input_file):
    """Driver for the binning procedures."""

    # Load input data
    input_data = InputData(input_file)

    # Resolve list of (inputfilename, outputfilename) pairs to process serially
    # at the top level.  Each pair is handed off to the parallel or serial binner.
    try:
        outputfilenames_given = len(input_data.outputfilename) > 0
    except Exception:
        outputfilenames_given = False

    if outputfilenames_given:
        if len(input_data.inputfilename) != len(input_data.outputfilename):
            print('ERROR: IN INPUT DATA FILE, THERE ARE NOT AS MANY INPUTFILES '
                  'GIVEN AS SUGGESTIONS FOR THE OUTPUTFILENAME.')
            raise ValueError('Mismatched inputfilename / outputfilename lengths')

    inputfilenames = input_data.inputfilename
    if outputfilenames_given:
        outputfilenames = input_data.outputfilename

    for i in range(len(inputfilenames)):
        input_data.inputfilename = inputfilenames[i]
        if outputfilenames_given:
            input_data.outputfilename = outputfilenames[i]

        _run_binning(input_data)


def _run_binning(idata):
    """Dispatch to parallel or serial binning depending on the MPI environment."""

    if _MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        size = comm.size
        rank = comm.rank
    else:
        size = 1
        rank = 0

    if size > 1:
        # Parallel path
        from Binning.modules.binning_org    import mainOrgBinning
        from Binning.modules.binning_worker import mainWorkerBinning

        if rank == 0:
            mainOrgBinning(idata, comm)
        else:
            mainWorkerBinning(idata, comm)

        comm.Barrier()

    else:
        # Serial path (single rank or no mpi4py)
        binning_pyinterface(idata)


############################################################################
# STAND-ALONE RUN
############################################################################
if __name__ == '__main__':
    import sys
    input_file = sys.argv[1]
    call_binning(input_file)
#
# END OF FILE
