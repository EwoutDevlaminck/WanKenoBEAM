"""Read 2D density field of a tokamaks from standard equilibrium
formats.
"""

__author__ = 'Ewout Devlaminck'
__version__ = 'Revision: '
__date__ = 'Date: '
__copyright__ = ' '
__license__ = 'Max-Planck-Institut fuer Plasmaphysik'

# Import statement
import numpy as np

#
# --- functions ---
#

# Read a nefile (torbeam, torray, lhbeam, ...)
def read(data_directory):
    """Read a nefile and store data in numpy
    arrays.

    USAGE: R, z, nefield = read_nefile(data_directory)
    
    Input variables:
       > data_directory, string with the path to the directory
         where the nefile is stored.

    Returns: the list (R, z, nefield) 
    where
       > R[iR, iz], 2d numpy array, with R on the 2d poloidal grid.
       > z[iR, iz], 2d numpy array, with z on the 2d poloidal grid.
       > nefield[iR, iz], 2d numpy array, with the 2D resolved density
         on grid points (R, z) in the poloidal plane;
         the indices iR and iz run over grid points in R and z, respectively

    WARNING: The grid in R and z is returned in cm, not in m!
    """

    # Open and read the nefile
    filename1 = data_directory + "/nefile"

    datafile = open(filename1)

    lines = datafile.readlines()
    
    # Close the data file
    datafile.close()

    # Read the number of grid points
    datastring = lines[1].split()
    nptR = int(datastring[0])
    nptz = int(datastring[1])
    try:
        datastring = lines[3].split()
        psi_sep = float(datastring[2])
    except IndexError:
        print('\n WARNING nefile: assuming psi = 1. at the separatrix. \n')
        psi_sep = 1.

    # Define lists for variables
    R_val = []         # major radius grid
    z_val = []         # z coordinate grid
    ne_val = []        # values of ne

    # Loop over remaining lines 
    nlines = len(lines)
    for i in range(0, nlines):
        data = lines[i].split()
        # Read grid in major radius
        try:

            test = (data[0] == 'R') or (data[0] == 'Radial')
        except:
            # ... when the line is blank cycle to the next line ...
            continue
        try:
            # ... in some format the information is on the second
            # element of the line ...
            test = test or (data[1] == 'X-coordinates')
        except:
            pass
        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                R_val.extend(values)
                if len(R_val) == nptR: break
        # Read grid in the vertical coordinate z
        try:
            # ... try to read the first element of the line ...
            test = (data[0] == 'Z') or (data[0] == 'Vertical')
        except:
            # ... when the line is blank cycle to the next line ...
            continue
        try:
            # ... in some format the information is on the second
            # element of the line ...
            test = test or (data[1] == 'Z-coordinates')
        except:
            pass
        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                z_val.extend(values)
                if len(z_val) == nptz: break

        # Read the density
        try:
            # ... try to read the first element of the line ...
            test = (data[0] == 'Ne')
        except:
            # ... when the line is blank cycle to the next line ...
            continue            

        if test:
            start = i + 1
            for line in lines[start:]:
                values = map(float, line.split())
                ne_val.extend(values)
                if len(ne_val) == nptR * nptz: break

  
    # Convert lists of grid points into numpy arrays (converted in cm)
    R = 100. * np.array(R_val)
    z = 100. * np.array(z_val)
    z, R = np.meshgrid(z, R)

    # Convert the magnetic field lists into a numpy array
    ne_val = np.array(ne_val)
    ne_val.shape = (nptz, nptR) #(data are written with z first)
    ne_val = ne_val.T #(transpose to have R first)
    #


    # Return
    return (R, z, ne_val)

#
# end of file
