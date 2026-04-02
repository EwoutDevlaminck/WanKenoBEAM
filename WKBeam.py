"""Main driver of the WKBeam code.

Run $ python WKBeam.py for help.

Examples:

  $ mpiexec -np <n> python WKBeam.py trace  <ray_tracing_configuration_file>
  $ mpiexec -np <n> python WKBeam.py bin    <binning_configuration_file>
  $ python WKBeam.py plotbin   <binned_hdf5_file> <ray_tracing_configuration_file>
  $ python WKBeam.py beamFluct <binned_hdf5_file> <ray_tracing_configuration_file>
  $ python WKBeam.py plot2d    <binning_configuration_file>
  $ python WKBeam.py plotabs   <binning_configuration_file>
  $ python WKBeam.py ploteq    <ray_tracing_configuration_file>
  $ python WKBeam.py plotfluct <ray_tracing_configuration_file>
  $ python WKBeam.py flux      <binning_configuration_file>

Note that the first argument is a control flag and the second (and optional
third) argument is an input file or configuration file, cf. examples.

The ray tracing and binning procedures can be called through mpiexec for
parallel execution.  Interactive use is recommended for testing purposes
only with a very small number of rays.

Mode descriptions
-----------------
trace     : Monte-Carlo ray tracing.  Requires mpiexec.
bin       : Bin ray-tracing output onto a spatial/spectral grid.  Supports
            mpiexec for parallel binning over files.
plotbin   : Plot the 2-D binned beam (Wfct) overlaid on equilibrium flux
            surfaces.  Requires the XZ-binned HDF5 file and the RayTracing
            config (used to load the equilibrium for flux-surface contours).
beamFluct : Same 2-D beam plot but overlaid with the fluctuation amplitude
            profile instead of the plain equilibrium.  Same two arguments as
            plotbin.
plot2d    : Plot the 2-D angular spectrum from an Angular binning config.
plotabs   : Plot the 1-D absorbed power profile from an Absorption config.
ploteq    : Plot the magnetic equilibrium from a RayTracing config.
plotfluct : Plot the fluctuation model from a RayTracing config.
flux      : Compute the energy flux through a surface from a binning config.
beam3d    : 3-D beam visualisation using Mayavi.
QLdiff    : Compute the quasi-linear diffusion tensor.
"""

# Load standard modules
import sys

# Define the dictionary of operation modes
# (If you update this dictionary, DO NOT forget to update the 
#  help message below and the module doc string.)
WKBeam_modes = {
    'trace': {
        'procedure': 'call_ray_tracer',
        'module': 'RayTracing.tracerays',
    },
    'bin': {
        'procedure': 'call_binning',
        'module': 'Binning.binrays',
    },
    'plotbin': {
        'procedure': 'plot_binned',
        'module': 'Tools.PlotData.PlotBinnedData.plotbinneddata', 
    },
    'plot2d': {
        'procedure': 'plot2d',
        'module': 'Tools.PlotData.PlotBinnedData.plot2d',
    },
    'plotabs': {
        'procedure': 'plot_abs',
        'module': 'Tools.PlotData.PlotAbsorptionProfile.plotabsprofile',
    },
    'ploteq': {
        'procedure': 'plot_eq',
        'module': 'Tools.PlotData.PlotEquilibrium.plotequilibrium',
    },
    'plotfluct': {
        'procedure': 'plot_fluct',
        'module': 'Tools.PlotData.PlotFluctuations.plotfluctuations',
    },
    'flux': {
        'procedure': 'flux_through_surface',
        'module': 'Tools.PlotData.PlotBinnedData.EnergyFlux',
    },
    'beam3d': {
        'procedure': 'plot_beam_with_mayavi',
        'module': 'Tools.PlotData.PlotBinnedData.Beam3D',
    },
    'beamFluct': {
    	'procedure': 'plot_beam_fluct',
    	'module': 'Tools.PlotData.PlotFluctuations.plotBeamFluctuations',
    },
    'QLdiff': {
    	'procedure': 'call_QLdiff',
    	'module': 'QL_diffusion.QL_diff_calc',
    },
}

# Help messange (list operation modes, etc...)
msg = """ 
 USAGE: 

   $ python WKBEam.py <mode_flag> <input_or_configuration_file>

 If mode_flag = trace, launch with mpiexec:

   $ mpiexec -np <n> python WKBEam.py trace <configuration_file>

 where n is the number of processors. 
    
 LIST OF VALID MODE FLAGS:
  1.  trace     - <ray_tracing_configuration_file>  (use mpiexec)
  2.  bin       - <binning_configuration_file>       (supports mpiexec)
  3.  plotbin   - <XZ_binned.hdf5> <ray_tracing_configuration_file>
                  Plot 2-D beam on equilibrium flux surfaces.
  4.  beamFluct - <XZ_binned.hdf5> <ray_tracing_configuration_file>
                  Plot 2-D beam overlaid with fluctuation amplitude profile.
  5.  plot2d    - <binning_configuration_file>
  6.  plotabs   - <list_of_binning_configuration_files>
  7.  ploteq    - <ray_tracing_configuration_file>
  8.  plotfluct - <ray_tracing_configuration_file>
  9.  flux      - <binning_configuration_file>
  10. beam3d    - <binning_configuration_file>
  11. QLdiff    - <QL_diffusion_configuration_file>
"""

# Check the console input
if len(sys.argv) < 3:
    print(msg)
    sys.exit()

# Mode flag
flag = sys.argv[1]

# Check the mode flag and execute accordingly
if flag not in WKBeam_modes.keys():
    print(msg)
    raise ValueError("mode_flag not understood.")
else:
    # Define the input
    if flag == 'plotabs':
        inputfile = sys.argv[2:]
    elif flag == 'plotbin' or flag == 'beamFluct':
    #Added by Ewout. To also plot the computed equilibrium flux surfaces on top of
    # the beam propagation
        inputfile = sys.argv[2:]
    else:
        inputfile = sys.argv[2]
        
    # Extract the name of the relevant procedure and module
    procedurename = WKBeam_modes[flag]['procedure']
    modulename = WKBeam_modes[flag]['module']

    # Import the relevant module
    exec('from '+modulename+' import '+procedurename)

    # Launch the appropriate application
    exec(procedurename+'(inputfile)')
#
# END OF FILE
