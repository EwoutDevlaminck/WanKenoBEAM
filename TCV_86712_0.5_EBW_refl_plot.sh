#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/RayTracing.txt
#python3 WKBeam.py plotbin /home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/output/XZ_binned.hdf5 $command1 &
#python3 WKBeam.py beamFluct /home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/output/XZ_binned.hdf5 $command1 &
#python3 WKBeam.py beam3d /home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/Flux3D_wall.txt &
python3 WKBeam.py flux /home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/Flux3D_wall.txt &
python3 WKBeam.py flux /home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/Flux3D_launcherport.txt &


wait
echo "All done!"
