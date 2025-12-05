#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_88397_0.5_EBW_refl/RayTracing.txt
python3 WKBeam.py flux /home/devlamin/WKbeam_simulations/TCV_88397_0.5_EBW_refl/Flux3D_innerwall.txt &
python3 WKBeam.py plotabs /home/devlamin/WKbeam_simulations/TCV_88397_0.5_EBW_refl/AbsorptionUni.txt &
python3 WKBeam.py beamFluct /home/devlamin/WKbeam_simulations/TCV_88397_0.5_EBW_refl/output/XZ_binned.hdf5 $command1 &
wait
echo "All done!"
