#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/RayTracing.txt
mpiexec -np 32 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/XZ.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/Flux3D_wall.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_86712_0.5_EBW_refl/Flux3D_launcherport.txt

wait

echo "Ray tracing and binning done!"
