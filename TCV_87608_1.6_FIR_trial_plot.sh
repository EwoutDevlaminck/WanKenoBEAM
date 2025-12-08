#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/RayTracing.txt
python3 WKBeam.py plot2d /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/Angular.txt &
#python3 WKBeam.py plotabs /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/Absorption.txt &
python3 WKBeam.py beamFluct /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/output/XZ_binned.hdf5 $command1 &
wait
echo "All done!"
