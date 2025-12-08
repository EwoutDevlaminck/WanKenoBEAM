#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_fluct/RayTracing.txt
mpiexec -np 32 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_fluct/Angular.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_fluct/Angular2.txt
#python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_fluct/Absorption.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_fluct/XZ.txt
