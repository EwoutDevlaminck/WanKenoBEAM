#!/bin/bash
command1=/home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/RayTracing.txt
mpiexec -np 32 python3 WKBeam.py trace $command1
wait
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/Angular.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/Angular2.txt
#python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/Absorption.txt
python3 WKBeam.py bin /home/devlamin/WKbeam_simulations/TCV_87608_1.6_FIR_trial/XZ.txt
