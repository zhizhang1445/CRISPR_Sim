#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=8:00:00
#SBATCH --job-name antigenicwavex5
 
# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=8

(python3 antigenicWaveSimulation.py 0 8 && echo "job 00 finished") &
(python3 antigenicWaveSimulation.py 1 8 && echo "job 01 finished") &
(python3 antigenicWaveSimulation.py 2 8 && echo "job 02 finished") &
(python3 antigenicWaveSimulation.py 3 8 && echo "job 03 finished") &
(python3 antigenicWaveSimulation.py 4 8 && echo "job 04 finished") &
wait