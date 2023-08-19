#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --job-name antigenicwave
source ~/.virtualenvs/Env4Zhi/bin/activate 

# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=40

cd $SCRATCH/Data

python3 antigenicWaveSimulation.py 80
wait