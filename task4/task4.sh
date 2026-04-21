#!/bin/bash
#======================================================
# Job script for PH510 Assignment 4 - Task 4
# MPI parallel scaling study for the VMC boson code
#======================================================
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=task4_scaling
#SBATCH --output=task4_%j.out

set -euo pipefail

module purge
module load mpi

# Force single-threaded numerical libraries so each MPI rank uses only
# its allocated core. This avoids hidden threading that can distort both
# timings and scaling behaviour.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "===================================="
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Running on  : $(hostname)"
echo "Python      : $(python3 --version)"
echo "Loaded modules:"
module list 2>&1
echo "===================================="

for P in 1 2 4 8 16; do
    echo
    echo "------------------------------------"
    echo "Running task4.py with ${P} MPI process(es)"
    echo "------------------------------------"

    mpirun -np "${P}" python3 task4.py --nprocs "${P}"

    echo "Completed ${P}-rank run."
done

echo
echo "===================================="
echo "All runs complete."
echo "Output files produced:"
ls -lh Data/task4_results_P*.txt Figures/task4_energy_surface_P*.png Figures/task4_energy_slices_P*.png
echo "===================================="
