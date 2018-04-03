#!/bin/bash
#SBATCH --job-name Run_tst_mpi2
#SBATCH -t 2:00:00
#SBATCH --output testing-%j.out
#SBATCH -N 5
#SBATCH -p fdr

module purge
module load python/3.6.3_anaconda3

echo "I am here"

mkdir -p output_sigamm/$SLURM_JOB_ID

MAINPY=$SLURM_SUBMIT_DIR/Run_tst_mpi_sigamm.py" -L 99 -N 4000 -mu_s 1 -mu_l 10 -mu_w 10 -stp_s 0.5 -stp_l 100 -pxl_w 1 -bnd_w 1 -data M31_skyline2_20db -fol ./data“

echo "============= MPI RUN ============="
mpirun --np 20*$SLURM_NNODES python $MAINPY
echo "==================================="

cp $SLURM_SUBMIT_DIR/testing-$SLURM_JOB_ID.out $SLURM_SUBMIT_DIR/output_sigamm/testing-$SLURM_JOB_ID.out
rm $SLURM_SUBMIT_DIR/testing-$SLURM_JOB_ID.out

echo OK
