#!/bin/bash
#SBATCH --job-name run_batch_test_slurm
#SBATCH -t 19:10:00
#SBATCH --output testing-%j.out
#SBATCH -N 10
#SBATCH -p fdr,qdr,qdr-gpu

module purge
module load python/3.6.3_anaconda3 intel/mpi/64/2017.4.196 intel/compiler/64/2017.4.196

echo "I am here"

nbcore=$(grep -i processor /proc/cpuinfo |wc -l)
nbproc=$((nbcore*$SLURM_NNODES))

echo "Number of processes for mpi: $nbproc"

mkdir -p output_sigamm/$SLURM_JOB_ID

MAINPY=$SLURM_SUBMIT_DIR/run_tst_mpi_sigamm.py" -L 256 -N 400 -mu_s 0 -mu_l 0 -mu_w 10 -stp_s 0 -stp_l 0 -pxl_w 1 -bnd_w 1 -data M31_skyline2_20db -fol data"

echo "============= MPI RUN ============="
mpirun --np $nbproc python $MAINPY
echo "==================================="

cp $SLURM_SUBMIT_DIR/testing-$SLURM_JOB_ID.out $SLURM_SUBMIT_DIR/output_sigamm/testing-$SLURM_JOB_ID.out
rm $SLURM_SUBMIT_DIR/testing-$SLURM_JOB_ID.out

echo OK
