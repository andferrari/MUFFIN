#!/bin/bash
#SBATCH --job-name run_batch_test_slurm
#SBATCH -t 0:02:00
#SBATCH --output testing-%j.out
#SBATCH -N 1
#SBATCH -p fdr,qdr

module purge
module load python/3.6.3_anaconda3 intel/mpi/64/2017.4.196 intel/compiler/64/2017.4.196

echo "I am here"

rootdir=$(dirname $0)
echo $rootdir

nbproc=$(grep -i processor /proc/cpuinfo |wc -l)

numberofnodes=$(($nbproc*$SLURM_NNODES))
echo $numberofnodes

mkdir -p output_sigamm/$SLURM_JOB_ID

MAINPY=$rootdir/run_tst_mpi_sigamm.py" -L 19 -N 2 -mu_s 1 -mu_l 10 -mu_w 10 -stp_s 0.5 -stp_l 100 -pxl_w 1 -bnd_w 1 -data M31_skyline2_20db -fol $rootdir/data"

echo "============= MPI RUN ============="
mpirun --np $numberofnodes python $MAINPY
echo "==================================="

mv testing-$SLURM_JOB_ID.out $rootdir/output_sigamm/testing-$SLURM_JOB_ID.out

echo OK
