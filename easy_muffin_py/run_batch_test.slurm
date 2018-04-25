#!/bin/bash
#SBATCH --job-name run_batch_test_slurm
#SBATCH -t 19:10:00
#SBATCH --output testing-%j.out
#SBATCH -N 10
#SBATCH -p fdr,qdr,qdr-gpu

module purge
module load python/3.6.3_anaconda3 intel/mpi/64/2017.4.196 intel/compiler/64/2017.4.196

prog=./run_tst_mpi_sigamm.py

if [ ! -x $prog ]
then 
    echo "Could not find $prog, did you launch the script from the right directory or used"  1>&2
    echo "sbatch -D <directory path to script>  run_batch_test.slurm"  1>&2
    echo " ?" 1>&2
    exit -1
fi

echo "Launched from: $SLURM_SUBMIT_DIR"

nbproc=$(grep -i processor /proc/cpuinfo |wc -l)

numberofcores=$(($nbproc*$SLURM_NNODES))
echo "Will use $numberofcores MPI processes."

mkdir -p output_sigamm/$SLURM_JOB_ID

MAINPY="$prog -L 19 -N 2 -mu_s 1 -mu_l 10 -mu_w 10 -stp_s 0.5 -stp_l 100 -pxl_w 1 -bnd_w 1 -data M31_skyline2_20db -fol ./data"

echo "============= MPI RUN ============="
mpirun --np $numberofcores python $MAINPY
echo "==================================="

cp -v ./testing-$SLURM_JOB_ID.out ./output_sigamm/testing-$SLURM_JOB_ID.out
rm ./testing-$SLURM_JOB_ID.out
