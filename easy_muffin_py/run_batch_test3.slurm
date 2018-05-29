#!/bin/bash
#SBATCH --job-name run_batch_test_slurm
#SBATCH -t 19:10:00
#SBATCH --output testing-%j.out
#SBATCH -N 1
#SBATCH -p fdr,qdr,qdr-gpu

module purge
module load python/3.6.3_anaconda3 
module load intel/mpi/64/2017.4.196 intel/compiler/64/2017.4.196
module load fftw3-intel17/3.3.7

prog=./run_tst_mpi_sigamm2.py

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

odir=output_sigamm/$SLURM_JOB_ID
mkdir -p $odir

MAINPY=$prog${arguments}
MAINPY="$MAINPY -o $odir"

echo $MAINPY 

echo "============= MPI RUN ============="
mpirun --np $numberofcores python $MAINPY
echo "==================================="

cp -v ./testing-$SLURM_JOB_ID.out ./output_sigamm/testing-$SLURM_JOB_ID.out
rm ./testing-$SLURM_JOB_ID.out
