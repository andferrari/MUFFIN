#!/bin/sh
#OAR -l core=33, walltime=48:00:00
#OAR -t smp
#OAR -O output/%jobid%.output.txt
#OAR -E output/%jobid%.error.txt


module purge
module load Python/3.5.1
module load openmpi/1.6.2

mkdir -p output/$OAR_JOB_ID


echo $OAR_NODEFILE :
cat $OAR_NODEFILE
echo

#MAINPY=$OAR_WORKDIR/Run_tst.py"  32 100 600 0.75 2.77 0.75 2.77 10 I_20XHALO_CL_RS_SKY_COMPOSED_128freq"
MAINPY=$OAR_WORKDIR/Run_tst.py"  32 0 900 10 2 10 2 10 Celine_4var"

cd $TMPDIR

NSLOTS=$(cat $OAR_NODEFILE | wc -l)
PREF=$(dirname `which mpirun` | awk -F'/[^/]*$' '{print $1}')

echo "============= MPI RUN ============="
mpirun --mca mpi_warn_on_fork 0 --prefix $PREF -np $NSLOTS -machinefile $OAR_NODEFILE python3.5 $MAINPY
echo "==================================="

echo OK
