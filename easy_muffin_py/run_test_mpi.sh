#!/bin/sh
#OAR -l core=16, walltime=48:00:00
#OAR -p host='smp01'
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

MAINPY=$OAR_WORKDIR/Run_tst.py"  32 3 100 1e-3 1e-3 1e-3 1e-3 Celine"

cd $TMPDIR

NSLOTS=$(cat $OAR_NODEFILE | wc -l)
PREF=$(dirname `which mpirun` | awk -F'/[^/]*$' '{print $1}')

echo "============= MPI RUN ============="
mpirun --mca mpi_warn_on_fork 0 --prefix $PREF -np $NSLOTS -machinefile $OAR_NODEFILE python3.5 $MAINPY
echo "==================================="

echo OK
