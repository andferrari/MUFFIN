#!/bin/sh
#OAR -l core=4,walltime=00:20:00
#OAR -O /home/rammanouil/SCRATCH/test_mpi_2nodes.%jobid%.output
#OAR -E /home/rammanouil/SCRATCH/test_mpi_2nodes.%jobid%.error
#OAR -p host='smp01'
#OAR -t smp

module purge
module load Python/3.5.1
module load openmpi/1.6.2

TMPDIR=$SCRATCHDIR/$OAR_JOB_ID
mkdir -p $TMPDIR
cd $TMPDIR

echo $OAR_NODEFILE :
cat $OAR_NODEFILE
echo

MAINPY=$OAR_WORKDIR/Run_tst.py"  10 10 10 10 2 10 2"

cd $TMPDIR

NSLOTS=$(cat $OAR_NODEFILE | wc -l)
PREF=$(dirname `which mpirun` | awk -F'/[^/]*$' '{print $1}')

echo "============= MPI RUN ============="
mpirun --mca mpi_warn_on_fork 0 --prefix $PREF -np $NSLOTS -machinefile $OAR_NODEFILE python3.5 $MAINPY
echo "==================================="

echo OK
