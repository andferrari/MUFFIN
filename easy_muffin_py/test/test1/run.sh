#!/usr/bin/env bash

mydir=$(dirname $0)

export SLURM_JOB_ID=$$
odir=$mydir/../../output_sigamm/$$
mkdir -p $odir
echo "Job id is $SLURM_JOB_ID"
echo "output directory is $odir"
echo "log will be in $odir/muffin.log"

mpirun -n 8 $mydir/../../run_tst_mpi_sigamm.py -L 19 -N 2 -mu_s 1 -mu_l 10 -mu_w 10 -stp_s 0.5 -stp_l 100 -pxl_w 1 -bnd_w 1 -data M31_skyline2_20db -fol $mydir/data 2>&1 | tee $odir/muffin.log


$mydir/../../compare_muffin_outputs.py --gold=$mydir/gold --out=$odir
