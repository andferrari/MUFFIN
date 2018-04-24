#!/usr/bin/env bash

nbproc=8
mydir=$(dirname $(realpath $0))
golddir=$mydir/gold0

while getopts "n:cvg:" opt
do
    case $opt in
	n)
	    nbproc=$OPTARG
	    ;;
	c)
	    coverage="coverage run --parallel-mode"
	    ;;
	v)
	    verbose="-v"
	    ;;
	g)
	    golddir=$OPTARG
	    ;;	
	\?)
	    echo "invald option -$OPTARG." >&2
	    exit 1
	    ;;
	:)
	    echo "option -$OPTARG requires an argument." >&2
	    exit 1
	    ;;
    esac
done
      
odir=$mydir/output_sigamm/$$
mkdir -pv $odir
echo "output directory is $odir"
echo "log will be in $odir/muffin.log"

mpirun -n $nbproc $coverage $mydir/../../run_tst_mpi_sigamm.py -L 19 -N 2 -mu_s 1 -mu_l 10 -mu_w 10 -stp_s 0.5 -stp_l 100 -pxl_w 1 -bnd_w 1 -data M31_skyline2_20db -fol $mydir/data --odir $odir 2>&1 | tee $odir/muffin.log

$mydir/../../compare_muffin_outputs.py --gold=$golddir --out=$odir --ratio 1e-14 --diff 1e-16 --cool $verbose
status=$?
if [ -n "$coverage" ]
then
    coverage report -m
    coverage html
fi

exit $status
