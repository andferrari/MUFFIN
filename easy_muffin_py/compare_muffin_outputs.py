#!/usr/bin/env python3

import numpy as np
import argparse
import os
import os.path as path
import sys

parser = argparse.ArgumentParser(description='Compare easy muffin outputs.')
parser.add_argument('--out','-o', metavar='OUTDIR', type=str, dest='outdir',
                    help='directory where data was produced.')
parser.add_argument('--gold','-g', metavar='GOLDDIR', type=str, dest='golddir',
                    help='directory containing the gold data.')
parser.add_argument('--verbosity', metavar='LEVEL', dest='verbosity', type=int, default=1,
                    help='verbosity level.')
parser.add_argument('--verbose','-v', dest='verbosity', action='store_const',
                    const=2, help='verbose output.')
parser.add_argument('--quiet','-q', dest='verbosity', action='store_const',
                    const=0, help='no output.')
parser.add_argument('--diff','-d', metavar='DIFF', type=float, dest='max_diff',
                    default=1e-15,
                    help='tolerance (as a delta).')
parser.add_argument('--ratio','-r', metavar='RATIO', type=float, dest='max_ratio',
                    default=1e-15,
                    help='tolerance (as a ratio).')

args = parser.parse_args()

passed=True

if args.golddir:
    for f in os.listdir(args.golddir):
        gpath = path.join(args.golddir, f)
        opath = path.join(args.outdir, f)
        if path.isfile(gpath):
            if    args.verbosity > 1: print(f"Need to compare {gpath} ", end="")
            elif  args.verbosity > 0: print(f"{f}: ", end="")
            if not path.isfile(opath):
                passed = False
                if   args.verbosity > 1: print(f"but {opath} could not be found.")
                elif args.verbosity > 0: print(f"not found.")
                continue
            else:
                if args.verbosity > 1: print(f"with {opath}.")
            gdata = np.load(gpath)
            odata = np.load(opath)
            gavrg = np.average(abs(gdata))
            oavrg = np.average(abs(odata))
            delta = abs(gdata-odata)
            max_delta = np.amax(delta)
            if   args.verbosity > 1: print(f"Max delta is {max_delta} (with average = {(gavrg+oavrg)/2})->", end="")
            elif args.verbosity > 0: print(f"delta: {max_delta}", end="")
            if max_delta > args.max_diff:
                passed = False
                if args.verbosity > 0: print(" KO")
            else:
                if args.verbosity > 0: print(" OK")

if passed:
    if args.verbosity > 0: print("PASSED")
    sys.exit(0)
else:
    if args.verbosity > 0: print("NO! THAT'S WRONG! WRONG! WRONG! WRONG! NO! NO! BAD! BAD! BAD!")
    sys.exit(-1)
