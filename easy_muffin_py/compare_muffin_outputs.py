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
parser.add_argument('--diff','-d', metavar='DIFF', type=float, dest='max_diff',
                    default=1e-15,
                    help='tolerance (as a delta).')

args = parser.parse_args()

passed=True
if args.golddir:
    print("Blob")
    for f in os.listdir(args.golddir):
        gpath = path.join(args.golddir, f)
        opath = path.join(args.outdir, f)
        if path.isfile(gpath):
            print(f"Need to compare {gpath} ", end="")
            if not path.isfile(opath):
                passed = False
                print(f"but {opath} could not be found.")
                continue
            else:
                print(f"with {opath}.")
            gdata = np.load(gpath)
            odata = np.load(opath)
            gavrg = np.average(abs(gdata))
            oavrg = np.average(abs(odata))
            delta = abs(gdata-odata)
            max_delta = np.amax(delta)
            print(f"Max delta is {max_delta} (with average = {(gavrg+oavrg)/2})->", end="")
            if max_delta > args.max_diff:
                passed = False
                print(" KO")
            else:
                print(" OK")

if passed:
    print("PASSED")
    sys.exit(0)
else:
    print("NO! THAT'S WRONG! WRONG! WRONG! WRONG! NO! NO! BAD! BAD! BAD!")
    sys.exit(-1)
