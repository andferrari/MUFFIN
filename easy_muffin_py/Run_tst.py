#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:09:36 2017

@author: rammanouil
"""
# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================
import os
import numpy as np
from astropy.io import fits
import sys
from deconv3d_tools import conv
from mpi4py import MPI
import deconv3d_mpi as dcvMpi
from gs_mpi import gs_mu_s, gs_mu_l
from tictoc import tic, toc


#%% ===========================================================================
# MPI Run
# =============================================================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# =============================================================================
# Input
# =============================================================================
if len(sys.argv)==8:
    L = int(sys.argv[1])
    nitermax1 = int(sys.argv[2])
    nitermax2 = int(sys.argv[3])
    mu_s_max = float(sys.argv[4])
    mu_l_max = float(sys.argv[5])
    mu_s = float(sys.argv[6])
    mu_l = float(sys.argv[7])
elif len(sys.argv)==1:
    L = 10
    nitermax1 = 10
    nitermax2 = 10
    mu_s_max = 0.2
    mu_l_max = 2.2
    mu_s = 0.2
    mu_l = 2.2
else:
    if rank==0:
        print('')
        print('-'*100)
        print('You should input: L nitermax1 nitermax2 mu_s_max mu_l_max mu_s mu_l')
        print('')
        print('L: number of bands to be considered')
        print('nitermax: maximum number of iterations in a MUFFIN loop with mu_s')
        print('nitermax: maximum number of iterations in a MUFFIN loop with mu_s & mu_l')
        print('mu_s_max: mu_s_max used to compute tau')
        print('mu_l_max: mu_l_max used to compute tau')
        print('')
        print('            **** ex: mpirun -np 4 python3 Run_GS.py 10 10 10 0.2 2.2 0.2 2.2                 ')
        print('')
        print('-'*100)
        print('')
    sys.exit()

if rank==0:
    print('')
    print('L: ',L)
    print('nitermax1: ',nitermax1)
    print('nitermax2: ',nitermax2)
    print('mu_s_max: ',mu_s_max)
    print('mu_l_max: ',mu_l_max)
    print('mu_s: ',mu_s)
    print('mu_l: ',mu_l)


# =============================================================================
# Load data
# =============================================================================
def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x

folder = 'data256'
file_in = 'M31_3d_conv_256_10db'
folder = os.path.join(os.getcwd(), folder)
# genname = os.path.join(folder, file_in)
genname = '/home/rammanouil/easy_muffin/easy_muffin_py/data256/M31_3d_conv_256_10db' 
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'
CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]
skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = sky[:,:,0:L]
sky2 = np.sum(sky*sky)
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size


#%% ===========================================================================
# Run
# =============================================================================
# DWT parameters
nb=('db1','db2','db3','db4','db5','db6','db7','db8')

args = {'mu_s':mu_s_max,'mu_l':mu_l_max,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,
        'mu_s_max':mu_s_max,'mu_s_min':0,'mu_l_min':0,'mu_l_max':mu_l_max}
tic()
EM= dcvMpi.EasyMuffinSURE(**args)
if rank==0:
    print('using tau: ',EM.tau)
    print('')
# loop with mu_s
EM.mu_l = 0
EM.mu_s = mu_s
EM.loop(nitermax1)
# loop with mu_l
EM.mu_l = mu_l
EM.loop(nitermax2)


#%% ===========================================================================
# Save results
# =============================================================================
if rank==0:
    np.save('x0_tst.npy',EM.xf)
    np.save('wmse_tst.npy',EM.wmselist)
    np.save('wmses_tst.npy',EM.wmselistsure)
    np.save('snr_tst.npy',EM.snrlist)
    np.save('mu_s_tst.npy',mu_s)
    np.save('mu_l_tst.npy',mu_l)
    toc()
