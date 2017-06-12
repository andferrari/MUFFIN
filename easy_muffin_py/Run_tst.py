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
if len(sys.argv)==10:
    L = int(sys.argv[1])
    nitermax1 = int(sys.argv[2])
    nitermax2 = int(sys.argv[3])
    mu_s_max = float(sys.argv[4])
    mu_l_max = float(sys.argv[5])
    mu_s = float(sys.argv[6])
    mu_l = float(sys.argv[7])
    mu_eps = float(sys.argv[8])
    data_suffix = sys.argv[9]
elif len(sys.argv)==1:
    L = 10
    nitermax1 = 10
    nitermax2 = 10
    mu_s_max = 0.2
    mu_l_max = 2.2
    mu_s = 0.2
    mu_l = 2.2
    mu_eps = 0
    data_suffix = 'M31_3d_conv_256_10db'
else:
    if rank==0:
        print('')
        print('-'*100)
        print('You should input: L nitermax1 nitermax2 mu_s_max mu_l_max mu_s mu_l mu_eps data_suffix')
        print('')
        print('L: number of bands to be considered')
        print('nitermax1: maximum number of iterations in a MUFFIN loop with mu_s')
        print('nitermax2: maximum number of iterations in a MUFFIN loop with mu_s & mu_l')
        print('mu_s_max: mu_s_max used to compute tau')
        print('mu_l_max: mu_l_max used to compute tau')
        print('mu_s, mu_l, and mu_eps: spatial, spectral and tikhonov reg.')
        print('data_suffix: name suffix of data in folder data256')
        print('')
        print('            **** ex: mpirun -np 4 python3 Run_GS.py 10 10 10 0.2 2.2 0.2 2.2 0.0 M31_3d_conv_256_10db                ')
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
    print('mu_eps: ',mu_eps)
    print('data_suffix',data_suffix)


# =============================================================================
# Load data
# =============================================================================
def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x

folder = 'data256'
file_in = data_suffix

if os.getenv('OAR_JOB_ID') is not None:
    folder = os.path.join(os.getenv('OAR_WORKDIR'), folder)
else:
    folder = os.path.join(os.getcwd(), folder)

# CAUTIONNNNNNNNNN TEMPORARY SOLUTION
folder = '/home/rammanouil/easy_muffin/easy_muffin_py/data256'

genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'
CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]
skyname = genname+'_sky.fits'

if os.path.isfile(skyname):
    if rank==0:
        print('')
        print('estimating variance')
    sky = checkdim(fits.getdata(skyname, ext=0))
    sky = sky[:,:,0:L]
    sky2 = np.sum(sky*sky)
    Noise = CubeDirty - conv(CubePSF,sky)
    var = np.sum(Noise**2)/Noise.size
    if rank==0:
        print('')
        print('setting var to ', var)
else:
    var = 0.0
    sky = None
    if rank==0:
        print('')
        print('setting var to ', var)


#%% ===========================================================================
# Run
# =============================================================================
# DWT parameters
#nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb = (7,0)

args = {'mu_s':mu_s_max,'mu_l':mu_l_max,'mu_eps':mu_eps,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,
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

    toc()

    if os.getenv('OAR_JOB_ID') is not None:
        #os.mkdir(os.getenv('OAR_JOB_ID'))
        os.chdir(os.path.join(os.getenv('OAR_WORKDIR'), 'output/'+os.getenv('OAR_JOB_ID')))
    else:
        os.chdir(os.path.join(os.getcwd(), 'output'))

    if sky is not None:
        np.save('x0_tst.npy',EM.xf)
        np.save('wmse_tst.npy',EM.wmselist)
        np.save('wmses_tst.npy',EM.wmselistsure)
        np.save('snr_tst.npy',EM.snrlist)
        np.save('mu_s_tst.npy',mu_s)
        np.save('mu_l_tst.npy',mu_l)
    else:
        np.save('x0_tst.npy',EM.xf)
        np.save('wmses_tst.npy',EM.wmselistsure)
        np.save('mu_s_tst.npy',mu_s)
        np.save('mu_l_tst.npy',mu_l)
