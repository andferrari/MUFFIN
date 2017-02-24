#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:35:07 2017
@author: rammanouil
"""
#%% ===========================================================================
# Import
# =============================================================================
import os
import numpy as np
from astropy.io import fits
import sys
from deconv3d_tools import conv
from mpi4py import MPI
import tictoc as tm
import deconv3d_mpi as dcvMpi


#%% ===========================================================================
# MPI Run
# =============================================================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# =============================================================================
# Input
# =============================================================================
if len(sys.argv)==11:
    L = int(sys.argv[1])
    nitermax = int(sys.argv[2])
    mu_s_min = float(sys.argv[3])
    mu_s_max = float(sys.argv[4])
    mu_l_min = float(sys.argv[5])
    mu_l_max = float(sys.argv[6])
    absolutePrecision = float(sys.argv[7])
    thresh = float(sys.argv[8])
    maxiter = int(sys.argv[9])
    nitermean = int(sys.argv[10])
elif len(sys.argv)==1:
    L = 6
    nitermax = 10
    mu_s_min = 0.
    mu_s_max = 2.
    mu_l_min = 0
    mu_l_max = 4.
    absolutePrecision = 0.1
    thresh = 1e-4
    nitermean = 5
    maxiter = 10
else:
    if rank==0:
        print('')
        print('-'*100)
        print('You should input: L nitermax mu_s_min mu_s_max mu_l_min mu_l_max absolutePrecision thresh maxiter nitermean where')
        print('')
        print('L: number of bands to be considered')
        print('nitermax: maximum number of iterations in a MUFFIN loop')
        print('mu_s_min: mu_s_min in gs search')
        print('mu_s_max: mu_s_max in gs search')
        print('mu_l_min: mu_l_min in gs search')
        print('mu_l_max: mu_l_max in gs search')
        print('absolutePrecision: minimum absolute difference between a and b in gs')
        print('thresh: minimum allowed variance for wmsesure (if niter>100)')
        print('maxiter: maximum number of iterations allowed in gs search')
        print('nitermean: number of iterations to compute mean mu from after greedy gs')
        print('')
        print('            **** ex: mpirun -np 4 python3 Run.py 6 10 0 2 0 4 0.1 1e-4 10 10                 ')
        print('')
        print('-'*100)
        print('')
    sys.exit()

if rank==0:
    print('')
    print('L: ',L)
    print('nitermax: ',nitermax)
    print('mu_s_min: ',mu_s_min)
    print('mu_s_max: ',mu_s_max)
    print('mu_l_min: ',mu_l_min)
    print('mu_l_max: ',mu_l_max)
    print('absolutePrecision: ',absolutePrecision)
    print('thresh: ',thresh)
    print('nitermean: ',nitermean)
    print('maxiter: ',maxiter)


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
genname = os.path.join(folder, file_in)
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
# Set parameters
# =============================================================================
# DWT parameters    
nb=('db1','db2','db3','db4','db5','db6','db7','db8')
mu_s = 0.
mu_l = 0.
# create class instance
tm.tic()
args = {'mu_s':mu_s,'mu_l':mu_l,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,
        'mu_s_max':mu_s_max,'mu_s_min':mu_s_min,'mu_l_min':mu_l_min,'mu_l_max':mu_l_max,
        'absolutePrecision':absolutePrecision,'thresh':thresh}
EM= dcvMpi.EasyMuffinSURE(**args)


#%% ===========================================================================
# MPI loop
# =============================================================================
EM.loop_mu_s(nitermax,maxiter)
EM.set_mean_mu(set_mu_s=True,niter=nitermean)
EM.loop_mu_l(nitermax,maxiter)
EM.set_mean_mu(set_mu_l=True,niter=nitermean)
EM.loop(nitermax)


#%% ===========================================================================
# Save results
# =============================================================================
if rank==0:
    tm.toc()
    np.save('x0.npy',EM.xf)
    np.save('wmse.npy',EM.wmselist)
    np.save('wmses.npy',EM.wmselistsure)
    np.save('snr.npy',EM.snrlist)
    np.save('mu_s.npy',EM.mu_slist)
    np.save('mu_l.npy',EM.mu_llist)
    
    