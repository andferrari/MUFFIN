#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:35:07 2017

@author: rammanouil
"""
# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import os
import numpy as np
from astropy.io import fits
import pylab as pl
import sys
from deconv3d_tools import conv

from mpi4py import MPI

import tictoc as tm


if len(sys.argv)>1:
    visu=int(sys.argv[1])
else:
    visu=1

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

L = 50

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = sky[:,:,0:L]
sky2 = np.sum(sky*sky)

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

#%% ===========================================================================
# MPI
# =============================================================================

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import deconv3d_mpi as dcvMpi

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nitermax = 200
mu_s = 0.
mu_l = 0.

if rank==0:
    print('')
    print('----------------------------------------------------------')
    print('                      MPI: Easy MUFFIN SURE')
    print('----------------------------------------------------------')
    print('')

# every processor creates EM -- inside each one will do its one part of the job
tm.tic()
EM= dcvMpi.EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)

# gs borne parameters 
EM.mu_s_max = 0.5
EM.mu_s_min = 0
EM.mu_l_min = 1
EM.mu_l_max = 2.5
# gs precision
EM.absolutePrecision = 0.05
# threshold wmsesure (stopping criteria) 
EM.thresh = 1e-4

EM.loop_mu_s(nitermax)
EM.set_mean_mu(set_mu_s=True,niter=20)
EM.loop_mu_l(nitermax)
EM.set_mean_mu(set_mu_l=True,niter=20)
EM.loop(nitermax)

#%% ===========================================================================
# Validating results
# =============================================================================

if rank==0:
    tm.toc()
    np.save('x0.npy',EM.xf)
    np.save('wmse.npy',EM.wmselist)
    np.save('wmses.npy',EM.wmselistsure)
    np.save('snr.npy',EM.snrlist)
    np.save('mu_s.npy',EM.mu_slist)
    np.save('mu_l.npy',EM.mu_llist)
