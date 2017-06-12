#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:08:49 2016

@author: antonyschutz
"""
# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import os
import numpy as np
from astropy.io import fits
import pylab as pl
import sys

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

folder = 'data'
file_in = 'm31_3d_conv_10db'

folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:5]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:5]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = np.transpose(sky)[:,:,0:5]
sky2 = np.sum(sky*sky)

#%% ===========================================================================
# MPI 
# =============================================================================

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import deconv3d_mpi as dcvMpi

import deconv3d as dcv

#nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb = (7,0)
nitermax = 10
mu_s = 0.5
mu_l = 0.5

if rank==0:
    print('')
    print('----------------------------------------------------------')
    print('                       Easy MUFFIN')
    print('----------------------------------------------------------')
    print('')
    tm.tic()
    EM0= dcv.EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty)
    EM0.loop(nitermax)
    tm.toc()
    
    print('')
    print('----------------------------------------------------------')
    print('                   MPI : Easy MUFFIN')
    print('----------------------------------------------------------')
    print('')

# every processor creates EM -- inside each one will do its one part of the job 
tm.tic()
EM= dcvMpi.EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty)
EM.loop(nitermax)

# Once job done - display results ... 
if rank == 0: # I look at the results in EM created by master node even though others also created EM instance
    tm.toc()
    
    print('')
    print('----------------------------------------------------------')
    print('                    Compare results')
    print('----------------------------------------------------------')
    print('')
    
    print('')
    
    print('snr: ',(np.linalg.norm(np.asarray(EM.snrlist)-np.asarray(EM0.snrlist),np.inf)))
    
    print('')
    
    print('cost: ',np.linalg.norm(np.asarray(EM.costlist)-np.asarray(EM0.costlist),np.inf))
    
    print('')
    
    print('psnr: ',np.linalg.norm(np.asarray(EM.psnrlist)-np.asarray(EM0.psnrlist),np.inf))
    
    print('')
    
    print('wmsem: ',np.linalg.norm(np.asarray(EM.wmselist)-np.asarray(EM0.wmselist),np.inf))
    
    print('')
    
    print('v-v0: ',np.linalg.norm(EM.v-EM0.v))
    
    print('')    
    print('vtt-vtt0: ',np.linalg.norm(EM.vtt-EM0.vtt))
    
    print('')
    print('Error with Muffin: ',(np.linalg.norm(EM.xf -EM0.x)))
    print('')
    
if rank ==0 and visu:
    pl.figure(3)
    pl.imshow(EM.xf[:,:,1])
    pl.title('xf')
    
    pl.figure(4)
    pl.imshow(EM0.x[:,:,1]) 
    pl.title('x0')
    
    pl.figure(5)
    pl.imshow(EM0.xtt[:,:,1])
    pl.title('xtt0')
    
    pl.figure(8)
    pl.imshow(EM.tf[:,:,1])
    pl.title('tf')

if rank ==1 and visu:
    pl.figure(6)
    pl.imshow(EM.xtt[:,:,1])
    pl.title('xtt')
    
    pl.figure(7)
    pl.imshow(EM.x[:,:,1])
    pl.title('x')
    
    pl.figure(9)
    pl.imshow(EM.t[:,:,1])
    pl.title('t')
    
if visu:
    pl.show()

