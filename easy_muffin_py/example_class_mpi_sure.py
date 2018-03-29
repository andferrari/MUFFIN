#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:08:49 2016

@author: antonyschutz
"""
# ==============================================================================
# Imports
# ==============================================================================
import os
import numpy as np
from astropy.io import fits
from deconv3d_tools import conv, fix_dim
import argparse
from mpi4py import MPI 
import tictoc as tm

import deconv3d_mpi as dcvMpi
import deconv3d as dcv

# =============================================================================
# Input
# =============================================================================
parser = argparse.ArgumentParser(description='Awesome Argument Parser')
parser.add_argument('-fol','--folder',help='Path to data')
parser.add_argument('-nam','--file_in',help='Data Prefix')

args = parser.parse_args()

folder = args.folder
file_in = args.file_in

genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = fix_dim(fits.getdata(psfname, ext=0))[:,:,0:5]
CubeDirty = fix_dim(fits.getdata(drtname, ext=0))[:,:,0:5]

skyname = genname+'_sky.fits'
sky = fits.getdata(skyname, ext=0)
sky = np.transpose(sky)[:,:,0:5]
sky2 = np.sum(sky*sky)

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

#%% ===========================================================================
# MPI 
# =============================================================================

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb=(7,0)
nitermax = 4
mu_s = 0.
mu_l = 0.5
fftw = 1

if rank==0:
    print('')
    print('----------------------------------------------------------')
    print('                       Easy MUFFIN')
    print('----------------------------------------------------------')
    print('')
    tm.tic()
    EM00= dcv.EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,fftw=fftw)
    EM00.loop(nitermax)
    tm.toc()
    
    print('')
    print('----------------------------------------------------------')
    print('                       Easy MUFFIN SURE')
    print('----------------------------------------------------------')
    print('')
    tm.tic()
    EM0= dcv.EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,fftw=fftw)
    EM0.loop(nitermax)
    tm.toc()

    print('')
    print('----------------------------------------------------------')
    print('                      MPI: Easy MUFFIN SURE')
    print('----------------------------------------------------------')
    print('')
    
# every processor creates EM -- inside each one will do its one part of the job 
tm.tic()
EM= dcvMpi.EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,fftw=fftw)
EM.loop(nitermax)


#%% ===========================================================================
# Validating results  
# =============================================================================
    
# Once job done - display results ... 
if rank == 0: # I look at the results in EM created by master node even though others also created EM instance
    tm.toc()
    
    print('')
    print('----------------------------------------------------------')
    print('                    Compare results')
    print('----------------------------------------------------------')
    print('')

    print('')
    
    print('snr: ',np.linalg.norm(np.asarray(EM.snrlist)-np.asarray(EM0.snrlist),np.inf))
    
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
