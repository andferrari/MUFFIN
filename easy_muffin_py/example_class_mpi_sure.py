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

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

#%% ===========================================================================
# MPI 
# =============================================================================

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import deconv3d_mpi as dcvMpi

import deconv3d as dcv

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nitermax = 3
mu_s = 0.5
mu_l = 0.5

if rank==0:
    print('')
    print('----------------------------------------------------------')
    print('                       Easy MUFFIN')
    print('----------------------------------------------------------')
    print('')
    tm.tic()
    EM00= dcv.EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
    EM00.loop(nitermax)
    tm.toc()
    
    print('')
    print('----------------------------------------------------------')
    print('                       Easy MUFFIN SURE')
    print('----------------------------------------------------------')
    print('')
    tm.tic()
    EM0= dcv.EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
    EM0.loop(nitermax)
    tm.toc()

    print('')
    print('----------------------------------------------------------')
    print('                      MPI: Easy MUFFIN SURE')
    print('----------------------------------------------------------')
    print('')
    
# every processor creates EM -- inside each one will do its one part of the job 
tm.tic()
EM= dcvMpi.EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
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

    print('local size:',EM.lst_nbf)
    print('Starting point:',EM.displacements)
    print('Nbr of elts:',EM.sendcounts)
    
    print('')
    
    print('snrm: ',EM.snrlist)
    print('snr0: ',EM0.snrlist)
    print('snr00: ',EM00.snrlist)
    
    print('')
    
    print('costm: ',EM.costlist)
    print('cost0: ',EM0.costlist)
    
    print('')
    
    print('psnrm: ',EM.psnrlist)
    print('psnr0: ',EM0.psnrlist)
    
    print('')
    
    print('wmsem: ',EM.wmselist)
    print('wmse0: ',EM0.wmselist)

    print('')
    
    print('wmsesurem: ', EM.wmselistsure)
    print('wmsesure0: ',EM0.wmselistsure)
       
    print('')
    
    print('psnrsurem: ', EM.psnrlistsure)
    print('psnrsure0: ',EM0.psnrlistsure)
    
    print('')
    
    print('delta: ',np.linalg.norm(EM.deltaf))
    print('delta0: ',np.linalg.norm(EM0.x-2*EM0.xt))
    
    print('')
    
    print('Jdelta: ',np.linalg.norm(EM.Jdeltaf))
    print('Jdelta0: ',np.linalg.norm(EM0.Jx-2*EM0.Jxt))
    
    print('')    
    print('v-v0: ',np.linalg.norm(EM.v-EM0.v))
    
    print('')    
    print('Jv-Jv0: ',np.linalg.norm(EM.Jv-EM0.Jv))
    
    print('')    
    print('vtt-vtt0: ',np.linalg.norm(EM.vtt-EM0.vtt))
    

mse = EM.mse()

if rank ==0:
    print('')
    print('msem: ',EM0.mse())
    print('mse0: ',mse)

    print('')
    print('normxm: ',(np.linalg.norm(EM.xf)))
    print('normx0: ',(np.linalg.norm(EM0.x)))
    print('')
    
    print('')
    print('Error with Muffin: ',(np.linalg.norm(EM.xf -EM0.x)))
    print('')
    
