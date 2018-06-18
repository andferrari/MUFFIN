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
# Terminal Input
# =============================================================================
parser = argparse.ArgumentParser(description='Awesome Argument Parser')
parser.add_argument('-fol','--folder',help='Path to data folder')
parser.add_argument('-nam','--file_in',help='Data Prefix')
parser.add_argument('-sav','--save',default=0,help='Save Output Variables')
parser.add_argument('-init','--init',default=0,type=int,help='Init with Saved Variables')
parser.add_argument('-fol_init','--folder_init',help='Path to init data folder')

args = parser.parse_args()

folder = args.folder
file_in = args.file_in
save = args.save
init = args.init
folder_init = args.folder_init

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

genname = os.path.join(folder, file_in)
psf_name = genname+'_psf.fits'
drt_name = genname+'_dirty.fits'

L = 5 
cube_psf = fix_dim(fits.getdata(psf_name, ext=0))[:,:,0:L]
cube_dirty = fix_dim(fits.getdata(drt_name, ext=0))[:,:,0:L]

sky_name = genname+'_sky.fits'
sky = fits.getdata(sky_name, ext=0)
sky = np.transpose(sky)[:,:,0:L]
sky2 = np.sum(sky*sky)

Noise = cube_dirty - conv(cube_psf,sky)
var = np.sum(Noise**2)/Noise.size

#%% ===========================================================================
# MPI Setting 
# =============================================================================

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#%% ==============================================================================
# Tsts EM and EMSURE 
# ==============================================================================

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb=(7,0)
nitermax = 6
mu_s = 1
mu_l = 1
fftw = 1

#if rank==0:
#    print('')
#    print('----------------------------------------------------------')
#    print('                       Easy MUFFIN')
#    print('----------------------------------------------------------')
#    print('')
#    tm.tic()
#    EM00= dcv.EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=cube_psf,dirty=cube_dirty,var=var,fftw=fftw,init=init,
#               fol_init=folder_init,save=0)
#    EM00.loop(nitermax)
#    tm.toc()
#    
#    print('')
#    print('----------------------------------------------------------')
#    print('                       Easy MUFFIN SURE')
#    print('----------------------------------------------------------')
#    print('')
#    tm.tic()
#    EM0= dcv.EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=cube_psf,dirty=cube_dirty,var=var,fftw=fftw,init=init,
#               fol_init=folder_init,save=0)
#    EM0.loop(nitermax)
#    tm.toc()
#
#    print('')
#    print('----------------------------------------------------------')
#    print('                      MPI: Easy MUFFIN SURE')
#    print('----------------------------------------------------------')
#    print('')
    
# every processor creates EM -- inside each one will do its one part of the job 
tm.tic()
EM= dcvMpi.EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=cube_psf,dirty=cube_dirty,var=var,step_mu=[5e-1,5e-1],fftw=fftw,init=init,
               fol_init=folder_init,save=save)
#EM.loop(nitermax)
EM.loop_fdmc(nitermax)

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

#    print('')
#    
#    print('snr: ',np.linalg.norm(np.asarray(EM.snrlist)-np.asarray(EM0.snrlist),np.inf))
#    
#    print('')
#    
#    print('cost: ',np.linalg.norm(np.asarray(EM.costlist)-np.asarray(EM0.costlist),np.inf))
#    
#    print('')
#    
#    print('psnr: ',np.linalg.norm(np.asarray(EM.psnrlist)-np.asarray(EM0.psnrlist),np.inf))
#    
#    print('')
#    
#    print('wmsem: ',np.linalg.norm(np.asarray(EM.wmselist)-np.asarray(EM0.wmselist),np.inf))
#    
#    print('')
#    
#    print('v-v0: ',np.linalg.norm(EM.v-EM0.v))
#    
#    print('')    
#    print('vtt-vtt0: ',np.linalg.norm(EM.vtt-EM0.vtt))
#    
#    print('')
#    print('Error with Muffin: ',(np.linalg.norm(EM.xf -EM0.x)))
#    print('')    
    
    print('')
    print('EM.dx_s: ',np.linalg.norm(EM.dx_sf))
    print('')    

    print('')
    print('EM.dx_l: ',np.linalg.norm(EM.dx_lf))
    print('')    

    print('')
    print('EM.dx2_s: ',np.linalg.norm(EM.dx2_sf))
    print('')    

    print('')
    print('EM.dx2_s: ',np.linalg.norm(EM.dx2_lf))
    print('')    

    print('')
    print('EMsfdmc.sugarfdmclist: ',EM.sugarfdmclist[0][:])
    print('')    

    print('')
    print('EMsfdmc.sugarfdmclist: ',EM.sugarfdmclist[1][:])
    print('')    

