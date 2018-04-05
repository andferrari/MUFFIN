#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:00:55 2018

@author: rammanouil
"""

#%% ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================
import os
import numpy as np
from astropy.io import fits
import sys
from deconv3d_tools import conv, fix_dim
from mpi4py import MPI
import deconv3D_mpi as dcvMpi
from tictoc import tic, toc
from datetime import datetime
import argparse

#%% ===========================================================================
# MPI
# =============================================================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# =============================================================================
# Input
# =============================================================================
parser = argparse.ArgumentParser(description='Awesome Argument Parser')
parser.add_argument('-L','--L',default=10,type=int,help='Number of bands')
parser.add_argument('-N','--nitermax',default=10,type=int,help='Number of iterations')
parser.add_argument('-mu_s','--mu_s',default=0.2,type=float,help='Spatial regularization tuning parameter')
parser.add_argument('-mu_l','--mu_l',default=2.2,type=float,help='Spectral regularization tuning parameter')
parser.add_argument('-mu_w','--mu_wiener',default=5e1,type=float,help='Weiner regularization tuning parameter for initialisation')
parser.add_argument('-stp_s','--step_mu_s',default=0.001,type=float,help='Gradient step for spatial regularization')
parser.add_argument('-stp_l','--step_mu_l',default=0.001,type=float,help='Gradient step for spectral regularization')
parser.add_argument('-data','--data_suffix',default='M31_3d_conv_256_10db',help='Suffix of data name')
parser.add_argument('-pxl_w','--pixelweight',default=0,type=int,help='Use different weight per pixel')
parser.add_argument('-bnd_w','--bandweight',default=0,type=int,help='Use different weight per band')
parser.add_argument('-fol','--folder',help='Path to data folder')

args = parser.parse_args()

L = args.L
nitermax = args.nitermax
mu_s = args.mu_s
mu_l = args.mu_l
mu_wiener = args.mu_wiener
step_mu_s = args.step_mu_s
step_mu_l = args.step_mu_l
step_mu = [step_mu_s,step_mu_l]
data_suffix = args.data_suffix
pxl_w = args.pixelweight
bnd_w = args.bandweight

folder = args.folder

# =============================================================================
# Load data
# =============================================================================

file_in = data_suffix
folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psf_name = genname+'_psf.fits'
drt_name = genname+'_dirty.fits'
sky_name = genname+'_sky.fits'

cube_psf = fix_dim(fits.getdata(psf_name, ext=0))[:,:,-L:]
cube_dirty = fix_dim(fits.getdata(drt_name, ext=0))[:,:,-L:]

if os.path.isfile(skyname):

    if rank==0:
        print(data_suffix)
        print('')
        print('estimating variance')

    sky = fix_dim(fits.getdata(skyname, ext=0))
    sky = sky[:,:,-L:]

    sky2 = np.sum(sky*sky)
    Noise = cube_dirty - conv(cube_psf,sky)
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

#pl.figure()
#pl.imshow(sky[:,:,1])
#
#pl.figure()
#pl.imshow(CubePSF[:,:,1])
#
#pl.figure()
#pl.imshow(CubeDirty[:,:,1])


#%% ===========================================================================
# Run
# =============================================================================
# DWT parameters
nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)

args = {'mu_s':mu_s,'mu_l':mu_l,'mu_wiener':mu_wiener,'nb':nb,'truesky':sky,'psf':cube_psf,'dirty':cube_dirty,'var':var,'step_mu':step_mu,'pixelweighton':pxl_w,'bandweighton':pxl_w}
tic()

EM= dcvMpi.EasyMuffinSURE(**args)
if rank==0:
    print('using tau: ',EM.tau)
    print('')

EM.loop_fdmc(nitermax)

#%% ===========================================================================
# Save results
# =============================================================================
if rank==0:
    drctry = os.path.join(os.getcwd(), 'output_sigamm/'+os.getenv('SLURM_JOB_ID'))
    print(drctry)
    #os.mkdir(drctry)
    os.chdir(drctry)

    toc()

    if sky is not None:
        np.save('x0_tst.npy',EM.xf)
        np.save('wmse_tst.npy',EM.wmselist)
        np.save('wmses_tst.npy',EM.wmselistsure)
        np.save('wmsesfdmc_tst.npy',EM.wmselistsurefdmc)
        np.save('snr_tst.npy',EM.snrlist)
        np.save('mu_s_tst.npy',EM.mu_slist)
        np.save('mu_l_tst.npy',EM.mu_llist)
        np.save('dxs.npy',EM.dx_sf)
        np.save('dxl.npy',EM.dx_lf)
        np.save('sugar0.npy',EM.sugarfdmclist[0])
        np.save('sugar1.npy',EM.sugarfdmclist[1])
        np.save('cost.npy',EM.costlist)
        np.save('psnrsure.npy',EM.psnrlistsure)
    else:
        np.save('x0_tst.npy',EM.xf)
        np.save('wmses_tst.npy',EM.wmselistsure)
        np.save('mu_s_tst.npy',mu_s)
        np.save('mu_l_tst.npy',mu_l)
