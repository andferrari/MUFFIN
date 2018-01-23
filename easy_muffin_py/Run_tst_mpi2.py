#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:00:55 2018

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
import deconv3D_mpi2 as dcvMpi
from tictoc import tic, toc
from datetime import datetime

#%% ===========================================================================
# MPI Run
# =============================================================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


#%% ===========================================================================
# Saving output results
# =============================================================================


class Logger():
    
    def __init__(self,filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

if rank==0:
    daytime = str(datetime.now())
    file = os.path.join(os.getcwd(),'output/'+daytime+'_out.txt')
    sys.stdout = Logger(file)
    #sys.stdout = open(file,'w')

# =============================================================================
# Input
# =============================================================================
if len(sys.argv)==8:
    L = int(sys.argv[1])
    nitermax = int(sys.argv[2])
    mu_s = float(sys.argv[3])
    mu_l = float(sys.argv[4])
    step_mu_l = float(sys.argv[5])
    step_mu_s = float(sys.argv[6])
    step_mu = [step_mu_l, step_mu_s]
    data_suffix = sys.argv[7]
elif len(sys.argv)==1:
    L = 10
    nitermax = 10
    mu_s = 0.2
    mu_l = 2.2
    data_suffix = 'M31_3d_conv_256_10db'
    step_mu_l = 1e-3
    step_mu_s = 1e-3
    step_mu = [step_mu_l, step_mu_s]
else:
    if rank==0:
        print('')
        print('-'*100)
        print('You should input: L nitermax mu_s mu_l data_suffix step_mu_l step_mu_s ')
        print('')
        print('L: number of bands to be considered')
        print('nitermax: maximum number of iterations ')
        print('mu_s, mu_l: spatial, spectral reg.')
        print('data_suffix: name suffix of data in folder data256')
        print('step_mu_l step_mu_s : step for gradient descent')
        print('')
        print('            **** ex: mpirun -np 4 python3 Run_GS.py 10 0.2 0.2 1e-3 1e-3 M31_3d_conv_256_10db                ')
        print('')
        print('-'*100)
        print('')
    sys.exit()

if rank==0:
    print('')
    print('L: ',L)
    print('nitermax: ',nitermax)
    print('mu_s: ',mu_s)
    print('mu_l: ',mu_l)
    print('data_suffix',data_suffix)
    print('step_mu: ',step_mu)

# =============================================================================
# Load data
# =============================================================================
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

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]
skyname = genname+'_sky.fits'

if os.path.isfile(skyname):
    if rank==0:
        print('')
        print('estimating variance')
        
    #sky = checkdim(fits.getdata(skyname, ext=0))[:,:,0:L]
    sky = checkdim(fits.getdata(skyname, ext=0))
    sky = np.transpose(sky)[:,:,0:L]

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

args = {'mu_s':mu_s,'mu_l':mu_l,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,'step_mu':step_mu}
tic()

EM= dcvMpi.EasyMuffinSURE(**args)
if rank==0:
    print('using tau: ',EM.tau)
    print('')

EM.loop_fdmc(nitermax)


#%% ==================================dct=========================================
# Save results
# =============================================================================
if rank==0:
    drctry = os.path.join(os.getcwd(), 'output/'+daytime)
    os.mkdir(drctry)
    os.chdir(drctry)

    toc()
        
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




