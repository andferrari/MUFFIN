#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:25:03 2018

@author: rammanouil
"""

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import os
import sys 
import numpy as np
from astropy.io import fits
from deconv3d_tools import conv
from mpi4py import MPI
import deconv3D_mpi2 as dcvMpi
from tictoc import tic, toc
from datetime import datetime 

# ==============================================================================
# MPI 
# ==============================================================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# ==============================================================================
# Save output results to terminal and txt file  
# ==============================================================================
class Logger():
    
    def __init__(self,filename):
        self.terminal = sys.stdout
        self.log = open(filename,'w')
        
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

if rank ==0:
    daytime = str(datetime.now())
    file = os.path.join(os.getcwd(),'output_Gradient/'+daytime+'_GradientOut.txt')
    sys.stdout = Logger(file)
 
# ==============================================================================
# Input from terminal 
# ==============================================================================
if len(sys.argv)==8:
    L = int(sys.argv[1])
    nitermax = int(sys.argv[2])
    mu_l_min = float(sys.argv[3])
    mu_l_max = float(sys.argv[4])
    mu_s = float(sys.argv[5])
    num = float(sys.argv[6])
    data_suffix = sys.argv[7]
else:
    if rank ==0:
        print('')
        print('-'*100)
        print('You shld input L nitermax mu_l_min mu_l_max mu_s num data_suffix')
        print('ex mpirun -np 4 python3 SUGARFDMC_CheckGradient.py 10 10 0 1 10 0.5 M31_3d_conv_256_10db')
    sys.exit()
        
if rank ==0:
    print('')
    print('L: ',L)
    print('nitermax: ',nitermax)
    print('mu_l_min: ',mu_l_min)
    print('mu_l_max: ',mu_l_max)
    print('mu_s: ',mu_s)
    print('num: ',num)
    print('data_suffix: ',data_suffix)

# ==============================================================================
# Load data  
# ==============================================================================

def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x


folder = 'data'
file_in = data_suffix

folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = np.transpose(sky)[:,:,0:L]
sky = sky[:,:,0:L]
sky2 = np.sum(sky*sky)

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

# ==============================================================================
# Run  
# ==============================================================================
nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)
step_mu = [0,0]

mu = np.linspace(mu_l_min,mu_l_max,num)

if rank == 0:
    Risk = []
    wmse = []
    wmsesure = []
    wmsesurefdmc = []
    sugarfdmc = {}
    sugarfdmc[0] = []
    sugarfdmc[1] = []
                  
for mu_l in mu:
    if rank==0:
        print('')
        print('testing mu_l = ',mu_l)
    
    args = {'mu_s':mu_s,'mu_l':mu_l,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,'step_mu':step_mu}
    tic()

    EMm2= dcvMpi.EasyMuffinSURE(**args)
    EMm2.loop_fdmc(nitermax)
    
    if rank == 0:
        Risk.append(EMm2.costlist[-1])
        wmse.append(EMm2.wmselist[-1])
        wmsesure.append(EMm2.wmselistsure[-1])
        wmsesurefdmc.append(EMm2.wmselistsurefdmc[-1])
        sugarfdmc[1].append(EMm2.sugarfdmclist[1][-1])
        
if rank ==0: 
    dRisk = np.diff(wmse)/np.diff(mu)

# ==============================================================================
# Save results   
# ==============================================================================

if rank ==0:
    drctry = os.path.join(os.getcwd(),'output_Gradient/'+daytime)
    os.mkdir(drctry)
    os.chdir(drctry)
    
    toc()
    
    np.save('Risk.npy',Risk)
    np.save('wmse.npy',wmse)
    np.save('wmsesure.npy',wmsesure)
    np.save('wmsesurefdmc.npy',wmsesurefdmc)
    np.save('sugarfdmc1.npy',sugarfdmc[1])
    np.save('dRisk.npy',dRisk)
    np.save('EMsugarfdmc1.npy',EMm2.sugarfdmclist[1])
    

## ==============================================================================
## Print Figures
#### ==============================================================================
import matplotlib.pyplot as pl 
daytime = '2018-01-29 09:56:39.940328'
daytime = '2018-01-29 10:25:24.076079'
drctry = os.path.join(os.getcwd(),'output_Gradient/'+daytime)
os.chdir(drctry)

Risk = np.load('Risk.npy')
wmse = np.load('wmse.npy')
wmsesure = np.load('wmsesure.npy')
wmsesurefdmc = np.load('wmsesurefdmc.npy')
sugarfdmc1 = np.load('sugarfdmc1.npy')
dRisk = np.load('dRisk.npy')
EMsugarfdmc1 = np.load('EMsugarfdmc1.npy')

pl.figure()
pl.plot(Risk,label='Risk')
pl.legend()

pl.figure()
pl.plot(wmse,label='wmse')
pl.plot(wmsesure,label='wmsesure')
pl.plot(wmsesurefdmc,label='wmsesurefdmc')
pl.legend()

pl.figure()
pl.plot(sugarfdmc1[1::],label='sugarfdmc1')
pl.plot(np.zeros(dRisk.size))
pl.legend()

pl.figure()
pl.plot(dRisk,label='dRisk')
pl.plot(np.zeros(dRisk.size))
pl.legend()

pl.figure()
pl.plot(sugarfdmc1[1::],'-*',label='sugarfdmc1')
pl.plot(dRisk,'-^',label='dRisk')
pl.plot(np.zeros(dRisk.size))
pl.legend()

pl.figure()
pl.plot(EMsugarfdmc1,label='EMsugarfdmc1')
pl.legend()

