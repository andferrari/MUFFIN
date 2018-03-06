#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:04:29 2018

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
if len(sys.argv)==9:
    L = int(sys.argv[1])
    nitermax = int(sys.argv[2])
    mu_s_min = float(sys.argv[3])
    mu_s_max = float(sys.argv[4])
    mu_l_min = float(sys.argv[5])
    mu_l_max = float(sys.argv[6])
    num = int(sys.argv[7])
    data_suffix = sys.argv[8]
else:
    if rank ==0:
        print('')
        print('-'*100)
        print('You shld input L nitermax mu_s_min mu_s_max mu_l_min mu_l_max num data_suffix')
        print('ex mpirun -np 4 python3 SUGARFDMC_CheckGradient.py 10 10 0 1 0 1 10 M31_3d_conv_256_10db')
    sys.exit()
        
if rank ==0:
    print('')
    print('L: ',L)
    print('nitermax: ',nitermax)
    print('mu_s_min: ',mu_s_min)
    print('mu_s_max: ',mu_s_max)
    print('mu_l_min: ',mu_l_min)
    print('mu_l_max: ',mu_l_max)
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
#sky = np.transpose(sky)[:,:,0:L]
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

mu_1,mu_2 = np.mgrid[mu_s_min:mu_s_max:np.complex(0,num),mu_l_min:mu_l_max:np.complex(0,num)]
mu1mu2 = np.vstack((mu_1.flatten(),mu_2.flatten())).T

if rank == 0:
    Risk = []
    wmse = []
    wmsesure = []
    wmsesurefdmc = []
    sugarfdmc = {}
    sugarfdmc[0] = []
    sugarfdmc[1] = []
    psnr = []
    snr = []
    psnrs = []

for mu in mu1mu2:
    if rank==0:
        print('')
        print('testing mu = ',mu)

    args = {'mu_s':mu[0],'mu_l':mu[1],'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,'step_mu':step_mu}
    tic()

    EMm2= dcvMpi.EasyMuffinSURE(**args)
    EMm2.loop_fdmc(nitermax)
    
    if rank == 0:
        Risk.append(EMm2.costlist[-1])
        wmse.append(EMm2.wmselist[-1])
        wmsesure.append(EMm2.wmselistsure[-1])
        wmsesurefdmc.append(EMm2.wmselistsurefdmc[-1])
        sugarfdmc[0].append(EMm2.sugarfdmclist[0][-1])
        sugarfdmc[1].append(EMm2.sugarfdmclist[1][-1])
        psnr.append(EMm2.snrlist[-1])
        psnrs.append(EMm2.psnrlistsure[-1])
        snr.append(EMm2.snrlist[-1])

if rank ==0:
    wmse_ = np.reshape(wmse,(num,num),order='C')
    dRisk0 = np.diff(wmse_,axis=0)/np.diff(mu_1,axis=0)
    dRisk1 = np.diff(wmse_,axis=1)/np.diff(mu_2,axis=1)

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
    np.save('sugarfdmc0.npy',sugarfdmc[0])
    np.save('sugarfdmc1.npy',sugarfdmc[1])
    np.save('dRisk0.npy',dRisk0)
    np.save('dRisk1.npy',dRisk1)
    np.save('EMsugarfdmc0.npy',EMm2.sugarfdmclist[0])
    np.save('EMsugarfdmc1.npy',EMm2.sugarfdmclist[1])
    np.save('snr.npy',snr)
    np.save('psnr.npy',psnr)
    np.save('psnrs.npy',psnrs)
    np.save('mu1.npy',mu_1)
    np.save('mu2.npy',mu_2)
    

## ==============================================================================
#%% Print Figures
## ==============================================================================
#import matplotlib.pyplot as pl
#from matplotlib import cm
#import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#
#daytime = '2018-01-29 14:50:52.277198'
#daytime = '2018-02-23 12:12:47.795495'
#daytime = '2018-02-22 18:20:25.570735'
#daytime = '2018-02-25 22:47:08.486893'
#drctry = os.path.join(os.getcwd(),'output_Gradient/'+daytime)
#os.chdir(drctry)
#
#Risk = np.load('Risk.npy')
#wmse = np.load('wmse.npy')
#wmsesure = np.load('wmsesure.npy')
#wmsesurefdmc = np.load('wmsesurefdmc.npy')
#sugarfdmc0 = np.load('sugarfdmc0.npy')
#sugarfdmc1 = np.load('sugarfdmc1.npy')
#dRisk0 = np.load('dRisk0.npy')
#
#
#dRisk1 = np.load('dRisk1.npy')
#EMsugarfdmc0 = np.load('EMsugarfdmc0.npy')
#EMsugarfdmc1 = np.load('EMsugarfdmc1.npy')
#snr = np.load('snr.npy')
#psnr = np.load('psnr.npy')
#psnrs = np.load('psnrs.npy')
#mu_1 = np.load('mu1.npy')
#mu_2 = np.load('mu2.npy')
#
#num = int(np.sqrt(np.size(Risk)))
#wmse_ = np.reshape(wmse,(num,num))
#wmsesure_ = np.reshape(wmsesure,(num,num))
#wmsesurefdmc_ = np.reshape(wmsesurefdmc,(num,num))
#
#dRisk0_ =  (dRisk0 )
#dRisk1_ = (dRisk1 )
#sugarfdmc0_ = np.reshape(sugarfdmc0,(num,num))
#sugarfdmc1_ = np.reshape(sugarfdmc1,(num,num))
#
#snr_ = np.reshape(snr,(num,num))
#psnr_ = np.reshape(psnr,(num,num))
#
##
#pl.figure()
#pl.plot(wmse,'*',label='wmse')
#pl.plot(wmsesure,label='wmsesure')
#pl.plot(wmsesurefdmc,label='wmsesurefdmc')
#pl.legend()
#
##
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1, mu_2, wmse_, cmap=cm.nipy_spectral,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.plot_wireframe(mu_1,mu_2 ,wmse_ )
##ax.scatter(mu_1, mu_2, wmse_, c='r', marker='o')
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('wmse_')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(wmse_**0.01,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
#ind = np.unravel_index(np.argmin(wmse_),wmse_.shape)
#mu_1_opt = mu_1[ind]
#mu_2_opt = mu_2[ind]
#
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(snr_,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
#
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1,mu_2,snr_,cmap=cm.nipy_spectral,linewidth=0,antialiased=False)
#fig.colorbar(surf,shrink=0.5,aspect=5)
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('snr')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1,mu_2,-psnr_,cmap=cm.nipy_spectral,linewidth=0,antialiased=False)
#fig.colorbar(surf,shrink=0.5,aspect=5)
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('psnr')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(psnr_**2,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
##
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1, mu_2, wmsesurefdmc_, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.plot_wireframe(mu_1,mu_2 ,wmse_ )
##ax.scatter(mu_1, mu_2, wmse_, c='r', marker='o')
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('wmse_')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(wmsesurefdmc_**0.1,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
##
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1, mu_2, wmsesure_, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.plot_wireframe(mu_1,mu_2 ,wmse_ )
##ax.scatter(mu_1, mu_2, wmse_, c='r', marker='o')
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('wmse_')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(wmsesure_**0.1,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
##
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1[1::,::], mu_2[1::,::], dRisk0_, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.plot_wireframe(mu_1,mu_2 ,wmse_ )
##ax.scatter(mu_1, mu_2, wmse_, c='r', marker='o')
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('wmse_')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(dRisk0_,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
##
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1[::,1::], mu_2[::,1::], dRisk1_, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.plot_wireframe(mu_1,mu_2 ,wmse_ )
##ax.scatter(mu_1, mu_2, wmse_, c='r', marker='o')
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('wmse_')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(dRisk1_,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
##
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1, mu_2, sugarfdmc0_ , cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.plot_wireframe(mu_1,mu_2 ,wmse_ )
##ax.scatter(mu_1, mu_2, wmse_, c='r', marker='o')
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('wmse_')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(sugarfdmc0_,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
##
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1, mu_2, sugarfdmc1_, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.plot_wireframe(mu_1,mu_2 ,wmse_ )
##ax.scatter(mu_1, mu_2, wmse_, c='r', marker='o')
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('wmse_')
#pl.show()
#
#fig = pl.figure()
#ax = fig.gca()
#im = ax.imshow(sugarfdmc1_,cmap=cm.nipy_spectral)
#fig.colorbar(im)
#
##
#fig = pl.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(mu_1, mu_2, snr_, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
##ax.plot_wireframe(mu_1,mu_2 ,wmse_ )
##ax.scatter(mu_1, mu_2, wmse_, c='r', marker='o')
#ax.set_xlabel('mu_1')
#ax.set_ylabel('mu_2')
#ax.set_zlabel('wmse_')
#pl.show()
#
#
#pl.figure()
#pl.plot(EMsugarfdmc0,'-*',label='EMsugarfdmc0')
#pl.legend()
#
#pl.figure()
#pl.plot(EMsugarfdmc1,'-*',label='EMsugarfdmc0')
#pl.legend()
#
#pl.figure()
#pl.plot(snr_)
