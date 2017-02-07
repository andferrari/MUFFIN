#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:08:49 2016

@author: antonyschutz
"""
#==============================================================================
# Imports
#==============================================================================
import os
import numpy as np 
from numpy.fft import fft2, ifft2, ifftshift
from astropy.io import fits
import pylab as pl 

#==============================================================================
# tools 
#==============================================================================
def checkdim(x):
    if len(x.shape)==4:
        x =np.squeeze(x)
        x = x.transpose((2,1,0))
    return x

def conv(x,y):
    tmp = ifftshift(ifft2(fft2(x,axes=(0,1))*fft2(y,axes=(0,1)),axes=(0,1)),axes=(0,1))
    return tmp.real
#==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
#==============================================================================    
folder='data256'
file_in = 'M31_3d_conv_256_10db'

folder = os.path.join(os.getcwd(),folder)
genname = os.path.join(folder,file_in) 
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,150:155]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,150:155]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))[:,:,150:155]
sky2 = np.sum(sky*sky)

from SuperNiceSpectraDeconv import SNSD 

#pl.figure()
#pl.imshow(CubePSF[:,:,0])
#
#pl.figure()
#pl.imshow(CubeDirty[:,:,0])
#
#pl.figure()
#pl.imshow(sky[:,:,0])

#==============================================================================
#%% Test IUWT 
#==============================================================================
#DM = SNSD(mu_s=.5,nb=(8,0),tau=1e-4,truesky=sky)
#
#DM.parameters()
#
#DM.setSpectralPSF(CubePSF)
#DM.setSpectralDirty(CubeDirty)
#
#[SpectralSkyModel, cost, snr]=DM.main()
#
## check results
#resid = sky-SpectralSkyModel
#print(10*np.log10( sky2 / np.sum(resid*resid)  ))
#
#pl.figure()
#pl.plot(cost,label='cost')
#pl.legend(loc='best')
#
#pl.figure()
#pl.plot(snr,label='snr')
#pl.legend(loc='best')

#==============================================================================
#%% Test DWT 
#==============================================================================

# compute noise var 
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

DM = SNSD(mu_s=0.8,mu_l=0,nitermax=1000,nb=('db1','db2','db3','db4','db5','db6','db7','db8'),var = var, truesky=sky)

DM.parameters()

DM.setSpectralPSF(CubePSF)
DM.setSpectralDirty(CubeDirty)

# easy MUFFIN
[SpectralSkyModel, cost_, snr_, psnr_ ]=DM.main()

# check results
resid = sky-SpectralSkyModel
print(10*np.log10( sky2 / np.sum(resid*resid)  ))

pl.figure()
pl.plot(cost_,label='cost')
pl.legend(loc='best')

pl.figure()
pl.plot(snr_,label='snr')
pl.legend(loc='best')

pl.figure()
pl.plot(psnr_,label='psnr')
pl.legend(loc='best')

#==============================================================================
#%% Test DWT + SURE estimation of mse  
#==============================================================================

## easy MUFFIN SURE 
#[SpectralSkyModel_s, cost_s, snr_s, psnr_true, psnr_est, wmse_true, wmse_est]=DM.main(method='sure')
#
## check results
#resid_s = sky-SpectralSkyModel_s
#print(10*np.log10( sky2 / np.sum(resid_s*resid_s)  ))
#
#pl.figure()
#pl.plot(cost_s,label='cost')
#pl.legend(loc='best')
#
#pl.figure()
#pl.plot(snr_s,label='snr')
#pl.legend(loc='best')
#
#pl.figure()
#pl.plot(psnr_true,label='psnr true')
#pl.plot(psnr_est,label='psnr est')
#pl.legend(loc='best')
#
#pl.figure()
#pl.plot(wmse_true,label='wmse true')
#pl.plot(wmse_est,label='wmse est')
#pl.legend(loc='best')

#==============================================================================
#%% Test DWT + SURE estimation of mse  
#==============================================================================

# easy MUFFIN SURE 
[SpectralSkyModel_gs, cost_gs, snr_gs, psnr_true_gs, psnr_est_gs, wmse_true_gs, wmse_est_gs, mu_s_gs,mu_l_gs]=DM.main(method='gs2')

# check results
resid_gs = sky-SpectralSkyModel_gs
print(10*np.log10( sky2 / np.sum(resid_gs*resid_gs)  ))

#pl.figure()
#pl.plot(cost_gs,label='cost')
#pl.legend(loc='best')
#
pl.figure()
pl.plot(snr_gs,label='snr')
pl.legend(loc='best')
#
pl.figure()
pl.plot(psnr_true_gs,label='psnr true')
pl.plot(psnr_est_gs,label='psnr est')
pl.legend(loc='best')
pl.xlabel('niter')
pl.ylabel('psnr')
#
#pl.figure()
#pl.plot(wmse_true_gs,label='wmse true')
#pl.plot(wmse_est_gs,label='wmse est')
#pl.legend(loc='best')

#pl.figure()
#pl.stem(mu_s_gs,label='mu_s')
#pl.legend(loc='best')
#
#
pl.figure()
pl.plot(snr_,label='snr')
pl.plot(snr_gs,label='snr_gs')
pl.legend(loc='best')
pl.xlabel('niter')
pl.ylabel('')
#
#pl.figure()
#pl.plot(cost_,label='cost')
#pl.plot(cost_s,label='cost_s')
#pl.plot(cost_gs,label='cost_gs')
#pl.legend(loc='best')

pl.figure()
pl.plot(mu_s_gs,label='mu_s')
pl.plot(mu_l_gs,label='mu_l')
pl.legend(loc='best')

np.save('mu_s_gs.npy', mu_s_gs)
np.save('mu_l_gs.npy', mu_l_gs)
np.save('snr_gs.npy', snr_gs)
np.save('snr_.npy', snr_)
np.save('psnr_true_gs.npy', psnr_true_gs)
np.save('psnr_est_gs.npy', psnr_est_gs)


psnr_true_gs = np.load('psnr_true_gs.npy')
psnr_est_gs = np.load('psnr_est_gs.npy')

pl.figure()
pl.plot(psnr_,label='psnr opt')
#pl.plot(psnr_true_gs,label='psnr true')
pl.plot(psnr_est_gs,label='psnr est')
pl.legend(loc='best')
pl.xlabel('niter')
pl.ylabel('psnr')

snr_gs = np.load('snr_gs.npy')
pl.figure()
pl.plot(snr_gs,label='gs')
pl.plot(snr_,label='opt')
pl.xlabel('niter')
pl.ylabel('snr')
pl.legend(loc='best')
