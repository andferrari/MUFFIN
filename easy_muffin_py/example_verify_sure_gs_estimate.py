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

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,150:151]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,150:151]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))[:,:,150:151]
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

DM = SNSD(mu_s=0.8,nitermax=10,nb=('db1','db2','db3','db4','db5','db6','db7','db8'),var = var, truesky=sky)

DM.parameters()

DM.setSpectralPSF(CubePSF)
DM.setSpectralDirty(CubeDirty)

# easy MUFFIN
[SpectralSkyModel, cost, snr]=DM.main()

# check results
resid = sky-SpectralSkyModel
print(10*np.log10( sky2 / np.sum(resid*resid)  ))

pl.figure()
pl.plot(cost,label='cost')
pl.legend(loc='best')

pl.figure()
pl.plot(snr,label='snr')
pl.legend(loc='best')
 
#==============================================================================
#%% Test DWT + SURE estimation of mse  
#==============================================================================

# easy MUFFIN SURE 
[SpectralSkyModel_s, cost_s, snr_s, psnr_true, psnr_est, wmse_true, wmse_est]=DM.main(method='sure')

# check results
resid_s = sky-SpectralSkyModel_s
print(10*np.log10( sky2 / np.sum(resid_s*resid_s)  ))

pl.figure()
pl.plot(cost_s,label='cost')
pl.legend(loc='best')

pl.figure()
pl.plot(snr_s,label='snr')
pl.legend(loc='best')

pl.figure()
pl.plot(psnr_true,label='psnr true')
pl.plot(psnr_est,label='psnr est')
pl.legend(loc='best')

pl.figure()
pl.plot(wmse_true,label='wmse true')
pl.plot(wmse_est,label='wmse est')
pl.legend(loc='best')

#==============================================================================
#%% Test DWT + SURE estimation of mse  
#==============================================================================

# easy MUFFIN SURE 
[SpectralSkyModel_s, cost_s, snr_s, psnr_true, psnr_est, wmse_true, wmse_est, mu_s_]=DM.main(method='gs')

# check results
resid_s = sky-SpectralSkyModel_s
print(10*np.log10( sky2 / np.sum(resid_s*resid_s)  ))

pl.figure()
pl.plot(cost_s,label='cost')
pl.legend(loc='best')

pl.figure()
pl.plot(snr_s,label='snr')
pl.legend(loc='best')

pl.figure()
pl.plot(psnr_true,label='psnr true')
pl.plot(psnr_est,label='psnr est')
pl.legend(loc='best')

pl.figure()
pl.plot(wmse_true,label='wmse true')
pl.plot(wmse_est,label='wmse est')
pl.legend(loc='best')

pl.figure()
pl.plot(mu_s_,label='mu_s')
pl.legend(loc='best')


