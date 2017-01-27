#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:08:49 2016

@author: antonyschutz
"""
#==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
#==============================================================================
import os
import numpy as np 
from astropy.io import fits
import pylab as pl 

def checkdim(x):
    if len(x.shape)==4:
        x =np.squeeze(x)
        x = x.transpose((2,1,0))
    return x
    
    
folder='data256'
file_in = 'M31_3d_conv_256_10db'

folder = os.path.join(os.getcwd(),folder)
genname = os.path.join(folder,file_in) 
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,250:251]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,250:251]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))[:,:,250:251]
sky2 = np.sum(sky*sky)

from SuperNiceSpectraDeconv import SNSD 

pl.figure()
pl.imshow(CubePSF[:,:,0])

pl.figure()
pl.imshow(CubeDirty[:,:,0])

pl.figure()
pl.imshow(sky[:,:,0])

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

DM = SNSD(mu_s=0.5,nitermax=50,nb=('db1','db2','db3','db4','db5','db6','db7','db8'),truesky=sky)

DM.parameters()

DM.setSpectralPSF(CubePSF)
DM.setSpectralDirty(CubeDirty)

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

#%% ============================================================================
# Search for best mu_s 
# ==============================================================================

from SuperNiceSpectraDeconv import SNSD

mu_s_ = np.linspace(0,2,num=21)
snr_ = np.zeros(mu_s_.shape)
niter = 0

for mu_s in mu_s_:
    print(mu_s)
    DM = SNSD(mu_s=mu_s,nitermax=1000,nb=('db1','db2','db3','db4','db5','db6','db7','db8'),truesky=sky)
    DM.parameters()
    DM.setSpectralPSF(CubePSF)
    DM.setSpectralDirty(CubeDirty)
    
    [SpectralSkyModel , cost, snr] = DM.main()
    
    resid = sky-SpectralSkyModel
    snr = 10*np.log10(sky2 / np.sum(resid*resid))
    
    snr_[niter] = snr
    niter+=1


pl.figure()
pl.plot(mu_s_,snr_)
pl.show()

pl.figure()
pl.plot(cost)
pl.show()

