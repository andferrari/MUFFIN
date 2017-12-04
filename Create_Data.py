#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:16:36 2017

@author: rammanouil
"""

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import os
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from astropy.io import fits
import pylab as pl

def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x

def conv(x,y):
    tmp = ifftshift(ifft2(fft2(x,axes=(0,1))*fft2(y,axes=(0,1)),axes=(0,1)),axes=(0,1))
    return tmp.real
    

folder = 'data256'
folder = os.path.join(os.getcwd(), folder)
file_in = 'M31_256freq'
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))
CubeDirty = checkdim(fits.getdata(drtname, ext=0))

skyname = os.path.join(folder,'M31_3d_sky_256freq.fits')
sky = fits.getdata(skyname, ext=0) 
sky = np.transpose(sky)

# Create dirty image 
CubeDirtyy = conv(CubePSF,sky)
CubeDirtyy = CubeDirtyy.real

# Add noise 
snr = 3
var = (np.sum(CubeDirtyy**2)/CubeDirtyy.size)/(10**(snr/10))
noise = np.random.normal(0,var**.5,np.shape(CubeDirtyy))
CubeDirtyy_10db = CubeDirtyy + noise 
Pb = np.sum(noise*noise)
Ps = np.sum(CubeDirtyy*CubeDirtyy)
snrr = 10*np.log10(Ps/Pb)

# save data cubes as M31_3d_conv_256_10db_( psf.fits; dirty.fits; sky.fits)
tmp = fits.PrimaryHDU(CubePSF)
tmp.writeto('M31_3d_conv_256_3db_psf.fits')

tmp = fits.PrimaryHDU(CubeDirtyy_10db)
tmp.writeto('M31_3d_conv_256_3db_dirty.fits')

tmp = fits.PrimaryHDU(sky)
tmp.writeto('M31_3d_conv_256_3db_sky.fits')

#pl.figure()
#pl.imshow(CubePSF[:,:,100])
#pl.colorbar()

#pl.figure()
#pl.imshow(CubeDirty[:,:,10])
#pl.colorbar()

#pl.figure()
#pl.imshow(sky[:,:,100])
#pl.colorbar()

#pl.figure()
#pl.imshow(CubeDirtyy[:,:,100])
#pl.colorbar()

#pl.figure()
#pl.imshow(CubeDirtyy_10db[:,:,100])
#pl.colorbar()
    