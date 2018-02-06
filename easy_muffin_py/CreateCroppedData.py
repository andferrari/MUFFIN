#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:27:53 2018

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
    

folder = 'data'
folder = os.path.join(os.getcwd(), folder)
file_in = 'm31_3d_conv'
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
skyname = genname+'_sky.fits'

CubePSF = 0*checkdim(fits.getdata(psfname, ext=0))[94:158,94:158,:]
CubePSF[int(np.shape(CubePSF)[0]*0.5),int(np.shape(CubePSF)[1]*0.5)] = 1

sky = fits.getdata(skyname, ext=0) 
sky = np.transpose(sky)
sky = sky[94:158,94:158,:]

# Create dirty image 
CubeDirtyy = conv(CubePSF,sky)
CubeDirtyy = CubeDirtyy.real

# Add noise 
snr = 10
var = (np.sum(CubeDirtyy**2)/CubeDirtyy.size)/(10**(snr/10))
noise = np.random.normal(0,var**.5,np.shape(CubeDirtyy))
CubeDirtyy_10db = CubeDirtyy + noise 
Pb = np.sum(noise*noise)
Ps = np.sum(CubeDirtyy*CubeDirtyy)
snrr = 10*np.log10(Ps/Pb)

# save data cubes as M31_3d_conv_256_10db_( psf.fits; dirty.fits; sky.fits)
tmp = fits.PrimaryHDU(CubePSF)
tmp.writeto('m31_3d_crpd2_10db_psf.fits')

tmp = fits.PrimaryHDU(CubeDirtyy_10db)
tmp.writeto('m31_3d_crpd2_10db_dirty.fits')

tmp = fits.PrimaryHDU(sky)
tmp.writeto('m31_3d_crpd2_10db_sky.fits')

pl.figure()
pl.imshow(CubePSF[:,:,1])
pl.colorbar()

pl.figure()
pl.imshow(sky[:,:,1])
pl.colorbar()

pl.figure()
pl.imshow(CubeDirtyy[:,:,1])
pl.colorbar()

pl.figure()
pl.imshow(CubeDirtyy_10db[:,:,1])
pl.colorbar()
    