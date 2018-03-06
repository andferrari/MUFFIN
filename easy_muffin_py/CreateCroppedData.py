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
file_in = 'halo_unmix_3spec'
#file_in = 'm31_3d_conv'
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
skyname = genname+'_sky.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))
#CubePSF[int(np.shape(CubePSF)[0]*0.5),int(np.shape(CubePSF)[1]*0.5)] = 1

sky = fits.getdata(skyname, ext=0) 
sky = np.transpose(sky)
sky = 560*sky[:,:,:256]

# Create dirty image 
CubeDirtyy = conv(CubePSF,sky)
CubeDirtyy = CubeDirtyy.real

# Add noise 
snr = 20
var = (np.sum(CubeDirtyy**2)/CubeDirtyy.size)/(10**(snr/10))
noise = np.random.normal(0,var**.5,np.shape(CubeDirtyy))
CubeDirtyy_20db = CubeDirtyy + noise 
Pb = np.sum(noise*noise)
Ps = np.sum(CubeDirtyy*CubeDirtyy)
snrr = 10*np.log10(Ps/Pb)

pl.figure()
pl.imshow(sky[:,:,10],cmap='nipy_spectral')
pl.colorbar()
pl.title('sky')

pl.figure()
pl.imshow(CubePSF[:,:,10],cmap='nipy_spectral')
pl.colorbar()
pl.title('psf')

pl.figure()
pl.imshow(CubeDirtyy[:,:,10],cmap='nipy_spectral')
pl.colorbar()
pl.title('dirty')

pl.figure()
pl.imshow(CubeDirtyy_20db[:,:,10],cmap='nipy_spectral')
pl.colorbar()
pl.title('dirty noise')

# save data cubes as M31_3d_conv_256_10db_( psf.fits; dirty.fits; sky.fits)
tmp = fits.PrimaryHDU(CubePSF)
tmp.writeto('halo_unmix_3spec_x560_20db_psf.fits')

tmp = fits.PrimaryHDU(CubeDirtyy_20db)
tmp.writeto('halo_unmix_3spec_x560_20db_dirty.fits')

tmp = fits.PrimaryHDU(sky)
tmp.writeto('halo_unmix_3spec_x560_20db_sky.fits')

#%% Load and check if were correctly saved 
def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x


file_in = 'halo_unmix_3spec_x560_20db'
psfname = file_in+'_psf.fits'
drtname = file_in+'_dirty.fits'
skyname = file_in+'_sky.fits'

L = 256 

CubePSF_ = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty_ = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]
#CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,-L:]
#CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,-L:]


sky_ = checkdim(fits.getdata(skyname, ext=0))
#sky = np.transpose(sky)[:,:,0:L]
    
sky_ = sky[:,:,0:L]
#sky = sky[:,:,-L:]
    
sky2 = np.sum(sky_*sky_)
Noise = CubeDirty_ - conv(CubePSF_,sky_)
var_ = np.sum(Noise**2)/Noise.size
    
pl.figure()
pl.imshow(CubePSF_[:,:,1],cmap='nipy_spectral')
pl.colorbar()

pl.figure()
pl.imshow(sky_[:,:,1],cmap='nipy_spectral')
pl.colorbar()

pl.figure()
pl.imshow(CubeDirty_[:,:,1],cmap='nipy_spectral')
pl.colorbar()
