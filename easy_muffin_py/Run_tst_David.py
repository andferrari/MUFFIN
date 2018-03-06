#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:41:28 2018

@author: rammanouil
"""

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================
import numpy as np
from astropy.io import fits
from deconv3d_tools import conv
import os
import matplotlib.pyplot as pl 

# =============================================================================
# Load data
# =============================================================================
def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x

folder = 'data_david'
file_in = 'M31_skyline2_crpd_20db'
L = 256

genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'
CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]
skyname = genname+'_sky.fits'

sky = checkdim(fits.getdata(skyname, ext=0))
sky = sky[:,:,0:L]
sky2 = np.sum(sky*sky)
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

print('')
print('setting var to ', var)


fig = pl.figure()
ax = fig.add_subplot(1,3,1)
ax.imshow(CubePSF[:,:,1])
ax = fig.add_subplot(1,3,2)
ax.imshow(sky[:,:,1])
ax = fig.add_subplot(1,3,3)
ax.imshow(CubeDirty[:,:,1])

from deconv3d import EasyMuffinSURE

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)
nitermax = 500
mu_s = 1
mu_l = 1
step_mu = [0,0]

args = {'mu_s':mu_s,'mu_l':mu_l,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,'step_mu':step_mu}

EM1= EasyMuffinSURE(**args)

print('using tau: ',EM1.tau)
print('')

EM1.loop_fdmc(nitermax)

SpectralSkyModel1 = EM1.xt
cost1 = EM1.costlist
snr1 = EM1.snrlist
psnr1 = EM1.psnrlist
wmse1 = EM1.wmselist


nb=('db1','db2','db3','db4','db5','db6','db7','db8','I')
#nb = (7,0)

args = {'mu_s':mu_s,'mu_l':mu_l,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,'step_mu':step_mu}

EM2= EasyMuffinSURE(**args)

print('using tau: ',EM2.tau)
print('')

EM2.loop_fdmc(nitermax)

SpectralSkyModel2 = EM2.xt
cost2 = EM2.costlist
snr2 = EM2.snrlist
psnr2 = EM2.psnrlist
wmse2 = EM2.wmselist

fig = pl.figure()
ax = fig.add_subplot(1,2,1)
ax.plot(EM1.xt[5,12,:])
ax = fig.add_subplot(1,2,2)
ax.plot(EM2.xt[5,12,:])



