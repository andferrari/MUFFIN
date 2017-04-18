#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:08:49 2016

@author: antonyschutz
"""
# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import os
import numpy as np
from astropy.io import fits
import pylab as pl
from deconv3d_tools import conv

def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x

folder = 'data256'
file_in = 'M31_3d_conv_256_10db'

folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = np.squeeze(checkdim(fits.getdata(psfname, ext=0))[:,:,0:1])
CubeDirty = np.squeeze(checkdim(fits.getdata(drtname, ext=0))[:,:,0:1])

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = sky[:,:,0:1]
sky = np.squeeze(sky)
sky2 = np.sum(sky*sky)

#pl.figure()
#pl.imshow(sky)
#pl.figure()
#pl.imshow(CubePSF)
#pl.figure()
#pl.imshow(CubeDirty)

#%% ==============================================================================
#
# ==============================================================================

from deconv2d import EasyMuffin, EasyMuffinSURE

#%% ===========================================================================
# Set parameters
# =============================================================================

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
mu_s = 0.7
sigma=10
truesky=sky
psf=CubePSF
dirty=CubeDirty
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
nitermax = 5

args = {'mu_s':mu_s,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var}

EM= EasyMuffin(**args)
EM.loop(nitermax)
SpectralSkyModel2 = EM.xt
cost2 = EM.costlist
snr2 = EM.snrlist
psnr2 = EM.psnrlist
wmse2 = EM.wmselist

EMs= EasyMuffinSURE(**args)
EMs.loop(nitermax)
SpectralSkyModel3 = EMs.xt
cost3 = EMs.costlist
snr3 = EMs.snrlist
psnr3 = EMs.psnrlist
psnrsure3 = EMs.psnrlistsure
wmse3 = EMs.wmselist
wmsesure3 = EMs.wmselistsure

pl.figure()
pl.plot(snr2,label='snr2')
pl.plot(snr3,'*',label='snr3')
pl.legend(loc='best')

pl.figure()
pl.plot(cost2,label='cost2')
pl.plot(cost3,'*',label='cost3')
pl.legend(loc='best')

pl.figure()
pl.plot(psnr2,label='psnr2')
pl.plot(psnr3,'*',label='psnr3')
pl.plot(psnrsure3,'*',label='psnrsure3')
pl.legend(loc='best')

pl.figure()
pl.plot(wmse2,label='wmse2')
pl.plot(wmse3,label='wmse3')
pl.plot(wmsesure3,'*',label='wmsesure3')
pl.legend(loc='best')

#%% ===========================================================================
# Find best mu_s mu_l using greedy approach
# =============================================================================
nb=('db1','db2','db3','db4','db5','db6','db7','db8')
mu_s = 0.7
sigma=10
truesky=sky
psf=CubePSF
dirty=CubeDirty
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
nitermax = 50
mu_s_max = 5
mu_s_min = 0
absolutePrecision = 0.01
thresh = 1e-20

args = {'mu_s':mu_s,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,
        'mu_s_max':mu_s_max,'mu_s_min':mu_s_min,'absolutePrecision':absolutePrecision,'thresh':thresh}

EM= EasyMuffinSURE(**args)

#%% ===========================================================================
# MPI loop
# =============================================================================
nitermax = 100
maxiter = 30
nitermean = 20
EM.loop_mu_s(nitermax,maxiter)
EM.set_mean_mu_s(niter=nitermean)
EM.loop(nitermax)

pl.figure()
pl.plot(EM.mu_slist)

pl.figure()
pl.plot(EM.snrlist)
