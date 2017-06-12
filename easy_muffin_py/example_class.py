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


folder = 'data256Eusipco'
file_in = 'M31_3d_conv_256_10db'

folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:5]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:5]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = sky[:,:,0:5]
sky2 = np.sum(sky*sky)

fig = pl.figure()
ax = fig.add_subplot(1,3,1)
ax.imshow(CubePSF[:,:,1])
ax = fig.add_subplot(1,3,2)
ax.imshow(sky[:,:,1])
ax = fig.add_subplot(1,3,3)
ax.imshow(CubeDirty[:,:,1])

#%% ==============================================================================
#
# ==============================================================================

from SuperNiceSpectraDeconv import SNSD
from deconv3d import EasyMuffin, EasyMuffinSURE

#nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb = (7,0)
nitermax = 10

mu_s = 0.5
mu_l = 2

DM = SNSD(mu_s=mu_s, mu_l = mu_l, nb=nb,nitermax=nitermax,truesky=sky)
DM.parameters()
DM.setSpectralPSF(CubePSF)
DM.setSpectralDirty(CubeDirty)
(SpectralSkyModel , cost, snr, psnr) = DM.main()

EM= EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty)
EM.loop(nitermax)
SpectralSkyModel2 = EM.xt
cost2 = EM.costlist
snr2 = EM.snrlist
psnr2 = EM.psnrlist
wmse2 = EM.wmselist

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EMs= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
EMs.loop(nitermax)
SpectralSkyModel3 = EMs.xt
cost3 = EMs.costlist
snr3 = EMs.snrlist
psnr3 = EMs.psnrlist
psnrsure3 = EMs.psnrlistsure
wmse3 = EMs.wmselist
wmsesure3 = EMs.wmselistsure

pl.figure()
pl.plot(snr,label='snr1')
pl.plot(snr2,label='snr2')
pl.plot(snr3,'*',label='snr3')
pl.legend(loc='best')

pl.figure()
pl.plot(cost/(EM.nxy*EM.nxy*EM.nfreq),label='cost1')
pl.plot(cost2,label='cost2')
pl.plot(cost3,'*',label='cost3')
pl.legend(loc='best')

pl.figure()
pl.plot(psnr,label='psnr1')
pl.plot(psnr2,label='psnr2')
pl.plot(psnr3,'*',label='psnr3')
pl.plot(psnrsure3,'*',label='psnrsure3')
pl.legend(loc='best')

pl.figure()
pl.plot(wmse2,label='wmse2')
pl.plot(wmse3,label='wmse3')
pl.plot(wmsesure3,'*',label='wmsesure3')
pl.legend(loc='best')

