# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import numpy as np
from astropy.io import fits
import pylab as pl
from deconv3d_tools import conv 

psfname = 'data_Mircea/dirty_images/PSF.fits'
drtname = 'data_Mircea/dirty_images/Hoag_dirty.fits'
skyname = 'data_Mircea/true_images/Hoag.fits'

CubePSF = fits.getdata(psfname, ext=0)
CubeDirty = fits.getdata(drtname, ext=0)

sky = fits.getdata(skyname, ext=0)
sky2 = np.sum(sky*sky)

pl.figure()
pl.imshow(sky)
pl.figure()
pl.imshow(CubePSF)
pl.figure()
pl.imshow(CubeDirty)

#%% ==============================================================================
#
# ==============================================================================

from deconv2d import EasyMuffin, EasyMuffinSURE

#%% ===========================================================================
# Set parameters
# =============================================================================

# nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb = 8
mu_s = 0.1
sigma=10
truesky=sky
psf=CubePSF
dirty=CubeDirty
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
nitermax = 10

args = {'mu_s':mu_s,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var}

EM= EasyMuffin(**args)
EM.loop(nitermax)
SpectralSkyModel2 = EM.xt
cost2 = EM.costlist
snr2 = EM.snrlist
psnr2 = EM.psnrlist
wmse2 = EM.wmselist

EMs= EasyMuffinSURE(**args)
EMs.loop_fdmc(nitermax)
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
