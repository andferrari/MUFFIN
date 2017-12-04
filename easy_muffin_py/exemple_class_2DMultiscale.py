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

from deconv2d import EasyMuffin, EasyMuffinSURE
from deconv2dMultiscale import EasyMuffin as EasyMuffinm 
from deconv2dMultiscale import EasyMuffinSURE as EasyMuffinSUREm 

#%% ===========================================================================
# Set parameters
# =============================================================================

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb = (3,0)
mu_s = 1.6
sigma=10
truesky=sky
psf=CubePSF
dirty=CubeDirty
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
nitermax = 500

args = {'mu_s':mu_s,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var}

EM= EasyMuffin(**args)
EM.loop(nitermax)

pl.figure()
pl.plot(EM.psnrlist)
 
#%% ===========================================================================
# Set parameters
# =============================================================================

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
mu_s = [0.1]*np.size(nb)
nb = (3,0)
mu_s = [1.6]*nb[0] # mu_s = [8.4, 1.6 ]
sigma=10
truesky=sky
psf=CubePSF
dirty=CubeDirty
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
nitermax = 500

args = {'mu_s':mu_s,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var}

EMm2= EasyMuffinm(**args)
EMm2.loop(nitermax)

pl.figure()
pl.plot(EMm2.psnrlist)


#%% ===========================================================================
# Set parameters
# =============================================================================

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb = (8,0)
mu_s = 0.4
sigma=10
truesky=sky
psf=CubePSF
dirty=CubeDirty
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
nitermax = 5000
step_mus =10

args = {'mu_s':mu_s,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,'step_mus':step_mus}

EMM_= EasyMuffinSURE(**args)
EMM_.loop_fdmc(nitermax)

pl.figure()
pl.plot(EMM_.mu_slist)

pl.figure()
pl.plot(EMM_.snrlist)

pl.figure()
pl.plot(EMM_.psnrlist)

pl.figure()
pl.plot(EMM_.sugarfdmclist)
pl.plot(np.zeros(np.shape(EMM_.sugarfdmclist)))

#%% ===========================================================================
# Set parameters
# =============================================================================

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
mu_s = [1]*np.size(nb)
nb = (2,0)
mu_s = [1]*nb[0]
#nb = (8,0)
#mu_s = [0.1]*nb[0]
sigma=10
truesky=sky
psf=CubePSF
dirty=CubeDirty
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
nitermax = 3000
step_mus =10

args = {'mu_s':mu_s,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,'step_mus':step_mus}

EMM= EasyMuffinSUREm(**args)
EMM.loop_fdmc(nitermax)

pl.figure()
for counter,b in enumerate(EMM.nbw_decomp):
    pl.plot(EMM.mu_slist[counter][:900:],label=counter)
pl.legend()

pl.figure()
pl.plot(EMM.snrlist[:900:])

pl.figure()
pl.plot(EMM.psnrlist[:900:])

pl.figure()
for counter,b in enumerate(EMM.nbw_decomp):
    pl.plot(EMM.sugarfdmclist[b][:940:],label=counter)
pl.legend()
pl.plot(np.zeros(np.shape(EMM.sugarfdmclist[b][:])))

pl.figure()
pl.plot(EMM.wmselistsure[:900:])
pl.plot(EMM.wmselist[:900:])


