#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 15:47:05 2017

@author: rammanouil
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
file_in = 'M31_3d_conv_256_50db'

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


#%% ==============================================================================
#
# ==============================================================================

from deconv3d import EasyMuffin, EasyMuffinSUREDescent

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)
nitermax = 1000

mu_s = 0.1
mu_l = 0

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

#EM= EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty)
#EM.loop(nitermax)

EM= EasyMuffinSUREDescent(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
EM.loop(nitermax)


