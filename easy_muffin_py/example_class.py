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


def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x


folder = 'data'
file_in = 'm31_3d_conv_10db'

folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))
CubeDirty = checkdim(fits.getdata(drtname, ext=0))

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = np.transpose(sky)
sky2 = np.sum(sky*sky)

# ==============================================================================
#
# ==============================================================================

from SuperNiceSpectraDeconv import SNSD
from deconv3d import EasyMuffin

DM = SNSD(mu_s=.5, nb=8,nitermax=3,truesky=sky)

DM.parameters()

DM.setSpectralPSF(CubePSF)
DM.setSpectralDirty(CubeDirty)

(SpectralSkyModel , cost, snr) = DM.main()


EM= EasyMuffin(mu_s=.5, nb=8,truesky=sky,psf=CubePSF,dirty=CubeDirty)

(SpectralSkyModel2 , cost2, snr2) = EM.loop(nitermax=3)
