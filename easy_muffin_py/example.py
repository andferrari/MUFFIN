#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:08:49 2016

@author: antonyschutz
"""
#==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
#==============================================================================
import os
import numpy as np 
from astropy.io import fits

def checkdim(x):
    if len(x.shape)==4:
        x =np.squeeze(x)
    x = x.transpose((2,1,0))
    return x
    
    
folder='/Users/antonyschutz/Documents/easy_muffin_py/data/'
file_in = 'm31_3d_conv_10db'

folder = os.path.join(os.getcwd(),folder)
genname = os.path.join(folder,file_in) 
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))
CubeDirty = checkdim(fits.getdata(drtname, ext=0))

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0)) 
sky2 = np.sum(sky*sky)
#==============================================================================
# 
#==============================================================================

from SuperNiceSpectraDeconv import SNSD 

# %% test IUWT
DM = SNSD(mu_s=.5,nb=(8,0))

DM.parameters()

DM.setSpectralPSF(CubePSF)
DM.setSpectralDirty(CubeDirty)

SpectralSkyModel=DM.main()

# %% test DWT
DM = SNSD(mu_s=.5,nb=('db1','db2'))

DM.parameters()

DM.setSpectralPSF(CubePSF)
DM.setSpectralDirty(CubeDirty)

SpectralSkyModel=DM.main()

##==============================================================================
## Check results 
##==============================================================================
#
#resid = sky-SpectralSkyModel
#print(10*np.log10( sky2 / np.sum(resid*resid)  ))