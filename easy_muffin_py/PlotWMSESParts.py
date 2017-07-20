#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:50:30 2017

@author: rammanouil
"""

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

from deconv3d import EasyMuffinSURE

#nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb = (7,0)
nitermax = 100

mu_s = 0.5
mu_l = 2

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
sub1 = EMs.wmselistsuresub1
sub2 = EMs.wmselistsuresub2
sub3 = EMs.wmselistsuresub3
somme = [sub1[i]+sub2[i]+sub3[i] for i,elt in enumerate(sub1)]

pl.figure()
pl.plot(wmsesure3,label='wmse (sure)')
pl.plot(sub1,label='LS')
pl.plot(sub2,label='Jac.')
pl.plot(sub3,label='var')
pl.plot(somme,'*',label='tst')
pl.legend(loc='best')


