#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:30:26 2017

@author: rammanouil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:24:36 2017

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


folder = 'DataCyril'
file_in = 'test.cube'

folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'.psf.fits'
dirtyname = genname+'.dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[231:743,231:743,0:2]
CubeDirty = checkdim(fits.getdata(dirtyname, ext=0))[231:743,231:743,0:2]

sky = np.zeros(np.shape(CubePSF))
sky[np.int(np.shape(sky)[0]/2),np.int(np.shape(sky)[1]/2),:] = 10
sky2 = np.sum(sky*sky)

dirty2 = conv(CubePSF,sky)
pl.figure()
pl.imshow(dirty2[:,:,0])
pl.colorbar()

pl.figure()
pl.imshow(CubeDirty[:,:,0]-dirty2[:,:,0])
pl.colorbar()

print(np.linalg.norm(CubeDirty-conv(CubePSF,sky)))

pl.figure()
pl.imshow(CubePSF[:,:,0])
pl.colorbar()
pl.figure()
pl.imshow(CubeDirty[:,:,0])
pl.colorbar()
pl.figure()
pl.imshow(sky[:,:,0])
pl.colorbar()

from deconv3d import EasyMuffin, EasyMuffinSURE

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)
nitermax = 100

mu_s = 0.1
mu_l = 3.0

#%% ==============================================================================
#
# ==============================================================================

EM= EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,mu_wiener=0.08)
EM.loop(nitermax)
SpectralSkyModel2 = EM.xt
cost2 = EM.costlist
snr2 = EM.snrlist
psnr2 = EM.psnrlist
wmse2 = EM.wmselist

pl.figure()
pl.imshow(EM.x[:,:,0],cmap='nipy_spectral')
pl.colorbar()

pl.figure()
pl.imshow(EM.dirty[:,:,0],cmap='nipy_spectral')
pl.colorbar()


pl.figure()
pl.plot(snr2,label='snr2')
pl.legend(loc='best')

pl.figure()
pl.plot(cost2,label='cost2')
pl.legend(loc='best')

pl.figure()
pl.plot(psnr2,label='psnr2')
pl.legend(loc='best')

pl.figure()
pl.plot(wmse2,label='wmse2')
pl.legend(loc='best')

###

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EMsfdmc= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[1e0,1e0],mu_wiener=0.08)
EMsfdmc.loop_fdmc(nitermax)
SpectralSkyModel4 = EMsfdmc.xt
cost4 = EMsfdmc.costlist
snr4 = EMsfdmc.snrlist
psnr4 = EMsfdmc.psnrlist
psnrsure4 = EMsfdmc.psnrlistsure
wmse4 = EMsfdmc.wmselist
wmsesure4 = EMsfdmc.wmselistsure

pl.figure()
pl.plot(EMsfdmc.mu_slist)

pl.figure()
pl.plot(EMsfdmc.mu_llist)



