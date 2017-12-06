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

CubePSF = checkdim(fits.getdata(psfname, ext=0))[231:743,231:743,0:2]

sky = np.zeros(np.shape(CubePSF))
sky[np.int(np.shape(sky)[0]/2),np.int(np.shape(sky)[1]/2),:] = 10
sky2 = np.sum(sky*sky)

CubeDirty = conv(CubePSF,sky)

pl.figure()
pl.imshow(CubePSF[:,:,0],cmap='nipy_spectral')
pl.title('psf')
pl.colorbar()
pl.figure()
pl.imshow(CubeDirty[:,:,0],cmap='nipy_spectral')
pl.title('dirty')
pl.colorbar()
pl.figure()
pl.imshow(sky[:,:,0],cmap='nipy_spectral')
pl.title('sky')
pl.colorbar()

# Add noise 
snr = 30
var = (np.sum(CubeDirty**2)/CubeDirty.size)/(10**(snr/10))
noise = np.random.normal(0,var**.5,np.shape(CubeDirty))
CubeDirty = CubeDirty + noise 
Pb = np.sum(noise*noise)
Ps = np.sum(CubeDirty*CubeDirty)
snrr = 10*np.log10(Ps/Pb)

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

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EMs= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,mu_wiener=0.08)
EMs.loop(nitermax)
SpectralSkyModel3 = EMs.xt
cost3 = EMs.costlist
snr3 = EMs.snrlist
psnr3 = EMs.psnrlist
psnrsure3 = EMs.psnrlistsure
wmse3 = EMs.wmselist
wmsesure3 = EMs.wmselistsure

pl.figure()
pl.imshow(EM.x[:,:,0],cmap='nipy_spectral')
pl.colorbar()

pl.figure()
pl.imshow(EM.dirty[:,:,0],cmap='nipy_spectral')
pl.colorbar()

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

#sub1 = EMs.wmselistsuresub1
#sub2 = EMs.wmselistsuresub2
#sub3 = EMs.wmselistsuresub3
#somme = [sub1[i]+sub2[i]+sub3[i] for i,elt in enumerate(sub1)]
#pl.figure()
#pl.plot(wmsesure3,label='wmse (sure)')
#pl.plot(sub1,label='LS')
#pl.plot(sub2,label='Jac.')
#pl.plot(sub3,label='var')
#pl.plot(somme,'*',label='tst')
#pl.legend(loc='best')

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


pl.figure()
pl.plot(EMsfdmc.sugarfdmclist[0])

pl.figure()
pl.plot(EMsfdmc.sugarfdmclist[1])

mu_s = 0.1
mu_l = 0
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EMsfdmc2= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[1e0,1e0])
EMsfdmc2.loop_fdmc(nitermax)

pl.figure()
pl.plot(EMsfdmc2.mu_slist)

pl.figure()
pl.plot(EMsfdmc2.mu_llist)


pl.figure()
pl.plot(EMsfdmc2.sugarfdmclist[0])

pl.figure()
pl.plot(EMsfdmc2.sugarfdmclist[1])

mu_s = 0.1
mu_l = 0.000001
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EMsfdmc3= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[1e0,1e0])
EMsfdmc3.loop_fdmc(nitermax)

pl.figure()
pl.plot(EMsfdmc3.mu_slist)

pl.figure()
pl.plot(EMsfdmc3.mu_llist)

pl.figure()
pl.plot(EMsfdmc3.sugarfdmclist[0])

pl.figure()
pl.plot(EMsfdmc3.sugarfdmclist[1])


