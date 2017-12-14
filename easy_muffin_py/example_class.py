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

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:2]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:2]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = sky[:,:,0:2]
sky2 = np.sum(sky*sky)

fig = pl.figure()
ax = fig.add_subplot(1,3,1)
ax.imshow(CubePSF[:,:,1])
ax = fig.add_subplot(1,3,2)
ax.imshow(sky[:,:,1])
ax = fig.add_subplot(1,3,3)
ax.imshow(CubeDirty[:,:,1])

from SuperNiceSpectraDeconv import SNSD
from deconv3d import EasyMuffin, EasyMuffinSURE

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)
nitermax = 5000

mu_s = 1
mu_l = 1

#%% ==============================================================================
#
# ==============================================================================

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
EMs= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,mu_wiener=5e1)
EMs.loop(nitermax)
SpectralSkyModel3 = EMs.xt
cost3 = EMs.costlist
snr3 = EMs.snrlist
psnr3 = EMs.psnrlist
psnrsure3 = EMs.psnrlistsure
wmse3 = EMs.wmselist
wmsesure3 = EMs.wmselistsure

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EMsfdmc= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[5e-1,5e-1])
EMsfdmc.loop_fdmc(nitermax)
SpectralSkyModel4 = EMsfdmc.xt
cost4 = EMsfdmc.costlist
snr4 = EMsfdmc.snrlist
psnr4 = EMsfdmc.psnrlist
psnrsure4 = EMsfdmc.psnrlistsure
wmse4 = EMsfdmc.wmselist
wmsesurefdmc = EMsfdmc.wmselistsurefdmc

pl.figure()
pl.plot(EMsfdmc.mu_llist)
pl.figure()
pl.plot(EMsfdmc.mu_slist)

pl.figure()
pl.plot(snr,label='snr1')
pl.plot(snr2,label='snr2')
pl.plot(snr3,'*',label='snr3')
pl.plot(snr4,'*',label='snr4')
pl.legend(loc='best')

pl.figure()
pl.plot(cost/(EM.nxy*EM.nxy*EM.nfreq),label='cost1')
pl.plot(cost2,label='cost2')
pl.plot(cost3,'*',label='cost3')
pl.plot(cost4,'*',label='cost4')
pl.legend(loc='best')

pl.figure()
pl.plot(psnr,label='psnr1')
pl.plot(psnr2,label='psnr2')
pl.plot(psnr3,'*',label='psnr3')
pl.plot(psnrsure3,'*',label='psnrsure3')
pl.plot(psnrsure4,'*',label='psnrsure4')
pl.legend(loc='best')

pl.figure()
pl.plot(wmse2,label='wmse2')
pl.plot(wmse3,label='wmse3')
pl.plot(wmsesure3,'*',label='wmsesure3')
pl.plot(wmsesurefdmc,'*',label='wmsesurefdmc')
pl.legend(loc='best')

nitermax = 4000
EMsfdmc1= EasyMuffinSURE(mu_s=0.5, mu_l = 2, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[5e-1,5e-1])
EMsfdmc1.loop_fdmc(nitermax)

EMsfdmc1= EasyMuffinSURE(mu_s=0.1, mu_l = 3, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[5e-1,5e-1])
EMsfdmc1.loop_fdmc(nitermax)

EMsfdmc2= EasyMuffinSURE(mu_s=0.5, mu_l = 0.5, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[5e-1,5e-1])
EMsfdmc2.loop_fdmc(nitermax)

EMsfdmc3= EasyMuffinSURE(mu_s=1, mu_l = 1, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[5e-1,5e-1])
EMsfdmc3.loop_fdmc(nitermax)

EMsfdmc4= EasyMuffinSURE(mu_s=1.5, mu_l = 1.5, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[5e-1,5e-1])
EMsfdmc4.loop_fdmc(nitermax)

EMsfdmc4= EasyMuffinSURE(mu_s=2, mu_l = 2, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[5e-1,5e-1])
EMsfdmc4.loop_fdmc(nitermax)

EMsfdmc6= EasyMuffinSURE(mu_s=0.5, mu_l = 0.5, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mu=[5e-1,5e-1])
EMsfdmc6.alpha_s = np.ones(EMsfdmc6.nfreq)
EMsfdmc6.alpha_l = np.ones((EMsfdmc6.nxy,EMsfdmc6.nxy))
EMsfdmc6.loop_fdmc(nitermax)

pl.figure()
pl.plot(EMsfdmc1.mu_llist)
pl.plot(EMsfdmc1.mu_slist)

pl.figure()
pl.plot(EMsfdmc2.mu_llist)
pl.plot(EMsfdmc2.mu_slist)

pl.figure()
pl.plot(EMsfdmc3.mu_llist)
pl.plot(EMsfdmc3.mu_slist)

pl.figure()
pl.plot(EMsfdmc4.mu_llist)
pl.plot(EMsfdmc4.mu_slist)

pl.figure()
pl.plot(EMsfdmc1.mu_llist)
pl.plot(EMsfdmc1.mu_slist)
pl.plot(EMsfdmc2.mu_llist,'--')
pl.plot(EMsfdmc2.mu_slist,'--')
pl.plot(EMsfdmc3.mu_llist,'--')
pl.plot(EMsfdmc3.mu_slist,'--')
pl.plot(EMsfdmc4.mu_llist,'-.')
pl.plot(EMsfdmc4.mu_slist,'-.')

pl.figure()
pl.plot(EMsfdmc1.mu_llist)
pl.plot(EMsfdmc2.mu_llist,'--')
pl.plot(EMsfdmc3.mu_llist,'--')
pl.plot(EMsfdmc4.mu_llist,'-.')

pl.figure()
pl.plot(EMsfdmc1.mu_slist)
pl.plot(EMsfdmc2.mu_slist,'--')
pl.plot(EMsfdmc3.mu_slist,'--')
pl.plot(EMsfdmc4.mu_slist,'-.')


pl.figure()
pl.plot(EMsfdmc1.snrlist)
pl.plot(EMsfdmc2.snrlist,'-*')
pl.plot(EMsfdmc3.snrlist,'--')
pl.plot(EMsfdmc4.snrlist,'-.')

pl.figure()
pl.plot(EMsfdmc1.wmselistsurefdmc)
pl.plot(EMsfdmc2.wmselistsurefdmc,'-*')
pl.plot(EMsfdmc3.wmselistsurefdmc,'--')
pl.plot(EMsfdmc4.wmselistsurefdmc,'-.')


pl.figure()
pl.plot(EMsfdmc6.mu_llist)
pl.plot(EMsfdmc6.mu_slist)
pl.figure()
pl.plot(EMsfdmc6.snrlist)

pl.figure()
pl.plot(EMsfdmc2.wmselist)
pl.plot(EMsfdmc2.wmselistsure)
pl.plot(EMsfdmc2.wmselistsurefdmc)



