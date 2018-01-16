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
from deconv3d_tools import conv
import pylab as pl 

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

CubePSF = np.squeeze(checkdim(fits.getdata(psfname, ext=0))[:,:,0:1])
CubeDirty = np.squeeze(checkdim(fits.getdata(drtname, ext=0))[:,:,0:1])

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = np.squeeze(sky[:,:,0:1])
sky2 = np.sum(sky*sky)


#%% ==============================================================================
#
# ==============================================================================

from deconv2d import EasyMuffinSURE

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)
nitermax = 3000

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

mu_s = 0.5
EM= EasyMuffinSURE(mu_s=mu_s, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mus=1.5)
EM.loop_fdmc(nitermax)

mu_s = 0.1
EM2= EasyMuffinSURE(mu_s=mu_s, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mus=1.5)
EM2.loop_fdmc(nitermax)

mu_s = 1
EM3= EasyMuffinSURE(mu_s=mu_s, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var,step_mus=1.5)
EM3.loop_fdmc(nitermax)


pl.figure()
pl.plot(EM2.wmselist,label='wmse')
pl.plot(EM2.wmselistsure,'+',label='wmse sure')
pl.plot(EM2.wmselistsurefdmc,'*',label='wmse sure_fdmc')
pl.legend()

pl.figure()
pl.plot(EM2.sugarfdmclist,'-*',label='sugarfdmc')
pl.plot(np.zeros(np.size(EM2.sugarfdmclist)),label='0')
pl.legend()

pl.figure()
pl.plot(EM2.mu_slist,label='2')
pl.plot(EM3.mu_slist,label='3')
pl.plot(EM.mu_slist,label='1')
pl.legend()

pl.figure()
pl.plot(EM2.costlist)

pl.figure()
pl.plot(EM2.snrlist,label='2')
pl.plot(EM3.snrlist,label='3')
pl.legend()

#%% 

from deconv2d import EasyMuffinSURE

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)
nitermax = 200

mu_s_ = np.linspace(0.1,1,10)
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size

Risk = []
wmse = []
wmsesure = []
wmsesurefdmc = []
sugarfdmc = []

for mu_s in mu_s_:
    EM= EasyMuffinSURE(mu_s=mu_s, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
    EM.loop_fdmc(nitermax)
    Risk.append((np.linalg.norm(EM.x-sky)**2)/(EM.nxy**2))
    wmse.append(EM.wmselist[-1])
    wmsesure.append(EM.wmselistsure[-1])
    wmsesurefdmc.append(EM.wmselistsurefdmc[-1])
    sugarfdmc.append(EM.sugarfdmclist[-1])

np.save('Risk.npy',Risk)
np.save('wmse.npy',wmse)
np.save('wmsesure.npy',wmsesure)
np.save('wmsesurefdmc.npy',wmsesurefdmc)
np.save('sugarfdmc.npy',sugarfdmc)

pl.figure()
pl.semilogx(mu_s_,wmse,label='wmse')
pl.semilogx(mu_s_,wmsesurefdmc,'*-',label='wmse sure_fdmc')
pl.semilogx(mu_s_,wmsesure,'o-',label='wmse sure')
pl.legend()

pl.figure()
pl.semilogx(mu_s_,Risk,'-*',label='Risk')
pl.legend()

idxmusopt = np.argmin(Risk)
musopt = mu_s_[idxmusopt]
print(musopt)


pl.figure()
pl.semilogx(mu_s_,sugarfdmc,label='sugarfdmc')
pl.legend()

dRisk = np.diff(Risk)/np.diff(mu_s_)

pl.figure()
pl.semilogx(mu_s_[:-1:],dRisk,label='dRisk')
pl.legend()

dwmsesurefdmc = np.diff(wmsesurefdmc)/np.diff(mu_s_)
dwmse = np.diff(wmse)/np.diff(mu_s_)
pl.figure()
pl.semilogx(mu_s_[:-1:],dwmse,'-^',label='dwmse')
pl.semilogx(mu_s_,sugarfdmc,'-+',label='sugarfdmc')
pl.semilogx(mu_s_[:-1:],0*mu_s_[:-1:])
pl.legend()

pl.figure()
pl.plot(EM.sugarfdmclist,label='sugarfdmclist')
pl.plot(np.zeros(np.size(EM.sugarfdmclist)))
pl.legend()



#%% Plusieurs realisations du bruit 

from deconv3d_tools import conv 

from deconv2d import EasyMuffinSURE

nrep_ = 10
Risk_ = []
wmse_ = []
wmsesure_ = []
wmsesurefdmc_ = []
sugarfdmc_ = []

for nrep in range(nrep_):
    
    # Create dirty image 
    CubeDirty = conv(CubePSF,sky)
    CubeDirty = CubeDirty.real
    # Add noise 
    snr = 30
    var = (np.sum(CubeDirty**2)/CubeDirty.size)/(10**(snr/10))
    noise = np.random.normal(0,var**.5,np.shape(CubeDirty))
    CubeDirty = CubeDirty + noise 
    
    nb=('db1','db2','db3','db4','db5','db6','db7','db8')
    #nb = (7,0)
    nitermax = 200

    mu_s_ = 10**np.linspace(-1.5,0,30)
    Noise = CubeDirty - conv(CubePSF,sky)
    var = np.sum(Noise**2)/Noise.size

    Risk = []
    wmse = []
    wmsesure = []
    wmsesurefdmc = []
    sugarfdmc = []

    for mu_s in mu_s_:
        EM= EasyMuffinSURE(mu_s=mu_s, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
        EM.loop_fdmc(nitermax)
        Risk.append((np.linalg.norm(EM.x-sky)**2)/(EM.nxy**2))
        wmse.append(EM.wmselist[-1])
        wmsesure.append(EM.wmselistsure[-1])
        wmsesurefdmc.append(EM.wmselistsurefdmc[-1])
        sugarfdmc.append(EM.sugarfdmclist[-1])
        
    Risk_.append(Risk)
    wmse_.append(wmse)
    wmsesure_.append(wmsesure)
    wmsesurefdmc_.append(wmsesurefdmc)
    sugarfdmc_.append(sugarfdmc)
    
Risk_mean = [sum(i)/nrep for i in zip(*Risk_)]
wmse_mean = [sum(i)/nrep for i in zip(*wmse_)]
wmsesure_mean = [sum(i)/nrep for i in zip(*wmsesure_)]
wmsesurefdmc_mean = [sum(i)/nrep for i in zip(*wmsesurefdmc_)]
sugarfdmc_mean = [sum(i)/nrep for i in zip(*sugarfdmc_)]


np.save('Risk_.npy',Risk_)
np.save('wmse_.npy',wmse_)
np.save('wmsesure_.npy',wmsesure_)
np.save('wmsesurefdmc_.npy',wmsesurefdmc_)
np.save('sugarfdmc_.npy',sugarfdmc_)

np.save('Risk._meannpy',Risk_mean)
np.save('wmse_mean.npy',wmse_mean)
np.save('wmsesure_mean.npy',wmsesure_mean)
np.save('wmsesurefdmc_mean.npy',wmsesurefdmc_mean)
np.save('sugarfdmc_mean.npy',sugarfdmc_mean)


pl.figure()
pl.semilogx(mu_s_,wmse_mean,label='wmse')
pl.semilogx(mu_s_,wmsesurefdmc_mean,'*',label='wmse sure_fdmc')
pl.semilogx(mu_s_,wmsesure_mean,'*',label='wmse sure')
pl.legend()


pl.figure()
pl.semilogx(mu_s_,Risk_mean,label='Risk')
pl.legend()

idxmusopt = np.argmin(Risk)
musopt = mu_s_[idxmusopt]
print(musopt)


dwmsesurefdmc = np.diff(wmsesurefdmc_mean)/np.diff(mu_s_)
dwmse = np.diff(wmse_mean)/np.diff(mu_s_)
pl.figure()
pl.semilogx(mu_s_[:-1:],dwmse,'-^',label='dwmse')
pl.semilogx(mu_s_[:-1:],dwmsesurefdmc,'-^',label='dwmsesurefdmc')
pl.semilogx(mu_s_,sugarfdmc_mean,'-+',label='sugarfdmc')
pl.semilogx(mu_s_[:-1:],0*mu_s_[:-1:])
pl.legend()

pl.figure()
pl.plot(EM.sugarfdmclist,label='sugarfdmclist')
pl.plot(np.zeros(np.size(EM.sugarfdmclist)))
pl.legend()





