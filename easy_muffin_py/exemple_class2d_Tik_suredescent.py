#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:19:02 2017

@author: rammanouil
"""

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import os
import numpy as np
from astropy.io import fits
from deconv3d_tools import conv, myfft2, myifft2, abs2
import pylab as pl 
from deconv2d_Tik import TikhonovSURE, Tikhonov
import scipy.stats as st

def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x

folder = 'data256Eusipco'
file_in = 'M31_3d_conv_256_20db'

folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

psf = np.squeeze(checkdim(fits.getdata(psfname, ext=0))[:,:,0:1])
dirty = np.squeeze(checkdim(fits.getdata(drtname, ext=0))[:,:,0:1])

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = np.squeeze(sky[:,:,0:1])
sky2 = np.sum(sky*sky)

#%% Lena 

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

n = 512
sky = pl.imread('lena.png')
psf = gkern(n,50) 
dirty_ = conv(psf,sky)
snr = 20
var = (np.sum(dirty_**2)/dirty_.size)/(10**(snr/10))
noise = np.random.normal(0,var**.5,np.shape(dirty_))
dirty = dirty_ + noise 
Pb = np.sum(noise*noise)
Ps = np.sum(dirty*dirty)
snrr = 10*np.log10(Ps/Pb)
print(snrr)

pl.figure()
pl.imshow(psf)
pl.colorbar()

pl.figure()
pl.imshow(dirty)
pl.colorbar()

pl.figure()
pl.imshow(dirty_)
pl.colorbar()

pl.figure()
pl.imshow(sky)
pl.colorbar()


#%% ==============================================================================
#
# ==============================================================================

nitermax = 1000
mu_s = 2
eps_mu_s = 0.1
eps_x = 1e-3
beta = np.max(abs2(myfft2(psf)))
eps_x = 0.9/(beta/2 + mu_s**2)
    
Noise = dirty - conv(psf,sky)
var = np.sum(Noise**2)/Noise.size

EM= TikhonovSURE(mu_s=mu_s,truesky=sky,psf=psf,dirty=dirty,var=var,eps_mu_s=eps_mu_s, eps_x = eps_x)
EM.loop_mu_s(nitermax)

mu_s = 10
eps_mu_s = 0.01
eps_x = 1e-3
beta = np.max(abs2(myfft2(psf)))
eps_x = 0.9/(beta/2 + mu_s**2)

EM2= TikhonovSURE(mu_s=mu_s,truesky=sky,psf=psf,dirty=dirty,var=var,eps_mu_s=eps_mu_s, eps_x = eps_x)
EM2.loop_mu_s(nitermax)

pl.figure()
pl.imshow(EM.dirty)
pl.colorbar()
pl.figure()
pl.imshow(EM.x)
pl.colorbar()
pl.figure()
pl.imshow(EM.truesky)
pl.colorbar()


pl.figure()
pl.plot(EM.wmselist,label='wmse')
pl.plot(EM.wmselistsure,'+',label='wmse sure')
pl.legend()


pl.figure()
pl.plot(EM.mu_slist)

pl.figure()
pl.plot(EM.costlist)

pl.figure()
pl.plot(EM.snrlist)

pl.figure()
pl.plot(EM2.mu_slist)

pl.figure()
pl.plot(EM2.costlist)

pl.figure()
pl.plot(EM2.snrlist)


#%% 

mu_s_ = np.logspace(-1.2,1.5,100)
Noise = dirty - conv(psf,sky)
var = np.sum(Noise**2)/Noise.size

Risk = []
wmse = []
snr = []

for mu_s in mu_s_:
    psftik = 1/( np.abs(myfft2(psf))**2 + mu_s)
    x = myifft2(psftik*myfft2(conv(psf,dirty)))
    wmse.append((np.linalg.norm(conv(psf,x-sky))**2)/(512*512))
    Risk.append((np.linalg.norm(x-sky)**2)/(512*512))
    snr.append(10*np.log10(np.linalg.norm(x)**2/np.linalg.norm(x-sky)**2))

#np.save('Risk.npy',Risk)
#np.save('wmse.npy',wmse)
#np.save('wmsesure.npy',wmsesure)

idxmusopt = np.argmin(wmse)
musopt = mu_s_[idxmusopt]
print(musopt)

pl.figure()
pl.semilogx(mu_s_,wmse,label='wmse')
pl.semilogx(musopt,wmse[idxmusopt],'*')
#pl.plot(mu_s_,wmsesure,'o-',label='wmse sure')
pl.legend()

pl.figure()
pl.semilogx(mu_s_,snr,label='wmse')
#pl.plot(mu_s_,wmsesure,'o-',label='wmse sure')
pl.legend()

pl.figure()
pl.semilogx(mu_s_,Risk,'-*',label='Risk')
pl.legend()

idxmusopt = np.argmin(Risk)
musopt = mu_s_[idxmusopt]
print(musopt)

idxmusopt = np.argmax(snr)
musopt = mu_s_[idxmusopt]
print(musopt)

idxmusopt = np.argmin(wmse)
musopt = mu_s_[idxmusopt]
print(musopt)

pl.figure()
pl.imshow(x.real)
pl.colorbar()

#%% verification de J_R

nitermax = 500
mu_s_ = np.logspace(-1.2,1,50)
mu_s_ = np.linspace(0,5,50)
Noise = dirty - conv(psf,sky)
var = np.sum(Noise**2)/Noise.size
eps_mu_s = 0
beta = np.max(abs2(myfft2(psf)))


wmse = []
wmse2 = []
wmse3 = []
J_R = []


for mu_s in mu_s_:
    psftik = 1/( np.abs(myfft2(psf))**2 + mu_s)
    x = myifft2(psftik*myfft2(conv(psf,dirty)))
    wmse.append((np.linalg.norm(conv(psf,x-sky))**2)/(512*512))
    eps_x = 0.9/(beta/2 + mu_s**2)
    EM= TikhonovSURE(mu_s=mu_s,truesky=sky,psf=psf,dirty=dirty,var=var,eps_mu_s=eps_mu_s, eps_x = eps_x)
    EM.loop_mu_s(nitermax)
    wmse2.append(EM.wmselist[-1])
    wmse3.append(EM.wmselistsure[-1])
    J_R.append(EM.J_R)

Jwmse = np.diff(wmse)/np.diff(mu_s_)

idxmusopt = np.argmin(wmse)
musopt = mu_s_[idxmusopt]
print(musopt)

pl.figure()
pl.plot(mu_s_,wmse,label='wmse')
pl.plot(musopt,wmse[idxmusopt],'*')
pl.legend()

pl.figure()
pl.plot(mu_s_[0:-1:],Jwmse,label='Jwmse')
pl.legend()
pl.plot(mu_s_[0:-1:],np.zeros(np.size(Jwmse)))
pl.plot(musopt,Jwmse[idxmusopt],'*')


Jwmse2 = np.diff(wmse2)/np.diff(mu_s_)
Jwmse3 = np.diff(wmse3)/np.diff(mu_s_)

idxmusopt = np.argmin(wmse2)
musopt = mu_s_[idxmusopt]
print(musopt)

pl.figure()
pl.plot(mu_s_,wmse2,label='wmse2')
pl.plot(musopt,wmse2[idxmusopt],'*')
pl.legend()

pl.figure()
pl.plot(mu_s_[0:-1:],Jwmse2,label='Jwmse2')
pl.legend()
pl.plot(mu_s_[0:-1:],np.zeros(np.size(Jwmse2)))
pl.plot(musopt,Jwmse2[idxmusopt],'*')


idxmusopt = np.argmin(wmse3)
musopt = mu_s_[idxmusopt]
print(musopt)

pl.figure()
pl.plot(mu_s_,wmse2,label='wmse2')
pl.plot(mu_s_,wmse3,label='wmse3')
pl.plot(musopt,wmse3[idxmusopt],'*')
pl.legend()

pl.figure()
pl.plot(mu_s_[1::],Jwmse3,label='Jwmse3')
pl.legend()
pl.plot(mu_s_[1::],np.zeros(np.size(Jwmse3)))
pl.plot(musopt,Jwmse3[idxmusopt],'*')


pl.figure()
pl.plot(mu_s_,J_R,label='J_R')
pl.plot(mu_s_,np.zeros(np.size(J_R)))
pl.plot(musopt,J_R[idxmusopt],'*')
pl.legend()

tmp  = [i/np.size(sky) for i in J_R]

pl.figure()
pl.plot(mu_s_,tmp,label='J_R')
pl.plot(mu_s_[0:-1:],Jwmse2,'.-',label='Jwmse2')
pl.plot(mu_s_,np.zeros(np.size(J_R)))
pl.plot(musopt,tmp[idxmusopt],'*')
pl.legend()


