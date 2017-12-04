#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:32:36 2017

@author: rammanouil
"""

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import os
import numpy as np
from astropy.io import fits
from deconv3d_tools import conv, myfft2, myifft2, abs2, myifftshift
import pylab as pl 
from deconv2d_Tik import TikhonovSURE, Tikhonov
import scipy.stats as st
import tictoc as tm

from matplotlib2tikz import save as tikz_save

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


#%% 

mu_s_ = np.logspace(-1,1,100)
mu_s_ = np.logspace(-1,1,100)
#mu_s_ = np.logspace(-3,-2,100) # Lena 
Noise = dirty - conv(psf,sky)
var = np.sum(Noise**2)/Noise.size

Risk = []
wmse = []
snr = []
N = np.size(sky)

for mu_s in mu_s_:
    psftik = 1/( np.abs(myfft2(psf))**2 + mu_s)
    x = myifft2(psftik*myfft2(conv(psf,dirty)))
    wmse.append((np.linalg.norm(conv(psf,x-sky))**2)/N)
    Risk.append((np.linalg.norm(x-sky)**2)/N)
    snr.append(10*np.log10(np.linalg.norm(x)**2/np.linalg.norm(x-sky)**2))

idxmusopt = np.argmin(wmse)
musopt = mu_s_[idxmusopt]
print(musopt)

pl.figure()
pl.semilogx(mu_s_,wmse)
pl.semilogx(musopt,wmse[idxmusopt],'*',label='optimal value: '+r'$\theta^\star$')
pl.legend()
pl.xlabel(r'$\theta$')
pl.ylabel('WMSE')

N = np.size(sky)
wmse_analytique = []
sigmai2 = (np.abs(myifftshift(myfft2(psf)))**2)
xbreve2 = N*np.abs(myifftshift(myifft2(sky)))**2
for mu_s in mu_s_:
    tmp = np.sum(sigmai2*(sigmai2*var + xbreve2*(mu_s**2))/((sigmai2+mu_s)**2))
    wmse_analytique.append(tmp/N)

pl.figure()
pl.semilogx(mu_s_,wmse_analytique)
pl.legend()
pl.xlabel(r'$\theta$')
pl.ylabel('WMSE')


#musopt_formula = np.sum(sigmai2**2)*var/np.sum(sigmai2**2*xbreve2)
#print(musopt_formula)

mu_s_ = np.logspace(-1,10,1000)
dwmse_analytique = []
for mu_s in mu_s_:
    tmp = np.sum(2*sigmai2**2*(xbreve2*mu_s- var)/((sigmai2+mu_s)**3))
    dwmse_analytique.append(tmp/N)

pl.figure()
pl.semilogx(mu_s_,dwmse_analytique)
pl.semilogx(mu_s_,np.zeros(np.shape(dwmse_analytique)))
pl.legend()
pl.xlabel(r'$\theta$')
pl.ylabel('WMSE')

tmp1 = np.diff(wmse)
tmp2 = np.diff(mu_s_)
dwmse = [tmp1[i]/tmp2[i] for i in range(np.size(tmp1))]

pl.figure()
pl.semilogx(mu_s_[1::],dwmse)
pl.semilogx(mu_s_[1::],np.zeros(np.shape(dwmse)))
pl.legend()
pl.xlabel(r'$\theta$')
pl.ylabel('WMSE')

pl.figure()
hist = np.histogram(sigmai2[:],10)
pl.hist(hist[0],hist[1])

tmp = np.sort(np.vectorize(sigmai2))
pl.figure()
pl.plot(tmp)

#mse = []
#mse_analytique = []
#beta = np.sqrt(N)*myifftshift(myifft2(dirty))
#beta2 = N*np.abs(myifftshift(myifft2(dirty)))**2
#sigma = myifftshift(myfft2(psf))
#sigma2 = np.abs(myifftshift(myfft2(psf)))**2
#eps = (myifftshift(myfft2(Noise)))
#for mu_s in mu_s_:
#    tmp = np.sum((beta*sigma/(sigma2+mu_s)-(beta-eps)/(sigma))**2)
#    mse_analytique.append(tmp)
#    psftik = 1/( np.abs(myfft2(psf))**2 + mu_s)
#    x = myifft2(psftik*myfft2(conv(psf,dirty)))
#    mse.append((np.linalg.norm(x-sky)**2))
#
#pl.figure()
#pl.semilogx(mu_s_,mse_analytique)
#pl.legend()
#pl.xlabel(r'$\theta$')
#pl.ylabel('MSE - analytique')
#pl.figure()
#pl.semilogx(mu_s_,mse)
#pl.legend()
#pl.xlabel(r'$\theta$')
#pl.ylabel('WMSE')





