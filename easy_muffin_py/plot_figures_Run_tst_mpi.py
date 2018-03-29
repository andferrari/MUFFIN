#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:45:05 2018

@author: rammanouil
"""

import os
import numpy as np
import matplotlib.pyplot as pl 
from astropy.io import fits
from deconv3d_tools import fix_dim

#%% Path to folder 
day_time = '1881888'
label = '256 bands'

folder = 'data_david'
file_in = 'M31_skyline2_20db'
folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
skyname = genname+'_sky.fits'
sky = fix_dim(fits.getdata(skyname, ext=0))
dirtyname = genname+'_dirty.fits'
CubeDirty = fix_dim(fits.getdata(dirtyname, ext=0))

drctry = os.path.join(os.getcwd(),'output/'+daytime)
os.chdir(drctry)

x0_tst =np.load('x0_tst.npy')
wmse_tst=np.load('wmse_tst.npy')
wmses_tst=np.load('wmses_tst.npy')
wmsesfdmc_tst=np.load('wmsesfdmc_tst.npy')
snr_tst=np.load('snr_tst.npy')
mu_s_tst=np.load('mu_s_tst.npy')
mu_l_tst=np.load('mu_l_tst.npy')
dxs=np.load('dxs.npy')
dxl=np.load('dxl.npy')
sugar0=np.load('sugar0.npy')
sugar1=np.load('sugar1.npy')
cost=np.load('cost.npy')
psnr = np.load('psnrsure.npy')

os.chdir('../..')

pl.figure(1)

pl.subplot(2,4,1)
pl.plot(x0_tst[5,12,:],label=label)
pl.plot(sky[5,12,:])
pl.title('Spectre 1')

pl.subplot(2,4,2)
pl.plot(x0_tst[12,12,:],label=label)
pl.plot(sky[12,12,:])
pl.title('Spectre 2')

pl.subplot(2,4,3)
pl.plot(x0_tst[20,20,:],label=label)
pl.plot(sky[20,20,:])
pl.title('Spectre 3')

N = snr_tst.size 
pl.subplot(2,4,4)
pl.plot(snr_tst[:N],label=label)
pl.legend()
pl.title('SNR')

pl.subplot(2,4,5)
pl.plot(psnr[:N],label=label)
pl.legend()
pl.title('PSNR')

pl.subplot(2,4,6)
pl.plot(mu_s_tst[:N],label=label)
pl.legend()
pl.title('mu_s')

pl.subplot(2,4,7)
pl.plot(mu_l_tst[:N],label=label)
pl.legend()
pl.title('mu_l')

pl.subplot(2,4,8)
pl.plot(wmse_tst,label='wmse')
pl.plot(wmses_tst,'-*',label='wmses')
pl.plot(wmsesfdmc_tst,'-^',label='wmses_fdmc')
pl.legend()
pl.title('wmse')

#%%

pl.figure()
pl.imshow(x0_tst[:,:,100])
pl.colorbar()

pl.figure()
pl.imshow(sky[:,:,100])
pl.colorbar()

pl.figure(1)
for i in range(60):
    pl.clf()
    pl.plot(x0_tst[i,1,:],label='Spectre_{:01d}'.format(i))
    pl.plot(sky[i,1,:])
    pl.legend()
    pl.ylim((x0_tst.min(),x0_tst.max()))
    #pl.savefig('temp_{:03d}.png'.format(i))

N = snr_tst.size 
pl.figure()
pl.plot(snr_tst[:N],label='snr_tst')
pl.legend()

pl.figure()
pl.plot(sugar0[1:N],label='sugar0')
pl.plot(0*sugar0[1:N])
pl.legend()

pl.figure()
pl.plot(sugar1[1:N],label='sugar1')
pl.plot(0*sugar1[1:N])
pl.legend()

pl.figure()
pl.plot(cost[:N],label='cost')
pl.legend()

pl.figure()
pl.plot(psnr)
pl.title('psnr')


