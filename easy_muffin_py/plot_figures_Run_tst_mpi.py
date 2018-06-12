#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:45:05 2018

@author: rammanouil
"""

import os
import numpy as np
import matplotlib.pyplot as pl 
import argparse

# =============================================================================
# Terminal Input
# =============================================================================
parser = argparse.ArgumentParser(description='Awesome Argument Parser')
parser.add_argument('-res_fol','--res_folder',help='Path to results folder')

args = parser.parse_args()

res_fol = args.res_folder # 

#%% Path to folder 

drctry = res_fol
os.chdir(drctry)

x0_tst =np.load('x0_tst.npy')
wmse_tst=np.load('wmse_tst.npy')
snr_tst=np.load('snr_tst.npy')
cost=np.load('cost.npy')

wmses_tst=np.load('wmses_tst.npy')
wmsesfdmc_tst=np.load('wmsesfdmc_tst.npy')
mu_s_tst=np.load('mu_s_tst.npy')
mu_l_tst=np.load('mu_l_tst.npy')
dxs=np.load('dxs.npy')
dxl=np.load('dxl.npy')
sugar0=np.load('sugar0.npy')
sugar1=np.load('sugar1.npy')
psnr = np.load('psnrsure.npy')

os.chdir('../..')

N = snr_tst.size 
#%%

pl.figure(1)

pl.subplot(2,4,1)
pl.plot(sugar0[1:N],label='sugar0')
pl.plot(0*sugar0[1:N])
pl.legend()

pl.subplot(2,4,2)
pl.plot(sugar1[1:N],label='sugar1')
pl.plot(0*sugar1[1:N])
pl.legend()

pl.subplot(2,4,3)
pl.plot(cost[:N],label='cost')
pl.legend()

pl.subplot(2,4,4)
pl.plot(snr_tst[:N],label='snr')
pl.legend()
pl.title('SNR')

pl.subplot(2,4,5)
pl.plot(psnr[:N],label='psnr')
pl.legend()
pl.title('PSNR')

pl.subplot(2,4,6)
pl.plot(mu_s_tst[:N],label='mu_s')
pl.legend()
pl.title('mu_s')

pl.subplot(2,4,7)
pl.plot(mu_l_tst[:N],label='mu_l')
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
pl.imshow(x0_tst[:,:,100],cmap='nipy_spectral')
pl.colorbar()

pl.figure()
pl.imshow(sky[:,:,100],cmap='nipy_spectral')
pl.colorbar()

i=186
j=101
pl.figure()
pl.plot(sky[i,j,:])
pl.plot(x0_tst[i,j,:])

i=121
j=152
pl.figure()
pl.plot(sky[i,j,:])
pl.plot(x0_tst[i,j,:])

i=140
j=112
pl.figure()
pl.plot(sky[i,j,:])
pl.plot(x0_tst[i,j,:])

pl.figure(1)
for i in range(60):
    pl.clf()
    pl.plot(x0_tst[i,1,:],label='Spectre_{:01d}'.format(i))
    pl.plot(sky[i,1,:])
    pl.legend()
    pl.ylim((x0_tst.min(),x0_tst.max()))
    #pl.savefig('temp_{:03d}.png'.format(i))

pl.show()










