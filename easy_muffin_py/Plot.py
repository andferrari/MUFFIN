#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:29:59 2017

@author: rammanouil
"""

import pylab as pl 
import numpy as np 
import os

folder = 'ResClaude'
folder = os.path.join(os.getcwd(), folder)

mu_l_name = os.path.join(folder,'mu_l.npy')
mu_s_name = os.path.join(folder,'mu_s.npy')
snr_name = os.path.join(folder,'snr.npy')
wmse_name = os.path.join(folder,'wmse.npy')
wmses_name = os.path.join(folder,'wmses.npy')
x0_name = os.path.join(folder,'x0.npy')

mu_l = np.load(mu_l_name) 
mu_s = np.load(mu_s_name)
snr = np.load(snr_name)
wmse = np.load(wmse_name)
wmses = np.load(wmses_name)
x0 = np.load(x0_name)

pl.figure()
pl.plot(mu_s,label=r'$\mu_s$')
pl.plot(mu_l,label=r'$\mu_l$')
pl.legend(loc='best')
pl.xlabel('It.')

pl.figure()
pl.plot(mu_l,label=r'$\mu_l$')
pl.legend(loc='best')
pl.xlabel('It.')

pl.figure()
pl.plot(mu_s[1::],label=r'$\mu_s$')
pl.legend(loc='best')
pl.xlabel('It.')

pl.figure()
pl.plot(snr)
pl.xlabel('It.')
pl.ylabel('SNR')
pl.title('SNR variation w.r.t. the nbr. of iterations')

pl.figure()
pl.plot(10*np.log10(wmse),label='wmse (opt.)')
pl.plot(10*np.log10(wmses),'+',label='wmse (greedy)')
pl.legend(loc='best')
pl.xlabel('It.')
pl.title('wmse (dB): with opt. parameters & greedy approach')

#%% 

snr_gs_name = os.path.join(folder,'snr_gs.npy')
snr_gs = np.load(snr_gs_name) 
mu_s_gs_name = os.path.join(folder,'mu_s_gs.npy')
mu_s_gs = np.load(mu_s_gs_name) 
mu_l_gs_name = os.path.join(folder,'mu_l_gs.npy')
mu_l_gs = np.load(mu_l_gs_name) 

pl.figure()
pl.plot(snr_gs,label='snr (opt.)')
pl.plot(snr,label='snr (greedy)')
pl.legend(loc='best')
pl.title('snr (dB): with opt. parameters & greedy approach')

mu_s_gs_ = np.tile(mu_s_gs,(len(mu_s)))
mu_l_gs_ = np.tile(mu_l_gs,(len(mu_s)))
pl.figure()
pl.plot(mu_s,label=r'$\mu_s$ (greedy)')
pl.plot(mu_s_gs_,label=r'$\mu_s$ (opt.)')
pl.plot(mu_l,label=r'$\mu_l$ (greedy)')
pl.plot(mu_l_gs_,label=r'$\mu_l$ (opt.)')
pl.legend(loc='best')
pl.xlabel('It.')
pl.title('Optimal & greedy regularization parameter values')

print('')
print('greedy gs: (mu_s,mu_l)=(',mu_s[-1],',',mu_l[-1],')')
print('gs: (mu_s,mu_l)=(',mu_s_gs,',',mu_l_gs,')')

#%%

snr_tst_name = os.path.join(folder,'snr_tst.npy')
snr_tst = np.load(snr_tst_name) 

snr_tst2_name = os.path.join(folder,'snr_tst2.npy')
snr_tst2 = np.load(snr_tst2_name) 

pl.figure()
pl.plot(snr,label='snr greedy gs')
pl.plot(snr_gs,label='snr gs')
pl.plot(snr_tst,label='snr tst')
pl.plot(snr_tst2,label='snr tst2')
pl.legend(loc='best')

#%% 
from astropy.io import fits

def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x

folder = 'data256'
file_in = 'M31_3d_conv_256_10db'
folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

L = 100

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]
skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = sky[:,:,0:L]

l = 1
pl.figure()
pl.imshow(CubeDirty[:,:,l])
pl.colorbar()
pl.figure()
pl.imshow(x0[:,:,l])
pl.colorbar()
pl.figure()
pl.imshow(sky[:,:,l])
pl.colorbar()

x = 130
y = 140
pl.figure()
pl.plot(sky[x,y,:],label='Sky spectrum')
pl.plot(CubeDirty[x,y,:],label='Dirty spectrum')
pl.plot(x0[x,y,:],'+',label='Reconstructed spectrum')
pl.legend(loc='best')

        




