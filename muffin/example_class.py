#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:08:49 2016

@author: antonyschutz
"""
# ==============================================================================
# Imports
# ==============================================================================

import os
import numpy as np
from astropy.io import fits
import pylab as pl
from deconv3d_tools import conv, fix_dim
from tictoc import tic, toc
from super_nice_spectra_deconv import SNSD
from deconv3d import EasyMuffin, EasyMuffinSURE
import argparse 

# =============================================================================
# Terminal Input
# =============================================================================
parser = argparse.ArgumentParser(description='Awesome Argument Parser')
parser.add_argument('-fol','--folder',help='Path to data folder')
parser.add_argument('-nam','--file_in',help='Data Prefix')
parser.add_argument('-sav','--save',default=0,type=int,help='Save Output Variables')
parser.add_argument('-init','--init',default=0,type=int,help='Init with Saved Variables')
parser.add_argument('-fol_init','--folder_init',help='Path to init data folder')
parser.add_argument('-N','--niter',default=3,type=int,help='Save Output Variables')

args = parser.parse_args()

folder = args.folder
file_in = args.file_in
save = args.save
init = args.init
folder_init = args.folder_init
N = args.niter

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

genname = os.path.join(folder, file_in)
psf_name = genname+'_psf.fits'
drt_name = genname+'_dirty.fits'
L = 5
cube_psf = fix_dim(fits.getdata(psf_name, ext=0))[:,:,0:L]
cube_dirty = fix_dim(fits.getdata(drt_name, ext=0))[:,:,0:L]

sky_name = genname+'_sky.fits'
sky = fix_dim(fits.getdata(sky_name, ext=0))
sky = np.transpose(sky)[:,:,0:L]
sky2 = np.sum(sky*sky)

fig = pl.figure()
ax = fig.add_subplot(1,3,1)
ax.imshow(cube_psf[:,:,1])
ax = fig.add_subplot(1,3,2)
ax.imshow(sky[:,:,1])
ax = fig.add_subplot(1,3,3)
ax.imshow(cube_dirty[:,:,1])

#%% ==============================================================================
# Tsts SNSD EM and EMSURE 
# ==============================================================================

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nb = (7,0)
nitermax = N

mu_s = 1
mu_l = 1
fftw = 0

#tic()
#DM = SNSD(mu_s=mu_s, mu_l = mu_l, nb=nb,nitermax=nitermax,truesky=sky)
#DM.parameters()
#DM.setSpectralPSF(cube_psf)
#DM.setSpectralDirty(cube_dirty)
#(SpectralSkyModel , cost, snr, psnr) = DM.main()
#toc()
#
#tic()
#EM= EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=cube_psf,dirty=cube_dirty,fftw=fftw,init=init,
#               fol_init=folder_init, save=0)
#EM.loop(nitermax)
#SpectralSkyModel2 = EM.xt
#cost2 = EM.costlist
#snr2 = EM.snrlist
#psnr2 = EM.psnrlist
#wmse2 = EM.wmselist
#toc()
#
#tic()
#Noise = cube_dirty - conv(cube_psf,sky)
#var = np.sum(Noise**2)/Noise.size
#EMs= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=cube_psf,dirty=cube_dirty,var=var,mu_wiener=5e1,fftw=fftw,init=init,
#               fol_init=folder_init,save=0)
#EMs.loop(nitermax)
#SpectralSkyModel3 = EMs.xt
#cost3 = EMs.costlist
#snr3 = EMs.snrlist
#psnr3 = EMs.psnrlist
#psnrsure3 = EMs.psnrlistsure
#wmse3 = EMs.wmselist
#wmsesure3 = EMs.wmselistsure
#toc()

Noise = cube_dirty - conv(cube_psf,sky)
var = np.sum(Noise**2)/Noise.size
EMsfdmc= EasyMuffinSURE(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=cube_psf,dirty=cube_dirty,var=var,step_mu=[5e-1,5e-1],fftw=fftw,init=init,
               fol_init=folder_init,save=save)
EMsfdmc.loop_fdmc(nitermax)
#EMsfdmc= EasyMuffin(mu_s=mu_s, mu_l = mu_l, nb=nb,truesky=sky,psf=cube_psf,dirty=cube_dirty,var=var,fftw=fftw,init=init,
#               fol_init=folder_init,save=save)
#EMsfdmc.loop(nitermax)
SpectralSkyModel4 = EMsfdmc.xt
cost4 = EMsfdmc.costlist
snr4 = EMsfdmc.snrlist
psnr4 = EMsfdmc.psnrlist
#psnrsure4 = EMsfdmc.psnrlistsure
wmse4 = EMsfdmc.wmselist
#wmsesurefdmc = EMsfdmc.wmselistsurefdmc

#%% ==============================================================================
# Plot some results 
# ===================================================================bool(dct)===========

#pl.figure()
#pl.plot(EMsfdmc.mu_llist)
#pl.figure()
#pl.plot(EMsfdmc.mu_slist)
#
#pl.figure()
#pl.plot(snr,label='snr1')
#pl.plot(snr2,label='snr2')
#pl.plot(snr3,'*',label='snr3')
#pl.plot(snr4,'*',label='snr4')
#pl.legend(loc='best')
#
#pl.figure()
#pl.plot(cost/(EM.nxy*EM.nxy*EM.nfreq),label='cost1')
#pl.plot(cost2,label='cost2')
#pl.plot(cost3,'*',label='cost3')
#pl.plot(cost4,'*',label='cost4')
#pl.legend(loc='best')
#
#pl.figure()
#pl.plot(psnr,label='psnr1')
#pl.plot(psnr2,label='psnr2')
#pl.plot(psnr3,'*',label='psnr3')
#pl.plot(psnrsure3,'*',label='psnrsure3')
#pl.plot(psnrsure4,'*',label='psnrsure4')
#pl.legend(loc='best')
#
#pl.figure()
#pl.plot(wmse2,label='wmse2')
#pl.plot(wmse3,label='wmse3')
#pl.plot(wmsesure3,'*',label='wmsesure3')
#pl.plot(wmsesurefdmc,'*',label='wmsesurefdmc')
#pl.legend(loc='best')
#
#pl.show()


print('')
print('EMsfdmc.dx_s: ',np.linalg.norm(EMsfdmc.dx_s))
print('')    

print('')
print('EMsfdmc.dx_l: ',np.linalg.norm(EMsfdmc.dx_l))
print('')    

print('')
print('EMsfdmc.dx2_s: ',np.linalg.norm(EMsfdmc.dx2_s))
print('')    

print('')
print('EMsfdmc.dx2_l: ',np.linalg.norm(EMsfdmc.dx2_l))
print('')    
 
print('')
print('EMsfdmc.sugarfdmclist: ',EMsfdmc.sugarfdmclist[0][:])
print('')    

print('')
print('EMsfdmc.sugarfdmclist: ',EMsfdmc.sugarfdmclist[1][:])
print('')    

