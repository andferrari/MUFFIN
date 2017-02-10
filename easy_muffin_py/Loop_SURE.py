#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:39:55 2017

@author: rammanouil
"""

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

folder='data256'
file_in = 'M31_3d_conv_256_10db'

folder = os.path.join(os.getcwd(),folder)
genname = os.path.join(folder,file_in) 
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,250:255]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,250:255]

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))[:,:,250:255]
sky2 = np.sum(sky*sky)

pl.figure()
pl.imshow(CubePSF[:,:,0])

pl.figure()
pl.imshow(CubeDirty[:,:,0])

pl.figure()
pl.imshow(sky[:,:,0])

#%% ---------------------------------------------------------------------------
# from SuperNiceSpectraDeconv import SNSD 
#from deconv3d import EasyMuffin
#
#
#nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nitermax = 100
#
##%% 
#
#EM= EasyMuffin(mu_s=.2, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty)
#mu_s = EM.gs_mu_s(nitermax=nitermax,maxiter=3)
#mu_s_list = EM.mu_s_lst
#mse_lst = EM.mse_lst
#
#pl.figure()
#pl.stem(mu_s_list,mse_lst)
#
#mu_s_lst2 = []
#mse_lst2 = []
#
#for mu_s in [0.1, 0.5, 0.9, 1.5, 2]:
#    print(mu_s)
#    EM = EasyMuffin(mu_s=mu_s, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty)
#    EM.loop(nitermax)
#    mu_s_lst2.append(mu_s)
#    mse_lst2.append(EM.mse())
#
#pl.figure()
#pl.stem(mu_s_lst2,mse_lst2)
#
#pl.figure()
#pl.plot(EM.wmse(),label='wmse3')
#pl.plot(EM.wmsesure(),'*',label='wmsesure3')
#pl.legend(loc='best')

#%% ---------------------------------------------------------------------------
from deconv3d import EasyMuffinSURE

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nitermax = 600
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EM00= EasyMuffinSURE(mu_s=1.0, mu_l = 0, nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
EM00.loop(nitermax)

np.save('EM00_.npy',EM00)

pl.figure()
pl.plot(EM00.wmselist,label='wmse3')
pl.plot(EM00.wmselistsure,'*',label='wmsesure3')
pl.legend(loc='best')

pl.figure()
pl.plot(EM00.costlist,label='cost')

pl.figure()
pl.plot(EM00.snrlist,label='cost')

#%% ---------------------------------------------------------------------------
from deconv3d import EasyMuffinSURE

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nitermax = 100
Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EM0= EasyMuffinSURE(mu_s=0., mu_l=0., nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)
EM0.loop(nitermax)

np.save('EM0_.npy',EM0)

pl.figure()
pl.plot(EM0.wmselist,label='wmse3')
pl.plot(EM0.wmselistsure,label='wmsesure3')
pl.legend(loc='best')


#%% 

from deconv3d import EasyMuffinSURE

nb=('db1','db2','db3','db4','db5','db6','db7','db8')
nitermax = 100

Noise = CubeDirty - conv(CubePSF,sky)
var = np.sum(Noise**2)/Noise.size
EM= EasyMuffinSURE(mu_s=1, mu_l=0,nb=nb,truesky=sky,psf=CubePSF,dirty=CubeDirty,var=var)

EM.loop_mu_s(nitermax)
np.save('EM_.npy',EM)
EM.loop_mu_l(nitermax)
np.save('EM_.npy',EM)


for i in range(10):
    EM.loop_mu_s(3)
    EM.loop_mu_l(3)
    

#np.save('EM_.npy',EM)

pl.figure()
pl.plot(EM.mu_slist,label='mu_s')
pl.plot(EM.mu_llist,label='mu_l')
pl.legend(loc='best')

pl.figure()
pl.plot(EM.wmselist,label='wmse')
pl.plot(EM.wmselistsure,label='wmsesure')
pl.legend(loc='best')

pl.figure()
pl.plot(EM.costlist,label='cost')
pl.legend(loc='best')

pl.figure()
pl.plot(EM.snrlist,label='snr')
pl.legend(loc='best')

##%%

pl.figure()
pl.plot(EM00.wmselist,label='wmse00')
pl.plot(EM0.wmselist,label='wmse0')
pl.plot(EM.wmselist,label='wmse')
pl.legend(loc='best')

pl.figure()
pl.plot(EM00.snrlist,label='snr00')
pl.plot(EM0.snrlist,label='snr0')
pl.plot(EM.snrlist,label='snr')
pl.legend(loc='best')
#
#
pl.figure()
pl.plot(EM00.costlist,label='cost00')
pl.plot(EM0.costlist,label='cost0')
pl.plot(EM.costlist,label='cost')
pl.legend(loc='best')
