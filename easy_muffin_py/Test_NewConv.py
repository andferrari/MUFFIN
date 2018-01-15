#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:02:55 2017

@author: rammanouil
"""

import numpy as np 
from numpy.fft import fft2, ifft2, ifftshift

def myfft2(x):
    return fft2(x,axes=(0,1))

def myifft2(x):
    return ifft2(x,axes=(0,1))

def myifftshift(x):
    return ifftshift(x,axes=(0,1))

def conv(x,y,ref='min'):
    if x.shape[0]==y.shape[0]:
        tmp = myifftshift(myifft2(myfft2(x)*myfft2(y)))
    elif x.shape[0]>y.shape[0]:
        z = np.zeros((x.shape[0],x.shape[0]))
        z[:y.shape[0],:y.shape[1]]=y
        z = myifftshift(z)
        tmp = myifftshift(myifft2(myfft2(x)*myfft2(z)))
    else:
        z = np.zeros((y.shape[0],y.shape[0]))
        z[:x.shape[0],:x.shape[1]]=x
        z = myifftshift(z)
        tmp = myifftshift(myifft2(myfft2(z)*myfft2(y)))
    
    if ref=='min':
        Nout = np.minimum(x.shape[0],y.shape[0])
        tmp = myifftshift(tmp)
        tmp = tmp[:Nout,:Nout]
    
    return tmp.real

from skimage import data
import pylab as pl 
 
image = data.coins()[:303,:303]
image = image.astype(np.float)

pl.figure()
pl.imshow(image)
pl.colorbar()

filt = np.ones((5,5))
filt = filt/np.sum(filt)

filt = np.zeros((2*image.shape[0],2*image.shape[1]))
filt[:4,:4]=1
filt = myifftshift(filt)

image_filt = conv(image,filt,'min')

pl.figure()
pl.imshow(image_filt)

