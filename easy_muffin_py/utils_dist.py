#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:28:24 2017

@author: rflamary
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from scipy.fftpack import dct,idct
import pywt


import time
__time_tic_toc=time.time()

def tic():
    """ Python implementation of Matlab tic() function """
    global __time_tic_toc
    __time_tic_toc=time.time()

def toc(message='Elapsed time : {} s'):
    """ Python implementation of Matlab toc() function """
    t=time.time()
    print(message.format(t-__time_tic_toc))
    return t-__time_tic_toc

def toq():
    """ Python implementation of Julia toc() function """
    t=time.time()
    return t-__time_tic_toc

#==============================================================================
# TOOLS        
#==============================================================================
def defadj(x):
    return x[::-1,::-1,:] 

def sat(x):
    """ Soft thresholding on array x"""
    return np.minimum(np.abs(x), 1.0)*np.sign(x)

def abs2(x):
    return x.real*x.real+x.imag*x.imag  
#==============================================================================
# MYFFT definition for fast change of library And TOOLS
#==============================================================================
def myfft2(x):
    return fft2(x,axes=(0,1))

def myifft2(x):
    return ifft2(x,axes=(0,1))    
    
def myifftshift(x):
    return ifftshift(x,axes=(0,1))   
    
    
#==============================================================================
# IUWT from IUWT.jl from PyMoresane
#==============================================================================
def iuwt_decomp(x, scale, store_c0=False):

    filter = (1./16,4./16,6./16,4./16,1./16)

    coeff = np.zeros((x.shape[0],x.shape[1],scale), dtype=np.float)
    c0 = x

    for i in range(scale):
        c = a_trous(c0,filter,i)
        c1 = a_trous(c,filter,i)
        coeff[:,:,i] = c0 - c1
        c0 = c

    if store_c0:
        return coeff,c0
    else:
        return coeff

    
def iuwt_recomp(x, scale, c0=False):

    filter = (1./16,4./16,6./16,4./16,1./16)

    max_scale = x.shape[2] + scale

    if c0 != False:
        recomp = c0
    else:
        recomp = np.zeros((x.shape[0],x.shape[1]), dtype=np.float)


    for i in range(max_scale-1,-1,-1):
        recomp = a_trous(recomp,filter,i) + x[:,:,i-scale]

#    if scale > 0:
#        for i in range(scale,0,-1):
#            recomp = a_trous(recomp,filter,i)


    return recomp
    
    
def iuwt_decomp_adj(u,scale):
    htu = iuwt_decomp(u[:,:,0],1)[:,:,0]
    for k in range(1,scale):
        htu += iuwt_decomp(u[:,:,k],k)[:,:,k]
    return htu

def a_trous(C0, filter, scale):
    """
    Copy form https://github.com/ratt-ru/PyMORESANE
    The following is a serial implementation of the a trous algorithm. Accepts the following parameters:

    INPUTS:
    filter      (no default):   The filter-bank which is applied to the components of the transform.
    C0          (no default):   The current array on which filtering is to be performed.
    scale       (no default):   The scale for which the decomposition is being carried out.

    OUTPUTS:
    C1                          The result of applying the a trous algorithm to the input.
    """
    tmp = filter[2]*C0

    tmp[(2**(scale+1)):,:] += filter[0]*C0[:-(2**(scale+1)),:]
    tmp[:(2**(scale+1)),:] += filter[0]*C0[(2**(scale+1))-1::-1,:]

    tmp[(2**scale):,:] += filter[1]*C0[:-(2**scale),:]
    tmp[:(2**scale),:] += filter[1]*C0[(2**scale)-1::-1,:]

    tmp[:-(2**scale),:] += filter[3]*C0[(2**scale):,:]
    tmp[-(2**scale):,:] += filter[3]*C0[:-(2**scale)-1:-1,:]

    tmp[:-(2**(scale+1)),:] += filter[4]*C0[(2**(scale+1)):,:]
    tmp[-(2**(scale+1)):,:] += filter[4]*C0[:-(2**(scale+1))-1:-1,:]

    C1 = filter[2]*tmp

    C1[:,(2**(scale+1)):] += filter[0]*tmp[:,:-(2**(scale+1))]
    C1[:,:(2**(scale+1))] += filter[0]*tmp[:,(2**(scale+1))-1::-1]

    C1[:,(2**scale):] += filter[1]*tmp[:,:-(2**scale)]
    C1[:,:(2**scale)] += filter[1]*tmp[:,(2**scale)-1::-1]

    C1[:,:-(2**scale)] += filter[3]*tmp[:,(2**scale):]
    C1[:,-(2**scale):] += filter[3]*tmp[:,:-(2**scale)-1:-1]

    C1[:,:-(2**(scale+1))] += filter[4]*tmp[:,(2**(scale+1)):]
    C1[:,-(2**(scale+1)):] += filter[4]*tmp[:,:-(2**(scale+1))-1:-1]

    return C1

#==============================================================================
# DIRTY INITIALIZATION FOR ADMM 
#==============================================================================
        
def init_dirty_admm(dirty, psf, psfadj, mu):
    """ Initialization with Wiener Filter """
    A = 1.0/( abs2( myfft2(psf ) ) + mu  )
    B = myifftshift( myifft2( myfft2(dirty) * myfft2(psfadj) ) )
    result = myifft2( A * myfft2(B.real) )
    return result.real    
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        