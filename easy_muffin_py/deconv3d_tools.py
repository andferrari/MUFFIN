#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:42:26 2017

@author: rammanouil
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
import pywt

#==============================================================================
# Compute tau
#==============================================================================
def compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp):

    beta = np.max(abs2(myfft2(psf)))

    #print('nbw_decomp=',len(nbw_decomp))

    tau = 0.9/(beta/2  + sigma*(mu_s**2)*len(nbw_decomp) + sigma*(mu_l**2))
    tau = tau
    return tau
#==============================================================================
# tools for Jacobians comp.
#==============================================================================
def Heavy(x):
    return (np.sign(x)+1)/2

def Rect(x):
    return Heavy(x+1)-Heavy(x-1)

#==============================================================================
# TOOLS
#==============================================================================
def defadj(x):
    return np.roll(np.roll(x[::-1,::-1,:],1,axis=0),1,axis=1)

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

def conv(x,y):
    tmp = myifftshift(myifft2(myfft2(x)*myfft2(y)))
    return tmp.real
#==============================================================================
# DWT from adapted to same style as IUWT.jl from PyMoresane
#==============================================================================
def dwt_decomp(x, list_wavelet, store_c0=False):
    out = {}
    coef = []
    for base in list_wavelet:
        a,(b,c,d) = pywt.dwt2(x, base)
        coef.append((a,(b,c,d)))
        out[base] = np.vstack( ( np.hstack((a,b)) , np.hstack((c,d)) ) )
    if store_c0:
        return out,coef
    else:
        return out

def dwt_recomp(x_in, nbw, c0=False):
    list_wavelet = nbw[0:-1]
    out = 0
    for n,base in enumerate(list_wavelet):
        x = x_in[base]
        ny,nx = x.shape
        y2 = int(ny/2)
        x2 = int(nx/2)
        a = x[:y2,:x2]
        b = x[:y2,x2:]
        c = x[y2:,:x2]
        d = x[y2:,x2:]
        out += pywt.idwt2( (a,(b,c,d)), base )
    return out

#==============================================================================
# IUWT from IUWT.jl from PyMoresane
#==============================================================================
def iuwt_decomp(x, scale, store_c0=False):

#    filter = (1./16,4./16,6./16,4./16,1./16)
#    coeff = np.zeros((x.shape[0],x.shape[1],scale), dtype=np.float)
    coeff = {}
    c0 = x

#    for i in range(scale):
    for i in scale:
        c = a_trous(c0,i)
        c1 = a_trous(c,i)
#        coeff[:,:,i] = c0 - c1
        coeff[i] = c0 - c1
        c0 = c

    if store_c0:
        return coeff,c0
    else:
        return coeff


def iuwt_recomp(x, scale, c0=False):

#    filter = (1./16,4./16,6./16,4./16,1./16)

    max_scale = len(x) + scale

    if c0 != False:
        recomp = c0
    else:
        recomp = np.zeros((x[0].shape[0],x[0].shape[1]), dtype=np.float)


    for i in range(max_scale-1,-1,-1):
        recomp = a_trous(recomp,i) + x[i-scale]

#    if scale > 0:
#        for i in range(scale,0,-1):
#            recomp = a_trous(recomp,filter,i)


    return recomp


def iuwt_decomp_adj(u,scale):
    htu = iuwt_decomp(u[:,:,0],1)[:,:,0]
    for k in range(1,scale):
        htu += iuwt_decomp(u[:,:,k],k)[:,:,k]
    return htu

def a_trous(C0, scale):
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
    filter = (1./16,4./16,6./16,4./16,1./16)

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
# DIRTY INITIALIZATION FOR wienner
#==============================================================================

def init_dirty_wiener(dirty, psf, psfadj, mu):
    """ Initialization with Wiener Filter """
    A = 1.0/( abs2( myfft2(psf ) ) + mu  )
    B = myifftshift( myifft2( myfft2(dirty) * myfft2(psfadj) ) )
    result = myifft2( A * myfft2(B.real) )
    return result.real

#==============================================================================
# mpi splitting
#==============================================================================

def optimal_split(ntot,nsplit):
    if (ntot % nsplit)==0:
        x=int(ntot/nsplit)
        return [x for i in range(nsplit)]
    else:
        x=int(np.ceil(ntot/nsplit))
        y=int(ntot-x*(nsplit-1))


        ret=[x for i in range(nsplit-1)]
        ret.append(y)
        ret=np.array(ret)

        if y<1:
            ret[y-2:]-=1
            ret[-1]=1

        return ret.tolist()

#==============================================================================
# tools for golden section search
#==============================================================================

#def gs_search(f, a, b, args=(),absolutePrecision=1e-2,maxiter=100):
#
#    gr = (1+np.sqrt(5))/2
#    c = b - (b - a)/gr
#    d = a + (b - a)/gr
#    niter = 0
#
#    while abs(a - b) > absolutePrecision and niter < maxiter:
#        if f( *((c,) + args) ) < f( *((d,) + args) ):
#            b = d
#        else:
#            a = c
#
#        c = b - (b - a)/gr
#        d = a + (b - a)/gr
#        niter+=1
#
#    return (a + b)/2
