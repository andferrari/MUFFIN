#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:59:48 2016

@author: antonyschutz
"""
#import matplotlib.pylab as pl
import numpy as np
import os
from numpy.fft import fft2, ifft2, ifftshift
from scipy.fftpack import dct,idct
from astropy.io import fits
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
# MYFFT definition for fast change of library And TOOLS
#==============================================================================
def myfft2(x):
    return fft2(x,axes=(0,1))

def myifft2(x):
    return ifft2(x,axes=(0,1))

def myifftshift(x):
    return ifftshift(x,axes=(0,1))

def sat(x):
    """ Soft thresholding on array x"""
    return np.minimum(np.abs(x), 1.0)*np.sign(x)

def abs2(x):
    return x.real*x.real+x.imag*x.imag

def init_dirty_admm(dirty, psf, psfadj, mu):
    """ Initialization with Wiener Filter """
    A = 1.0/( abs2( myfft2(psf ) ) + mu  )
    B = myifftshift( myifft2( myfft2(dirty) * myfft2(psfadj) ) )
    result = myifft2( A * myfft2(B.real) )
    return result.real

def checkdim(x):
    if len(x.shape)==4:
        x =np.squeeze(x)
    x = x.transpose((2,1,0))
    return x

def defadj(x):
    return x[::-1,::-1,:]
#==============================================================================
# MAIN FONCTION
#==============================================================================
def easy_muffin(mu_s,mu_l,nb,nitermax,file_in,folder='data/',filename_x0=''):



    print('loading...')

    folder = os.path.join(os.getcwd(),folder)
    genname = os.path.join(folder,file_in)

    psfname = genname+'_psf.fits'
    drtname = genname+'_dirty.fits'
    skyname = genname+'_sky.fits'

    psf = checkdim(fits.getdata(psfname, ext=0))
    dirty = checkdim(fits.getdata(drtname, ext=0))
    sky = checkdim(fits.getdata(skyname, ext=0))

    psfadj = defadj(psf)

    print('psf size ', psf.shape)
    print('drt size ', dirty.shape)
    print('sky size ', sky.shape)

    nfreq = dirty.shape[2]
    nxy = dirty.shape[0]


    if filename_x0:
        x = np.load(filename_x0)
    else:
        x = init_dirty_admm(dirty, psf, psfadj, 5e1)


    # precomputations
    print("precomputations...")

    hth_fft = np.zeros((nxy,nxy,nfreq), dtype=np.complex)
    fty = np.zeros((nxy,nxy,nfreq), dtype=np.float)


    psfadj_fft = myfft2(psfadj)
    hth_fft = myfft2( myifftshift( myifft2( psfadj_fft * myfft2(psf) ) ) )
    tmp = myifftshift(myifft2(myfft2(dirty)*psfadj_fft))
    fty = tmp.real


    wstu = np.zeros((nxy,nxy), dtype=np.float)
    Delta_freq = np.zeros((nxy,nxy), dtype=np.float)
#    tmp_spat_cal = np.zeros((nxy,nxy,nb), dtype=np.float)
    xt = np.zeros((nxy,nxy,nfreq), dtype=np.float)

    # Iteration
    print("iterate...")

    loop = True
    niter = 0

    resid = sky-x
    snr = np.zeros(nitermax+1, dtype=np.float)
    sky2 = np.sum(sky*sky)
    snr[niter] = 10*np.log10( sky2 / np.sum(resid*resid)  )


    u = np.zeros((nxy,nxy,nfreq,nb), dtype=np.float)
    v = np.zeros((nxy,nxy,nfreq), dtype=np.float)

    tau = 1e-4
    sigma = 1
    print( niter, ' ', snr[niter])

    while loop and niter<nitermax:
        niter+=1
        t = idct(v, axis=2, norm='ortho') # to check

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(x) * hth_fft ) )
        Delta_freq = tmp.real- fty

        for freq in range(nfreq):

            # compute iuwt adjoint
            wstu = iuwt_recomp(np.squeeze(u[:,:,freq,:]), 0)

            # compute xt
            xt[:,:,freq] = np.maximum(x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu + mu_l*t[:,:,freq]), 0.0 )

            # update u
            tmp_spat_scal = iuwt_decomp(2*xt[:,:,freq] - x[:,:,freq],nb)
            for b in range(nb):
                u[:,:,freq,b] = sat( u[:,:,freq,b] + sigma*mu_s*tmp_spat_scal[:,:,b])

        # update v
        v = sat(v + sigma*mu_l*dct(2*xt - x, axis=2, norm='ortho'))
        x = xt.copy()

        resid = sky-x
        snr[niter] = 10*np.log10( sky2 / np.sum(resid*resid)  )

        print( niter, ' ', snr[niter])



    return xt, snr
