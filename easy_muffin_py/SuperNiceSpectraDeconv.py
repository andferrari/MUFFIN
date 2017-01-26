#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:07:31 2016

@author: antonyschutz
"""
import numpy as np 
from numpy.fft import fft2, ifft2, ifftshift
from scipy.fftpack import dct,idct
import pywt

class SNSD: 
    def __init__(self,nitermax=10,
                 mu_s=0.5,
                 mu_l=0.0,
                 nb=(8,0),
                 tau = 1e-4,
                 sigma = 1,
                 dirtyinit=[],
                 truesky=[]):    
        
        print('')
        
        if nitermax< 1: 
            print('mu_s must be a positive integer, nitermax=10')
            nitermax=10
        
        if type(nb) is not tuple: 
            print('nb must be a tuple of wavelets for dwt ')
            print('or a list of 2 integer for IUWT')
            print('first integer is the scale of decomposition (default:8)')
            print('second integer the scale or recomposition (default:0)')
            nb = (8,0)
            
        if mu_s< 0 : 
            print('mu_s must be non negative, mu_s=.5')
            mu_s=0.5 
            
        if mu_l< 0 : 
            print('mu_l must be non negative, mu_l=0.')
            mu_s=0.0
  
        if tau< 0 : 
            print('mu_s must be non negative, tau=1e-4')
            tau=1e-4
            
        if sigma< 0 : 
            print('mu_s must be positive, sigma=1.')
            sigma=1.     
            
        #======================================================================
        # INITIALIZATION and INITIALIZATION FUNCTION             
        #======================================================================

        self.nitermax = int(nitermax)
        self.nb = nb        
        self.mu_s = mu_s
        self.mu_l = mu_l
        self.sigma = sigma        
        self.tau = tau
        self.dirtyinit = dirtyinit 
        self.truesky = truesky

    def parameters(self):
        print('')
        print('nitermax: ',self.nitermax)
        print('nb: ',self.nb)
        print('mu_s: ',self.mu_s)
        print('mu_l: ',self.mu_l)
        print('tau: ',self.tau)
        print('sigma: ',self.sigma)
        
  
    def setSpectralPSF(self,psf):
        """ check dimension and process psf then store in self """
        # check dimension and process psf then store in self
        self.psf = psf
        
    def setSpectralDirty(self,dirty):
        # check dimension and process dirty then store in self        
        self.dirty = dirty   

        #======================================================================
        # MAIN PROGRAM - EASY MUFFIN         
        #======================================================================

    def main(self):   
        
        psf = self.psf
        dirty = self.dirty
        dirtyinit = self.dirtyinit 
        
        nitermax = self.nitermax 
        nb = self.nb
        
        mu_s = self.mu_s
        mu_l = self.mu_l 
        tau = self.tau 
        sigma = self.sigma 
        
        truesky = self.truesky
        
        return easy_muffin(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,truesky)
        
#==============================================================================
# MAIN FONCTION 
#==============================================================================
def easy_muffin(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,truesky=None):
                  
    
    psfadj = defadj(psf)
    
    print('')
    print('psf size ', psf.shape)
    print('drt size ', dirty.shape)
    
    nfreq = dirty.shape[2]
    nxy = dirty.shape[0]

    if dirtyinit:
        x = dirtyinit
    else:
        x = init_dirty_admm(dirty, psf, psfadj, 5e1)
        
    if truesky.any():
        snr = np.zeros((nitermax))
        truesky2 = np.sum(truesky*truesky)
        
    cost = np.zeros((nitermax))
    
    # precomputations
    print('')    
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

 
    if type(nb[0]) == int:
        Decomp = iuwt_decomp
        Recomp = iuwt_recomp   
        nbw_decomp = [f for f in range(nb[0])]
        nbw_recomp = nb[-1] 

        print('')
        print('IUWT: tau = ', tau)
                    
    else:
        Decomp = dwt_decomp
        Recomp = dwt_recomp        
        nbw_decomp = nb
        nbw_recomp = nb 
        
        tau = compute_tau_DWT(psf,nfreq,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('DWT: tau = ', tau)
        
    u = {}   
    for freq in range(nfreq):        
        u[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)
             
    
    # Iteration 
    print('')    
    print("iterate...")

    loop = True
    niter = 0
    
#    u = np.zeros((nxy,nxy,nfreq,nb), dtype=np.float)
    v = np.zeros((nxy,nxy,nfreq), dtype=np.float)
          
    print('iteration: ',niter)
        
    while loop and niter<nitermax:
        
        # compute snr if truesky given
        if truesky.any():
            resid = truesky - x
            snr[niter]= 10*np.log10(truesky2 / np.sum(resid*resid))
            
        niter+=1 

        t = idct(v, axis=2, norm='ortho') # to check 

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(x) * hth_fft ) ) 
        Delta_freq = tmp.real- fty
        for freq in range(nfreq):
            
            # compute iuwt adjoint
            wstu = Recomp(u[freq], nbw_recomp)

            # compute xt
            xt[:,:,freq] = np.maximum(x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu + mu_l*t[:,:,freq]), 0.0 )

            # update u
            tmp_spat_scal = Decomp(2*xt[:,:,freq] - x[:,:,freq] , nbw_decomp)

            for b in nbw_decomp:
                u[freq][b] = sat( u[freq][b] + sigma*mu_s*tmp_spat_scal[b])
                
        # update v
        v = sat(v + sigma*mu_l*dct(2*xt - x, axis=2, norm='ortho'))
        x = xt.copy()
               
        # compute cost 
        tmp = dirty - myifftshift(myifft2(myfft2(x)*myfft2(psf)))
        LS_cst = 0.5*(np.linalg.norm(tmp)**2)
        tmp = 0.
        for freq in range(nfreq):
            tmp1 = Decomp(x[:,:,freq],nbw_decomp)
            for b in nbw_decomp:
                tmp = tmp + np.sum(np.abs(tmp1[b]))
        Spt_cst = mu_s*tmp
        Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
        cost[niter-1] = LS_cst + Spt_cst + Spc_cst 

        print('iteration: ',niter)

    if truesky.any():
        return xt, cost, snr
    else:
        return xt, cost
        
#==============================================================================
# Compute tau        
#==============================================================================
def compute_tau_DWT(psf,nfreq,mu_s,mu_l,sigma,nbw_decomp):
    
    beta = np.max(np.abs(myfft2(psf))**2)
    
    print('nbw_decomp=',len(nbw_decomp))    

    tau = 0.9/(beta/2  + sigma*(mu_s**2)*len(nbw_decomp) + sigma*(mu_l**2))
    return tau

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
# DIRTY INITIALIZATION FOR ADMM 
#==============================================================================
        
def init_dirty_admm(dirty, psf, psfadj, mu):
    """ Initialization with Wiener Filter """
    A = 1.0/( abs2( myfft2(psf ) ) + mu  )
    B = myifftshift( myifft2( myfft2(dirty) * myfft2(psfadj) ) )
    result = myifft2( A * myfft2(B.real) )
    return result.real    
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        