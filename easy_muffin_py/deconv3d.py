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
import copy 


class EasyMuffin(): 
    def __init__(self,
                 mu_s=0.5,
                 mu_l=0.0,
                 nb=(8,0),
                 tau = 1e-4,
                 sigma = 1,
                 var = 0,
                 dirtyinit=[],
                 dirty=[],
                 truesky=[],
                 psf=[]):    
        
        print('')
        

        
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

        self.nb = nb        
        self.mu_s = mu_s
        self.mu_l = mu_l
        self.sigma = sigma        
        self.tau = tau
        self.dirtyinit = dirtyinit 
        self.truesky = truesky
        self.psf = psf
        self.dirty=dirty
        self.var = var
        
        self.init_algo()
        
    def init_algo(self):
        """Initialization of te algorithm (all intermediate variables)"""
        
        self.psfadj = defadj(self.psf)
        
        print('')
        print('psf size ', self.psf.shape)
        print('drt size ', self.dirty.shape)
        
        self.nfreq = self.dirty.shape[2]
        self.nxy = self.dirty.shape[0]
        
    
        if self.dirtyinit:
            self.x = self.dirtyinit
        else:
            self.x = init_dirty_wiener(self.dirty, self.psf, self.psfadj, 5e1)
            
        if self.truesky.any():
            self.snrlist = []
            self.truesky2 = np.sum(self.truesky*self.truesky)
            
        self.costlist = []
        
        # precomputations
        print('')    
        print("precomputations...")
        
        self.hth_fft = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.complex) 
        self.fty = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float) 
        
        self.psfadj_fft = myfft2(self.psfadj)               
        self.hth_fft = myfft2( myifftshift( myifft2( self.psfadj_fft * myfft2(self.psf) ) ) )        
        tmp = myifftshift(myifft2(myfft2(self.dirty)*self.psfadj_fft))
        self.fty = tmp.real
    
    
        self.wstu = np.zeros((self.nxy,self.nxy), dtype=np.float) 
        self.Delta_freq = np.zeros((self.nxy,self.nxy), dtype=np.float) 
        self.xt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float) 
    
     
        if type(self.nb[0]) == int:
            self.Decomp = iuwt_decomp
            self.Recomp = iuwt_recomp   
            self.nbw_decomp = [f for f in range(self.nb[0])]
            self.nbw_recomp = self.nb[-1] 
    
            print('')
            print('IUWT: tau = ', self.tau)
                        
        else:
            self.Decomp = dwt_decomp
            self.Recomp = dwt_recomp        
            self.nbw_decomp =self.nb
            self.nbw_recomp = self.nb 
            
            self.tau = compute_tau_DWT(self.psf,self.mu_s,self.mu_l,self.sigma,self.nbw_decomp)
            print('')
            print('DWT: tau = ', self.tau)
            
        
            
        self.u = {}   
        for freq in range(self.nfreq):        
            self.u[freq] = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)
                 
        
        # Iteration 
        print('')    
        print("iterate...")
    

        self.nitertot = 0
        
        # compute cost 
        self.costlist.append(self.cost()) 
    
        # compute snr if truesky given
        if self.truesky.any():
            self.snrlist.append(self.snr())
                
        self.v = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float)      
        
    def cost(self):
        """Compute cost for current iterate x"""
        tmp = self.dirty - myifftshift(myifft2(myfft2(self.x)*myfft2(self.psf)))
        LS_cst = 0.5*(np.linalg.norm(tmp)**2)
        tmp = 0.
        for freq in range(self.nfreq):
            tmp1 = self.Decomp(self.x[:,:,freq],self.nbw_decomp)
            for b in self.nbw_decomp:
                tmp = tmp + np.sum(np.abs(tmp1[b]))
        Spt_cst = self.mu_s*tmp
        Spc_cst = self.mu_l*np.sum(np.abs(dct(self.x,axis=2,norm='ortho')))
        return LS_cst + Spt_cst + Spc_cst         
    
    def snr(self):
        resid = self.truesky - self.x
        return 10*np.log10(self.truesky2 / np.sum(resid*resid))      
    
    def update(self):
        
        t = idct(self.v, axis=2, norm='ortho') # to check 

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(self.x) *self.hth_fft ) ) 
        Delta_freq = tmp.real- self.fty
        for freq in range(self.nfreq):
            
            # compute iuwt adjoint
            wstu = self.Recomp(self.u[freq], self.nbw_recomp)

            # compute xt
            self.xt[:,:,freq] = np.maximum(self.x[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + self.mu_s*wstu + self.mu_l*t[:,:,freq]), 0.0 )

            # update u
            tmp_spat_scal = self.Decomp(2*self.xt[:,:,freq] - self.x[:,:,freq] , self.nbw_decomp)

            for b in self.nbw_decomp:
                self.u[freq][b] = sat( self.u[freq][b] + self.sigma*self.mu_s*tmp_spat_scal[b])
                
        # update v
        self.v = sat(self.v + self.sigma*self.mu_l*dct(2*self.xt - self.x, axis=2, norm='ortho'))
        self.x = self.xt.copy()
               
        # compute cost 
        self.costlist.append(self.cost())

        # compute snr if truesky given
        if self.truesky.any():
            self.snrlist.append(self.snr())

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

    def loop(self,nitermax=10):   
        """ main loop """

        if nitermax< 1: 
            print('nitermax must be a positive integer, nitermax=10')
            nitermax=10
            
        for niter in range(nitermax):
            
            self.update()

            print('iteration: ',niter)        
       
        if self.truesky.any():
            return self.xt, np.array(self.costlist), np.array(self.snrlist)
        else:
            return self.xt, np.array(self.costlist)




#==============================================================================
# Compute tau        
#==============================================================================
def compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp):
    
    beta = np.max(abs2(myfft2(psf)))
    
    print('nbw_decomp=',len(nbw_decomp))    

    tau = 0.9/(beta/2  + sigma*(mu_s**2)*len(nbw_decomp) + sigma*(mu_l**2))
    tau = tau/100
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
# tools for golden section search  
#==============================================================================

def gs_search(f, a, b, args=(),absolutePrecision=1e-2,maxiter=100):

    gr = (1+np.sqrt(5))/2
    c = b - (b - a)/gr
    d = a + (b - a)/gr
    niter = 0
    
    while abs(a - b) > absolutePrecision and niter < maxiter:
        if f( *((c,) + args) ) < f( *((d,) + args) ):
            b = d
        else:
            a = c
                    
        c = b - (b - a)/gr
        d = a + (b - a)/gr
        niter+=1

    return (a + b)/2
        