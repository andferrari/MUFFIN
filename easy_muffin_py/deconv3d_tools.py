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

    print('1')
    beta = np.max(abs2(myfft2(psf)))
    print('2')

    #print('nbw_decomp=',len(nbw_decomp))

    tau = 0.9/(beta/2  + sigma*(mu_s**2)*len(nbw_decomp) + sigma*(mu_l**2))
    tau = tau
    return tau

def compute_tau_2D(psf,mu_s,sigma,nbw_decomp):

    beta = np.max(abs2(myfft2(psf)))
    tau = 0.9/(beta/2  + sigma*(mu_s**2)*len(nbw_decomp))
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
    if x.ndim==3:
        return np.roll(np.roll(x[::-1,::-1,:],1,axis=0),1,axis=1)
    else:
        return np.roll(np.roll(x[::-1,::-1],1,axis=0),1,axis=1)

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
#def dwt_decomp(x, list_wavelet, store_c0=False):
#    out = {}
#    coef = []
#    for base in list_wavelet:
#        a,(b,c,d) = pywt.dwt2(x, base)
#        coef.append((a,(b,c,d)))
#        out[base] = np.vstack( ( np.hstack((a,b)) , np.hstack((c,d)) ) )
#    if store_c0:
#        return out,coef
#    else:
#        return out
#
#def dwt_recomp(x_in, nbw, c0=False):
#    list_wavelet = nbw[0:-1]
#    out = 0
#    for n,base in enumerate(list_wavelet):
#        x = x_in[base]
#        ny,nx = x.shape
#        y2 = int(ny/2)
#        x2 = int(nx/2)
#        a = x[:y2,:x2]
#        b = x[:y2,x2:]
#        c = x[y2:,:x2]
#        d = x[y2:,x2:]
#        out += pywt.idwt2( (a,(b,c,d)), base )
#    return out

def dwt_recomp(wt_coeffs, DWT_list, level = 1.):
	"""This function computes the inverse disctrete wavelet transform on a union of wavelet basis.
		wt_coeff : matrix of decompostion coefficients (approximation and details in two columns).
		signal_length = length of the original signal we want retrieve. 
		DWT_list : list of the used wavelet's name."""
	N2 =  wt_coeffs[DWT_list[0]].shape[0] #image is supposed to be square
	res = np.zeros((N2,N2))
    
	for i in DWT_list:
		ima_dec = wt_coeffs[i]
		level = pywt.dwt_max_level(N2, pywt.Wavelet(i)) ## modif 04-04-17
		wt_coeffs_2D = organize_dwt_coeff(ima_dec, level)
		ima_rec = pywt.waverec2(wt_coeffs_2D, i, 'per')
		res = res + ima_rec

	return res

def dwt_decomp(x_im, DWT_list, level = 1.):
	"""This function computes the disctrete wavelet transform on a union of wavelet basis.
		x = signal to be decomposed.
		DWT_list : list of the used wavelet's name."""
	res = {}
	(N1, N2) = np.shape(x_im) 
	
	for i in DWT_list:
		coeffs = pywt.wavedec2(x_im, i, 'per', level = None)## modif 04-04-17
		level = pywt.dwt_max_level(N2, pywt.Wavelet(i)) ## modif 04-04-17
		ima_dec = lecture_dwt_coeff(coeffs, level, N2, N2)
		res[i] = ima_dec
	
	return res


def lecture_dwt_coeff(coeff, level, LI, COL) :
	""" This function reorganizes as an image the DWT coefficients with the following structure : [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)] which is compatible with the pywt functions."""
	im = np.zeros((LI, COL))
	
	ind_li = 0
	ind_col = 0

	for i in range(len(coeff)):
		
		if i == 0:
			ind_li = int(LI/(2**(int(level) -i)))
			ind_col = int(COL/(2**(int(level) -i)))
			im[:ind_li, :ind_col] = coeff[i]

		else:
			a,b = coeff[i][0].shape
			im[:ind_li, ind_col:ind_col + b] = coeff[i][0]
			
			im[ind_li:ind_li+a,:ind_col] = coeff[i][1]
			
			im[ind_li:ind_li+a,ind_col:ind_col+b] = coeff[i][2]
			ind_li = ind_li + b
			ind_col = ind_col + a
	return 	im


def organize_dwt_coeff(im, level) :
	""" This function reorganizes an image as the DWT coefficients with the following structure : [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)] which is compatible with the pywt functions."""
	coeff = []
	LI, COL = im.shape
	ind_li = 0
	ind_col = 0

	for i in range(level+1):
		
		if i == 0:
			ind_li = int(LI/(2**(int(level) -i)))
			ind_col = int(COL/(2**(int(level) -i)))
			coeff_tab = im[:ind_li, :ind_col]
			coeff.append(coeff_tab)

		else:
			a = int(LI/(2**(level+1 -i)))
			b = int(COL/(2**(level+1 -i)))
			coeff_tab0 = im[:ind_li, ind_col:ind_col + b]
			coeff_tab1 = im[ind_li:ind_li+a,:ind_col]
			coeff_tab2 = im[ind_li:ind_li+a,ind_col:ind_col+b]
			ind_li = ind_li + b
			ind_col = ind_col + a
			coeff.append( ( coeff_tab0, coeff_tab1, coeff_tab2))
	return 	coeff


#def dwt_decomp(x, list_wavelet, store_im=False):
#    out = {} # Dictionary of coef for each wavelet
#    coefs = [] # List of all wavelet coeficients
#
#    for base in list_wavelet:
#        coef = pywt.wavedec2(x,base,mode='periodization')
#        coefs.append(coef)
#
#        tmp = []
#        for i,coefscale in enumerate(coef):
#            if i==0:
#                tmp = coef[0]
#            else:
#                (LH, HL, HH) = coef[i]
#                tmp = np.vstack( ( np.hstack((tmp,LH)) , np.hstack((HL,HH)) ) )
#        out[base] = tmp
#
#    if store_im:
#        return coefs, out
#    else:
#        return coefs
#
#
#def dwt_recomp(x_in, list_wavelet):
#    out = 0
#    for n,base in enumerate(list_wavelet):
#        x = x_in[n] # coef a,b,c,d at db1 for ex
#        out +=pywt.waverec2(x, base, mode='periodization')
#
#    return out

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


def iuwt_recomp(x, scale, c0=[]):

#    filter = (1./16,4./16,6./16,4./16,1./16)

    max_scale = len(x) + scale

    if c0.any():
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
    htu = iuwt_decomp(u[0],[0])[0]
    scale = len(u)
    for k in range(1,scale): #  1 à 7
        scale_decomp = [i for i in range(k+1)]
        htu += iuwt_decomp(u[k],scale_decomp)[k]
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
