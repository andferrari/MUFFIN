#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:08:49 2016

@author: antonyschutz
"""
# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================

import os
import numpy as np
from astropy.io import fits
import pylab as pl
import utils_dist as ud


from numpy.fft import fft2, ifft2, ifftshift
from scipy.fftpack import dct,idct

from mpi4py import MPI



#==============================================================================
# MAIN FONCTION
#==============================================================================
def easy_muffin(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit):


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

    # Iteration
    print('')
    print("iterate...")

    loop = True
    niter = 0

    u = np.zeros((nxy,nxy,nfreq,nb), dtype=np.float)
    v = np.zeros((nxy,nxy,nfreq), dtype=np.float)

    print('iteration: ',niter)

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
            u[:,:,freq,:] = sat( u[:,:,freq,:] + sigma*mu_s*tmp_spat_scal)

        # update v
        v = sat(v + sigma*mu_l*dct(2*xt - x, axis=2, norm='ortho'))
        x = xt.copy()

        print('iteration: ',niter)

    return xt


def easy_muffin_mpi(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit):

    nfreq = dirty.shape[2]
    nxy = dirty.shape[0]

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nbw=size-1
    idw=rank-1

    # max step possible
    nf2=int(np.ceil(nfreq*1.0/nbw))

    lst_nbf=[nfreq]
    for i in range(nbw):
        step=min(nf2,max(nfreq-sum(lst_nbf[1:]),0))
        lst_nbf.append(step)

    nbsum=0
    sendcounts=[0,]
    displacements=[0,]
    for i in range(nbw):
        displacements.append(nbsum)
        taille=nxy*nxy*lst_nbf[i+1]
        sendcounts.append(taille)
        nbsum+=taille
    #print(sendcounts)


    if rank==0:
        print('')
        print('psf size ', psf.shape)
        print('drt size ', dirty.shape)

        print('')
        print("precomputations...")



    if rank==0:
        print('local size={}'.format(lst_nbf))

    if not rank==0: # local init
        nfreq=lst_nbf[rank]
        psf=psf[:,:,idw*nf2:idw*nf2+nfreq]
        dirty=dirty[:,:,idw*nf2:idw*nf2+nfreq]

        psfadj = defadj(psf)

        psfadj_fft = myfft2(psfadj)
        hth_fft = myfft2( myifftshift( myifft2( psfadj_fft * myfft2(psf) ) ) )        
        tmp = myifftshift(myifft2(myfft2(dirty)*psfadj_fft))
        fty = tmp.real

        if dirtyinit:
            x = dirtyinit
        else:
            x = init_dirty_admm(dirty, psf, psfadj, 5e1)

        xt = np.zeros((nxy,nxy,nfreq), dtype=np.float, order='F')
        u = np.zeros((nxy,nxy,nfreq,nb), dtype=np.float, order='F')
        tloc = np.zeros((nxy,nxy,nfreq), dtype=np.float, order='F')
        t=np.zeros((0,))
        delta=np.zeros((0,))
        xf=None
    else: # global init
        v = np.zeros((nxy,nxy,nfreq), dtype=np.float, order='F')
        t = np.zeros((nxy,nxy,nfreq), dtype=np.float, order='F')
        tloc = np.zeros((0), dtype=np.float, order='F')
        xt = np.zeros((0), dtype=np.float, order='F')
        xf = np.zeros((nxy,nxy,nfreq), dtype=np.float, order='F')
        delta=np.zeros((nxy,nxy,nfreq), dtype=np.float, order='F')
        deltal=np.zeros((0,))

    loop = True
    niter = 0
#
    while loop and niter<nitermax:
        niter+=1

        comm.Scatterv([t,sendcounts,displacements,MPI.DOUBLE],tloc,root=0)
#        if rank==0:
#            for i in range(0,nbw):
#                comm.Send(t[:,:,i*nf2:i*nf2+lst_nbf[i+1]].copy(), dest=i+1, tag=10)
#                #print('sent master {}'.format(i))
#
#        else:
        if not rank==0:
            #comm.Recv(tloc, source=0, tag=10)
            tmp = myifftshift( myifft2( myfft2(x) * hth_fft ) )
            Delta_freq = tmp.real- fty
            for freq in range(nfreq):

                # compute iuwt adjoint
                wstu = iuwt_recomp(np.squeeze(u[:,:,freq,:]), 0)

                # compute xt
                xt[:,:,freq] = np.maximum(x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu + mu_l*tloc[:,:,freq]), 0.0 )

                # update u
                tmp_spat_scal = iuwt_decomp(2*xt[:,:,freq] - x[:,:,freq],nb)
                u[:,:,freq,:] = sat( u[:,:,freq,:] + sigma*mu_s*tmp_spat_scal)
            deltal=2*xt - x
            #print('Computed local {}'.format(rank))

            # sent local delta to main worker
            #comm.Send(deltal,dest=0,tag=12)

        #comm.Barrier()
        comm.Gatherv(deltal,[delta,sendcounts,displacements,MPI.DOUBLE],root=0)

        # update v and t
        if rank==0:

#            for i in range(0,nbw):
#                deltatemp=np.zeros((nxy,nxy,lst_nbf[i+1]))
#                comm.Recv(deltatemp, source=i+1, tag=12)
#                delta[:,:,i*nf2:i*nf2+lst_nbf[i+1]]=deltatemp[:,:,:lst_nbf[i+1]]
#                #print('recieve delta master {}'.format(i))
            v = sat(v + sigma*mu_l*dct(delta, axis=2, norm='ortho'))
            t = idct(v, axis=2, norm='ortho') # to check
            print('iteration: ',niter)
        else:
            x = xt.copy()


    # get x from all workers
#    if rank==0:
#        xtemp=np.zeros((nxy,nxy,nf2))
#        for i in range(0,nbw):
#            xtemp=np.zeros((nxy,nxy,lst_nbf[i+1]))
#            comm.Recv(xtemp, source=i+1, tag=13)
#            xf[:,:,i*nf2:i*nf2+lst_nbf[i+1]]=xtemp[:,:,:lst_nbf[i+1]]
#    else:
#        comm.Send(xt,dest=0,tag=13)
    #comm.Barrier()
    comm.Gatherv(xt,[xf,sendcounts,displacements,MPI.DOUBLE],root=0)

    return xf

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

def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x


folder = 'data'
file_in = 'm31_3d_conv_10db'

folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))
CubeDirty = checkdim(fits.getdata(drtname, ext=0))

skyname = genname+'_sky.fits'
sky = checkdim(fits.getdata(skyname, ext=0))
sky = np.transpose(sky)
sky2 = np.sum(sky*sky)

# ==============================================================================
#
# ==============================================================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

nitermax=10
mu_s=0.5
mu_l=0.
nb=8
tau = 1e-4
sigma = 1

if rank==0:
    ud.tic()
    x=easy_muffin(CubePSF,CubeDirty,nitermax,nb,mu_s,mu_l,tau,sigma,[])
    ud.toc()

ud.tic()

x2=easy_muffin_mpi(CubePSF,CubeDirty,nitermax,nb,mu_s,mu_l,tau,sigma,[])
if rank==0:
    ud.toc()
    print('Error with Muffin: {:e}'.format(np.linalg.norm(x-x2)))
    
    pl.figure()
    pl.imshow(x[:,:,1])
    pl.figure()
    pl.imshow(x2[:,:,1])
    pl.show()


