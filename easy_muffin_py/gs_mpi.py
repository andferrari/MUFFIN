#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:44:12 2017

@author: rammanouil
"""

from deconv3d_tools import compute_tau_DWT
import deconv3d_mpi as dcvMpi
import numpy as np

def gs_mu_s(comm,rank,mu_s,mu_l,nb,sky,CubePSF,CubeDirty,var,
        mu_s_max,mu_s_min,mu_l_min,mu_l_max,absolutePrecision,thresh,nitermax=10,maxiter=100):
    
    args = {'mu_s':mu_s,'mu_l':mu_l,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,
        'mu_s_max':mu_s_max,'mu_s_min':mu_s_min,'mu_l_min':mu_l_min,'mu_l_max':mu_l_max,
        'absolutePrecision':absolutePrecision,'thresh':thresh}
    
    a = mu_s_min
    b = mu_s_max
    gr = (1+np.sqrt(5))/2
    c = b - (b - a)/gr
    d = a + (b - a)/gr
    niter = 0

    while abs(a - b) > absolutePrecision and niter < maxiter:

        EM= dcvMpi.EasyMuffinSURE(**args)
        if rank==0:
            EM.tau = compute_tau_DWT(EM.psf,EM.mu_s_max,EM.mu_l_max,EM.sigma,EM.nbw_decomp)
            print('using smallest (safest) tau: ',EM.tau)
            print('')
        EM.tau = comm.bcast(EM.tau,root=0)
        EM.mu_s = c
        EM.loop(nitermax)
        res1 = EM.wmse() 
        res1 = comm.bcast(res1,root=0)

        EM= dcvMpi.EasyMuffinSURE(**args)
        if rank==0:
            EM.tau = compute_tau_DWT(EM.psf,EM.mu_s_max,EM.mu_l_max,EM.sigma,EM.nbw_decomp)
            print('using smallest (safest) tau: ',EM.tau)
            print('')
        EM.tau = comm.bcast(EM.tau,root=0)
        EM.mu_s = d
        EM.loop(nitermax)
        res2 = EM.wmse()
        res2 = comm.bcast(res2,root=0)

        if res1 < res2:
            b = d
        else:
            a = c

        c = b - (b - a)/gr
        d = a + (b - a)/gr
        niter+=1

    return (a+b)/2


def gs_mu_l(comm,rank,mu_s,mu_l,nb,sky,CubePSF,CubeDirty,var,
        mu_s_max,mu_s_min,mu_l_min,mu_l_max,absolutePrecision,thresh,mu_s_gs,nitermax=10,maxiter=100):

    args = {'mu_s':mu_s,'mu_l':mu_l,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,
        'mu_s_max':mu_s_max,'mu_s_min':mu_s_min,'mu_l_min':mu_l_min,'mu_l_max':mu_l_max,
        'absolutePrecision':absolutePrecision,'thresh':thresh}
    
    a = mu_l_min
    b = mu_l_max
    gr = (1+np.sqrt(5))/2
    c = b - (b - a)/gr
    d = a + (b - a)/gr
    niter = 0
    
    while abs(a - b) > absolutePrecision and niter < maxiter:

        EM= dcvMpi.EasyMuffinSURE(**args)
        if rank==0:
            EM.tau = compute_tau_DWT(EM.psf,EM.mu_s_max,EM.mu_l_max,EM.sigma,EM.nbw_decomp)
            print('using smallest (safest) tau: ',EM.tau)
            print('')
        EM.tau = comm.bcast(EM.tau,root=0)
        EM.mu_s = mu_s_gs
        EM.loop(nitermax)
        EM.mu_l = c
        EM.loop(nitermax)
        res1 = EM.wmse() 
        res1 = comm.bcast(res1,root=0)

        EM= dcvMpi.EasyMuffinSURE(**args)
        if rank==0:
            EM.tau = compute_tau_DWT(EM.psf,EM.mu_s_max,EM.mu_l_max,EM.sigma,EM.nbw_decomp)
            print('using smallest (safest) tau: ',EM.tau)
            print('')
        EM.tau = comm.bcast(EM.tau,root=0)
        EM.mu_s = mu_s_gs
        EM.loop(nitermax)
        EM.mu_l = d
        EM.loop(nitermax)
        res2 = EM.wmse()
        res2 = comm.bcast(res2,root=0)

        if res1 < res2:
            b = d
        else:
            a = c

        c = b - (b - a)/gr
        d = a + (b - a)/gr
        niter+=1

    return (a+b)/2

    