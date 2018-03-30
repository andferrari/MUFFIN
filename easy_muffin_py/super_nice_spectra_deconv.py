#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:07:31 2016

@author: antonyschutz
"""
import numpy as np
from scipy.fftpack import dct,idct
import copy
from deconv3d_tools import compute_tau_DWT, defadj, init_dirty_wiener, sat, heavy, rect
from deconv3d_tools import myfft2, myifft2, myifftshift, conv
from deconv3d_tools import iuwt_decomp, iuwt_decomp_adj, dwt_decomp, dwt_recomp

class SNSD:
    def __init__(self,nitermax=10,
                 mu_s=0.5,
                 mu_l=0.0,
                 nb=(8,0),
                 tau = 1e-4,
                 sigma = 10,
                 var = 0,
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
        self.var = var

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

    def main(self,method='easy_muffin'):

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

        if method == 'easy_muffin':
            return easy_muffin(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,truesky)
        elif method == 'sure':
            var = self.var
            return easy_muffin_sure(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,var,truesky)
        elif method == 'gs':
            var = self.var
            return easy_muffin_sure_gs(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,var,truesky)
        elif method=='la':
            var = self.var
            return easy_muffin_sure_gs_la(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,var,truesky)
        elif method=='gs2':
            var = self.var
            return easy_muffin_sure_gs2(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,var,truesky)


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
        x = init_dirty_wiener(dirty, psf, psfadj, 5e1)

    if truesky.any():
        snr = np.zeros((nitermax+1))
        truesky2 = np.sum(truesky*truesky)

    cost = np.zeros((nitermax+1))
    psnr_ = np.zeros((nitermax+1))

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
        Recomp = iuwt_decomp_adj
        nbw_decomp = [f for f in range(nb[0])]
        nbw_recomp = nb[-1]

        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('IUWT: tau = ', tau)

    else:
        Decomp = dwt_decomp
        Recomp = dwt_recomp
        nbw_decomp = nb
        nbw_recomp = nb

        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
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
    cost[niter] = LS_cst + Spt_cst + Spc_cst

    # compute snr if truesky given
    if truesky.any():
        resid = truesky - x
        snr[niter]= 10*np.log10(truesky2 / np.sum(resid*resid))

    psnr_num = np.sum((dirty-truesky)**2)/(nxy*nxy*nfreq)

    # psnr_true
    psnr_[niter] = 10*np.log10(psnr_num/((np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)))

    v = np.zeros((nxy,nxy,nfreq), dtype=np.float)

    print('iteration: ',niter)

    while loop and niter<nitermax:

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
        cost[niter+1] = LS_cst + Spt_cst + Spc_cst

        psnr_[niter+1] = 10*np.log10(psnr_num/((np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)))

        # compute snr if truesky given
        if truesky.any():
            resid = truesky - x
            snr[niter+1]= 10*np.log10(truesky2 / np.sum(resid*resid))

        niter+=1

        print('iteration: ',niter,' ',mu_s,' ',mu_l)

    if truesky.any():
        return xt, cost, snr, psnr_
    else:
        return xt, cost

def easy_muffin_sure(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,var=0,truesky=None):

    psfadj = defadj(psf)

    print('')
    print('psf size ', psf.shape)
    print('drt size ', dirty.shape)

    nfreq = dirty.shape[2]
    nxy = dirty.shape[0]

    if dirtyinit:
        x = dirtyinit
    else:
        x = init_dirty_wiener(dirty, psf, psfadj, 5e1)

    if truesky.any():
        snr = np.zeros((nitermax+1))
        truesky2 = np.sum(truesky*truesky)

    cost = np.zeros((nitermax+1))
    wmse_true = np.zeros((nitermax+1))
    wmse_est = np.zeros((nitermax+1))
    psnr_true = np.zeros((nitermax+1))
    psnr_est = np.zeros((nitermax+1))
    psnr_num = np.sum((dirty-truesky)**2)/(nxy*nxy*nfreq)

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
    xt = np.zeros((nxy,nxy,nfreq), dtype=np.float)


    if type(nb[0]) == int:
        Decomp = iuwt_decomp
        Recomp = iuwt_decomp_adj
        nbw_decomp = [f for f in range(nb[0])]
        nbw_recomp = nb[-1]
        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('IUWT: tau = ', tau)

    else:
        Decomp = dwt_decomp
        Recomp = dwt_recomp
        nbw_decomp = nb
        nbw_recomp = nb
        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('DWT: tau = ', tau)

    u = {}
    for freq in range(nfreq):
        u[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)

    v = np.zeros((nxy,nxy,nfreq), dtype=np.float)

    # compute Hn
    Hn = np.zeros((nxy,nxy,nfreq))
    np.random.seed(1)
    n = np.random.binomial(1,0.5,(nxy,nxy,nfreq))
    n[n==0] = -1
    Hn = conv(n,psfadj) # wrong

    # init Jacobians
    Jv = np.zeros((nxy,nxy,nfreq))
    Jx = init_dirty_wiener(n, psf, psfadj, 5e1)
    Jxt = np.zeros((nxy,nxy,nfreq))
    Ju = {}
    for freq in range(nfreq):
        Ju[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)

    # Iteration
    print('')
    print("iterate...")

    loop = True
    niter = 0

    print('iteration: ',niter)

    #=================================
    # Initialise evaluation metrics
    #=================================

    # snr
    if truesky.any():
        resid = truesky - x
        snr[niter]= 10*np.log10(truesky2 / np.sum(resid*resid))

    # cost
    tmp = dirty - conv(x,psf)
    LS_cst = (np.linalg.norm(tmp)**2)
    tmp = 0.
    for freq in range(nfreq):
        tmp1 = Decomp(x[:,:,freq],nbw_decomp)
        for b in nbw_decomp:
            tmp = tmp + np.sum(np.abs(tmp1[b]))
    Spt_cst = mu_s*tmp
    Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
    cost[niter] = 0.5*LS_cst + Spt_cst + Spc_cst

    # wmse_true
    wmse_true[niter] = (np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)

    # wmse_est (wmse given by SURE)
    tmp = n*conv(Jx,psf)
    wmse_est[niter] = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

    # psnr_true
    psnr_true[niter] = 10*np.log10(psnr_num/wmse_true[niter])

    # psnr_est
    psnr_est[niter] = 10*np.log10(psnr_num/wmse_est[niter])


    while loop and niter<nitermax:

        #=================================
        # MUFFIN Alg.
        #=================================

        t = idct(v, axis=2, norm='ortho') # to check
        Jt = idct(Jv, axis=2,norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(x) * hth_fft ) )
        Delta_freq = tmp.real- fty
        tmp = myifftshift( myifft2( myfft2(Jx) * hth_fft ) )
        JDelta_freq = tmp.real- Hn

        for freq in range(nfreq):

            # compute iuwt adjoint
            wstu = Recomp(u[freq], nbw_recomp)
            Js_l = Recomp(Ju[freq], nbw_recomp)

            # compute xt
            xtt = x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu + mu_l*t[:,:,freq])
            xt[:,:,freq] = np.maximum(xtt, 0.0 )
            Jxtt = Jx[:,:,freq] - tau*(JDelta_freq[:,:,freq] + mu_s*Js_l + mu_l*Jt[:,:,freq])
            Jxt[:,:,freq] = heavy(xtt)*Jxtt

            # update u
            tmp_spat_scal = Decomp(2*xt[:,:,freq] - x[:,:,freq] , nbw_decomp)
            tmp_spat_scal_J = Decomp(2*Jxt[:,:,freq] - Jx[:,:,freq] , nbw_decomp)
            for b in nbw_decomp:
                utt = u[freq][b] + sigma*mu_s*tmp_spat_scal[b]
                u[freq][b] = sat( utt )
                Jutt = Ju[freq][b] + sigma*mu_s*tmp_spat_scal_J[b]
                Ju[freq][b] = rect( utt )*Jutt


        # update v
        vtt = v + sigma*mu_l*dct(2*xt - x, axis=2, norm='ortho')
        v = sat(vtt)
        Jvtt = Jv + sigma*mu_l*dct(2*Jxt - Jx, axis=2, norm='ortho')
        Jv = rect(vtt)*Jvtt

        x = xt.copy()
        Jx = Jxt.copy()

        #=================================
        # Compute evaluation metrics
        #=================================

        # snr
        if truesky.any():
            resid = truesky - x
            snr[niter+1]= 10*np.log10(truesky2 / np.sum(resid*resid))

        # cost
        tmp = dirty - conv(x,psf)
        LS_cst = (np.linalg.norm(tmp)**2)
        tmp = 0.
        for freq in range(nfreq):
            tmp1 = Decomp(x[:,:,freq],nbw_decomp)
            for b in nbw_decomp:
                tmp = tmp + np.sum(np.abs(tmp1[b]))
        Spt_cst = mu_s*tmp
        Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
        cost[niter+1] = 0.5*LS_cst + Spt_cst + Spc_cst

        # wmse_true
        wmse_true[niter+1] = (np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)

        # wmse_est (wmse given by SURE)
        tmp = n*conv(Jx,psf)
        wmse_est[niter+1] = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

        # psnr_true
        psnr_true[niter+1] = 10*np.log10(psnr_num/wmse_true[niter])

        # psnr_est
        psnr_est[niter+1] = 10*np.log10(psnr_num/wmse_est[niter])

        niter+=1
        print('iteration: ',niter)

    if truesky.any():
        return xt, cost, snr, psnr_true, psnr_est, wmse_true, wmse_est
    else:
        return xt, cost


def easy_muffin_sure_gs(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,var=0,truesky=None):

    psfadj = defadj(psf)

    print('')
    print('psf size ', psf.shape)
    print('drt size ', dirty.shape)

    nfreq = dirty.shape[2]
    nxy = dirty.shape[0]

    if dirtyinit:
        x = dirtyinit
    else:
        x = init_dirty_wiener(dirty, psf, psfadj, 5e1)

    if truesky.any():
        snr = np.zeros((nitermax+1))
        truesky2 = np.sum(truesky*truesky)

    cost = np.zeros((nitermax+1))
    wmse_true = np.zeros((nitermax+1))
    wmse_est = np.zeros((nitermax+1))
    psnr_true = np.zeros((nitermax+1))
    psnr_est = np.zeros((nitermax+1))
    psnr_num = np.sum((dirty-truesky)**2)/(nxy*nxy*nfreq)
    mu_s_ = np.zeros(nitermax)

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
    xt = np.zeros((nxy,nxy,nfreq), dtype=np.float)


    if type(nb[0]) == int:
        Decomp = iuwt_decomp
        Recomp = iuwt_decomp_adj
        nbw_decomp = [f for f in range(nb[0])]
        nbw_recomp = nb[-1]
        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('IUWT: tau = ', tau)

    else:
        Decomp = dwt_decomp
        Recomp = dwt_recomp
        nbw_decomp = nb
        nbw_recomp = nb
        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('DWT: tau = ', tau)

    u = {}
    for freq in range(nfreq):
        u[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)

    v = np.zeros((nxy,nxy,nfreq), dtype=np.float)

    # compute Hn
    Hn = np.zeros((nxy,nxy,nfreq))
    np.random.seed(1)
    n = np.random.binomial(1,0.5,(nxy,nxy,nfreq))
    n[n==0] = -1
    Hn = conv(n,psfadj)

    # init Jacobians
    Jv = np.zeros((nxy,nxy,nfreq))
    Jx = init_dirty_wiener(n, psf, psfadj, 5e1)
    Jxt = np.zeros((nxy,nxy,nfreq))
    Ju = {}
    for freq in range(nfreq):
        Ju[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)

    # Iteration
    print('')
    print("iterate...")

    loop = True
    niter = 0

    #=================================
    # initialise evaluation metrics
    #=================================

    # snr
    if truesky.any():
        resid = truesky - x
        snr[niter]= 10*np.log10(truesky2 / np.sum(resid*resid))

    # cost
    tmp = dirty - conv(x,psf)
    LS_cst = (np.linalg.norm(tmp)**2)
    tmp = 0.
    for freq in range(nfreq):
        tmp1 = Decomp(x[:,:,freq],nbw_decomp)
        for b in nbw_decomp:
            tmp = tmp + np.sum(np.abs(tmp1[b]))
    Spt_cst = mu_s*tmp
    Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
    cost[niter] = 0.5*LS_cst + Spt_cst + Spc_cst

    # wmse_true
    wmse_true[niter] = (np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)

    # wmse_est (wmse given by SURE)
    tmp = n*conv(Jx,psf)
    wmse_est[niter] = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

    # psnr_true
    psnr_true[niter] = 10*np.log10(psnr_num/wmse_true[niter])

    # psnr_est
    psnr_est[niter] = 10*np.log10(psnr_num/wmse_est[niter])

    tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)

    print('iteration: ',niter)

    while loop and niter<nitermax:

        #=================================
        # MUFFIN Alg.
        #=================================

        t = idct(v, axis=2, norm='ortho') # to check
        Jt = idct(Jv, axis=2,norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(x) * hth_fft ) )
        Delta_freq = tmp.real- fty
        tmp = myifftshift( myifft2( myfft2(Jx) * hth_fft ) )
        JDelta_freq = tmp.real- Hn

        # set m_s using gs (golden section search)
        args = (t,Jt,Delta_freq,JDelta_freq,u,Ju,x,tau,mu_l,Jx,xt,nbw_recomp,nbw_decomp,Recomp,Decomp,Jxt,sigma,v,Jv,dirty,psf,n,nxy,nfreq,var)
        mu_s = gs_search(One_MUFFIN_iter, a=0.5, b=1.0, args=args,absolutePrecision=1e-1,maxiter=100)
        mu_s_[niter] = mu_s
        print('mu_s',mu_s)


        for freq in range(nfreq):

            # compute iuwt adjoint
            wstu = Recomp(u[freq], nbw_recomp)
            Js_l = Recomp(Ju[freq], nbw_recomp)

            # compute xt
            xtt = x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu + mu_l*t[:,:,freq])
            xt[:,:,freq] = np.maximum(xtt, 0.0 )
            Jxtt = Jx[:,:,freq] - tau*(JDelta_freq[:,:,freq] + mu_s*Js_l + mu_l*Jt[:,:,freq])
            Jxt[:,:,freq] = heavy(xtt)*Jxtt

            # update u
            tmp_spat_scal = Decomp(2*xt[:,:,freq] - x[:,:,freq] , nbw_decomp)
            tmp_spat_scal_J = Decomp(2*Jxt[:,:,freq] - Jx[:,:,freq] , nbw_decomp)
            for b in nbw_decomp:
                utt = u[freq][b] + sigma*mu_s*tmp_spat_scal[b]
                u[freq][b] = sat( utt )
                Jutt = Ju[freq][b] + sigma*mu_s*tmp_spat_scal_J[b]
                Ju[freq][b] = rect( utt )*Jutt


        # update v
        vtt = v + sigma*mu_l*dct(2*xt - x, axis=2, norm='ortho')
        v = sat(vtt)
        Jvtt = Jv + sigma*mu_l*dct(2*Jxt - Jx, axis=2, norm='ortho')
        Jv = rect(vtt)*Jvtt

        x = xt.copy()
        Jx = Jxt.copy()

        #=================================
        # Compute evaluation metrics
        #=================================

        # snr
        if truesky.any():
            resid = truesky - x
            snr[niter+1]= 10*np.log10(truesky2 / np.sum(resid*resid))

        # cost
        tmp = dirty - conv(x,psf)
        LS_cst = (np.linalg.norm(tmp)**2)
        tmp = 0.
        for freq in range(nfreq):
            tmp1 = Decomp(x[:,:,freq],nbw_decomp)
            for b in nbw_decomp:
                tmp = tmp + np.sum(np.abs(tmp1[b]))
        Spt_cst = mu_s*tmp
        Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
        cost[niter+1] = 0.5*LS_cst + Spt_cst + Spc_cst

        # wmse_true
        wmse_true[niter+1] = (np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)

        # wmse_est (wmse given by SURE)
        tmp = n*conv(Jx,psf)
        wmse_est[niter+1] = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

        # psnr_true
        psnr_true[niter+1] = 10*np.log10(psnr_num/wmse_true[niter])

        # psnr_est
        psnr_est[niter+1] = 10*np.log10(psnr_num/wmse_est[niter])

        niter+=1
        print('iteration: ',niter)

    if truesky.any():
        return xt, cost, snr, psnr_true, psnr_est, wmse_true, wmse_est, mu_s_
    else:
        return xt, cost


def easy_muffin_sure_gs2(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,var=0,truesky=None):

    psfadj = defadj(psf)

    print('')
    print('psf size ', psf.shape)
    print('drt size ', dirty.shape)

    nfreq = dirty.shape[2]
    nxy = dirty.shape[0]

    if dirtyinit:
        x = dirtyinit
    else:
        x = init_dirty_wiener(dirty, psf, psfadj, 5e1)

    if truesky.any():
        snr = np.zeros((nitermax+1))
        truesky2 = np.sum(truesky*truesky)

    cost = np.zeros((nitermax+1))
    wmse_true = np.zeros((nitermax+1))
    wmse_est = np.zeros((nitermax+1))
    psnr_true = np.zeros((nitermax+1))
    psnr_est = np.zeros((nitermax+1))
    psnr_num = np.sum((dirty-truesky)**2)/(nxy*nxy*nfreq)
    mu_s_ = np.zeros(nitermax)
    mu_l_ = np.zeros(nitermax)

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
    xt = np.zeros((nxy,nxy,nfreq), dtype=np.float)


    if type(nb[0]) == int:
        Decomp = iuwt_decomp
        Recomp = iuwt_decomp_adj
        nbw_decomp = [f for f in range(nb[0])]
        nbw_recomp = nb[-1]
        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('IUWT: tau = ', tau)

    else:
        Decomp = dwt_decomp
        Recomp = dwt_recomp
        nbw_decomp = nb
        nbw_recomp = nb
        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('DWT: tau = ', tau)

    u = {}
    for freq in range(nfreq):
        u[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)

    v = np.zeros((nxy,nxy,nfreq), dtype=np.float)

    # compute Hn
    Hn = np.zeros((nxy,nxy,nfreq))
    np.random.seed(1)
    n = np.random.binomial(1,0.5,(nxy,nxy,nfreq))
    n[n==0] = -1
    Hn = conv(n,psfadj)

    # init Jacobians
    Jv = np.zeros((nxy,nxy,nfreq))
    Jx = init_dirty_wiener(n, psf, psfadj, 5e1)
    Jxt = np.zeros((nxy,nxy,nfreq))
    Ju = {}
    for freq in range(nfreq):
        Ju[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)

    # Iteration
    print('')
    print("iterate...")

    loop = True
    niter = 0

    #=================================
    # initialise evaluation metrics
    #=================================

    # snr
    if truesky.any():
        resid = truesky - x
        snr[niter]= 10*np.log10(truesky2 / np.sum(resid*resid))

    # cost
    tmp = dirty - conv(x,psf)
    LS_cst = (np.linalg.norm(tmp)**2)
    tmp = 0.
    for freq in range(nfreq):
        tmp1 = Decomp(x[:,:,freq],nbw_decomp)
        for b in nbw_decomp:
            tmp = tmp + np.sum(np.abs(tmp1[b]))
    Spt_cst = mu_s*tmp
    Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
    cost[niter] = 0.5*LS_cst + Spt_cst + Spc_cst

    # wmse_true
    wmse_true[niter] = (np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)

    # wmse_est (wmse given by SURE)
    tmp = n*conv(Jx,psf)
    wmse_est[niter] = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

    # psnr_true
    psnr_true[niter] = 10*np.log10(psnr_num/wmse_true[niter])

    # psnr_est
    psnr_est[niter] = 10*np.log10(psnr_num/wmse_est[niter])

    tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)

    print('iteration: ',niter)

    mu_l = 0

    while loop and niter<nitermax:

        #=================================
        # MUFFIN Alg.
        #=================================

        t = idct(v, axis=2, norm='ortho') # to check
        Jt = idct(Jv, axis=2,norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(x) * hth_fft ) )
        Delta_freq = tmp.real- fty
        tmp = myifftshift( myifft2( myfft2(Jx) * hth_fft ) )
        JDelta_freq = tmp.real- Hn

        # set m_s using gs (golden section search)
        if niter < nitermax/2:
            args = (t,Jt,Delta_freq,JDelta_freq,u,Ju,x,tau,mu_l,Jx,xt,nbw_recomp,nbw_decomp,Recomp,Decomp,Jxt,sigma,v,Jv,dirty,psf,n,nxy,nfreq,var)
            mu_s = gs_search(One_MUFFIN_iter, a=0.5, b=1.0, args=args,absolutePrecision=1e-1,maxiter=100)
        else:
            args = (t,Jt,Delta_freq,JDelta_freq,u,Ju,x,tau,mu_s,Jx,xt,nbw_recomp,nbw_decomp,Recomp,Decomp,Jxt,sigma,v,Jv,dirty,psf,n,nxy,nfreq,var)
            mu_l = gs_search(One_MUFFIN_iter_mu_l, a=0.5, b=3.0, args=args,absolutePrecision=1e-1,maxiter=100)

        mu_s_[niter] = mu_s
        mu_l_[niter] = mu_l
        print('mu_s',mu_s)


        for freq in range(nfreq):

            # compute iuwt adjoint
            wstu = Recomp(u[freq], nbw_recomp)
            Js_l = Recomp(Ju[freq], nbw_recomp)

            # compute xt
            xtt = x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu + mu_l*t[:,:,freq])
            xt[:,:,freq] = np.maximum(xtt, 0.0 )
            Jxtt = Jx[:,:,freq] - tau*(JDelta_freq[:,:,freq] + mu_s*Js_l + mu_l*Jt[:,:,freq])
            Jxt[:,:,freq] = heavy(xtt)*Jxtt

            # update u
            tmp_spat_scal = Decomp(2*xt[:,:,freq] - x[:,:,freq] , nbw_decomp)
            tmp_spat_scal_J = Decomp(2*Jxt[:,:,freq] - Jx[:,:,freq] , nbw_decomp)
            for b in nbw_decomp:
                utt = u[freq][b] + sigma*mu_s*tmp_spat_scal[b]
                u[freq][b] = sat( utt )
                Jutt = Ju[freq][b] + sigma*mu_s*tmp_spat_scal_J[b]
                Ju[freq][b] = rect( utt )*Jutt


        # update v
        vtt = v + sigma*mu_l*dct(2*xt - x, axis=2, norm='ortho')
        v = sat(vtt)
        Jvtt = Jv + sigma*mu_l*dct(2*Jxt - Jx, axis=2, norm='ortho')
        Jv = rect(vtt)*Jvtt

        x = xt.copy()
        Jx = Jxt.copy()

        #=================================
        # Compute evaluation metrics
        #=================================

        # snr
        if truesky.any():
            resid = truesky - x
            snr[niter+1]= 10*np.log10(truesky2 / np.sum(resid*resid))

        # cost
        tmp = dirty - conv(x,psf)
        LS_cst = (np.linalg.norm(tmp)**2)
        tmp = 0.
        for freq in range(nfreq):
            tmp1 = Decomp(x[:,:,freq],nbw_decomp)
            for b in nbw_decomp:
                tmp = tmp + np.sum(np.abs(tmp1[b]))
        Spt_cst = mu_s*tmp
        Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
        cost[niter+1] = 0.5*LS_cst + Spt_cst + Spc_cst

        # wmse_true
        wmse_true[niter+1] = (np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)

        # wmse_est (wmse given by SURE)
        tmp = n*conv(Jx,psf)
        wmse_est[niter+1] = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

        # psnr_true
        psnr_true[niter+1] = 10*np.log10(psnr_num/wmse_true[niter])

        # psnr_est
        psnr_est[niter+1] = 10*np.log10(psnr_num/wmse_est[niter])

        niter+=1
        print('iteration: ',niter)

    if truesky.any():
        return xt, cost, snr, psnr_true, psnr_est, wmse_true, wmse_est, mu_s_, mu_l_
    else:
        return xt, cost



def easy_muffin_sure_gs_la(psf,dirty,nitermax,nb,mu_s,mu_l,tau,sigma,dirtyinit,var=0,truesky=None):

    psfadj = defadj(psf)

    print('')
    print('psf size ', psf.shape)
    print('drt size ', dirty.shape)

    nfreq = dirty.shape[2]
    nxy = dirty.shape[0]

    if dirtyinit:
        x = dirtyinit
    else:
        x = init_dirty_wiener(dirty, psf, psfadj, 5e1)

    if truesky.any():
        snr = np.zeros((nitermax+1))
        truesky2 = np.sum(truesky*truesky)

    cost = np.zeros((nitermax+1))
    wmse_true = np.zeros((nitermax+1))
    wmse_est = np.zeros((nitermax+1))
    psnr_true = np.zeros((nitermax+1))
    psnr_est = np.zeros((nitermax+1))
    psnr_num = np.sum((dirty-truesky)**2)/(nxy*nxy*nfreq)
    mu_s_ = np.zeros(nitermax)

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
    xt = np.zeros((nxy,nxy,nfreq), dtype=np.float)


    if type(nb[0]) == int:
        Decomp = iuwt_decomp
        Recomp = iuwt_decomp_adj
        nbw_decomp = [f for f in range(nb[0])]
        nbw_recomp = nb[-1]
        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('IUWT: tau = ', tau)

    else:
        Decomp = dwt_decomp
        Recomp = dwt_recomp
        nbw_decomp = nb
        nbw_recomp = nb
        tau = compute_tau_DWT(psf,mu_s,mu_l,sigma,nbw_decomp)
        print('')
        print('DWT: tau = ', tau)

    u = {}
    for freq in range(nfreq):
        u[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)

    v = np.zeros((nxy,nxy,nfreq), dtype=np.float)

    # compute Hn
    Hn = np.zeros((nxy,nxy,nfreq))
    np.random.seed(1)
    n = np.random.binomial(1,0.5,(nxy,nxy,nfreq))
    n[n==0] = -1
    Hn = conv(n,psfadj)

    # init Jacobians
    Jv = np.zeros((nxy,nxy,nfreq))
    Jx = init_dirty_wiener(n, psf, psfadj, 5e1)
    Jxt = np.zeros((nxy,nxy,nfreq))
    Ju = {}
    for freq in range(nfreq):
        Ju[freq] = Decomp(np.zeros((nxy,nxy)) , nbw_decomp)

    # Iteration
    print('')
    print("iterate...")

    loop = True
    niter = 0

    #=================================
    # initialise evaluation metrics
    #=================================

    # snr
    if truesky.any():
        resid = truesky - x
        snr[niter]= 10*np.log10(truesky2 / np.sum(resid*resid))

    # cost
    tmp = dirty - conv(x,psf)
    LS_cst = (np.linalg.norm(tmp)**2)
    tmp = 0.
    for freq in range(nfreq):
        tmp1 = Decomp(x[:,:,freq],nbw_decomp)
        for b in nbw_decomp:
            tmp = tmp + np.sum(np.abs(tmp1[b]))
    Spt_cst = mu_s*tmp
    Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
    cost[niter] = 0.5*LS_cst + Spt_cst + Spc_cst

    # wmse_true
    wmse_true[niter] = (np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)

    # wmse_est (wmse given by SURE)
    tmp = n*conv(Jx,psf)
    wmse_est[niter] = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

    # psnr_true
    psnr_true[niter] = 10*np.log10(psnr_num/wmse_true[niter])

    # psnr_est
    psnr_est[niter] = 10*np.log10(psnr_num/wmse_est[niter])


    tau = compute_tau_DWT(psf,1,mu_l,sigma,nbw_decomp)

    print('iteration: ',niter)

    while loop and niter<nitermax:

        #=================================
        # MUFFIN Alg.
        #=================================

        # set m_s using gs (golden section search)
        if np.mod(niter,20)==0:
            args = (u,Ju,x,tau,mu_l,Jx,xt,nbw_recomp,nbw_decomp,Recomp,Decomp,Jxt,sigma,v,Jv,dirty,psf,n,nxy,nfreq,var,hth_fft,fty,Hn)
            mu_s = gs_search(One_MUFFIN_iter_la, a=0.5, b=1.0, args=args,absolutePrecision=1e-1,maxiter=100)

        mu_s_[niter] = mu_s

        t = idct(v, axis=2, norm='ortho') # to check
        Jt = idct(Jv, axis=2,norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(x) * hth_fft ) )
        Delta_freq = tmp.real- fty
        tmp = myifftshift( myifft2( myfft2(Jx) * hth_fft ) )
        JDelta_freq = tmp.real- Hn

        for freq in range(nfreq):

            # compute iuwt adjoint
            wstu = Recomp(u[freq], nbw_recomp)
            Js_l = Recomp(Ju[freq], nbw_recomp)

            # compute xt
            xtt = x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu + mu_l*t[:,:,freq])
            xt[:,:,freq] = np.maximum(xtt, 0.0 )
            Jxtt = Jx[:,:,freq] - tau*(JDelta_freq[:,:,freq] + mu_s*Js_l + mu_l*Jt[:,:,freq])
            Jxt[:,:,freq] = heavy(xtt)*Jxtt

            # update u
            tmp_spat_scal = Decomp(2*xt[:,:,freq] - x[:,:,freq] , nbw_decomp)
            tmp_spat_scal_J = Decomp(2*Jxt[:,:,freq] - Jx[:,:,freq] , nbw_decomp)
            for b in nbw_decomp:
                utt = u[freq][b] + sigma*mu_s*tmp_spat_scal[b]
                u[freq][b] = sat( utt )
                Jutt = Ju[freq][b] + sigma*mu_s*tmp_spat_scal_J[b]
                Ju[freq][b] = rect( utt )*Jutt


        # update v
        vtt = v + sigma*mu_l*dct(2*xt - x, axis=2, norm='ortho')
        v = sat(vtt)
        Jvtt = Jv + sigma*mu_l*dct(2*Jxt - Jx, axis=2, norm='ortho')
        Jv = Rect(vtt)*Jvtt

        x = xt.copy()
        Jx = Jxt.copy()

        #=================================
        # Compute evaluation metrics
        #=================================

        # snr
        if truesky.any():
            resid = truesky - x
            snr[niter+1]= 10*np.log10(truesky2 / np.sum(resid*resid))

        # cost
        tmp = dirty - conv(x,psf)
        LS_cst = (np.linalg.norm(tmp)**2)
        tmp = 0.
        for freq in range(nfreq):
            tmp1 = Decomp(x[:,:,freq],nbw_decomp)
            for b in nbw_decomp:
                tmp = tmp + np.sum(np.abs(tmp1[b]))
        Spt_cst = mu_s*tmp
        Spc_cst = mu_l*np.sum(np.abs(dct(x,axis=2,norm='ortho')))
        cost[niter+1] = 0.5*LS_cst + Spt_cst + Spc_cst

        # wmse_true
        wmse_true[niter+1] = (np.linalg.norm(conv(psf,truesky-x))**2)/(nxy*nxy*nfreq)

        # wmse_est (wmse given by SURE)
        tmp = n*conv(Jx,psf)
        wmse_est[niter+1] = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

        # psnr_true
        psnr_true[niter+1] = 10*np.log10(psnr_num/wmse_true[niter])

        # psnr_est
        psnr_est[niter+1] = 10*np.log10(psnr_num/wmse_est[niter])

        niter+=1
        print('iteration: ',niter)

    if truesky.any():
        return xt, cost, snr, psnr_true, psnr_est, wmse_true, wmse_est, mu_s_
    else:
        return xt, cost

#==============================================================================
# tools for golden section search
#==============================================================================

def gs_search(f, a, b, args=(),absolutePrecision=0.2,maxiter=100):

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

def One_MUFFIN_iter(mu_s,t,Jt,Delta_freq,JDelta_freq,u,Ju,x,tau,mu_l,Jx,xt,nbw_recomp,nbw_decomp,Recomp,Decomp,Jxt,sigma,v,Jv,dirty,psf,n,nxy,nfreq,var):

    xt_ = xt.copy()
    Jxt_ = Jxt.copy()
    u_ = copy.deepcopy(u)
    Ju_ = copy.deepcopy(Ju)


    for freq in range(nfreq):

            # compute iuwt adjoint
            wstu_ = Recomp(u_[freq], nbw_recomp)
            Js_l_ = Recomp(Ju_[freq], nbw_recomp)

            # compute xt
            xtt_ = x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu_ + mu_l*t[:,:,freq])
            xt_[:,:,freq] = np.maximum(xtt_, 0.0 ) ###
            Jxtt_ = Jx[:,:,freq] - tau*(JDelta_freq[:,:,freq] + mu_s*Js_l_ + mu_l*Jt[:,:,freq])
            Jxt_[:,:,freq] = heavy(xtt_)*Jxtt_ ##

            # update u
            tmp_spat_scal_ = Decomp(2*xt_[:,:,freq] - x[:,:,freq] , nbw_decomp)
            tmp_spat_scal_J_ = Decomp(2*Jxt_[:,:,freq] - Jx[:,:,freq] , nbw_decomp)
            for b in nbw_decomp:
                 utt_ = u_[freq][b] + sigma*mu_s*tmp_spat_scal_[b]
                 u_[freq][b] = sat( utt_ ) ##
                 Jutt_ = Ju_[freq][b] + sigma*mu_s*tmp_spat_scal_J_[b]
                 Ju_[freq][b] = rect( utt_ )*Jutt_ ##


    # update v
    vtt_ = v + sigma*mu_l*dct(2*xt_ - x, axis=2, norm='ortho')
    v = sat(vtt_)
    Jvtt_ = Jv + sigma*mu_l*dct(2*Jxt_ - Jx, axis=2, norm='ortho')
    Jv = rect(vtt_)*Jvtt_

    # wmse_est (wmse given by SURE)
    tmp = dirty - conv(xt_,psf)
    LS_cst = (np.linalg.norm(tmp)**2)
    tmp = n*conv(Jxt_,psf)
    wmse_est_ = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

    #print('(mu_s,wmse)=',mu_s,' ',wmse_est_)
    return wmse_est_


def One_MUFFIN_iter_mu_l(mu_l,t,Jt,Delta_freq,JDelta_freq,u,Ju,x,tau,mu_s,Jx,xt,nbw_recomp,nbw_decomp,Recomp,Decomp,Jxt,sigma,v,Jv,dirty,psf,n,nxy,nfreq,var):

    xt_ = xt.copy()
    Jxt_ = Jxt.copy()
    u_ = copy.deepcopy(u)
    Ju_ = copy.deepcopy(Ju)


    for freq in range(nfreq):

            # compute iuwt adjoint
            wstu_ = Recomp(u_[freq], nbw_recomp)
            Js_l_ = Recomp(Ju_[freq], nbw_recomp)

            # compute xt
            xtt_ = x[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu_ + mu_l*t[:,:,freq])
            xt_[:,:,freq] = np.maximum(xtt_, 0.0 ) ###
            Jxtt_ = Jx[:,:,freq] - tau*(JDelta_freq[:,:,freq] + mu_s*Js_l_ + mu_l*Jt[:,:,freq])
            Jxt_[:,:,freq] = heavy(xtt_)*Jxtt_ ##

            # update u
            tmp_spat_scal_ = Decomp(2*xt_[:,:,freq] - x[:,:,freq] , nbw_decomp)
            tmp_spat_scal_J_ = Decomp(2*Jxt_[:,:,freq] - Jx[:,:,freq] , nbw_decomp)
            for b in nbw_decomp:
                 utt_ = u_[freq][b] + sigma*mu_s*tmp_spat_scal_[b]
                 u_[freq][b] = sat( utt_ ) ##
                 Jutt_ = Ju_[freq][b] + sigma*mu_s*tmp_spat_scal_J_[b]
                 Ju_[freq][b] = rect( utt_ )*Jutt_ ##


    # update v
    vtt_ = v + sigma*mu_l*dct(2*xt_ - x, axis=2, norm='ortho')
    v = sat(vtt_)
    Jvtt_ = Jv + sigma*mu_l*dct(2*Jxt_ - Jx, axis=2, norm='ortho')
    Jv = rect(vtt_)*Jvtt_

    # wmse_est (wmse given by SURE)
    tmp = dirty - conv(xt_,psf)
    LS_cst = (np.linalg.norm(tmp)**2)
    tmp = n*conv(Jxt_,psf)
    wmse_est_ = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

    #print('(mu_s,wmse)=',mu_s,' ',wmse_est_)
    return wmse_est_

#==============================================================================
# tools for golden section search + Look ahead
#==============================================================================

def One_MUFFIN_iter_la(mu_s,u,Ju,x,tau,mu_l,Jx,xt,nbw_recomp,nbw_decomp,Recomp,Decomp,Jxt,sigma,v,Jv,dirty,psf,n,nxy,nfreq,var,hth_fft,fty,Hn):

    xt_ = xt.copy()
    Jxt_ = Jxt.copy()
    u_ = copy.deepcopy(u)
    Ju_ = copy.deepcopy(Ju)
    x_ = x.copy()
    Jx_ = Jx.copy()
    v_ = v.copy()
    Jv_ = Jv.copy()

    for i in range(10):

        t = idct(v_, axis=2, norm='ortho') # to check
        Jt = idct(Jv_, axis=2,norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(x_) * hth_fft ) )
        Delta_freq = tmp.real- fty
        tmp = myifftshift( myifft2( myfft2(Jx_) * hth_fft ) )
        JDelta_freq = tmp.real- Hn

        for freq in range(nfreq):

                # compute iuwt adjoint
                wstu = Recomp(u_[freq], nbw_recomp)
                Js_l = Recomp(Ju_[freq], nbw_recomp)

                # compute xt
                xtt = x_[:,:,freq] - tau*(Delta_freq[:,:,freq] + mu_s*wstu + mu_l*t[:,:,freq])
                xt_[:,:,freq] = np.maximum(xtt, 0.0 ) ###
                Jxtt = Jx_[:,:,freq] - tau*(JDelta_freq[:,:,freq] + mu_s*Js_l + mu_l*Jt[:,:,freq])
                Jxt_[:,:,freq] = heavy(xtt)*Jxtt ##

                # update u
                tmp_spat_scal = Decomp(2*xt_[:,:,freq] - x_[:,:,freq] , nbw_decomp)
                tmp_spat_scal_J = Decomp(2*Jxt_[:,:,freq] - Jx_[:,:,freq] , nbw_decomp)
                for b in nbw_decomp:
                    utt = u_[freq][b] + sigma*mu_s*tmp_spat_scal[b]
                    u_[freq][b] = sat( utt ) ##
                    Jutt = Ju_[freq][b] + sigma*mu_s*tmp_spat_scal_J[b]
                    Ju_[freq][b] = rect( utt )*Jutt ##


        # update v
        vtt = v_ + sigma*mu_l*dct(2*xt_ - x_, axis=2, norm='ortho')
        v_ = sat(vtt)
        Jvtt = Jv_ + sigma*mu_l*dct(2*Jxt_ - Jx_, axis=2, norm='ortho')
        Jv_ = rect(vtt)*Jvtt

        x_ = xt_.copy()
        Jx_ = Jxt_.copy()

        # wmse_est (wmse given by SURE)
        tmp = dirty - conv(xt_,psf)
        LS_cst = (np.linalg.norm(tmp)**2)
        tmp = n*conv(Jxt_,psf)
        wmse_est = LS_cst/(nxy*nxy*nfreq) - var + 2*(var/(nxy*nxy*nfreq))*(np.sum(tmp))

        #print('(mu_s,wmse)=',mu_s,' ',wmse_est)

    return wmse_est
