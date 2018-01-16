#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:05:32 2017

@author: rammanouil
"""

import numpy as np
from scipy.fftpack import dct,idct

from deconv3d_tools import compute_tau_DWT, defadj, init_dirty_wiener, sat, Heavy, Rect
from deconv3d_tools import myfft2, myifft2, myifftshift, conv, optimal_split
from deconv3d_tools import iuwt_decomp, iuwt_decomp_adj, dwt_decomp, dwt_recomp
from mpi4py import MPI
import sys

str_cost="| {:5d} | {:6.6e} |"
str_cost_title="| {:5s} | {:12s} |\n"+"-"*24

str_cst_snr="| {:5d} | {:6.6e} | {:6.6e} |"
str_cst_snr_title="-"*39+"\n"+"| {:5s} | {:12s} | {:12s} |\n"+"-"*39

str_cost_wmsesure = "| {:5d} | {:6.6e} | {:6.6e} |"
str_cost_wmsesure_title = "-"*39+"\n"+"| {:5s} | {:12s} | {:12s} |\n"+"-"*39

str_cst_snr_wmse_wmsesure = "| {:5d} | {:6.6e} | {:6.6e} | {:6.6e} | {:6.6e} |"
str_cst_snr_wmse_wmsesure_title="-"*69+"\n"+"| {:5s} | {:12s} | {:12s} | {:12s} | {:12s} |\n"+"-"*69

str_cost_wmsesure_mu = "| {:5d} | {:6.6e} | {:6.6e} | {:6.6e} | {:6.6e} |"
str_cost_wmsesure_mu_title = "-"*69+"\n"+"| {:5s} | {:12s} | {:12s} | {:12s} | {:12s} |\n"+"-"*69

str_cst_snr_wmse_wmsesure_mu = "| {:5d} | {:6.6e} | {:6.6e} | {:6.6e} | {:6.6e} | {:6.6e} | {:6.6e} |"
str_cst_snr_wmse_wmsesure_mu_title="-"*99+"\n"+"| {:5s} | {:12s} | {:12s} | {:12s} | {:12s} | {:12s} | {:12s} |\n"+"-"*99

#Global variable -
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

nbw = size
idw = rank - 1

class EasyMuffin():
    def __init__(self,
                 mu_s=0.5,
                 mu_l=0.0,
                 mu_wiener = 5e1,
                 nb=(8,0),
                 tau = 1e-4,
                 sigma = 10,
                 var = 0,
                 dirtyinit=[],
                 dirty=[],
                 truesky=[],
                 psf=[]):

        if idw > nbw:
            comm.MPI_Finalize()

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
            print('tau must be non negative, tau=1e-4')
            tau=1e-4

        if sigma< 0 :
            print('sigma must be positive, sigma=10.')
            sigma=10.

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
        self.mu_wiener = mu_wiener

        self.nfreq = self.dirty.shape[2]
        self.nxy = self.dirty.shape[0]

        # Partitioning the frequency bands
        self.lst_nbf = optimal_split(self.nfreq,nbw)
        self.lst_nbf[0:0] = [self.nfreq]
        self.nf2 = self.lst_nbf[1:-1:1]
        self.nf2 = np.cumsum(self.nf2)
        self.nf2 = self.nf2.tolist()
        self.nf2[0:0] = [0]

        if rank ==0:
            print('')
            print(self.lst_nbf)
            print(self.nf2)

        nbsum = 0
        self.sendcounts = [0,]
        self.displacements = [0,]
        for i in range(nbw):
            self.displacements.append(nbsum)
            taille = self.nxy*self.nxy*self.lst_nbf[i+1]
            self.sendcounts.append(taille)
            nbsum+=taille

        if nbw > self.nfreq:
            if rank==0:
                print('----------------------------------------------------------------')
                print('   mpi: !!!! You cannoy have more workers than bands !!!!')
                print('----------------------------------------------------------------')
                sys.exit()
            else:
                sys.exit()

        np.random.seed(1)
        self.n = np.random.binomial(1,0.5,(self.nxy,self.nxy,self.nfreq))
        self.n[self.n==0] = -1

        self.init_algo()

    def init_algo(self):
        """Initialization of the algorithm (all intermediate variables)"""
        
        if rank ==0:
            
            print('psf size ', self.psf.shape)
            print('drt size ', self.dirty.shape)
            
            # precomputations
            print('')
            print('precomputations...')
            print('1')
            
            # x initialization
            self.xf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
            self.x = np.zeros((0))
            self.xt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
            
            self.vtt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
            self.v = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
            
            self.tf = np.zeros((self.nxy,self.nxy,self.nfreq),dtype=np.float,order='F')
            self.t = np.zeros((0))
            
            self.deltaf = np.zeros((self.nxy,self.nxy,self.nfreq),dtype=np.float,order='F')
            self.delta = np.zeros((0))
    
            if type(self.nb[0]) == int:
                self.Decomp = iuwt_decomp
                self.Recomp = iuwt_decomp_adj ### adjoint pas recomp
                self.nbw_decomp = [f for f in range(self.nb[0])]
                self.nbw_recomp = self.nb[-1]
                self.tau = compute_tau_DWT(self.psf,self.mu_s,self.mu_l,self.sigma,self.nbw_decomp)
                print('')
                print('IUWT: tau = ', self.tau)
                print('')
            else:
                self.Decomp = dwt_decomp
                self.Recomp = dwt_recomp
                self.nbw_decomp =self.nb
                self.nbw_recomp = self.nb
                self.tau = compute_tau_DWT(self.psf,self.mu_s,self.mu_l,self.sigma,self.nbw_decomp)
                print('')
                print('DWT: tau = ', self.tau)
                print('')
                
            # Compute spatial and spectral scaling parameters
            test = 0
            if test ==1:
                self.alpha_l = 1/(np.sum(self.dirty**2,2)+1e-1) # image
                self.alpha_l = conv(self.alpha_l,np.ones((3,3)),'max')
                self.alpha_l = self.alpha_l/self.alpha_l.max()
            else:
                self.alpha_l = np.ones((self.nxy,self.nxy))
            
        else:
            
            self.nfreq = self.lst_nbf[rank]
            self.psf = np.asfortranarray(self.psf[:,:,self.nf2[idw]:self.nf2[idw]+self.nfreq])
            self.dirty = np.asfortranarray(self.dirty[:,:,self.nf2[idw]:self.nf2[idw]+self.nfreq])
            
            if self.truesky is not None:
                self.truesky = self.truesky[:,:,self.nf2[idw]:self.nf2[idw]+self.nfreq]
            
            self.psfadj = defadj(self.psf)
            
            # x initialization
            if self.dirtyinit:
                self.x = np.asfortranarray(self.dirtyinit)
            else:
                self.x = np.asfortranarray(init_dirty_wiener(self.dirty, self.psf, self.psfadj, self.mu_wiener))
    
            # initializing alg. variables
            self.hth_fft = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.complex, order='F')
            self.fty = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.psfadj_fft = myfft2(self.psfadj)
            self.hth_fft = myfft2( myifftshift( myifft2( self.psfadj_fft * myfft2(self.psf) ) ) )
            tmp = myifftshift(myifft2(myfft2(self.dirty)*self.psfadj_fft))
            self.fty = tmp.real
            self.wstu = np.zeros((self.nxy,self.nxy), dtype=np.float, order='F')
            self.Delta_freq = np.zeros((self.nxy,self.nxy), dtype=np.float, order='F')
            self.xtt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.xt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.xf = np.zeros((0))
            self.t = np.zeros((self.nxy,self.nxy,self.nfreq),dtype=np.float,order='F')
            self.tf = np.zeros((0))
            self.delta = np.zeros((self.nxy,self.nxy,self.nfreq),dtype=np.float,order='F')
            self.deltaf = np.zeros((0))
    
            if type(self.nb[0]) == int:
                self.Decomp = iuwt_decomp
                self.Recomp = iuwt_decomp_adj ### adjoint pas recomp
                self.nbw_decomp = [f for f in range(self.nb[0])]
                self.nbw_recomp = self.nb[-1]
                #self.tau = compute_tau_DWT(self.psf,self.mu_s,self.mu_l,self.sigma,self.nbw_decomp)
                print('')
                print('IUWT: tau = ', self.tau)
                print('')
            else:
                self.Decomp = dwt_decomp
                self.Recomp = dwt_recomp
                self.nbw_decomp =self.nb
                self.nbw_recomp = self.nb
                #self.tau = compute_tau_DWT(self.psf,self.mu_s,self.mu_l,self.sigma,self.nbw_decomp)
                print('')
                print('DWT: tau = ', self.tau)
                print('')
                
            self.tau = 0
    
            self.utt = {}
            for freq in range(self.nfreq):
                self.utt[freq] = self.Decomp(np.zeros((self.nxy,self.nxy),order='F') , self.nbw_decomp)
            self.u = {}
            for freq in range(self.nfreq):
                self.u[freq] = self.Decomp(np.zeros((self.nxy,self.nxy),order='F') , self.nbw_decomp)
                
            # Compute spatial and spectral scaling parameters
            test = 0
            if test ==1:
                self.alpha_s = 1/(np.sum(np.sum(self.dirty**2,0),0)+1e-1) # col. vector
                self.alpha_s = self.alpha_s/self.alpha_s.max()
            else:
                self.alpha_s = np.ones(self.nfreq)

        self.tau = comm.bcast(self.tau,root=0) # root bcasts tau to everyone else 
        self.nitertot = 0
        
        self.costlist = []
        comm.Gatherv(self.x,[self.xf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        self.costlist.append(self.cost())
        
        # compute snr, psnr, wmse
        if self.truesky.any():
            self.snrlist = []
            self.wmselist = []
            self.psnrlist = []
            self.truesky2 = np.sum(self.truesky*self.truesky)
            self.psnrnum = np.sum((self.dirty-self.truesky)**2)/(self.nxy*self.nxy*self.nfreq)
            
            self.snrlist.append(self.snr())
            self.psnrlist.append(self.psnr())
            self.wmselist.append(self.wmse())
            if rank==0:
                print('The snr initialization is ',self.snrlist[0])
                print('')

    def cost(self):
        if not rank==0:
            """Compute cost for current iterate x"""
            tmp = self.dirty - myifftshift(myifft2(myfft2(self.x)*myfft2(self.psf)))
            LS_cst = 0.5*(np.linalg.norm(tmp)**2)
            tmp = 0.
            for freq in range(self.nfreq):
                tmp1 = self.Decomp(self.x[:,:,freq],self.nbw_decomp)
                for b in self.nbw_decomp:
                    tmp = tmp + np.sum(np.abs(tmp1[b]*self.alpha_s[freq]))
            Spt_cst = self.mu_s*tmp
            cst = Spt_cst + LS_cst
        else:
            Spc_cst = self.mu_l*np.sum(np.abs(dct(self.x*self.alpha_l[...,None],axis=2,norm='ortho')))
            cst = Spc_cst
            
        cst_list = comm.gather(cst)
        
        if rank==0:
            return sum(cst_list)/(self.nxy*self.nxy*self.nfreq)
        else:
            return cst

    def snr(self):
        if not rank==0:
            resid = self.truesky - self.x
            resid = np.float(np.sum(resid*resid))
        else:
            resid = 0
            
        rlist = comm.gather(resid)
        
        if rank==0:
            return 10*np.log10(self.truesky2 / np.sum(rlist))
        else:
            return resid

    def psnr(self):
        if not rank==0:
            resid = np.linalg.norm(conv(self.psf,self.truesky-self.x))**2
        else:
            resid = 0
            
        rlist = comm.gather(resid)
        
        if rank==0:
            return 10*np.log10(self.psnrnum / (np.sum(rlist)/(self.nxy*self.nxy*self.nfreq)))
        else:
            return resid

    def wmse(self):
        if not rank == 0:
            resid = np.linalg.norm(conv(self.psf,self.truesky-self.x))**2
        else:
            resid = 0
            
        rlist = comm.gather(resid)
        
        if rank==0:
            return np.sum(rlist)/(self.nxy*self.nxy*self.nfreq)
        else:
            return resid

    def mse(self):
        if not rank==0:
            resid = np.linalg.norm(self.truesky-self.x)**2
        else:
            resid = 0
            
        rlist = comm.gather(resid)
        
        if rank==0:
            return np.sum(rlist)/(self.nxy*self.nxy*self.nfreq)
        else:
            return resid

   #======================================================================
   # MAIN Iteration - EASY MUFFIN
   #======================================================================

    def update(self):
        
        if rank ==0:
            # rank 0 computes idct 
            self.tf = np.asfortranarray(idct(self.v, axis=2, norm='ortho')) # to check
            
        comm.Scatterv([self.tf,self.sendcounts,self.displacements,MPI.DOUBLE],self.t,root=0)

        if not rank==0:
            # compute gradient
            tmp = myifftshift( myifft2( myfft2(self.x) *self.hth_fft ) )
            Delta_freq = tmp.real- self.fty

            for freq in range(self.nfreq):
                # compute iuwt adjoint
                wstu = self.Recomp(self.u[freq], self.nbw_recomp)
                # compute xt
                self.xtt[:,:,freq] = self.x[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + self.mu_s*self.alpha_s[freq]*wstu + self.mu_l*self.alpha_l*self.t[:,:,freq])
                self.xt[:,:,freq] = np.maximum(self.xtt[:,:,freq], 0.0 )
                # update u
                tmp_spat_scal = self.Decomp(2*self.xt[:,:,freq] - self.x[:,:,freq] , self.nbw_decomp)
                for b in self.nbw_decomp:
                    self.utt[freq][b] = self.u[freq][b] + self.sigma*self.mu_s*self.alpha_s[freq]*tmp_spat_scal[b]
                    self.u[freq][b] = sat(self.utt[freq][b])
                    
            self.delta = np.asfortranarray(2*self.xt-self.x)

        comm.gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)

        if rank==0:
            # update v
            self.vtt = self.v + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(2*self.xt - self.x, axis=2, norm='ortho')
            self.v = sat(self.vtt)
            self.x = self.xt.copy(order='F')

        # compute cost snr, psnr, wmse if truesky given
        self.costlist.append(self.cost())
        if self.truesky is not None:
            self.snrlist.append(self.snr())
            self.psnrlist.append(self.psnr())
            self.wmselist.append(self.wmse())

   #======================================================================
   # MAIN PROGRAM - EASY MUFFIN
   #======================================================================

    def loop(self,nitermax=10):
        """ main loop """
        if nitermax< 1:
            if rank==0:
                print('nitermax must be a positive integer, nitermax=10')
            nitermax=10

        # Iterations
        for niter in range(nitermax):
            self.update()
            if rank==0:
                if self.truesky.any():
                    if (niter % 20) ==0:
                        print(str_cst_snr_title.format('It.','Cost','SNR'))
                    print(str_cst_snr.format(niter,self.costlist[-1],self.snrlist[-1]))
                else:
                    if (niter % 20) ==0:
                        print(str_cost_title.format('It.','Cost'))
                    print(str_cost.format(niter,self.costlist[-1]))


class EasyMuffinSURE(EasyMuffin):

    def __init__(self,
                 mu_s=0.5,
                 mu_l=0.0,
                 mu_wiener = 5e1,
                 nb=(8,0),
                 tau = 1e-4,
                 sigma = 10,
                 var = 0,
                 dirtyinit=[],
                 dirty=[],
                 truesky=[],
                 psf=[],
                 step_mu = [0,0]):

        super(EasyMuffinSURE,self).__init__(
                 mu_s,
                 mu_l,
                 mu_wiener,
                 nb,
                 tau,
                 sigma,
                 var,
                 dirtyinit,
                 dirty,
                 truesky,
                 psf)

        self.step_mu = step_mu


    def init_algo(self):

            super(EasyMuffinSURE,self).init_algo()

            # compute Hn
            self.Hn = np.zeros((self.nxy,self.nxy,self.nfreq))
#            np.random.seed(1)
#            self.n = np.random.binomial(1,0.5,(self.nxy,self.nxy,self.nfreq))
#            self.n[self.n==0] = -1
            self.Hn = conv(self.n,self.psfadj)

            # init Jacobians
            self.Jv = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.Jx = init_dirty_wiener(self.n, self.psf, self.psfadj, self.mu_wiener)
            self.Jxt = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.Ju = {}
            for freq in range(self.nfreq):
                self.Ju[freq] = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

            # psnr, and wmse estimated using psure
            self.wmselistsure = []
            self.wmselistsure.append(self.wmsesure())

            if self.truesky.any():
                self.psnrlistsure = []
                self.psnrlistsure.append(self.psnrsure())

            # mu_s list
            self.mu_slist = []
            self.mu_slist.append(self.mu_s)

            # mu_l list
            self.mu_llist = []
            self.mu_llist.append(self.mu_l)

            # fdmc variables
            self.sugarfdmclist = {}
            self.sugarfdmclist[0] = [0]
            self.sugarfdmclist[1] = [0]
            np.random.seed(1)
            self.delta = np.random.randn(self.nxy,self.nxy,self.nfreq)
            self.eps = 20*(self.var**0.5)*((self.nxy**2)**(-0.3)) # à verifier
            self.dirty2 = self.dirty + self.eps*self.delta

            self.xt2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float)
            self.u2 = {}
            for freq in range(self.nfreq):
                self.u2[freq] = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

            if self.dirtyinit:
                self.x2 = self.dirtyinit
            else:
                self.x2 = init_dirty_wiener(self.dirty2, self.psf, self.psfadj, self.mu_wiener)

            self.v2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float)

            self.fty2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float)
            tmp = myifftshift(myifft2(myfft2(self.dirty2)*self.psfadj_fft))
            self.fty2 = tmp.real

            self.xtt2 = self.xtt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.utt2 = {}
            for freq in range(self.nfreq):
                self.utt2[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), order='F') , self.nbw_decomp)
            self.vtt2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float)
            self.wmselistsurefdmc = []
            self.wmselistsurefdmc.append(self.wmsesurefdmc())

            self.dv_s = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.dx_s = init_dirty_wiener(self.n, self.psf, self.psfadj, self.mu_wiener)
            self.dxt_s = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.du_s = {}
            for freq in range(self.nfreq):
                self.du_s[freq] = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

            self.dv_l = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.dx_l = init_dirty_wiener(self.n, self.psf, self.psfadj, self.mu_wiener)
            self.dxt_l = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.du_l = {}
            for freq in range(self.nfreq):
                self.du_l[freq] = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

            self.dv2_s = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.dx2_s = init_dirty_wiener(self.n, self.psf, self.psfadj, self.mu_wiener)
            self.dxt2_s = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.du2_s = {}
            for freq in range(self.nfreq):
                self.du2_s[freq] = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

            self.dv2_l = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.dx2_l = init_dirty_wiener(self.n, self.psf, self.psfadj, self.mu_wiener)
            self.dxt2_l = np.zeros((self.nxy,self.nxy,self.nfreq))
            self.du2_l = {}
            for freq in range(self.nfreq):
                self.du2_l[freq] = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

    def wmsesure(self):
        tmp = self.dirty - conv(self.x,self.psf)
        LS_cst = np.linalg.norm(tmp)**2/(self.nxy*self.nxy*self.nfreq)
        tmp = self.n*conv(self.Jx,self.psf)
        tmp = 2*(self.var/(self.nxy*self.nxy*self.nfreq))*(np.sum(tmp))
        return LS_cst - self.var + tmp

    def wmsesurefdmc(self):
        tmp = self.dirty - conv(self.x,self.psf)
        LS_cst = np.linalg.norm(tmp)**2
        tmp = ((conv(self.x2,self.psf) - conv(self.x,self.psf))*self.delta)/self.eps
        return LS_cst/(self.nxy*self.nxy*self.nfreq) - self.var + 2*(self.var/(self.nxy*self.nxy*self.nfreq))*(np.sum(tmp))

    def psnrsure(self):
        return 10*np.log10(self.psnrnum/self.wmsesure())

    def update_Jacobians(self):
        Jt = idct(self.Jv, axis=2,norm='ortho')
        # compute gradient
        tmp = myifftshift( myifft2( myfft2(self.Jx) * self.hth_fft ) )
        JDelta_freq = tmp.real- self.Hn
        for freq in range(self.nfreq):
            # compute iuwt adjoint
            Js_l = self.Recomp(self.Ju[freq], self.nbw_recomp)
            # compute xt
            Jxtt = self.Jx[:,:,freq] - self.tau*(JDelta_freq[:,:,freq] + self.mu_s*self.alpha_s[freq]*Js_l + self.mu_l*self.alpha_l*Jt[:,:,freq])
            self.Jxt[:,:,freq] = Heavy(self.xtt[:,:,freq])*Jxtt
            # update u
            tmp_spat_scal_J = self.Decomp(2*self.Jxt[:,:,freq] - self.Jx[:,:,freq] , self.nbw_decomp)
            for b in self.nbw_decomp:
                Jutt = self.Ju[freq][b] + self.sigma*self.mu_s*self.alpha_s[freq]*tmp_spat_scal_J[b]
                self.Ju[freq][b] = Rect( self.utt[freq][b] )*Jutt
        # update v
        Jvtt = self.Jv + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(2*self.Jxt - self.Jx, axis=2, norm='ortho')
        self.Jv = Rect(self.vtt)*Jvtt
        self.Jx = self.Jxt.copy()
        # wmsesure
        self.wmselistsure.append(self.wmsesure())
        # psnrsure
        if self.truesky.any():
            self.psnrlistsure.append(self.psnrsure())

        return self.wmselistsure[-1]

    def loop(self,nitermax=10):
        """ main loop """

        if nitermax < 1:
            print('nitermax must be a positive integer, nitermax=10')
            nitermax=10
        for niter in range(nitermax):
            self.mu_slist.append(self.mu_s)
            self.mu_llist.append(self.mu_l)
            super(EasyMuffinSURE,self).update()
            self.update_Jacobians()
            self.nitertot+=1

            if self.truesky.any():
                if (niter % 20) ==0:
                    print(str_cst_snr_wmse_wmsesure_title.format('It.','Cost','SNR','WMSE','WMSES'))
                #print(str_cst_snr_wmse_wmsesure.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsure[-1]))
                print(str_cst_snr_wmse_wmsesure.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsure[-1]))
            else:
                if (niter % 20) ==0:
                    print(str_cost_wmsesure_title.format('It.','Cost','WMSES'))
                print(str_cost_wmsesure.format(niter,self.costlist[-1],self.wmselistsure[-1]))


    # run update with y + eps*delta
    def update2(self):
        t = idct(self.v2, axis=2, norm='ortho') # to check
        # compute gradient
        tmp = myifftshift( myifft2( myfft2(self.x2) *self.hth_fft ) )
        Delta_freq = tmp.real- self.fty2
        for freq in range(self.nfreq):
            # compute iuwt adjoint
            wstu = self.Recomp(self.u2[freq], self.nbw_recomp)
            # compute xt
            self.xtt2[:,:,freq] = self.x2[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + self.mu_s*self.alpha_s[freq]*wstu + self.mu_l*self.alpha_l*t[:,:,freq])
            self.xt2[:,:,freq] = np.maximum(self.xtt2[:,:,freq], 0.0 )
            # update u
            tmp_spat_scal = self.Decomp(2*self.xt2[:,:,freq] - self.x2[:,:,freq] , self.nbw_decomp)
            for b in self.nbw_decomp:
                self.utt2[freq][b] = self.u2[freq][b] + self.sigma*self.mu_s*self.alpha_s[freq]*tmp_spat_scal[b]
                self.u2[freq][b] = sat(self.utt2[freq][b])
        # update v
        self.vtt2 = self.v2 + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(2*self.xt2 - self.x2, axis=2, norm='ortho')
        self.v2 = sat(self.vtt2)
        self.x2 = self.xt2.copy()
        self.wmselistsurefdmc.append(self.wmsesurefdmc())

    def dx_mu(self):

        dt_s = idct(self.dv_s, axis=2, norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(self.dx_s) *self.hth_fft ) )
        Delta_freq = tmp.real #- self.fty
        for freq in range(self.nfreq):

            # compute iuwt adjoint
            wstu = self.Recomp(self.u[freq], self.nbw_recomp) + self.mu_s*self.alpha_s[freq]*self.Recomp(self.du_s[freq], self.nbw_recomp)

            # compute xt
            dxtt_s = self.dx_s[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + wstu + self.mu_l*self.alpha_l*dt_s[:,:,freq])
            self.dxt_s[:,:,freq] = Heavy(self.xtt[:,:,freq] )*dxtt_s

            # update u
            tmp_spat_scal = self.Decomp((2*self.xt[:,:,freq] - self.x[:,:,freq]) + self.mu_s*self.alpha_s[freq]*(2*self.dxt_s[:,:,freq] - self.dx_s[:,:,freq]), self.nbw_decomp)


            for b in self.nbw_decomp:
                dutt_s = self.du_s[freq][b] + self.sigma*tmp_spat_scal[b]
                self.du_s[freq][b] = Rect(self.utt[freq][b])*dutt_s

        # update v
        dvtt_s = self.dv_s + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(2*self.dxt_s - self.dx_s, axis=2, norm='ortho')
        self.dv_s = sat(self.vtt)*dvtt_s

        self.dx_s = self.dxt_s.copy()

        dt_l = idct(self.dv_l*self.mu_l*self.alpha_l[...,None] + self.v, axis=2, norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(self.dx_l) *self.hth_fft ) )
        Delta_freq = tmp.real #- self.fty
        for freq in range(self.nfreq):

            # compute iuwt adjoint
            wstu = self.mu_s*self.alpha_s[freq]*self.Recomp(self.du_l[freq], self.nbw_recomp)

            # compute xt
            dxtt_l = self.dx_l[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + wstu + dt_l[:,:,freq])
            self.dxt_l[:,:,freq] = Heavy(self.xtt[:,:,freq] )*dxtt_l

            # update u
            tmp_spat_scal = self.Decomp(self.mu_s*self.alpha_s[freq]*(2*self.dxt_l[:,:,freq] - self.dx_l[:,:,freq]), self.nbw_decomp)

            for b in self.nbw_decomp:
                dutt_l = self.du_l[freq][b] + self.sigma*tmp_spat_scal[b]
                self.du_l[freq][b] = Rect(self.utt[freq][b])*dutt_l

        # update v
        dvtt_l = self.dv_l + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(2*self.dxt_l - self.dx_l, axis=2, norm='ortho') + self.sigma*dct(2*self.xt - self.x, axis=2, norm='ortho')
        self.dv_l = sat(self.vtt)*dvtt_l

        self.dx_l = self.dxt_l.copy()


    def dx2_mu(self):
        dt_s = idct(self.dv2_s, axis=2, norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(self.dx2_s) *self.hth_fft ) )
        Delta_freq = tmp.real #- self.fty
        for freq in range(self.nfreq):

            # compute iuwt adjoint
            wstu = self.Recomp(self.u2[freq], self.nbw_recomp) + self.mu_s*self.alpha_s[freq]*self.Recomp(self.du2_s[freq], self.nbw_recomp)

            # compute xt
            dxtt_s = self.dx2_s[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + wstu + self.mu_l*self.alpha_l*dt_s[:,:,freq])
            self.dxt2_s[:,:,freq] = Heavy(self.xtt2[:,:,freq] )*dxtt_s

            # update u
            tmp_spat_scal = self.Decomp((2*self.xt2[:,:,freq] - self.x2[:,:,freq]) + self.mu_s*self.alpha_s[freq]*(2*self.dxt2_s[:,:,freq] - self.dx2_s[:,:,freq]), self.nbw_decomp)


            for b in self.nbw_decomp:
                dutt_s = self.du2_s[freq][b] + self.sigma*tmp_spat_scal[b]
                self.du2_s[freq][b] = Rect(self.utt2[freq][b])*dutt_s

        # update v
        dvtt2_s = self.dv2_s + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(2*self.dxt2_s - self.dx2_s, axis=2, norm='ortho')
        self.dv2_s = sat(self.vtt2)*dvtt2_s

        self.dx2_s = self.dxt2_s.copy()


        dt_l = idct(self.dv2_l*self.mu_l*self.alpha_l[...,None] + self.v2, axis=2, norm='ortho')

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(self.dx2_l) *self.hth_fft ) )
        Delta_freq = tmp.real #- self.fty
        for freq in range(self.nfreq):

            # compute iuwt adjoint
            wstu = self.mu_s*self.alpha_s[freq]*self.Recomp(self.du2_l[freq], self.nbw_recomp)

            # compute xt
            dxtt_l = self.dx2_l[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + wstu + dt_l[:,:,freq])
            self.dxt2_l[:,:,freq] = Heavy(self.xtt2[:,:,freq] )*dxtt_l

            # update u
            tmp_spat_scal = self.Decomp(self.mu_s*self.alpha_s[freq]*(2*self.dxt2_l[:,:,freq] - self.dx2_l[:,:,freq]), self.nbw_decomp)

            for b in self.nbw_decomp:
                dutt_l = self.du2_l[freq][b] + self.sigma*tmp_spat_scal[b]
                self.du2_l[freq][b] = Rect(self.utt2[freq][b])*dutt_l

        # update v
        dvtt2_l = self.dv2_l + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(2*self.dxt2_l - self.dx2_l, axis=2, norm='ortho') + self.sigma*dct(2*self.xt2 - self.x, axis=2, norm='ortho')
        self.dv2_l = sat(self.vtt2)*dvtt2_l

        self.dx2_l = self.dxt2_l.copy()


    def sugarfdmc(self):

        tmp = 2*conv(self.psf,self.dx_s)*(conv(self.psf,self.x)-self.dirty) + 2*self.var*conv(self.psf,self.dx2_s-self.dx_s)*self.delta/self.eps
        res1 = np.sum(tmp)/(self.nxy*self.nxy)

        tmp = 2*conv(self.psf,self.dx_l)*(conv(self.psf,self.x)-self.dirty) + 2*self.var*conv(self.psf,self.dx2_l-self.dx_l)*self.delta/self.eps
        res2 = np.sum(tmp)/(self.nxy*self.nxy)

        self.sugarfdmclist[0].append(res1)
        self.sugarfdmclist[1].append(res2)

    def loop_fdmc(self,nitermax=10):

        if nitermax < 1:
            print('nitermax must be a positve integer, nitermax=10')
            nitermax=10

        for niter in range(nitermax):
            self.mu_slist.append(self.mu_s)
            self.mu_llist.append(self.mu_l)
            super(EasyMuffinSURE,self).update()
            self.update_Jacobians()

            self.update2() #
            self.dx_mu() #
            self.dx2_mu() #
            self.sugarfdmc()

            if niter>1 and niter%10==0:
                self.GradDes_mu(self.step_mu)
                if niter>1000 and niter%100==0:
                    self.step_mu = [tmp/1.2 for tmp in self.step_mu]

            self.nitertot+=1

            if self.truesky.any():
                if (niter % 20) ==0:
                    print(str_cst_snr_wmse_wmsesure_mu_title.format('It.','Cost','SNR','WMSE','WMSES','mu_s','mu_l'))
                print(str_cst_snr_wmse_wmsesure_mu.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsurefdmc[-1],self.mu_slist[-1],self.mu_llist[-1]))
            else:
                if (niter % 20) ==0:
                    print(str_cost_wmsesure_mu_title.format('It.','Cost','WMSES','mu_s','mu_l'))
                print(str_cost_wmsesure_mu.format(niter,self.costlist[-1],self.wmselistsurefdmc[-1],self.mu_slist[-1],self.mu_llist[-1]))

    def GradDes_mu(self,step=[1e-3,1e-3]):
        self.mu_s = np.maximum(self.mu_s - step[0]*self.sugarfdmclist[0][-1],0)
        self.mu_l = np.maximum(self.mu_l - step[1]*self.sugarfdmclist[1][-1],0)
