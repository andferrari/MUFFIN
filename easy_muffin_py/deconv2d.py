#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:03:34 2017

@author: rammanouil
"""

import numpy as np

from deconv3d_tools import compute_tau_2D, defadj, init_dirty_wiener, sat, Heavy, Rect
from deconv3d_tools import myfft2, myifft2, myifftshift, conv
from deconv3d_tools import iuwt_decomp, iuwt_recomp, dwt_decomp, dwt_recomp
import copy


str_cost="| {:5d} | {:6.6e} |"
str_cost_title="| {:5s} | {:12s} |\n"+"-"*24

str_cst_snr="| {:5d} | {:6.6e} | {:6.6e} |"
str_cst_snr_title="-"*39+"\n"+"| {:5s} | {:12s} | {:12s} |\n"+"-"*39

str_cost_wmsesure = "| {:5d} | {:6.6e} | {:6.6e} |"
str_cost_wmsesure_title = "-"*39+"\n"+"| {:5s} | {:12s} | {:12s} |\n"+"-"*39

str_cst_snr_wmse_wmsesure = "| {:5d} | {:6.6e} | {:6.6e} | {:6.6e} | {:6.6e} |"
str_cst_snr_wmse_wmsesure_title="-"*69+"\n"+"| {:5s} | {:12s} | {:12s} | {:12s} | {:12s}  |\n"+"-"*69
                  
str_cost_wmsesure_mu = "| {:5d} | {:6.6e} | {:6.6e} | {:6.6e} |"
str_cost_wmsesure_mu_title = "-"*54+"\n"+"| {:5s} | {:12s} | {:12s} | {:12s} | {:12s} |\n"+"-"*54

str_cst_snr_wmse_wmsesure_mu = "| {:5d} | {:6.6e} | {:6.6e} | {:6.6e} | {:6.6e} | {:6.6e} |"
str_cst_snr_wmse_wmsesure_mu_title="-"*84+"\n"+"| {:5s} | {:12s} | {:12s} | {:12s} | {:12s} | {:12s} |\n"+"-"*84

                                      
class EasyMuffin():
    def __init__(self,
                 mu_s=0.5,
                 nb=(8,0),
                 tau = 1e-4,
                 sigma = 10,
                 var = 0,
                 dirtyinit=[],
                 dirty=[],
                 truesky=[],
                 psf=[],
                 mu_s_min=0,
                 mu_s_max=1,
                 absolutePrecision=0.1,
                 thresh=1e-4):


        if type(nb) is not tuple:
            print('nb must be a tuple of wavelets for dwt ')
            print('or a list of 2 integer for IUWT')
            print('first integer is the scale of decomposition (default:8)')
            print('second integer the scale or recomposition (default:0)')
            nb = (8,0)

        if mu_s< 0 :
            print('mu_s must be non negative, mu_s=.5')
            mu_s=0.5

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
        self.sigma = sigma
        self.tau = tau
        self.dirtyinit = dirtyinit
        self.truesky = truesky
        self.psf = psf
        self.dirty=dirty
        self.var = var

        self.mu_s_min = mu_s_min
        self.mu_s_max = mu_s_max
        self.absolutePrecision = absolutePrecision
        
        self.init_algo()

    def init_algo(self):
        """Initialization of te algorithm (all intermediate variables)"""

        self.psfadj = defadj(self.psf)

        print('psf size ', self.psf.shape)
        print('drt size ', self.dirty.shape)

        # precomputations
        print('')
        print("precomputations...")
        
        self.nxy = self.dirty.shape[0]

        if self.dirtyinit:
            self.x = self.dirtyinit
        else:
            self.x = init_dirty_wiener(self.dirty, self.psf, self.psfadj, 5e1)

        self.hth_fft = np.zeros((self.nxy,self.nxy), dtype=np.complex)
        self.fty = np.zeros((self.nxy,self.nxy), dtype=np.float)

        self.psfadj_fft = myfft2(self.psfadj)
        self.hth_fft = myfft2( myifftshift( myifft2( self.psfadj_fft * myfft2(self.psf) ) ) )
        tmp = myifftshift(myifft2(myfft2(self.dirty)*self.psfadj_fft))
        self.fty = tmp.real

        self.wstu = np.zeros((self.nxy,self.nxy), dtype=np.float)
        self.Delta_freq = np.zeros((self.nxy,self.nxy), dtype=np.float)
        self.xtt = np.zeros((self.nxy,self.nxy), dtype=np.float)
        self.xt = np.zeros((self.nxy,self.nxy), dtype=np.float)

        if type(self.nb[0]) == int:
            self.Decomp = iuwt_decomp
            self.Recomp = iuwt_recomp
            self.nbw_decomp = [f for f in range(self.nb[0])]
            self.nbw_recomp = self.nb[-1]

            print('')
            print('IUWT: tau = ', self.tau)
            print('')

        else:
            self.Decomp = dwt_decomp
            self.Recomp = dwt_recomp
            self.nbw_decomp =self.nb
            self.nbw_recomp = self.nb

            self.tau = compute_tau_2D(self.psf,self.mu_s,self.sigma,self.nbw_decomp)
            print('')
            print('DWT: tau = ', self.tau)
            print('')


        self.utt = {}
        self.utt = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

        self.u = {}
        self.u = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

        self.nitertot = 0

        # compute cost
        self.costlist = []
        self.costlist.append(self.cost())

        # snr, psnr, wmse
        if self.truesky.any():
            self.snrlist = []
            self.truesky2 = np.sum(self.truesky*self.truesky)
            self.wmselist = []
            self.psnrlist = []
            self.psnrnum = np.sum((self.dirty-self.truesky)**2)/(self.nxy*self.nxy)

        # compute snr if truesky given
        if self.truesky.any():
            self.snrlist.append(self.snr())
            self.psnrlist.append(self.psnr())
            self.wmselist.append(self.wmse())


    def cost(self):
        """Compute cost for current iterate x"""
        tmp = self.dirty - myifftshift(myifft2(myfft2(self.x)*myfft2(self.psf)))
        LS_cst = 0.5*(np.linalg.norm(tmp)**2)
        
        tmp = 0.
        tmp1 = self.Decomp(self.x[:,:],self.nbw_decomp)
        for b in self.nbw_decomp:
            tmp = tmp + np.sum(np.abs(tmp1[b]))
            
        Spt_cst = self.mu_s*tmp
        
        return (LS_cst + Spt_cst)/(self.nxy*self.nxy)


    def snr(self,change=True):
        if change:
            resid = self.truesky - self.x
            return 10*np.log10(self.truesky2 / np.sum(resid*resid))
        else:
            resid = self.truesky - self.x2_
            return 10*np.log10(self.truesky2 / np.sum(resid*resid))


    def psnr(self):
        resid = (np.linalg.norm(conv(self.psf,self.truesky-self.x))**2)/(self.nxy*self.nxy)
        return 10*np.log10(self.psnrnum / resid)

    def wmse(self,change=True):
        if change:
            return (np.linalg.norm(conv(self.psf,self.truesky-self.x))**2)/(self.nxy*self.nxy)
        else:
            return (np.linalg.norm(conv(self.psf,self.truesky-self.x2_))**2)/(self.nxy*self.nxy)

    def mse(self):
        return (np.linalg.norm(self.truesky-self.x)**2)/(self.nxy*self.nxy)

    def update(self,change=True):
        
        if change:
            xt_ = self.xt # xt_ is a pointer
            u_ = self.u
            x_ = self.x
        else:
            xt_ = self.xt.copy() # xt is a new local matrix
            u_  = copy.deepcopy(self.u)
            x_  = self.x.copy()            

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(x_) *self.hth_fft ) )
        Delta_freq = tmp.real- self.fty

        # compute iuwt adjoint
        wstu = self.Recomp(u_, self.nbw_recomp)
  
        # compute xt
        self.xtt = x_ - self.tau*(Delta_freq + self.mu_s*wstu )
        xt_ = np.maximum(self.xtt, 0.0 )
        #print('x',np.linalg.norm(xt_))
        
        # update u
        tmp_spat_scal = self.Decomp(2*xt_ - x_ , self.nbw_decomp)

        for b in self.nbw_decomp:
            self.utt[b] = u_[b] + self.sigma*self.mu_s*tmp_spat_scal[b]
            u_[b] = sat(self.utt[b])

        if change:
            self.x = xt_.copy()
            # compute cost
            self.costlist.append(self.cost())
            # compute snr, psnr, wmse if truesky given
            if self.truesky.any():
                self.snrlist.append(self.snr())
                self.psnrlist.append(self.psnr())
                self.wmselist.append(self.wmse())
        else:
            self.x2_ = xt_.copy()


   #======================================================================
   # MAIN PROGRAM - EASY MUFFIN
   #======================================================================

    def loop(self,nitermax=10):
        """ main loop """

        if nitermax< 1:
            print('nitermax must be a positive integer, nitermax=10')
            nitermax=10

        # Iteration

        for niter in range(nitermax):

            self.update()
            
            if self.truesky.any():
                if (niter % 20) ==0:
                    print(str_cst_snr_title.format('It.','Cost','SNR'))
                    
                print(str_cst_snr.format(niter,self.costlist[-1],self.snrlist[-1]))
            else:
                if (niter % 20) ==0:
                    print(str_cost_title.format('It.','Cost'))
                    
                print(str_cost.format(niter,self.costlist[-1]))


    def gs_mu_s(self,nitermax=10,a=0,b=2,absolutePrecision=1e-1,maxiter=100):

        gr = (1+np.sqrt(5))/2
        c = b - (b - a)/gr
        d = a + (b - a)/gr
        niter = 0

        self.mu_s_lst = []
        self.mse_lst = []

        while abs(a - b) > absolutePrecision and niter < maxiter:

            self.mu_s = c
            self.init_algo()
            self.mu_s_lst.append(c)
            self.loop(nitermax)
            res1 = self.wmse()
            self.mse_lst.append(res1)

            self.mu_s = d
            self.init_algo()
            self.mu_s_lst.append(d)
            self.loop(nitermax)
            res2 = self.wmse()
            self.mse_lst.append(res2)

            # if f( *((c,) + args) ) < f( *((d,) + args) ):
            if res1 < res2:
                b = d
            else:
                a = c

            c = b - (b - a)/gr
            d = a + (b - a)/gr
            niter+=1

        return (a+b)/2


class EasyMuffinSURE(EasyMuffin):

    def __init__(self,
                 mu_s=0.5,
                 nb=(8,0),
                 tau = 1e-4,
                 sigma = 10,
                 var = 0,
                 dirtyinit=[],
                 dirty=[],
                 truesky=[],
                 psf=[],
                 mu_s_min=0,
                 mu_s_max=1,
                 absolutePrecision=0.1,
                 thresh=1e-4):

        # stopping criteria for loop_mu_s/_mu_l
        self.thresh = thresh
        
        super(EasyMuffinSURE,self).__init__(
                 mu_s,
                 nb,
                 tau,
                 sigma,
                 var,
                 dirtyinit,
                 dirty,
                 truesky,
                 psf,
                 mu_s_min,
                 mu_s_max,
                 absolutePrecision) 


    def init_algo(self):

            super(EasyMuffinSURE,self).init_algo()

            # compute Hn
            self.Hn = np.zeros((self.nxy,self.nxy))
            np.random.seed(1)
            self.n = np.random.binomial(1,0.5,(self.nxy,self.nxy))
            self.n[self.n==0] = -1
            self.Hn = conv(self.n,self.psfadj)

            # init Jacobians
            self.Jx = init_dirty_wiener(self.n, self.psf, self.psfadj, 5e1)
            self.Jxt = np.zeros((self.nxy,self.nxy))
            self.Ju = {}
            self.Ju = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)

            # psnr, and wmse estimated using psure
            self.wmselistsure = []
            self.wmselistsure.append(self.wmsesure())

            if self.truesky.any():
                self.psnrlistsure = []
                self.psnrlistsure.append(self.psnrsure())

            # mu_s list
            self.mu_slist = []
            self.mu_slist.append(self.mu_s)
            

    def wmsesure(self,change=True):

        if change:
            tmp = self.dirty - conv(self.x,self.psf)
            LS_cst = np.linalg.norm(tmp)**2
            tmp = self.n*conv(self.Jx,self.psf)

            return LS_cst/(self.nxy*self.nxy) - self.var + 2*(self.var/(self.nxy*self.nxy))*(np.sum(tmp))
        else:
            tmp = self.dirty - conv(self.x2_,self.psf)
            LS_cst = np.linalg.norm(tmp)**2
            tmp = self.n*conv(self.Jx2_,self.psf)

            return LS_cst/(self.nxy*self.nxy) - self.var + 2*(self.var/(self.nxy*self.nxy))*(np.sum(tmp))

    def psnrsure(self):

        return 10*np.log10(self.psnrnum/self.wmsesure())


    def update_Jacobians(self,change=True):

        if change:
            Jxt_ = self.Jxt # pointer
            Ju_ = self.Ju
            Jx_ = self.Jx
            
        else:
            Jxt_ = self.Jxt.copy() # new matrix
            Ju_  = copy.deepcopy(self.Ju)
            Jx_  = self.Jx.copy()
            

        # compute gradient
        tmp = myifftshift( myifft2( myfft2(Jx_) * self.hth_fft ) )
        JDelta_freq = tmp.real- self.Hn

        # compute iuwt adjoint
        Js_l = self.Recomp(Ju_, self.nbw_recomp)

        # compute xt
        Jxtt = Jx_ - self.tau*(JDelta_freq + self.mu_s*Js_l)
        Jxt_ = Heavy(self.xtt)*Jxtt

        # update u
        tmp_spat_scal_J = self.Decomp(2*Jxt_[:,:] - Jx_[:,:] , self.nbw_decomp)
        for b in self.nbw_decomp:
            Jutt = Ju_[b] + self.sigma*self.mu_s*tmp_spat_scal_J[b]
            Ju_[b] = Rect( self.utt[b] )*Jutt

        if change:
            self.Jx = Jxt_.copy()

            # wmsesure
            self.wmselistsure.append(self.wmsesure())

            # psnrsure
            if self.truesky.any():
                self.psnrlistsure.append(self.psnrsure())
        else:
            self.Jx2_ = Jxt_.copy()

            return self.wmsesure(change=False)


    def loop(self,nitermax=10,change=True):
        """ main loop """

        if nitermax < 1:
            print('nitermax must be a positive integer, nitermax=10')
            nitermax=10

        for niter in range(nitermax):
            self.mu_slist.append(self.mu_s)
            super(EasyMuffinSURE,self).update(change)
            self.update_Jacobians(change)
            self.nitertot+=1

            if self.truesky.any():
                if (niter % 20) ==0:
                    print(str_cst_snr_wmse_wmsesure_title.format('It.','Cost','SNR','WMSE','WMSES'))                    
                print(str_cst_snr_wmse_wmsesure.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsure[-1]))
            else:
                if (niter % 20) ==0:
                    print(str_cost_wmsesure_title.format('It.','Cost','WMSES'))                    
                print(str_cost_wmsesure.format(niter,self.costlist[-1],self.wmselistsure[-1]))
                


    def loop_mu_s(self,nitermax=10,maxiter=100):
        """ main loop """
        
        if self.nitertot==0:
            self.tau = compute_tau_2D(self.psf,self.mu_s_max,self.sigma,self.nbw_decomp)
            print('')
            print('setting tau to smallest (safest): ',self.tau)
            print('')

        if nitermax < 1:
            print('nitermax must be a positive integer, nitermax=10')
            nitermax=10

        std = 1000 # stopping criteria std of wmsesure
        niter = 0
        
        while (niter<nitermax) and (std>self.thresh):
            self.mu_s = self.golds_search_mu_s(a=self.mu_s_min, b=self.mu_s_max, maxiter=maxiter)
            super(EasyMuffinSURE,self).update()
            self.update_Jacobians()

            self.mu_slist.append(self.mu_s)
            self.nitertot+=1
              
            if self.truesky.any():
                if (niter % 20) ==0:
                    print(str_cst_snr_wmse_wmsesure_mu_title.format('It.','Cost','SNR','WMSE','WMSES','mu_s'))                    
                print(str_cst_snr_wmse_wmsesure_mu.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsure[-1],self.mu_slist[-1]))
            else:
                if (niter % 20) ==0:
                    print(str_cost_wmsesure_mu_title.format('It.','Cost','WMSES','mu_s','mu_l'))                    
                print(str_cost_wmsesure_mu.format(niter,self.costlist[-1],self.wmselistsure[-1],self.mu_slist[-1],self.mu_llist[-1]))
                    
            std = np.var(self.wmselistsure[niter-50::])
                
            niter+=1


    def golds_search_mu_s(self,a, b, maxiter=100):

        absolutePrecision = self.absolutePrecision
        gr = (1+np.sqrt(5))/2
        c = b - (b - a)/gr
        d = a + (b - a)/gr
        niter = 0

        while abs(a - b) > absolutePrecision and niter < maxiter:
            #print('(a,f(a))=(',a,',',self.f_gs_mu_s(a),') - (b,f(b))=(',b,',',self.f_gs_mu_s(b),') - (c,f(c))=(',c,',',self.f_gs_mu_s(c),') - (d,f(d))=(',d,',',self.f_gs_mu_s(d),') ')
            #print('')
            if self.f_gs_mu_s(c) < self.f_gs_mu_s(d):
                b = d
            else:
                a = c

            c = b - (b - a)/gr
            d = a + (b - a)/gr
            niter+=1

        #print('quit gs')
        return (a + b)/2


    def f_gs_mu_s(self,a):
        mu_s_0 = self.mu_s
        self.mu_s = a
        super(EasyMuffinSURE,self).update(change=False)
        res = self.update_Jacobians(change=False)
        self.mu_s = mu_s_0
        return res

    def set_mean_mu_s(self,niter=100):
        self.mu_s = np.mean(self.mu_slist[max(1,self.nitertot-niter)::])
            
            