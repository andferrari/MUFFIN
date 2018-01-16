#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:07:31 2016

@author: antonyschutz
"""
import numpy as np
from scipy.fftpack import dct,idct
from deconv3d_tools import compute_tau_DWT, defadj, init_dirty_wiener, sat, Heavy, Rect
from deconv3d_tools import myfft2, myifft2, myifftshift, conv, optimal_split
from deconv3d_tools import iuwt_decomp, iuwt_decomp_adj, dwt_decomp, dwt_recomp
import copy
import sys
from mpi4py import MPI


str_cost="| {:5d} | {:6.6e} |"
str_cost_title="-"*24+"\n"+"| {:5s} | {:12s} |\n"+"-"*24
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

# Global variable -
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

nbw = size - 1
idw = rank - 1

class EasyMuffin():
    def __init__(self,
                 mu_s=0.5,
                 mu_l=0.0,
                 mu_eps=0.0,
                 nb=(8,0),
                 tau = 1e-4,
                 sigma = 10,
                 var = 0,
                 dirtyinit=[],
                 dirty=[],
                 truesky=None,
                 psf=[],
                 mu_s_min=0,
                 mu_s_max=1,
                 mu_l_min=0,
                 mu_l_max=3,
                 absolutePrecision=0.1):

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

        if mu_eps< 0 :
            print('mu_eps must be non negative, mu_eps=0.')
            mu_eps=0.0

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
        self.mu_eps = mu_eps
        self.sigma = sigma
        self.tau = tau
        self.dirtyinit = dirtyinit
        self.truesky = truesky
        self.psf = psf
        self.dirty=dirty
        self.var = var

        self.mu_s_min = mu_s_min
        self.mu_s_max = mu_s_max
        self.mu_l_min = mu_l_min
        self.mu_l_max = mu_l_max
        self.absolutePrecision = absolutePrecision

        self.nfreq = self.dirty.shape[2]
        self.nxy = self.dirty.shape[0]

        # new solution for partition
        self.lst_nbf = optimal_split(self.nfreq,nbw)
        self.lst_nbf[0:0] = [self.nfreq]

        self.nf2 = self.lst_nbf[1:-1:1]
        self.nf2 = np.cumsum(self.nf2)
        self.nf2 = self.nf2.tolist()
        self.nf2[0:0]=[0]

        if rank==0:
            print('')
            print(self.lst_nbf)
            print(self.nf2)

        # old solution

#        self.nf2 =int(np.ceil(self.nfreq*1.0/nbw))
#        self.lst_nbf=[self.nfreq]

#        for i in range(nbw):
#            step=min(self.nf2,max(self.nfreq-sum(self.lst_nbf[1:]),0))
#            self.lst_nbf.append(step)

#        if self.lst_nbf[-1]==0:
#            self.nf2=int(np.floor(self.nfreq*1.0/(nbw)))

#            self.lst_nbf=[self.nfreq]
#            for i in range(nbw-1):
#                self.lst_nbf.append(self.nf2)

#            self.lst_nbf.append(self.nfreq-sum(self.lst_nbf[1:]))

#        if rank == 0:
#            print(self.lst_nbf)
#            print(self.nf2)


        nbsum=0
        self.sendcounts=[0,]
        self.displacements=[0,]
        for i in range(nbw):
            self.displacements.append(nbsum)
            taille=self.nxy*self.nxy*self.lst_nbf[i+1]
            self.sendcounts.append(taille)
            nbsum+=taille

        if rank ==0:
            print('')
            print('local size:',self.lst_nbf)
            print('Starting point:',self.displacements)
            print('Nbr of elts:',self.sendcounts)
            print('')

        if nbw > self.nfreq:
            if rank ==0:
                print('----------------------------------------------------------------')
                print('   mpi: !!!! You cannot have more workers than bands !!!! ')
                print('----------------------------------------------------------------')
                sys.exit()
            else:
                sys.exit()


        np.random.seed(1)
        self.n = np.random.binomial(1,0.5,(self.nxy,self.nxy,self.nfreq))
        self.n[self.n==0] = -1

        self.init_algo()

    def init_algo(self):
        """Initialization of te algorithm (all intermediate variables)"""


        if rank == 0:

            print('psf size ', self.psf.shape)
            print('drt size ', self.dirty.shape)

            # precomputations
            print('')
            print("precomputations...")
            print('1')

            self.vtt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.v = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            print('2')

            self.tf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.t = np.zeros((0)) # even master has to send tloc when gatherv
            print('3')

            self.xf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.x = np.zeros((0))
            self.xt = np.zeros((0))
            print('4')

            self.x2f_ = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.x2_ = np.zeros((0))
            print('5')

            self.deltaf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.delta = np.zeros((0))
            print('6')

            if type(self.nb[0]) == int:
                self.nbw_decomp = [f for f in range(self.nb[0])]
                self.tau = compute_tau_DWT(self.psf,self.mu_s,self.mu_l,self.sigma,self.nbw_decomp) # WRONGG
                print('')
                print('IUWT: tau = ', self.tau)
                print('')
            else:
                self.nbw_decomp =self.nb
                self.tau = compute_tau_DWT(self.psf,self.mu_s,self.mu_l,self.sigma,self.nbw_decomp)
                print('')
                print('DWT: tau = ', self.tau)
                print('')

        else:

            self.nfreq = self.lst_nbf[rank]
            self.psf = np.asfortranarray(self.psf[:,:,self.nf2[idw]:self.nf2[idw]+self.nfreq])
            self.dirty = np.asfortranarray(self.dirty[:,:,self.nf2[idw]:self.nf2[idw]+self.nfreq])

            if self.truesky is not None:
                self.truesky = self.truesky[:,:,self.nf2[idw]:self.nf2[idw]+self.nfreq]

            self.psfadj = defadj(self.psf)

            if self.dirtyinit:
                self.x = np.asfortranarray(self.dirtyinit) ####################
            else:
                if self.mu_eps ==0:
                    self.x =  np.asfortranarray(init_dirty_wiener(self.dirty, self.psf, self.psfadj, 5e1))
                else:
                    self.x =  np.asfortranarray(init_dirty_wiener(self.dirty, self.psf, self.psfadj, 5e1))


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

            self.x2_ = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.x2f_ = np.zeros((0))

            self.t = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.tf = np.zeros((0)) # even master has to send tloc when gatherv

            self.delta = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.deltaf = np.zeros((0))

            if type(self.nb[0]) == int:
                self.Decomp = iuwt_decomp
                self.Recomp = iuwt_decomp_adj
                self.nbw_decomp = [f for f in range(self.nb[0])]
                self.nbw_recomp = self.nb[-1]

            else:
                self.Decomp = dwt_decomp
                self.Recomp = dwt_recomp
                self.nbw_decomp =self.nb
                self.nbw_recomp = self.nb

            self.tau = 0

            self.utt = {}
            for freq in range(self.nfreq):
                self.utt[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), order='F') , self.nbw_decomp)

            self.u = {}
            for freq in range(self.nfreq):
                self.u[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), order='F') , self.nbw_decomp)

        self.tau = comm.bcast(self.tau,root=0)

        self.nitertot = 0

        # compute cost
        self.costlist = []

        comm.Gatherv(self.x,[self.xf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        self.costlist.append(self.cost())

        # snr, psnr, wmse
        if self.truesky is not None:
            self.snrlist = []
            self.truesky2 = np.sum(self.truesky*self.truesky)
            self.wmselist = []
            self.psnrlist = []
            self.psnrnum = np.sum((self.dirty-self.truesky)**2)/(self.nxy*self.nxy*self.nfreq)

        # compute snr if truesky given
        if self.truesky is not None:
            self.snrlist.append(self.snr())
            self.psnrlist.append(self.psnr())
            self.wmselist.append(self.wmse())
            if rank==0:
                print('The snr of the initialisation is ',self.snrlist[0])
                print('')


    def cost(self):
        if not rank==0:
            tmp = self.dirty - myifftshift(myifft2(myfft2(self.x)*myfft2(self.psf)))
            LS_cst = 0.5*(np.linalg.norm(tmp)**2) + 0.5*self.mu_eps*(np.linalg.norm(self.x)**2)
            tmp = 0.
            for freq in range(self.nfreq):
                tmp1 = self.Decomp(self.x[:,:,freq],self.nbw_decomp)
                for b in self.nbw_decomp:
                    tmp = tmp + np.sum(np.abs(tmp1[b]))
            Spt_cst = self.mu_s*tmp
            cst = LS_cst + Spt_cst
        else:
            Spc_cst = self.mu_l*np.sum(np.abs(dct(self.xf,axis=2,norm='ortho')))
            cst = Spc_cst

        cst_list = comm.gather(cst)

        if rank == 0:
            return sum(cst_list)/(self.nxy*self.nxy*self.nfreq)
        else:
            return cst


    def snr(self,change=True):
        if change:
            if not rank == 0:
                resid = self.truesky - self.x
                resid = float(np.sum(resid*resid))
            else:
                resid = 0

            rlist = comm.gather(resid)

            if rank == 0:
                return 10*np.log10(self.truesky2 / sum(rlist))
            else:
                return resid

        else:
            resid = self.truesky - self.x2_ ####################
            return 10*np.log10(self.truesky2 / np.sum(resid*resid)) ####################


    def psnr(self):
        if not rank == 0:
            resid = (np.linalg.norm(conv(self.psf,self.truesky-self.x))**2)
        else:
            resid = 0

        rlist = comm.gather(resid)

        if rank == 0:
            return 10*np.log10(self.psnrnum / (sum(rlist)/(self.nxy*self.nxy*self.nfreq)))
        else:
            return resid


    def wmse(self,change=True):
        if change:
            if not rank == 0:
                resid = (np.linalg.norm(conv(self.psf,self.truesky-self.x))**2)
            else:
                resid = 0

            rlist = comm.gather(resid)

            if rank == 0 :
                return sum(rlist)/(self.nxy*self.nxy*self.nfreq)
            else:
                return resid
        else:
            return (np.linalg.norm(conv(self.psf,self.truesky-self.x2_))**2)/(self.nxy*self.nxy*self.nfreq) ########@


    def mse(self):
        if not rank==0:
            resid = np.linalg.norm(self.truesky-self.x)**2
        else:
            resid = 0

        rlist = comm.gather(resid)

        if rank == 0:
            return sum(rlist)/(self.nxy*self.nxy*self.nfreq)
        else:
            return resid


    def update(self,change=True):


        if change:
            if rank==0:
                #x_ = self.x
                v_ = self.v
            else:
                xt_ = self.xt # xt_ is a pointer
                u_ = self.u
                x_ = self.x
        else:
            if rank ==0:
                v_  = self.v.copy(order='F')
                #x_  = self.x.copy()
            else:
                xt_ = self.xt.copy(order='F') # xt is a new local matrix
                u_  = copy.deepcopy(self.u)
                x_  = self.x.copy(order='F')

        if rank==0:
            # update t
            self.tf = np.asfortranarray(idct(v_, axis=2, norm='ortho'))

        comm.Scatterv([self.tf,self.sendcounts,self.displacements,MPI.DOUBLE],self.t,root=0)

        if not rank ==0:
            # compute gradient
            tmp = myifftshift( myifft2( myfft2(x_) *self.hth_fft ) )
            Delta_freq = tmp.real- self.fty + self.mu_eps*x_

            for freq in range(self.nfreq):
                # compute iuwt adjoint
                wstu = self.Recomp(u_[freq], self.nbw_recomp)

                # compute xt
                self.xtt[:,:,freq] = x_[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + self.mu_s*wstu + self.mu_l*self.t[:,:,freq])
                xt_[:,:,freq] = np.maximum(self.xtt[:,:,freq], 0.0 )

                # update u
                tmp_spat_scal = self.Decomp(2*xt_[:,:,freq] - x_[:,:,freq] , self.nbw_decomp)

                for b in self.nbw_decomp:
                    self.utt[freq][b] = u_[freq][b] + self.sigma*self.mu_s*tmp_spat_scal[b]
                    u_[freq][b] = sat(self.utt[freq][b])

            self.delta = np.asfortranarray(2*xt_ - x_)

        comm.Gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)

        if rank ==0:
            # update v
            self.vtt = np.asfortranarray(v_ + self.sigma*self.mu_l*dct(self.deltaf, axis=2, norm='ortho'))
            v_ = sat(self.vtt)

        if change:
            if rank==0:
                self.v = v_.copy(order='F')
            else:
                self.x = self.xt.copy(order='F')

            comm.Gatherv(self.x,[self.xf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)

            # compute cost
            self.costlist.append(self.cost())

            # compute snr, psnr, wmse if truesky given
            if self.truesky is not None:
                self.snrlist.append(self.snr())
                self.psnrlist.append(self.psnr())
                self.wmselist.append(self.wmse())
        else:
            if not rank==0:
                self.x2_ = xt_.copy(order='F')

            comm.Gatherv(self.x2_,[self.x2f_,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)


    def parameters(self):
        print('')
        print('nitermax: ',self.nitermax)
        print('nb: ',self.nb)
        print('mu_s: ',self.mu_s)
        print('mu_l: ',self.mu_l)
        print('tau: ',self.tau)
        print('sigma: ',self.sigma)

   #======================================================================
   # MAIN PROGRAM - EASY MUFFIN
   #======================================================================

    def loop(self,nitermax=10):
        """ main loop """

        if nitermax< 0:
            if rank==0:
                print('nitermax must be a positive integer, nitermax=10')
            nitermax=10

        for niter in range(nitermax):

            self.update()

            if rank==0:
                if self.truesky is not None:
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
                 mu_eps=0.0,
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
                 mu_l_min=0,
                 mu_l_max=3,
                 absolutePrecision=0.1,
                 thresh=1e-4):

        # stopping criteria for loop_mu_s/_mu_l
        self.thresh = thresh


        super(EasyMuffinSURE,self).__init__(
                 mu_s,
                 mu_l,
                 mu_eps,
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
                 mu_l_min,
                 mu_l_max,
                 absolutePrecision)

    def init_algo(self):

            super(EasyMuffinSURE,self).init_algo()

            if rank==0:
                self.psfadj = defadj(self.psf)

                # init Jacobians
                self.Jv = np.zeros((self.nxy,self.nxy,self.nfreq), order='F')

                self.Jtf = np.zeros((self.nxy,self.nxy,self.nfreq), order='F')
                self.Jt = np.zeros((0))

                self.Jx = np.zeros((0))

                self.Jdeltaf = np.zeros((self.nxy,self.nxy,self.nfreq), order='F')
                self.Jdelta = np.zeros((0))
            else:
                self.n = self.n[:,:,self.nf2[idw]:self.nf2[idw]+self.nfreq]

                # compute Hn
                self.Hn = np.zeros((self.nxy,self.nxy,self.nfreq), order='F')
                self.Hn = conv(self.n,self.psfadj)

                # init Jacobians
                self.Jt = np.zeros((self.nxy,self.nxy,self.nfreq), order='F')
                self.Jtf = np.zeros((0))

                self.Jx = np.asfortranarray(init_dirty_wiener(self.n, self.psf, self.psfadj, 5e1))
                self.Jxt = np.zeros((self.nxy,self.nxy,self.nfreq), order='F')

                self.Ju = {}
                for freq in range(self.nfreq):
                    self.Ju[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), order='F') , self.nbw_decomp)

                self.Jdelta = np.zeros((self.nxy,self.nxy,self.nfreq), order='F')
                self.Jdeltaf = np.zeros((0))

            # psnr, and wmse estimated using psure
            self.wmselistsure = []
            self.wmselistsure.append(self.wmsesure())

            if self.truesky is not None:
                self.psnrlistsure = []
                self.psnrlistsure.append(self.psnrsure())

            # mu_s list
            self.mu_slist = []
            self.mu_slist.append(self.mu_s)

            # mu_l list
            self.mu_llist = []
            self.mu_llist.append(self.mu_l)

    def wmsesure(self,change=True):

        if change:
            if not rank==0:
                tmp = self.dirty - conv(self.x,self.psf)
                LS_cst = np.linalg.norm(tmp)**2
                tmp = self.n*conv(self.Jx,self.psf)
                wmse = LS_cst + 2*self.var*np.sum(tmp)
            else:
                wmse = 0

            wmse_lst = comm.gather(wmse)

            if rank==0:
                return sum(wmse_lst)/(self.nxy*self.nxy*self.nfreq) - self.var
            else:
                return wmse_lst

        else: # x2_ Jx2_
            if not rank==0:
                tmp = self.dirty - conv(self.x2_,self.psf)
                LS_cst = np.linalg.norm(tmp)**2
                tmp = self.n*conv(self.Jx2_,self.psf)
                wmse = LS_cst + 2*self.var*np.sum(tmp)
            else:
                wmse = 0

            wmse_lst = comm.gather(wmse)

            if rank==0:
                return sum(wmse_lst)/(self.nxy*self.nxy*self.nfreq) - self.var
            else:
                return wmse_lst


    def psnrsure(self):
        if rank == 0:
            return 10*np.log10(self.psnrnum/self.wmselistsure[-1])
        else:
            return 0


    def update_Jacobians(self,change=True):

        if change:
            if rank ==0:
                Jv_ = self.Jv
            else:
                Jxt_ = self.Jxt # pointer
                Ju_ = self.Ju
                Jx_ = self.Jx

        else:
            if rank ==0:
                Jv_  = self.Jv.copy(order='F')
            else:
                Jxt_ = self.Jxt.copy(order='F') # new matrix
                Ju_  = copy.deepcopy(self.Ju)
                Jx_  = self.Jx.copy(order='F')

        if rank==0:
                self.Jtf = np.asfortranarray(idct(Jv_, axis=2,norm='ortho'))

        comm.Scatterv([self.Jtf,self.sendcounts,self.displacements,MPI.DOUBLE],self.Jt,root=0)

        if not rank ==0:
            # compute gradient
            tmp = myifftshift( myifft2( myfft2(Jx_) * self.hth_fft ) )
            JDelta_freq = tmp.real- self.Hn + self.mu_eps*Jx_

            for freq in range(self.nfreq):

                # compute iuwt adjoint
                Js_l = self.Recomp(Ju_[freq], self.nbw_recomp)

                # compute xt
                Jxtt = Jx_[:,:,freq] - self.tau*(JDelta_freq[:,:,freq] + self.mu_s*Js_l + self.mu_l*self.Jt[:,:,freq])
                Jxt_[:,:,freq] = Heavy(self.xtt[:,:,freq])*Jxtt

                # update u
                tmp_spat_scal_J = self.Decomp(2*Jxt_[:,:,freq] - Jx_[:,:,freq] , self.nbw_decomp)
                for b in self.nbw_decomp:
                    Jutt = Ju_[freq][b] + self.sigma*self.mu_s*tmp_spat_scal_J[b]
                    Ju_[freq][b] = Rect( self.utt[freq][b] )*Jutt

            self.Jdelta = np.asfortranarray(2*Jxt_ - Jx_)

        comm.Gatherv(self.Jdelta,[self.Jdeltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)

        if rank==0:
            # update v
            Jvtt = np.asfortranarray(Jv_ + self.sigma*self.mu_l*dct(self.Jdeltaf, axis=2, norm='ortho'))
            Jv_ = Rect(self.vtt)*Jvtt

        if change:
            if rank==0:
                self.Jv = Jv_.copy(order='F')
            else:
                self.Jx = self.Jxt.copy(order='F')

            # wmsesure
            self.wmselistsure.append(self.wmsesure())

            # psnrsure
            if self.truesky is not None:
                self.psnrlistsure.append(self.psnrsure())
        else:
            if not rank==0:
                self.Jx2_ = Jxt_.copy(order='F')

            wmsesure = self.wmsesure(change=False)

            return wmsesure


    def loop(self,nitermax=10,change=True):
        """ main loop """

        if nitermax < 0:
            if rank==0:
                print('nitermax must be a positive integer, nitermax=10')
            nitermax=10

        for niter in range(nitermax):
            self.mu_slist.append(self.mu_s)
            self.mu_llist.append(self.mu_l)
            super(EasyMuffinSURE,self).update(change)
            self.update_Jacobians(change)
            self.nitertot+=1

            if rank==0:
                if self.truesky is not None:
                    if (niter % 20) ==0:
                        print(str_cst_snr_wmse_wmsesure_mu_title.format('It.','Cost','SNR','WMSE','WMSES','mu_s','mu_l'))
                    print(str_cst_snr_wmse_wmsesure_mu.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsure[-1],self.mu_slist[-1],self.mu_llist[-1]))
                else:
                    if (niter % 20) ==0:
                        print(str_cost_wmsesure_mu_title.format('It.','Cost','WMSES','mu_s','mu_l'))
                    print(str_cost_wmsesure_mu.format(niter,self.costlist[-1],self.wmselistsure[-1],self.mu_slist[-1],self.mu_llist[-1]))


    def loop_mu_s(self,nitermax=10,maxiter=100):
        """ main loop """

        if self.nitertot==0:
            if rank==0:
                self.tau = compute_tau_DWT(self.psf,self.mu_s_max,self.mu_l_max,self.sigma,self.nbw_decomp)
                print('')
                print('setting tau to smallest (safest): ',self.tau)
                print('')

            self.tau = comm.bcast(self.tau,root=0)

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
            self.mu_llist.append(self.mu_l)
            self.nitertot+=1

            if rank==0:
                if self.truesky is not None:
                    if (niter % 20) ==0:
                        print(str_cst_snr_wmse_wmsesure_mu_title.format('It.','Cost','SNR','WMSE','WMSES','mu_s','mu_l'))
                    print(str_cst_snr_wmse_wmsesure_mu.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsure[-1],self.mu_slist[-1],self.mu_llist[-1]))
                else:
                    if (niter % 20) ==0:
                        print(str_cost_wmsesure_mu_title.format('It.','Cost','WMSES','mu_s','mu_l'))
                    print(str_cost_wmsesure_mu.format(niter,self.costlist[-1],self.wmselistsure[-1],self.mu_slist[-1],self.mu_llist[-1]))

            if rank==0 and niter>100:
                std = np.var(self.wmselistsure[niter-50::])

            std = comm.bcast(std,root=0)
            niter+=1


    def golds_search_mu_s(self,a, b, maxiter=100):

        absolutePrecision = self.absolutePrecision
        gr = (1+np.sqrt(5))/2
        c = b - (b - a)/gr
        d = a + (b - a)/gr
        niter = 0

        while abs(a - b) > absolutePrecision and niter < maxiter:

            if self.f_gs_mu_s(c) < self.f_gs_mu_s(d):
                b = d
            else:
                a = c

            c = b - (b - a)/gr
            d = a + (b - a)/gr
            niter+=1


        return (a + b)/2


    def f_gs_mu_s(self,a):

        mu_s_0 = self.mu_s
        self.mu_s = a
        super(EasyMuffinSURE,self).update(change=False)
        res = self.update_Jacobians(change=False)
        res = comm.bcast(res,root=0)
        self.mu_s = mu_s_0

        return res


    def loop_mu_l(self,nitermax=10,maxiter=100):
        """ main loop """

        if nitermax < 1:
            print('nitermax must be a positive integer, nitermax=10')
            nitermax=10

        std = 1000 # stopping criteria std of wmsesure
        niter = 0

        while (niter < nitermax) and (std>self.thresh):
            self.mu_l = self.golds_search_mu_l(a=self.mu_l_min, b=self.mu_l_max, maxiter=maxiter)
            super(EasyMuffinSURE,self).update()
            self.update_Jacobians()

            self.mu_llist.append(self.mu_l)
            self.mu_slist.append(self.mu_s)
            self.nitertot+=1

            if rank==0:
                if self.truesky is not None:
                    if (niter % 20) ==0:
                        print(str_cst_snr_wmse_wmsesure_mu_title.format('It.','Cost','SNR','WMSE','WMSES','mu_s','mu_l'))
                    print(str_cst_snr_wmse_wmsesure_mu.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsure[-1],self.mu_slist[-1],self.mu_llist[-1]))
                else:
                    if (niter % 20) ==0:
                        print(str_cost_wmsesure_mu_title.format('It.','Cost','WMSES','mu_s','mu_l'))
                    print(str_cost_wmsesure_mu.format(niter,self.costlist[-1],self.wmselistsure[-1],self.mu_slist[-1],self.mu_llist[-1]))

            if rank==0 and niter>100:
                std = np.var(self.wmselistsure[niter-50::])

            std = comm.bcast(std,root=0)
            niter+=1


    def golds_search_mu_l(self,a, b, maxiter=100):

        absolutePrecision = self.absolutePrecision
        gr = (1+np.sqrt(5))/2
        c = b - (b - a)/gr
        d = a + (b - a)/gr
        niter = 0

        while abs(a - b) > absolutePrecision and niter < maxiter:

            if self.f_gs_mu_l(c) < self.f_gs_mu_l(d):
                b = d
            else:
                a = c

            c = b - (b - a)/gr
            d = a + (b - a)/gr
            niter+=1

        return (a + b)/2


    def f_gs_mu_l(self,a):

        mu_l_0 = self.mu_l
        self.mu_l = a
        super(EasyMuffinSURE,self).update(change=False)
        res = self.update_Jacobians(change=False)
        res = comm.bcast(res,root=0)
        self.mu_l = mu_l_0

        return res

    def set_mean_mu(self,set_mu_l=False,set_mu_s=False,niter=100):

        if set_mu_l==True :
            self.mu_l = np.mean(self.mu_llist[max(1,self.nitertot-niter)::])

        if set_mu_s==True:
            self.mu_s = np.mean(self.mu_slist[max(1,self.nitertot-niter)::])
