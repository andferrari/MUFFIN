# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:05:32 2017

@author: rammanouil
"""
import numpy as np
from scipy.fftpack import dct,idct

from deconv3d_tools import compute_tau_DWT, defadj, init_dirty_wiener, sat, heavy, rect
from deconv3d_tools import myifftshift, optimal_split
from deconv3d_tools import iuwt_decomp, iuwt_decomp_adj, dwt_decomp, dwt_recomp, dwt_I_decomp, dwt_I_recomp
from deconv3d_tools import conv as convolve 
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

class EasyMuffin():
    def __init__(self,
                 comm = MPI.COMM_WORLD,
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
                 pixelweighton = 0,
                 bandweighton = 0,
                 fftw = 0,
                 init=0,
                 fol_init=0,
                 save=0,
                 odir='.'):
        
        self.comm = comm
        self.nbw = self.comm.Get_size() - 1
        self.idw = self.comm.Get_rank() - 1
        self.master = self.comm.Get_rank() == 0
        assert(type(nb) is tuple)
        assert(mu_s >= 0)
        assert(mu_l >= 0)
        assert(tau >= 0)
        assert(sigma >= 0)

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
        self.pixelweighton = pixelweighton
        self.bandweighton = bandweighton
        self.fftw_flag = fftw
        self.init = init
        self.fol_init = fol_init
        
        self.nfreq = self.dirty.shape[2]
        self.nxy = self.dirty.shape[0]

        # Partitioning the frequency bands
        self.lst_nbf = optimal_split(self.nfreq,self.nbw)
        self.lst_nbf[0:0] = [self.nfreq]
        self.nf2 = self.lst_nbf[1:-1:1]
        self.nf2 = np.cumsum(self.nf2)
        self.nf2 = self.nf2.tolist()
        self.nf2[0:0] = [0]
        
        self.save = save
        self.odir = odir

        if self.master:
            print('')
            print(self.lst_nbf)
            print(self.nf2)

        nbsum = 0
        self.sendcounts = [0,]
        self.displacements = [0,]
        for i in range(self.nbw):
            self.displacements.append(nbsum)
            taille = self.nxy*self.nxy*self.lst_nbf[i+1]
            self.sendcounts.append(taille)
            nbsum+=taille

        if self.nbw > self.nfreq:
            if self.master:
                print('----------------------------------------------------------------')
                print('   mpi: !!!! You cannot have more workers than bands !!!!')
                print('----------------------------------------------------------------')
                sys.exit()
            else:
                sys.exit()

        np.random.seed(1)
        self.n = np.random.binomial(1,0.5,(self.nxy,self.nxy,self.nfreq))
        self.n[self.n==0] = -1
        
        # fdmc variables
        self.eps = 4*(self.var**0.5)*((self.nxy**2)**(-0.3)) # à verifier
        np.random.seed(1)
        
        self.DeltaSURE = np.random.randn(self.nxy,self.nxy,self.nfreq)
        self.dirty2 = self.dirty + self.eps*self.DeltaSURE
                
        self.init_algo()

    def init_algo(self):
        """Initialization of the algorithm (all intermediate variables)"""
        
        if self.master:
            if self.fftw_flag==0:
                from deconv3d_tools import myfft2, myifft2
                self.fft2 = myfft2
                self.ifft2 = myifft2
            else:
                import pyfftw
                aa = pyfftw.empty_aligned(self.psf.shape,dtype='complex128')
                bb = pyfftw.empty_aligned(self.psf.shape,dtype='complex128')
                cc = pyfftw.empty_aligned(self.psf.shape,dtype='complex128')
                fft_object = pyfftw.FFTW(aa,bb,axes=(0,1),direction='FFTW_FORWARD',flags=('FFTW_MEASURE',),sads=1)
                ifft_object = pyfftw.FFTW(bb,cc,axes=(0,1),direction='FFTW_BACKWARD',flags=('FFTW_MEASURE',),threads=1)

                self.fft2 = fft_object
                self.ifft2 = ifft_object
            
            
            print('psf size ', self.psf.shape)
            print('drt size ', self.dirty.shape)
            
            # precomputations
            print('')
            print('precomputations...')
            print('')
            
            # x initialization
            self.xf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
            self.x = np.zeros((0))
            self.xtf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
            self.xt = np.zeros(0)
                
            self.vtt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
            self.v = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
#            if self.init:
#                self.v = np.load(self.fol_init+'/v.npy')
            
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
            elif self.nb[-1] == 'I':
                self.Decomp = dwt_I_decomp
                self.Recomp = dwt_I_recomp
                self.nbw_decomp = self.nb
                self.nbw_recomp = self.nb
                self.tau = compute_tau_DWT(self.psf,self.mu_s,self.mu_l,self.sigma,self.nbw_decomp)
                print('')
                print('DWT+I: tau = ', self.tau)
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
                
#            self.uf = {}
#            self.u = {}
            
            # Compute spatial and spectral scaling parameters
            if self.pixelweighton ==1:
                self.psfadj = defadj(self.psf)
                tmp = np.asfortranarray(init_dirty_wiener(self.dirty, self.psf, self.psfadj, self.mu_wiener))
                tmp = dct(tmp,axis=2,norm='ortho')
                self.alpha_l = 1/(np.sum(tmp**2,2)+1e-1) # image
                self.alpha_l = convolve(self.alpha_l,np.ones((3,3)),'max')
                self.alpha_l = self.alpha_l/self.alpha_l.max()
            else:
                self.alpha_l = np.ones((self.nxy,self.nxy))
            
        else:
            
            self.nfreq = self.lst_nbf[self.comm.Get_rank()]
            self.psf = np.asfortranarray(self.psf[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
            self.dirty = np.asfortranarray(self.dirty[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
            
            if self.truesky is not None:
                self.truesky = self.truesky[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq]
            
            if self.fftw_flag==0:
                from deconv3d_tools import myfft2, myifft2
                self.fft2 = myfft2
                self.ifft2 = myifft2
            else:
                import pyfftw
                aa = pyfftw.empty_aligned(self.psf.shape,dtype='complex128')
                bb = pyfftw.empty_aligned(self.psf.shape,dtype='complex128')
                cc = pyfftw.empty_aligned(self.psf.shape,dtype='complex128')
                fft_object = pyfftw.FFTW(aa,bb,axes=(0,1),direction='FFTW_FORWARD',flags=('FFTW_MEASURE',),threads=1)
                ifft_object = pyfftw.FFTW(bb,cc,axes=(0,1),direction='FFTW_BACKWARD',flags=('FFTW_MEASURE',),threads=1)

                self.fft2 = fft_object
                self.ifft2 = ifft_object
                
            self.psfadj = defadj(self.psf)
            
            # x initialization
            if self.dirtyinit:
                self.x = np.asfortranarray(self.dirtyinit)
#            elif self.init:
#                #print('')
#                #print('process ',self.comm.Get_rank(),'loading x_init from ',self.fol_init,' ... ')
#                self.x = np.load(self.fol_init+'/x0_tst.npy') 
#                self.x = np.asfortranarray(self.x[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
            else:
                self.x = np.asfortranarray(init_dirty_wiener(self.dirty, self.psf, self.psfadj, self.mu_wiener))
            
            # initializing alg. variables
            self.hth_fft = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.complex, order='F')
            self.fty = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.psfadj_fft = self.fft2(self.psfadj).copy()
            self.hth_fft = self.fft2( myifftshift( self.ifft2( self.psfadj_fft * self.fft2(self.psf) ) ) ).copy(order='F')
            tmp = myifftshift(self.ifft2(self.fft2(self.dirty)*self.psfadj_fft))
            self.fty = tmp.real.copy(order='F')
            self.wstu = np.zeros((self.nxy,self.nxy), dtype=np.float, order='F')
            self.xtt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.xt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
            self.xtf = np.zeros(0)
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
            elif self.nb[-1] == 'I':
                self.Decomp = dwt_I_decomp
                self.Recomp = dwt_I_recomp
                self.nbw_decomp = self.nb
                self.nbw_recomp = self.nb
            else:
                self.Decomp = dwt_decomp
                self.Recomp = dwt_recomp
                self.nbw_decomp =self.nb
                self.nbw_recomp = self.nb
                
            self.tau = 0
    
            self.utt = {}
            for freq in range(self.nfreq):
                self.utt[freq] = self.Decomp(np.zeros((self.nxy,self.nxy),order='F') , self.nbw_decomp)
            self.u = {}
            for freq in range(self.nfreq):
                self.u[freq] = self.Decomp(np.zeros((self.nxy,self.nxy),order='F') , self.nbw_decomp)
#            if self.init:
#                tmp = np.ndarray.tolist(np.load(self.fol_init+'/u.npy'))
#                for freq in range(self.nfreq) :
#                    self.u[freq] = tmp[self.nf2[self.idw]+freq]
#            self.uf = np.zeros((0))
                   
            # Compute spatial and spectral scaling parameters
            if self.bandweighton ==1:
                self.alpha_s = 1/(np.sum(np.sum(self.x**2,0),0)+1e-1) # col. vector
            else:
                self.alpha_s = np.ones(self.nfreq)
                
            self.alpha_l = np.ones((self.nxy,self.nxy))

        #self.sendcountsu = [ i*(self.nbw_decomp[-1]+1) for i in self.sendcounts]
        self.sendcountsu = [ i*np.size(self.nbw_decomp) for i in self.sendcounts]
        #self.displacementsu =[ i*(self.nbw_decomp[-1]+1) for i in self.displacements]
        self.displacementsu =[ i*np.size(self.nbw_decomp) for i in self.displacements]
        
#        if self.master:
#            self.uf_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F')
#        else:
#            # empty uf recv buffer at each worker node
#            self.uf_ = np.zeros((0))
                
                
        self.alpha_l = self.comm.bcast(self.alpha_l,root=0) 
        self.tau = self.comm.bcast(self.tau,root=0) # root bcasts tau to everyone else 
        self.nitertot = 0
        
        self.costlist = []
        self.comm.Gatherv(self.x,[self.xf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
        if self.master:
            self.u = {}
            self.uf = {}
            self.u_ = np.zeros((0))
            self.uf_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F')                
        else:
            self.uf = np.zeros((0))
            self.u_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F')
            self.uf_ = np.zeros((0))
                
        if self.init:
                
            if self.master:
                # load v for master 
                self.v = np.load(self.fol_init+'/v.npy')
                
                # load xf and scatter to nodes 
                self.xf = np.load(self.fol_init+'/x0_tst.npy') 
            
                # load u and scatter to nodes 
                self.uf = np.ndarray.tolist(np.load(self.fol_init+'/u.npy'))
            
                self.uf_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire  
        
                i = 0
                for val1 in self.uf.values():
                    for j in self.nbw_decomp:
                        self.uf_[:,:,i]=val1[j].copy()
                        i+=1
          
            self.comm.Scatterv([self.xf,self.sendcounts,self.displacements,MPI.DOUBLE],self.x,root=0)
            self.comm.Scatterv([self.uf_,self.sendcountsu,self.displacementsu,MPI.DOUBLE],self.u_,root=0)
            
            if not self.master:
                udicti = {}
                nfreqi=0
                for i in range(self.nfreq):
                    for j in self.nbw_decomp:
                        udicti[j]=self.u_[:,:,nfreqi]
                        nfreqi+=1
                    self.u[i]=udicti.copy()
                    
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
            if self.master:
                print('The snr initialization is ',self.snrlist[0])
                print('')

    def conv(self,x,y):
        tmp0 = self.fft2(x).copy(order='F')
        tmp = myifftshift(self.ifft2(tmp0*self.fft2(y)))
        return tmp.real
    
    def cost(self):
        if not self.master:
            """Compute cost for current iterate x"""
            tmp0 = self.fft2(self.x).copy(order='F')
            tmp = self.dirty - myifftshift(self.ifft2(tmp0*self.fft2(self.psf)))
            LS_cst = 0.5*(np.linalg.norm(tmp)**2)
            tmp = 0.
            for freq in range(self.nfreq):
                tmp1 = self.Decomp(self.x[:,:,freq],self.nbw_decomp)
                for b in self.nbw_decomp:
                    tmp = tmp + np.sum(np.abs(tmp1[b]*self.alpha_s[freq]))
            Spt_cst = self.mu_s*tmp
            cst = Spt_cst + LS_cst
        else:
            Spc_cst = self.mu_l*np.sum(np.abs(dct(self.xf*self.alpha_l[...,None],axis=2,norm='ortho')))
            cst = Spc_cst

        cst_list = self.comm.gather(cst)
        
        if self.master:
            return sum(cst_list)/(self.nxy*self.nxy*self.nfreq)
        else:
            return cst

    def snr(self):
        if not self.master:
            resid = self.truesky - self.x
            resid = np.float(np.sum(resid*resid))
        else:
            resid = 0
            
        rlist = self.comm.gather(resid)
        
        if self.master:
            return 10*np.log10(self.truesky2 / np.sum(rlist))
        else:
            return resid

    def psnr(self):
        if not self.master:
            resid = np.linalg.norm(self.conv(self.psf,self.truesky-self.x))**2
        else:
            resid = 0
            
        rlist = self.comm.gather(resid)
        
        if self.master:
            return 10*np.log10(self.psnrnum / (np.sum(rlist)/(self.nxy*self.nxy*self.nfreq)))
        else:
            return resid

    def wmse(self):
        if not self.master:
            resid = np.linalg.norm(self.conv(self.psf,self.truesky-self.x))**2
        else:
            resid = 0
            
        rlist = self.comm.gather(resid)
        
        if self.master:
            return np.sum(rlist)/(self.nxy*self.nxy*self.nfreq)
        else:
            return resid

    def mse(self):
        if not self.master:
            resid = np.linalg.norm(self.truesky-self.x)**2
        else:
            resid = 0
            
        rlist = self.comm.gather(resid)
        
        if self.master:
            return np.sum(rlist)/(self.nxy*self.nxy*self.nfreq)
        else:
            return resid

   #======================================================================
   # MAIN Iteration - EASY MUFFIN
   #======================================================================

    def update(self):
        
        
        if self.master:
            # rank 0 computes idct 
            self.tf = np.asfortranarray(idct(self.v, axis=2, norm='ortho')) # to check
            
        self.comm.Scatterv([self.tf,self.sendcounts,self.displacements,MPI.DOUBLE],self.t,root=0)

        if not self.master:
            # compute gradient
            tmp = myifftshift( self.ifft2( self.fft2(self.x) *self.hth_fft ) )
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
#                if freq==0 and self.idw==0:
#                    print('wstu1:',np.linalg.norm(wstu))
#                    print('xtt:',np.linalg.norm(self.xtt[:,:,freq] ))
#                    print('xt:',np.linalg.norm(self.xt[:,:,freq] ))
#                    print('')
                
            self.delta = np.asfortranarray(2*self.xt-self.x)

        self.comm.Gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)

        if self.master:
            # update v
            self.vtt = self.v + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(self.deltaf, axis=2, norm='ortho')
            self.v = sat(self.vtt)
        else:
            self.x = self.xt.copy(order='F')

        self.comm.Gatherv(self.x,[self.xf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        self.comm.Gatherv(self.xt,[self.xtf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
#        if self.master:
#            print('x:',np.linalg.norm(self.xf ))
#            print('xt:',np.linalg.norm(self.xtf ))

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
            if self.master:
                print('nitermax must be a positive integer, nitermax=10')
            nitermax=10

        # Iterations
        for niter in range(nitermax):
            self.update()
            if self.master:
                if self.truesky.any():
                    if (niter % 20) ==0:
                        print(str_cst_snr_title.format('It.','Cost','SNR'))
                    print(str_cst_snr.format(niter,self.costlist[-1],self.snrlist[-1]))
                else:
                    if (niter % 20) ==0:
                        print(str_cost_title.format('It.','Cost'))
                    print(str_cost.format(niter,self.costlist[-1]))
                    
        if self.save: 
            self.savexuv()
                    
    def savexuv(self):
 
        self.u_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node 
        
        i = 0
        for val1 in self.u.values():
            for j in self.nbw_decomp:
                self.u_[:,:,i]=val1[j].copy(order='F')
                i+=1
         
        if self.master:
            self.u_ = np.zeros((0))
               
        self.comm.Gatherv(self.u_, [self.uf_,self.sendcountsu,self.displacementsu,MPI.DOUBLE], root=0)
        
        if self.master:
            # re-ranger u en dictionnaire
            udicti = {}
            nfreqi = 0
             
            for i in range(self.nfreq):
                for j in self.nbw_decomp:
                    udicti[j]=self.uf_[:,:,nfreqi]
                    nfreqi+=1
                self.uf[i] = udicti.copy()
                
        if self.master:
            np.save(self.odir+'/x0_tst.npy',self.xf)
            np.save(self.odir+'/u.npy',self.uf)
            np.save(self.odir+'/v.npy',self.v)
            np.save(self.odir+'/cost.npy',self.costlist)
            
            if self.truesky is not None:
                np.save(self.odir+'/wmse_tst.npy',self.wmselist)
                np.save(self.odir+'/snr_tst.npy',self.snrlist)
                
        

class EasyMuffinSURE(EasyMuffin):

    def __init__(self,
                 comm = MPI.COMM_WORLD,
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
                 step_mu = [0,0],
                 pixelweighton = 0,
                 bandweighton = 0,
                 fftw = 0,
                 init=0,
                 fol_init=0,
                 save = 0,
                 odir='.'):
        
        self.step_mu = step_mu

        super(EasyMuffinSURE,self).__init__(
                 comm,
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
                 psf,
                 pixelweighton,
                 bandweighton,
                 fftw,
                 init,
                 fol_init,
                 save,
                 odir)


    def init_algo(self):

            super(EasyMuffinSURE,self).init_algo()
            
            if self.master:
                self.psfadj = defadj(self.psf)
                # init Jacobians
                self.Jv = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.Jtf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.Jt = np.zeros((0))
                self.Jx = np.zeros((0))
                 
            else:
                # compute Hn
                self.Hn = np.zeros((self.nxy,self.nxy,self.nfreq))
                self.n = self.n[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq]
                self.Hn = self.conv(self.n,self.psfadj)
                # init Jacobians
                self.Jt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.Jtf = np.zeros((0))
                self.Jx = np.asfortranarray(init_dirty_wiener(self.n, self.psf, self.psfadj, self.mu_wiener))
                self.Jxt = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.Ju = {}
                for freq in range(self.nfreq):
                    self.Ju[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), dtype=np.float,order='F') , self.nbw_decomp)

            # mu_s list
            self.mu_slist = []
            self.mu_slist.append(self.mu_s)

            # mu_l list
            self.mu_llist = []
            self.mu_llist.append(self.mu_l)


            if self.master:
                
                self.x2 = np.zeros((0))
                self.x2f = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                
                self.v2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.vtt2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dv_s = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dv_l = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dv2_s = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dv2_l = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                
                self.dx_sf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dx_lf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dx2_sf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F') #################" g pas vraiment besoin de ça
                self.dx2_lf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dx_s = np.zeros(0)
                self.dx_l = np.zeros(0)
                self.dx2_s = np.zeros(0)
                self.dx2_l = np.zeros(0)
                
                self.dxt_s  = np.zeros(0)
                self.dxt_l  = np.zeros(0)
                self.dxt2_s = np.zeros(0)
                self.dxt2_l = np.zeros(0)
                
                self.dt_sf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dt_s = np.zeros(0)
                self.dt_lf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dt_l = np.zeros(0)
                self.dt2_sf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dt2_s = np.zeros(0)
                self.dt2_lf = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dt2_l = np.zeros(0)
               
                self.t2 = np.zeros(0)
                self.t2f = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                
                self.xt2f = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.xt2 = np.zeros(0)
                
                self.u2f = {}
                self.u2 = {}
                self.du_sf = {}
                self.du_lf = {}
                self.du2_sf = {}
                self.du2_lf = {}
                self.du_s = {}
                self.du_l = {}
                self.du2_s = {}
                self.du2_l = {}
                
#                if self.init:
#                    self.v2 = np.load(self.fol_init+'/v2.npy')
#                    self.dv_s = np.load(self.fol_init+'/dv_s.npy')
#                    self.dv_l = np.load(self.fol_init+'/dv_l.npy')
#                    self.dv2_s = np.load(self.fol_init+'/dv2_s.npy')
#                    self.dv2_l = np.load(self.fol_init+'/dv2_l.npy')
            
            else:
                self.DeltaSURE  = np.asfortranarray(self.DeltaSURE[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
                self.dirty2 = np.asfortranarray(self.dirty2[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
                self.xt2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.xt2f = np.zeros(0)
                self.u2 = {}
                for freq in range(self.nfreq):
                    self.u2[freq] = self.Decomp(np.zeros((self.nxy,self.nxy)) , self.nbw_decomp)
                
                self.x2 = np.asfortranarray(init_dirty_wiener(self.dirty2, self.psf, self.psfadj, self.mu_wiener))
                self.x2f = np.zeros(0)
                self.fty2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                tmp = myifftshift(self.ifft2(self.fft2(self.dirty2)*self.psfadj_fft))
                self.fty2 = tmp.real.copy(order='F')
                self.xtt2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float, order='F')
                self.utt2 = {}
                for freq in range(self.nfreq):
                    self.utt2[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), dtype=np.float,order='F') , self.nbw_decomp)
                
                self.dx_s = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dx_l = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dx2_s = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dx2_l = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dx_sf = np.zeros(0)
                self.dx_lf = np.zeros(0)
                self.dx2_sf = np.zeros(0)
                self.dx2_lf = np.zeros(0)
                
                self.dxt_s = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dxt_l = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dxt2_s = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dxt2_l = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                
                self.dt_s = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dt_sf = np.zeros(0)
                self.dt_l = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dt_lf = np.zeros(0)
                self.dt2_s = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dt2_sf = np.zeros(0)
                self.dt2_l = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F')
                self.dt2_lf = np.zeros(0)
                
                self.du_l = {}
                for freq in range(self.nfreq):
                    self.du_l[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), dtype=np.float,order='F') , self.nbw_decomp)
                self.du_s = {}
                for freq in range(self.nfreq):
                    self.du_s[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), dtype=np.float,order='F') , self.nbw_decomp)
                self.du2_l = {}
                for freq in range(self.nfreq):
                    self.du2_l[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), dtype=np.float,order='F') , self.nbw_decomp)
                self.du2_s = {}
                for freq in range(self.nfreq):
                    self.du2_s[freq] = self.Decomp(np.zeros((self.nxy,self.nxy), dtype=np.float,order='F') , self.nbw_decomp)
                
#                if self.init:
#                    tmp = np.ndarray.tolist(np.load(self.fol_init+'/u2.npy'))
#                    for freq in range(self.nfreq) :
#                        self.u2[freq] = tmp[self.nf2[self.idw]+freq]
#                    
#                    tmp = np.ndarray.tolist(np.load(self.fol_init+'/du_s.npy'))
#                    for freq in range(self.nfreq) :
#                        self.du_s[freq] = tmp[self.nf2[self.idw]+freq]
#                        
#                    tmp = np.ndarray.tolist(np.load(self.fol_init+'/du_l.npy'))
#                    for freq in range(self.nfreq) :
#                        self.du_l[freq] = tmp[self.nf2[self.idw]+freq]
#                        
#                    tmp = np.ndarray.tolist(np.load(self.fol_init+'/du2_s.npy'))
#                    for freq in range(self.nfreq) :
#                        self.du2_s[freq] = tmp[self.nf2[self.idw]+freq]
#                    
#                    tmp = np.ndarray.tolist(np.load(self.fol_init+'/du2_l.npy'))
#                    for freq in range(self.nfreq) :
#                        self.du2_l[freq] = tmp[self.nf2[self.idw]+freq]
#                    
#                    self.x2 = np.load(self.fol_init+'/x2.npy') 
#                    self.x2 = np.asfortranarray(self.x2[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
#                
#                    self.dx_s = np.load(self.fol_init+'/dx_s.npy') 
#                    self.dx_s = np.asfortranarray(self.dx_s[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
#                    self.dx_l = np.load(self.fol_init+'/dx_l.npy') 
#                    self.dx_l = np.asfortranarray(self.dx_l[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
#                    self.dx2_l = np.load(self.fol_init+'/dx2_l.npy') 
#                    self.dx2_l = np.asfortranarray(self.dx2_l[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
#                    self.dx2_s = np.load(self.fol_init+'/dx2_s.npy') 
#                    self.dx2_s = np.asfortranarray(self.dx2_s[:,:,self.nf2[self.idw]:self.nf2[self.idw]+self.nfreq])
                    
                self.u2f = {}
                self.du_sf = {}
                self.du_lf = {}
                self.du2_sf = {}
                self.du2_lf = {}
            
                self.t2f = np.zeros(0)
                self.t2 = np.zeros((self.nxy,self.nxy,self.nfreq), dtype=np.float,order='F') 
            
            if self.init:
                
                self.step_mu = np.load(self.fol_init+'/step_mu.npy').tolist()
                
                if self.master:
                    tmp1 = np.load(self.fol_init+'/mu_s_tst.npy')
                    tmp2 = np.load(self.fol_init+'/mu_l_tst.npy')
                    tmp1 = tmp1[-1]
                    tmp2 = tmp2[-1]
                else:
                    tmp1 = 0
                    tmp2 = 0
                    
                self.mu_s = self.comm.bcast(tmp1,root=0)
                self.mu_l = self.comm.bcast(tmp2,root=0)    
                
                self.mu_slist[-1]=self.mu_s
                self.mu_llist[-1]=self.mu_l
            
                if self.master:                     
                    # load v at master node 
                    self.v2 = np.load(self.fol_init+'/v2.npy')
                    self.dv_s = np.load(self.fol_init+'/dv_s.npy')
                    self.dv_l = np.load(self.fol_init+'/dv_l.npy')
                    self.dv2_s = np.load(self.fol_init+'/dv2_s.npy')
                    self.dv2_l = np.load(self.fol_init+'/dv2_l.npy')
                
                    # load x2 dx_ dx2_ ... and send to nodes 
                    self.x2f = np.load(self.fol_init+'/x2.npy')
                    self.dx_sf = np.load(self.fol_init+'/dx_s.npy') 
                    self.dx_lf = np.load(self.fol_init+'/dx_l.npy') 
                    self.dx2_lf = np.load(self.fol_init+'/dx2_l.npy') 
                    self.dx2_sf = np.load(self.fol_init+'/dx2_s.npy') 
                
                self.comm.Scatterv([self.x2f,self.sendcounts,self.displacements,MPI.DOUBLE],self.x2,root=0)
                self.comm.Scatterv([self.dx_sf,self.sendcounts,self.displacements,MPI.DOUBLE],self.dx_s,root=0)
                self.comm.Scatterv([self.dx_lf,self.sendcounts,self.displacements,MPI.DOUBLE],self.dx_l,root=0)
                self.comm.Scatterv([self.dx2_sf,self.sendcounts,self.displacements,MPI.DOUBLE],self.dx2_s,root=0)
                self.comm.Scatterv([self.dx2_lf,self.sendcounts,self.displacements,MPI.DOUBLE],self.dx2_l,root=0)
            
                # u2
                if self.master:
                    self.uf = {}
                    self.uf_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F')  
                else:
                    self.u_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') 
                    
                if self.master:
                    self.uf = np.ndarray.tolist(np.load(self.fol_init+'/u2.npy'))
                    i = 0
                    for val1 in self.uf.values():
                        for j in self.nbw_decomp:
                            self.uf_[:,:,i]=val1[j].copy()
                            i+=1
          
                self.comm.Scatterv([self.uf_,self.sendcountsu,self.displacementsu,MPI.DOUBLE],self.u_,root=0)
            
                if not self.master:
                    udicti = {}
                    nfreqi=0
                    for i in range(self.nfreq):
                        for j in self.nbw_decomp:
                            udicti[j]=self.u_[:,:,nfreqi]
                            nfreqi+=1
                        self.u2[i]=udicti.copy()
                
                # du_s
                if self.master:
                    self.uf = {}
                    self.uf_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F')  
                else:
                    self.u_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') 
                    
                if self.master and self.init:
                    self.uf = np.ndarray.tolist(np.load(self.fol_init+'/du_s.npy'))
                    i = 0
                    for val1 in self.uf.values():
                        for j in self.nbw_decomp:
                            self.uf_[:,:,i]=val1[j].copy()
                            i+=1
          
                self.comm.Scatterv([self.uf_,self.sendcountsu,self.displacementsu,MPI.DOUBLE],self.u_,root=0)
            
            
                if not self.master:
                    udicti = {}
                    nfreqi=0
                    for i in range(self.nfreq):
                        for j in self.nbw_decomp:
                            udicti[j]=self.u_[:,:,nfreqi]
                            nfreqi+=1
                        self.du_s[i]=udicti.copy()
                
                # du_l
                if self.master:
                    self.uf = {}
                    self.uf_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F')  
                else:
                    self.u_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') 
                    
                if self.master:
                    self.uf = np.ndarray.tolist(np.load(self.fol_init+'/du_l.npy'))
                    i = 0
                    for val1 in self.uf.values():
                        for j in self.nbw_decomp:
                            self.uf_[:,:,i]=val1[j].copy()
                            i+=1
          
                self.comm.Scatterv([self.uf_,self.sendcountsu,self.displacementsu,MPI.DOUBLE],self.u_,root=0)
            
                if not self.master:
                    udicti = {}
                    nfreqi=0
                    for i in range(self.nfreq):
                        for j in self.nbw_decomp:
                            udicti[j]=self.u_[:,:,nfreqi]
                            nfreqi+=1
                        self.du_l[i]=udicti.copy()

                # du2_s
                if self.master:
                    self.uf = {}
                    self.uf_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F')  
                else:
                    self.u_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') 
                    
                if self.master:
                    self.uf = np.ndarray.tolist(np.load(self.fol_init+'/du2_s.npy'))
                    i = 0
                    for val1 in self.uf.values():
                        for j in self.nbw_decomp:
                            self.uf_[:,:,i]=val1[j].copy()
                            i+=1
          
                self.comm.Scatterv([self.uf_,self.sendcountsu,self.displacementsu,MPI.DOUBLE],self.u_,root=0)
            
                if not self.master:
                    udicti = {}
                    nfreqi=0
                    for i in range(self.nfreq):
                        for j in self.nbw_decomp:
                            udicti[j]=self.u_[:,:,nfreqi]
                            nfreqi+=1
                        self.du2_s[i]=udicti.copy()

                # du2_l
                if self.master:
                    self.uf = {}
                    self.uf_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F')  
                else:
                    self.u_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') 
                
                if self.master:
                    self.uf = np.ndarray.tolist(np.load(self.fol_init+'/du2_l.npy'))
                    i = 0
                    for val1 in self.uf.values():
                        for j in self.nbw_decomp:
                            self.uf_[:,:,i]=val1[j].copy()
                            i+=1
          
                self.comm.Scatterv([self.uf_,self.sendcountsu,self.displacementsu,MPI.DOUBLE],self.u_,root=0)
            
                if not self.master:
                    udicti = {}
                    nfreqi=0
                    for i in range(self.nfreq):
                        for j in self.nbw_decomp:
                            udicti[j]=self.u_[:,:,nfreqi]
                            nfreqi+=1
                        self.du2_l[i]=udicti.copy()
            
            self.comm.Gatherv(self.x2,[self.x2f,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
            
            self.sugarfdmclist = {}
            self.sugarfdmclist[0] = []
            self.sugarfdmclist[1] = []
            self.sugarfdmc()
            
            self.wmselistsurefdmc = []
            self.wmselistsure = []
            self.wmselistsure.append(self.wmsesure())
            
            if self.truesky.any():
                self.psnrlistsure = []
                self.psnrlistsure.append(self.psnrsure())
            
            self.wmselistsurefdmc.append(self.wmsesurefdmc())
            
    def wmsesure(self):
        if self.master:
            wmse = 0
        else:
            tmp = self.dirty - self.conv(self.x,self.psf)
            LS_cst = np.linalg.norm(tmp)**2
            tmp = self.n*self.conv(self.Jx,self.psf)
            wmse = LS_cst + 2*(self.var)*(np.sum(tmp))
        
        wmse_lst = self.comm.gather(wmse)
        
        if self.master:
            return sum(wmse_lst)/(self.nxy*self.nxy*self.nfreq) - self.var
        else:
            return wmse

    def wmsesurefdmc(self):
        
        if self.master:
            wmse = 0
        else:
            tmp = self.dirty - self.conv(self.x,self.psf)
            LS_cst = np.linalg.norm(tmp)**2        
            wmse = LS_cst + 2*(self.var/self.eps)*np.sum(  ( self.conv(self.x2,self.psf) - self.conv(self.x,self.psf) )*self.DeltaSURE  )
        
        wmse_lst = self.comm.gather(wmse)
        
        if self.master:
            return sum(wmse_lst)/(self.nxy*self.nxy*self.nfreq) - self.var
        else:
            return wmse
        
    def psnrsure(self):
        if self.master:
            return 10*np.log10(self.psnrnum/self.wmselistsure[-1])
        else:
            return 0

    def update_jacobians(self):
        if self.master:
            self.Jtf = np.asfortranarray(idct(self.Jv, axis=2,norm='ortho'))
        
        self.comm.Scatterv([self.Jtf,self.sendcounts,self.displacements,MPI.DOUBLE],self.Jt,root=0)
        
        if not self.master:
            tmp = myifftshift( self.ifft2( self.fft2(self.Jx) * self.hth_fft ) )
            JDelta_freq = tmp.real- self.Hn
            for freq in range(self.nfreq):
                # compute iuwt adjoint
                self.Js_l = self.Recomp(self.Ju[freq], self.nbw_recomp)
                # compute xt
                Jxtt = self.Jx[:,:,freq] - self.tau*(JDelta_freq[:,:,freq] + self.mu_s*self.alpha_s[freq]*self.Js_l + self.mu_l*self.alpha_l*self.Jt[:,:,freq])
                self.Jxt[:,:,freq] = heavy(self.xtt[:,:,freq])*Jxtt
                # update u
                tmp_spat_scal_J = self.Decomp(2*self.Jxt[:,:,freq] - self.Jx[:,:,freq] , self.nbw_decomp)
                for b in self.nbw_decomp:
                    Jutt = self.Ju[freq][b] + self.sigma*self.mu_s*self.alpha_s[freq]*tmp_spat_scal_J[b]
                    self.Ju[freq][b] = rect( self.utt[freq][b] )*Jutt
            self.delta = np.asfortranarray(2*self.Jxt-self.Jx)
            
        self.comm.Gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
        if self.master:
            Jvtt = self.Jv + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(self.deltaf, axis=2, norm='ortho')
            self.Jv = rect(self.vtt)*Jvtt
        else:
            self.Jx = self.Jxt.copy(order='F')
        
        # wmsesure
        self.wmselistsure.append(self.wmsesure())
        # psnrsure
        if self.truesky.any():
            self.psnrlistsure.append(self.psnrsure())
            
        return self.wmselistsure[-1]
        
    def loop(self,nitermax=10):
        """ main loop """

        if nitermax < 1:
            if self.master:
                print('nitermax must be a positive integer, nitermax=10')
                nitermax=10
        for niter in range(nitermax):
            self.mu_slist.append(self.mu_s)
            self.mu_llist.append(self.mu_l)
            super(EasyMuffinSURE,self).update()
            self.update_jacobians()
            self.nitertot+=1

            if self.master:
                if self.truesky.any():
                    if (niter % 20) ==0:
                        print(str_cst_snr_wmse_wmsesure_title.format('It.','Cost','SNR','WMSE','WMSES'))
                    print(str_cst_snr_wmse_wmsesure.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsure[-1]))
                else:
                    if (niter % 20) ==0:
                        print(str_cost_wmsesure_title.format('It.','Cost','WMSES'))
                    print(str_cost_wmsesure.format(niter,self.costlist[-1],self.wmselistsure[-1]))
         
        if self.save: 
            self.savexuv()                

    # run update with y + eps*delta
    def update2(self):
        if self.master:
            self.t2f = np.asfortranarray(idct(self.v2, axis=2, norm='ortho')) # to check
        
        self.comm.Scatterv([self.t2f,self.sendcounts,self.displacements,MPI.DOUBLE],self.t2,root=0)
        
        if not self.master:
            tmp = myifftshift( self.ifft2( self.fft2(self.x2) *self.hth_fft ) )
            Delta_freq = tmp.real- self.fty2
            for freq in range(self.nfreq):
                # compute iuwt adjoint
                wstu = self.Recomp(self.u2[freq], self.nbw_recomp)
                # compute xt
                self.xtt2[:,:,freq] = self.x2[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + self.mu_s*self.alpha_s[freq]*wstu + self.mu_l*self.alpha_l*self.t2[:,:,freq])
                self.xt2[:,:,freq] = np.maximum(self.xtt2[:,:,freq], 0.0 )
                # update u
                tmp_spat_scal = self.Decomp(2*self.xt2[:,:,freq] - self.x2[:,:,freq] , self.nbw_decomp)
                for b in self.nbw_decomp:
                    self.utt2[freq][b] = self.u2[freq][b] + self.sigma*self.mu_s*self.alpha_s[freq]*tmp_spat_scal[b]
                    self.u2[freq][b] = sat(self.utt2[freq][b])    
#                if freq==0 and self.idw==0:
#                    print('wstu1:',np.linalg.norm(wstu))
#                    print('xtt:',np.linalg.norm(self.xtt2[:,:,freq] ))
#                    print('xt:',np.linalg.norm(self.xt2[:,:,freq] ))
#                    print('')
                    
            self.delta = np.asfortranarray(2*self.xt2-self.x2)
        
        self.comm.Gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
        if self.master:
            self.vtt2 = self.v2 + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(self.deltaf, axis=2, norm='ortho')
            self.v2 = sat(self.vtt2)
        else:
            self.x2 = self.xt2.copy(order='F')    
            
        self.comm.Gatherv(self.x2,[self.x2f,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        self.wmselistsurefdmc.append(self.wmsesurefdmc())
        self.comm.Gatherv(self.xt2,[self.xt2f,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)   
        
#        if self.master:
#            print('x:',np.linalg.norm(self.x2f ))
#            print('xt:',np.linalg.norm(self.xt2f ))
        
    def dx_mu(self):
        
        if self.master:
            self.dt_sf = np.asfortranarray(idct(self.dv_s, axis=2, norm='ortho'))
            
        self.comm.Scatterv([self.dt_sf,self.sendcounts,self.displacements,MPI.DOUBLE],self.dt_s,root=0)
        
        if not self.master:
            tmp = myifftshift( self.ifft2( self.fft2(self.dx_s) *self.hth_fft ) )
            Delta_freq = tmp.real #- self.fty
            
            for freq in range(self.nfreq):

                # compute iuwt adjoint
                wstu = self.alpha_s[freq]*self.Recomp(self.u[freq], self.nbw_recomp) + self.mu_s*self.alpha_s[freq]*self.Recomp(self.du_s[freq], self.nbw_recomp)
                # compute xt
                dxtt_s = self.dx_s[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + wstu + self.mu_l*self.alpha_l*self.dt_s[:,:,freq])
                self.dxt_s[:,:,freq] = heavy(self.xtt[:,:,freq] )*dxtt_s
                # update u
                tmp_spat_scal = self.Decomp(self.alpha_s[freq]*(2*self.xt[:,:,freq] - self.x[:,:,freq]) + self.mu_s*self.alpha_s[freq]*(2*self.dxt_s[:,:,freq] - self.dx_s[:,:,freq]), self.nbw_decomp)
    
                for b in self.nbw_decomp:
                    dutt_s = self.du_s[freq][b] + self.sigma*tmp_spat_scal[b]
                    self.du_s[freq][b] = rect(self.utt[freq][b])*dutt_s
                
#                if freq==0 and self.idw==0:
#                    print('wstu1:',np.linalg.norm(wstu))
#                    print('xtt:',np.linalg.norm(self.xtt[:,:,freq] ))
#                    print('xt:',np.linalg.norm(self.dxt_s[:,:,freq] ))
#                    print('')
                    
            self.delta = np.asfortranarray(2*self.dxt_s-self.dx_s)
            
        self.comm.Gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)

        if self.master:
            # update v
            dvtt_s = self.dv_s + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(self.deltaf, axis=2, norm='ortho')
            self.dv_s = rect(self.vtt)*dvtt_s
        else:
            self.dx_s = self.dxt_s.copy(order='F')
            
#        self.comm.Gatherv(self.dx_s,[self.dx_sf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
#        if self.master:
#            print('x:',np.linalg.norm(self.dx_sf ))
        
        ##
        if self.master:
            self.dt_lf = np.asfortranarray(idct(self.dv_l*self.mu_l*self.alpha_l[...,None] + self.v*self.alpha_l[...,None], axis=2, norm='ortho'))

        self.comm.Scatterv([self.dt_lf,self.sendcounts,self.displacements,MPI.DOUBLE],self.dt_l,root=0)
        
        if not self.master:
            tmp = myifftshift( self.ifft2( self.fft2(self.dx_l) *self.hth_fft ) )
            Delta_freq = tmp.real
            for freq in range(self.nfreq):
                # compute iuwt adjoint
                wstu = self.mu_s*self.alpha_s[freq]*self.Recomp(self.du_l[freq], self.nbw_recomp)
    
                # compute xt
                dxtt_l = self.dx_l[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + wstu + self.dt_l[:,:,freq])
                self.dxt_l[:,:,freq] = heavy(self.xtt[:,:,freq] )*dxtt_l
    
                # update u
                tmp_spat_scal = self.Decomp(self.mu_s*self.alpha_s[freq]*(2*self.dxt_l[:,:,freq] - self.dx_l[:,:,freq]), self.nbw_decomp)
    
                for b in self.nbw_decomp:
                    dutt_l = self.du_l[freq][b] + self.sigma*tmp_spat_scal[b]
                    self.du_l[freq][b] = rect(self.utt[freq][b])*dutt_l
                    
            self.delta = np.asfortranarray(2*self.dxt_l - self.dx_l)
                
        self.comm.Gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
        if self.master:
            dvtt_l = self.dv_l + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(self.deltaf, axis=2, norm='ortho') + self.sigma*self.alpha_l[...,None]*dct(2*self.xtf - self.xf, axis=2, norm='ortho')
            self.dv_l = rect(self.vtt)*dvtt_l
        else:
            self.dx_l = self.dxt_l.copy(order='F')
            
#        self.comm.Gatherv(self.dx_l,[self.dx_lf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
    def dx2_mu(self):
        if self.master:
            self.dt2_sf = np.asfortranarray(idct(self.dv2_s, axis=2, norm='ortho'))
            
        self.comm.Scatterv([self.dt2_sf,self.sendcounts,self.displacements,MPI.DOUBLE],self.dt2_s,root=0)
        
        if not self.master:
            tmp = myifftshift( self.ifft2( self.fft2(self.dx2_s) *self.hth_fft ) )
            Delta_freq = tmp.real #- self.fty
            for freq in range(self.nfreq):
                # compute iuwt adjoint
                wstu = self.alpha_s[freq]*self.Recomp(self.u2[freq], self.nbw_recomp) + self.mu_s*self.alpha_s[freq]*self.Recomp(self.du2_s[freq], self.nbw_recomp)
                # compute xt
                dxtt_s = self.dx2_s[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + wstu + self.mu_l*self.alpha_l*self.dt2_s[:,:,freq])
                self.dxt2_s[:,:,freq] = heavy(self.xtt2[:,:,freq] )*dxtt_s
                # update u
                tmp_spat_scal = self.Decomp(self.alpha_s[freq]*(2*self.xt2[:,:,freq] - self.x2[:,:,freq]) + self.mu_s*self.alpha_s[freq]*(2*self.dxt2_s[:,:,freq] - self.dx2_s[:,:,freq]), self.nbw_decomp)   
                for b in self.nbw_decomp:
                    dutt_s = self.du2_s[freq][b] + self.sigma*tmp_spat_scal[b]
                    self.du2_s[freq][b] = rect(self.utt2[freq][b])*dutt_s
                
#                if freq==0 and self.idw==0:
#                    print('wstu1:',np.linalg.norm(wstu))
#                    print('xtt:',np.linalg.norm(self.xtt2[:,:,freq] ))
#                    print('xt:',np.linalg.norm(self.dxt2_s[:,:,freq] ))
#                    print('')
                    
            self.delta = np.asfortranarray(2*self.dxt2_s-self.dx2_s)
            
        self.comm.Gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
        if self.master:
            dvtt2_s = self.dv2_s + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(self.deltaf, axis=2, norm='ortho')
            self.dv2_s = rect(self.vtt2)*dvtt2_s
        else:
            self.dx2_s = self.dxt2_s.copy(order='F')

#        self.comm.Gatherv(self.dx2_s,[self.dx2_sf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
#        
#        if self.master:
#            print('x:',np.linalg.norm(self.dx2_sf ))
            
        if self.master:
            self.dt2_lf = np.asfortranarray(idct(self.dv2_l*self.mu_l*self.alpha_l[...,None] + self.v2*self.alpha_l[...,None], axis=2, norm='ortho'))
#            print('1:',np.linalg.norm(self.dv2_l))
#            print('2:',np.linalg.norm(self.v2))
#            print('3:',np.linalg.norm(self.dt2_lf))
        
        self.comm.Scatterv([self.dt2_lf,self.sendcounts,self.displacements,MPI.DOUBLE],self.dt2_l,root=0)
        
        if not self.master:
            tmp = myifftshift( self.ifft2( self.fft2(self.dx2_l) *self.hth_fft ) )
            Delta_freq = tmp.real #- self.fty
            for freq in range(self.nfreq):
                # compute iuwt adjoint
                wstu = self.mu_s*self.alpha_s[freq]*self.Recomp(self.du2_l[freq], self.nbw_recomp)
                # compute xt
                dxtt_l = self.dx2_l[:,:,freq] - self.tau*(Delta_freq[:,:,freq] + wstu + self.dt2_l[:,:,freq])
                self.dxt2_l[:,:,freq] = heavy(self.xtt2[:,:,freq] )*dxtt_l
                # update u
                tmp_spat_scal = self.Decomp(self.mu_s*self.alpha_s[freq]*(2*self.dxt2_l[:,:,freq] - self.dx2_l[:,:,freq]), self.nbw_decomp)
                for b in self.nbw_decomp:
                    dutt_l = self.du2_l[freq][b] + self.sigma*tmp_spat_scal[b]
                    self.du2_l[freq][b] = rect(self.utt2[freq][b])*dutt_l
                
#                if freq==0 and self.idw==0:
#                    print('4:',np.linalg.norm(self.dt2_l[:,:,freq]))
#                    print('')
                    
            self.delta = np.asfortranarray(2*self.dxt2_l-self.dx2_l)
            
        self.comm.Gatherv(self.delta,[self.deltaf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        
        if self.master:
            dvtt2_l = self.dv2_l + self.sigma*self.mu_l*self.alpha_l[...,None]*dct(self.deltaf, axis=2, norm='ortho') + self.sigma*self.alpha_l[...,None]*dct(2*self.xt2f - self.x2f, axis=2, norm='ortho')
            
            self.dv2_l = rect(self.vtt2)*dvtt2_l
        else:
            self.dx2_l = self.dxt2_l.copy(order='F')
            
#        self.comm.Gatherv(self.dx2_l,[self.dx2_lf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
        

    def sugarfdmc(self):
        if self.master:
            res1 = 0
            res2 = 0
        else:
            tmp = 2*self.conv(self.psf,self.dx_s)*(self.conv(self.psf,self.x)-self.dirty) + 2*self.var*self.conv(self.psf,self.dx2_s-self.dx_s)*self.DeltaSURE/self.eps
            res1 = np.sum(tmp)/(self.nxy*self.nxy)
            tmp = 2*self.conv(self.psf,self.dx_l)*(self.conv(self.psf,self.x)-self.dirty) + 2*self.var*self.conv(self.psf,self.dx2_l-self.dx_l)*self.DeltaSURE/self.eps
            res2 = np.sum(tmp)/(self.nxy*self.nxy)
        
        res1_lst = self.comm.gather(res1)
        res2_lst = self.comm.gather(res2)
        
        if self.master:
            res1 = sum(res1_lst)/self.nfreq
            res2 = sum(res2_lst)/self.nfreq
            
#        self.comm.Gatherv(self.dx_s,[self.dx_sf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
#        self.comm.Gatherv(self.x,[self.xf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
#        self.comm.Gatherv(self.dx2_s,[self.dx2_sf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
#        self.comm.Gatherv(self.dx_l,[self.dx_lf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
#        self.comm.Gatherv(self.dx2_l,[self.dx2_lf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
#        
#        if self.master:
#            tmp = 2*self.conv(self.psf,self.dx_sf)*(self.conv(self.psf,self.xf)-self.dirty) + 2*self.var*self.conv(self.psf,self.dx2_sf-self.dx_sf)*self.DeltaSURE/self.eps
#            res1 = np.sum(tmp)/(self.nxy*self.nxy*self.nfreq)
#
#            tmp = 2*self.conv(self.psf,self.dx_lf)*(self.conv(self.psf,self.xf)-self.dirty) + 2*self.var*self.conv(self.psf,self.dx2_lf-self.dx_lf)*self.DeltaSURE/self.eps
#            res2 = np.sum(tmp)/(self.nxy*self.nxy*self.nfreq)
#        
#        else:
#            res1 = 0
#            res2 = 0
            
        res1 = self.comm.bcast(res1,root=0) # root bcasts res1 to everyone else
        res2 = self.comm.bcast(res2,root=0) # root bcasts res2 to everyone else
        
        self.sugarfdmclist[0].append(res1)
        self.sugarfdmclist[1].append(res2)
            

    def loop_fdmc(self,nitermax=10):

        if nitermax < 1:
            if self.master:
                print('nitermax must be a positve integer, nitermax=10')
            nitermax=10
            
        for niter in range(nitermax):
            self.mu_slist.append(self.mu_s)
            self.mu_llist.append(self.mu_l)
            super(EasyMuffinSURE,self).update()
            self.update_jacobians()

            self.update2() #
            self.dx_mu() #
            self.dx2_mu() #
            self.sugarfdmc()
            
            if niter>1 and niter%30==0:
                self.graddes_mu(self.step_mu)
                if niter>5000 and niter%1000==0:
                    self.step_mu = [tmp/1.3 for tmp in self.step_mu]

            self.nitertot+=1

            if self.master:
                if self.truesky.any():
                    if (niter % 20) ==0:
                        print(str_cst_snr_wmse_wmsesure_mu_title.format('It.','Cost','SNR','WMSE','WMSES','mu_s','mu_l'))
                    print(str_cst_snr_wmse_wmsesure_mu.format(niter,self.costlist[-1],self.snrlist[-1],self.wmselist[-1],self.wmselistsurefdmc[-1],self.mu_slist[-1],self.mu_llist[-1]))
                else:
                    if (niter % 20) ==0:
                        print(str_cost_wmsesure_mu_title.format('It.','Cost','WMSES','mu_s','mu_l'))
                    print(str_cost_wmsesure_mu.format(niter,self.costlist[-1],self.wmselistsurefdmc[-1],self.mu_slist[-1],self.mu_llist[-1]))
        
        if self.save:
            self.comm.Gatherv(self.dx_l,[self.dx_lf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
            self.comm.Gatherv(self.dx_s,[self.dx_sf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)    
            self.comm.Gatherv(self.dx2_l,[self.dx2_lf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)
            self.comm.Gatherv(self.dx2_s,[self.dx2_sf,self.sendcounts,self.displacements,MPI.DOUBLE],root=0)

            self.savexuv_fdmc()
            self.savexuv()
            

    def savexuv_fdmc(self):

        self.u2_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node 
        self.u2f_ = np.zeros((0))
        if self.master:
            self.u2_ = np.zeros((0))
            self.u2f_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node   
        
        i = 0
        for val1 in self.u2.values():
            for j in self.nbw_decomp:
                self.u2_[:,:,i]=val1[j].copy(order='F')
                i+=1
               
        self.comm.Gatherv(self.u2_, [self.u2f_,self.sendcountsu,self.displacementsu,MPI.DOUBLE], root=0)
        
        if self.master:
            # re-ranger u en dictionnaire
            udicti = {}
            nfreqi = 0
             
            for i in range(self.nfreq):
                for j in self.nbw_decomp:
                    udicti[j]=self.u2f_[:,:,nfreqi]
                    nfreqi+=1
                self.u2f[i] = udicti.copy()
        ##########        
        self.u2_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node 
        if self.master:
            self.u2f_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node   

        i = 0
        for val1 in self.du_s.values():
            for j in self.nbw_decomp:
                self.u2_[:,:,i]=val1[j].copy(order='F')
                i+=1
               
        self.comm.Gatherv(self.u2_, [self.u2f_,self.sendcountsu,self.displacementsu,MPI.DOUBLE], root=0)
        
        if self.master:
            # re-ranger u en dictionnaire
            udicti = {}
            nfreqi = 0
             
            for i in range(self.nfreq):
                for j in self.nbw_decomp:
                    udicti[j]=self.u2f_[:,:,nfreqi]
                    nfreqi+=1
                self.du_sf[i] = udicti.copy()
        #############
        self.u2_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node 
        if self.master:
            self.u2f_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node   

        i = 0
        for val1 in self.du_l.values():
            for j in self.nbw_decomp:
                self.u2_[:,:,i]=val1[j].copy(order='F')
                i+=1
               
        self.comm.Gatherv(self.u2_, [self.u2f_,self.sendcountsu,self.displacementsu,MPI.DOUBLE], root=0)
        
        if self.master:
            # re-ranger u en dictionnaire
            udicti = {}
            nfreqi = 0
             
            for i in range(self.nfreq):
                for j in self.nbw_decomp:
                    udicti[j]=self.u2f_[:,:,nfreqi]
                    nfreqi+=1
                self.du_lf[i] = udicti.copy()
        ############### 
        self.u2_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node 
        if self.master:
            self.u2f_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node   

        i = 0
        for val1 in self.du2_s.values():
            for j in self.nbw_decomp:
                self.u2_[:,:,i]=val1[j].copy(order='F')
                i+=1
               
        self.comm.Gatherv(self.u2_, [self.u2f_,self.sendcountsu,self.displacementsu,MPI.DOUBLE], root=0)
        
        if self.master:
            # re-ranger u en dictionnaire
            udicti = {}
            nfreqi = 0
             
            for i in range(self.nfreq):
                for j in self.nbw_decomp:
                    udicti[j]=self.u2f_[:,:,nfreqi]
                    nfreqi+=1
                self.du2_sf[i] = udicti.copy()
        ################
        self.u2_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node 
        if self.master:
            self.u2f_ = np.zeros((self.nxy,self.nxy,self.nfreq*np.size(self.nbw_decomp)),dtype=np.float,order='F') # defaire self.u at each workers node   

        i = 0
        for val1 in self.du2_l.values():
            for j in self.nbw_decomp:
                self.u2_[:,:,i]=val1[j].copy()
                i+=1
               
        self.comm.Gatherv(self.u2_, [self.u2f_,self.sendcountsu,self.displacementsu,MPI.DOUBLE], root=0)
        
        if self.master:
            # re-ranger u en dictionnaire
            udicti = {}
            nfreqi = 0
             
            for i in range(self.nfreq):
                for j in self.nbw_decomp:
                    udicti[j]=self.u2f_[:,:,nfreqi]
                    nfreqi+=1
                self.du2_lf[i] = udicti.copy()

        if self.master:
            np.save(self.odir+'/x2.npy',self.x2f)
            np.save(self.odir+'/u2.npy',self.u2f)
            np.save(self.odir+'/v2.npy',self.v2)
            
            np.save(self.odir+'/dx_s.npy',self.dx_sf)
            np.save(self.odir+'/dx_l.npy',self.dx_lf)
            np.save(self.odir+'/dx2_s.npy',self.dx2_sf)
            np.save(self.odir+'/dx2_l.npy',self.dx2_lf)
            
            np.save(self.odir+'/dv_s.npy',self.dv_s)
            np.save(self.odir+'/dv_l.npy',self.dv_l)
            np.save(self.odir+'/dv2_s.npy',self.dv2_s)
            np.save(self.odir+'/dv2_l.npy',self.dv2_l)
            
            np.save(self.odir+'/du_s.npy',self.du_sf)
            np.save(self.odir+'/du_l.npy',self.du_lf)
            np.save(self.odir+'/du2_s.npy',self.du2_sf)
            np.save(self.odir+'/du2_l.npy',self.du2_lf)
            
            np.save(self.odir+'/wmses_tst.npy',self.wmselistsure)
            np.save(self.odir+'/wmsesfdmc_tst.npy',self.wmselistsurefdmc)
            np.save(self.odir+'/mu_s_tst.npy',self.mu_slist)
            np.save(self.odir+'/mu_l_tst.npy',self.mu_llist)
            np.save(self.odir+'/dxs.npy',self.dx_sf)
            np.save(self.odir+'/dxl.npy',self.dx_lf)
            np.save(self.odir+'/sugar0.npy',self.sugarfdmclist[0])
            np.save(self.odir+'/sugar1.npy',self.sugarfdmclist[1])
            np.save(self.odir+'/cost.npy',self.costlist)
            np.save(self.odir+'/psnrsure.npy',self.psnrlistsure)
            
            np.save(self.odir+'/step_mu.npy',self.step_mu)
                

    def graddes_mu(self,step=[1e-3,1e-3]):
        self.mu_s = np.maximum(self.mu_s - step[0]*self.sugarfdmclist[0][-1],0)
        self.mu_l = np.maximum(self.mu_l - step[1]*self.sugarfdmclist[1][-1],0)

