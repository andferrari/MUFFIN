#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:00:55 2018

@author: rammanouil
"""

# ==============================================================================
# OPEN PSF AND DIRTY CUBE - SKY to check results
# ==============================================================================
import os
import numpy as np
from astropy.io import fits
import sys
from deconv3d_tools import conv
from mpi4py import MPI
import deconv3D_mpi2 as dcvMpi
from tictoc import tic, toc
from datetime import datetime
import matplotlib.pyplot as pl 

#%% ===========================================================================
# MPI Run
# =============================================================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


#%% ===========================================================================
# Saving output results
# =============================================================================


class Logger():
    
    def __init__(self,filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

if rank==0:
    daytime = str(datetime.now())
    file = os.path.join(os.getcwd(),'output/'+daytime+'_out.txt')
    sys.stdout = Logger(file)
    #sys.stdout = open(file,'w')

# =============================================================================
# Input
# =============================================================================
if len(sys.argv)==8:
    L = int(sys.argv[1])
    nitermax = int(sys.argv[2])
    mu_s = float(sys.argv[3])
    mu_l = float(sys.argv[4])
    step_mu_l = float(sys.argv[5])
    step_mu_s = float(sys.argv[6])
    step_mu = [step_mu_l, step_mu_s]
    data_suffix = sys.argv[7]
elif len(sys.argv)==1:
    L = 10
    nitermax = 10
    mu_s = 0.2
    mu_l = 2.2
    data_suffix = 'M31_3d_conv_256_10db'
    step_mu_l = 1e-3
    step_mu_s = 1e-3
    step_mu = [step_mu_l, step_mu_s]
else:
    if rank==0:
        print('')
        print('-'*100)
        print('You should input: L nitermax mu_s mu_l data_suffix step_mu_l step_mu_s ')
        print('')
        print('L: number of bands to be considered')
        print('nitermax: maximum number of iterations ')
        print('mu_s, mu_l: spatial, spectral reg.')
        print('data_suffix: name suffix of data in folder data256')
        print('step_mu_l step_mu_s : step for gradient descent')
        print('')
        print('            **** ex: mpirun -np 4 python3 Run_GS.py 10 0.2 0.2 1e-3 1e-3 M31_3d_conv_256_10db                ')
        print('')
        print('-'*100)
        print('')
    sys.exit()

if rank==0:
    print('')
    print('L: ',L)
    print('nitermax: ',nitermax)
    print('mu_s: ',mu_s)
    print('mu_l: ',mu_l)
    print('data_suffix',data_suffix)
    print('step_mu: ',step_mu)

# =============================================================================
# Load data
# =============================================================================
def checkdim(x):
    if len(x.shape) == 4:
        x = np.squeeze(x)
        x = x.transpose((2, 1, 0))
    return x

folder = 'data'
#folder = 'data256Eusipco'
file_in = data_suffix
folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]
#CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,-L:]
#CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,-L:]
skyname = genname+'_sky.fits'

if os.path.isfile(skyname):
    if rank==0:
        print('')
        print('estimating variance')
        
    #sky = checkdim(fits.getdata(skyname, ext=0))[:,:,0:L]
    sky = checkdim(fits.getdata(skyname, ext=0))
    #sky = np.transpose(sky)[:,:,0:L]
    
    sky = sky[:,:,0:L]
    #sky = sky[:,:,-L:]
    
    sky2 = np.sum(sky*sky)
    Noise = CubeDirty - conv(CubePSF,sky)
    var = np.sum(Noise**2)/Noise.size
    if rank==0:
        print('')
        print('setting var to ', var)
else:
    var = 0.0
    sky = None
    if rank==0:
        print('')
        print('setting var to ', var)


#pl.figure()
#pl.imshow(sky[:,:,1])
#
#pl.figure()
#pl.imshow(CubePSF[:,:,1])
#
#pl.figure()
#pl.imshow(CubeDirty[:,:,1])


#%% ===========================================================================
# Run
# =============================================================================
# DWT parameters
nb=('db1','db2','db3','db4','db5','db6','db7','db8')
#nb = (7,0)

args = {'mu_s':mu_s,'mu_l':mu_l,'nb':nb,'truesky':sky,'psf':CubePSF,'dirty':CubeDirty,'var':var,'step_mu':step_mu}
tic()

EM= dcvMpi.EasyMuffinSURE(**args)
if rank==0:
    print('using tau: ',EM.tau)
    print('')

EM.loop_fdmc(nitermax)


if rank==2:
    print('')
    print('x - x2',np.linalg.norm(EM.x-EM.x2))
    print('')
    print('dx_s - dx2_s',np.linalg.norm(EM.dx2_s-EM.dx_s))
    print('')
    print('dx_l - dx2_l',np.linalg.norm(EM.dx2_l-EM.dx_l))
    print('')
    print('dx_s',np.linalg.norm(EM.dx_l))
    print('')
    print('dx_l',np.linalg.norm(EM.dx_s))
    
    
#%% ==================================dct=========================================
# Save results
# =============================================================================
if rank==0:
    drctry = os.path.join(os.getcwd(), 'output/'+daytime)
    os.mkdir(drctry)
    os.chdir(drctry)

    toc()
        
    if sky is not None:
        np.save('x0_tst.npy',EM.xf)
        np.save('wmse_tst.npy',EM.wmselist)
        np.save('wmses_tst.npy',EM.wmselistsure)
        np.save('wmsesfdmc_tst.npy',EM.wmselistsurefdmc)
        np.save('snr_tst.npy',EM.snrlist)
        np.save('mu_s_tst.npy',EM.mu_slist)
        np.save('mu_l_tst.npy',EM.mu_llist)
        np.save('dxs.npy',EM.dx_sf)
        np.save('dxl.npy',EM.dx_lf)
        np.save('sugar0.npy',EM.sugarfdmclist[0])
        np.save('sugar1.npy',EM.sugarfdmclist[1])
        np.save('cost.npy',EM.costlist)
        np.save('psnrsure.npy',EM.psnrlistsure)
    else:
        np.save('x0_tst.npy',EM.xf)
        np.save('wmses_tst.npy',EM.wmselistsure)
        np.save('mu_s_tst.npy',mu_s)
        np.save('mu_l_tst.npy',mu_l)


## ==============================================================================
## Print Figures
#%% ==============================================================================

#daytime = '2018-01-25 17:38:52.365628'
#daytime = '2018-01-26 10:07:59.906797' # mu_l 5
daytime = '2018-01-26 17:47:39.160249' # mu_l 0.5
daytime = '2018-01-31 14:24:39.166184'
daytime = '2018-01-31 14:48:01.443285'
daytime = '2018-01-31 14:52:59.876755'
daytime = '2018-01-31 14:59:00.813077'

## Batch 1 
daytime = '2018-01-31 15:20:52.673684' # 0 0 0 0 0
daytime = '2018-01-31 15:33:53.708948' # 0.27 1.9 0 0
daytime = '2018-01-31 15:34:41.792130' # 0 0 1 1
daytime = '2018-01-31 15:35:34.669448' # 0.5 1 1 1
daytime = '2018-01-31 15:36:18.280453' # 0.1 1.5 1 1
daytime = '2018-01-31 15:37:05.301404' # 0.1 1.9 1 0
daytime = '2018-01-31 15:37:58.528560' # 0.27 1.5 0 1

daytime = '2018-01-31 20:06:25.713816'

## Batch 2 
daytime = '2018-01-31 21:13:07.594892' # 1 1 10 100
daytime = '2018-01-31 21:13:37.427846' # 1 3 10 100
daytime = '2018-01-31 21:11:41.012484' # 0.2 2 10 100
daytime = '2018-01-31 21:12:52.923476' # 0.5 0.5 10 100
daytime = '2018-01-31 21:11:57.300465' # 1 4 10 100 
daytime = '2018-01-31 21:13:22.380551' # 2 4 10 100
daytime = '2018-01-31 21:12:21.326258' # 2 3 10 100 

## step 10 1000 
daytime = '2018-02-01 11:19:15.105930'

## step 10 100 5 bands 1 3 
daytime = '2018-02-01 16:31:29.877066'
daytime = '2018-02-02 13:31:31.861911' # 0.4 4 10 100
daytime = '2018-02-02 13:29:26.649440' # 0.1 3 1 10
daytime = '2018-02-02 13:28:51.909670' # 1 3 1 100 

## 250 bands 4000 iter 
daytime = '2018-02-10 16:15:20.548334' # 0.1 4 1 100 nb diff ok
daytime = '2018-02-09 11:58:12.946391' # 0.1 3 1 100 ok 
daytime = '2018-02-08 08:07:15.101651' # 1 3 100 100 No 
daytime = '2018-02-07 08:13:50.644883' # 1 3 10 100
daytime = '2018-02-03 00:16:27.218396' # 1 3 1 100 

daytime = '2018-02-12 11:10:21.922748'

## mu_s tests 
daytime = '2018-02-13 16:03:46.418454'
daytime = '2018-02-13 16:10:14.010312'
daytime = '2018-02-13 16:01:59.321735'

## bands 5 step 10 mu_s only 5 2 1 mu_l = 0 
daytime = '2018-02-14 09:18:14.057160_out'
daytime = '2018-02-14 09:17:31.803409_out'
daytime = '2018-02-14 09:16:52.473327_out'

## Tests with mu_l 
daytime = '2018-02-14 11:24:37.169880'
daytime = '2018-02-14 11:24:48.018588'
daytime = '2018-02-14 11:24:55.045829'
daytime = '2018-02-14 11:24:27.026967'
daytime = '2018-02-14 12:56:11.232973'
daytime = '2018-02-14 14:51:13.099343' # 5
daytime = '2018-02-14 15:13:55.998626' # 7
daytime = '2018-02-14 17:08:45.669524' # 50 bands 0 1 0 100


daytime = '2018-02-19 11:12:44.055675'
daytime = '2018-02-19 11:12:44.055675'
daytime = '2018-02-19 11:54:44.795678'
daytime = '2018-02-19 13:21:55.314376' ### sugar1 is almost zero !!!!
daytime = '2018-02-19 14:05:06.912266'
daytime = '2018-02-19 14:48:49.447477'  # testing the convergence of mu_l with a small step size equal to 1 

daytime = '2018-02-19 15:58:52.471774' # after bug fix !!!!

daytime = '2018-02-19 16:06:03.791489'

daytime = '2018-02-19 16:42:57.255836'

daytime = '2018-02-19 16:58:34.894112'

daytime = '2018-02-19 17:19:11.637618'

daytime = '2018-02-19 18:10:17.775849' ## mu_l reussi
daytime = '2018-02-19 18:10:34.171932' ## mu_l reussi 

daytime = '2018-02-20 10:52:09.110689'
daytime = '2018-02-20 11:17:59.922265'

daytime = '2018-02-20 11:50:05.374682'
daytime = '2018-02-20 12:05:54.468976'
daytime = '2018-02-20 14:24:38.453427'

daytime = '2018-02-20 14:55:24.585042'

daytime = '2018-02-20 16:12:22.842807'

daytime = '2018-02-20 16:51:46.223065'

daytime = '2018-02-20 17:16:32.304653'

daytime = '2018-02-20 17:48:32.102744'
daytime = '2018-02-20 18:48:08.794495'
daytime = '2018-02-20 18:46:15.482854'

daytime = '2018-02-21 10:34:36.483458'
daytime = '2018-02-21 10:44:22.506516'

# observing sugar0 for != values of mu_s 0.1 0.5 1 2 
daytime = '2018-02-21 10:48:05.412362' # mu_s à.1

daytime = '2018-02-21 10:49:32.011018' # mu_s 0.1  3000 iterations 

daytime = '2018-02-21 11:05:53.996008' # mu_s 0.5 

daytime = '2018-02-21 11:19:28.754747' # mu_s 2 

daytime = '2018-02-21 11:28:44.809348' # mu_s 1 3000 iterations 

daytime = '2018-02-21 11:40:29.991767' # mu_s 2 

daytime = '2018-02-21 11:46:09.257557'
daytime = '2018-02-21 11:47:25.538152'

daytime = '2018-02-21 12:40:28.527043'
daytime = '2018-02-21 12:40:14.077674'

daytime = '2018-02-21 15:15:17.733131' # small eps 
daytime = '2018-02-21 15:15:56.351190' # big eps 
daytime = '2018-02-21 15:17:57.928936'

daytime = '2018-02-21 15:25:07.228487'
daytime = '2018-02-21 15:29:02.736632'
daytime = '2018-02-21 15:34:21.028329'

daytime = '2018-02-23 14:52:11.335650' # crpd data 3 
daytime = '2018-02-23 15:45:13.074157'
daytime = '2018-02-23 16:28:55.735557'
daytime = '2018-02-23 16:36:44.152363'
daytime = '2018-02-23 16:47:46.331601' # eps 2 
daytime = '2018-02-23 16:54:56.176828' # eps 20 This is to see if the gradient is more stable than before 
daytime = '2018-02-23 17:08:54.637629'
daytime = '2018-02-23 17:31:25.077925'
daytime = '2018-02-23 17:45:33.514439'
daytime = '2018-02-23 18:13:43.094194'
daytime = '2018-02-23 12:12:47.795495'
daytime = '2018-02-23 18:39:03.351815'
daytime = '2018-02-23 19:00:33.079065'
daytime = '2018-02-23 19:10:33.542128'

daytime = '2018-02-24 19:33:27.592428'
daytime = '2018-02-24 19:40:04.740772'

daytime = '2018-02-24 19:40:04.740773'
daytime = '2018-02-24 19:43:48.217750'
daytime = '2018-02-24 19:45:12.267150'
daytime = '2018-02-24 19:47:05.339411'

daytime = '2018-02-24 19:49:25.072138'

daytime = '2018-02-24 19:50:40.723242'

daytime = '2018-02-24 19:52:04.852755'

daytime = '2018-02-24 19:53:41.672932'

daytime = '2018-02-24 20:13:35.049512'

daytime = '2018-02-24 20:48:19.560401'

daytime = '2018-02-24 20:50:11.108789'
daytime = '2018-02-24 20:51:17.999665'
daytime = '2018-02-24 20:52:22.922174' ########
daytime = '2018-02-24 20:54:07.924037'

daytime = '2018-02-24 20:58:15.046469'
daytime = '2018-02-24 21:26:42.118270'
daytime = '2018-02-24 21:56:12.330501'
daytime = '2018-02-24 22:26:01.479902'
daytime = '2018-02-24 22:54:53.892297'
daytime = '2018-02-24 23:24:29.770111'
daytime = '2018-02-24 23:53:52.776088'
daytime = '2018-02-25 00:23:45.144978'
daytime = '2018-02-25 00:53:29.438027'
daytime = '2018-02-25 01:23:18.837665'
daytime = '2018-02-25 16:14:29.168164'
daytime = '2018-02-25 16:20:26.736520'
daytime = '2018-02-25 16:22:16.663637'

daytime = '2018-02-25 16:24:44.670551'
daytime = '2018-02-25 16:27:29.303703'

daytime = '2018-02-25 16:37:00.534027' #############"
daytime = '2018-02-25 16:53:26.790553'
daytime = '2018-02-25 17:12:03.759797'
daytime = '2018-02-25 17:32:53.662015'
daytime = '2018-02-25 17:51:47.077818'
daytime = '2018-02-25 18:08:31.792172'
daytime = '2018-02-25 18:31:30.257714'

daytime = '2018-02-25 19:03:00.189657' # /
daytime = '2018-02-25 19:21:29.558812'
daytime = '2018-02-25 19:26:22.318666'
daytime = '2018-02-25 19:44:10.978363'

daytime = '2018-02-24 22:26:01.479902'

#%% 
drctry = os.path.join(os.getcwd(),'output/'+daytime)
os.chdir(drctry)

x0_tst =np.load('x0_tst.npy')
wmse_tst=np.load('wmse_tst.npy')
wmses_tst=np.load('wmses_tst.npy')
wmsesfdmc_tst=np.load('wmsesfdmc_tst.npy')
snr_tst=np.load('snr_tst.npy')
mu_s_tst=np.load('mu_s_tst.npy')
mu_l_tst=np.load('mu_l_tst.npy')
dxs=np.load('dxs.npy')
dxl=np.load('dxl.npy')
sugar0=np.load('sugar0.npy')
sugar1=np.load('sugar1.npy')
cost=np.load('cost.npy')
psnr = np.load('psnrsure.npy')
#psnrsure=np.load('psnrsure.npy')

os.chdir('../..')

pl.figure()
pl.imshow(x0_tst[:,:,0])
pl.colorbar()

pl.figure()
pl.plot(wmse_tst,label='wmse_tst')
pl.plot(wmses_tst,'-*',label='wmses_tst')
pl.plot(wmsesfdmc_tst,'-^',label='wmses_fdmctst')
pl.legend()

N = snr_tst.size 
pl.figure()
pl.plot(snr_tst[:N],label='snr_tst')
pl.legend()

pl.figure()
pl.plot(sugar0[1:N],label='sugar0')
pl.plot(0*sugar0[1:N])
pl.legend()

pl.figure()
pl.plot(sugar1[1:N],label='sugar1')
pl.plot(0*sugar1[1:N])
pl.legend()

#pl.figure()
#pl.plot(psnrsure[:N],label='psnrsure')
#pl.legend()

pl.figure()
pl.plot(cost[:N],label='cost')
pl.legend()

pl.figure()
pl.plot(psnr)
pl.title('psnr')

pl.figure()
pl.plot(mu_s_tst[:N],label='mu_s')
pl.legend()

pl.figure()
pl.plot(mu_l_tst[:N],label='mu_l')
pl.legend()

#%%#

## Batch 2 
daytime = []
daytime.append('2018-01-31 21:13:07.594892') # 1 1 10 100
daytime.append('2018-01-31 21:13:37.427846') # 1 3 10 100
daytime.append('2018-01-31 21:11:41.012484') # 0.2 2 10 100
daytime.append('2018-01-31 21:12:52.923476') # 0.5 0.5 10 100
daytime.append('2018-01-31 21:11:57.300465') # 1 4 10 100 
daytime.append('2018-01-31 21:13:22.380551') # 2 4 10 100
daytime.append('2018-01-31 21:12:21.326258') # 2 3 10 100 

mu_s_ = []
mu_l_ = []

for day in daytime :
    drctry = os.path.join(os.getcwd(),'output/'+day)
    os.chdir(drctry)
    mu_s_.append(np.load('mu_s_tst.npy'))
    mu_l_.append(np.load('mu_l_tst.npy'))
    os.chdir('../..')

pl.figure()
for mu in mu_s_:
    pl.plot(mu)
pl.title('mu_s')
    
pl.figure()
for mu in mu_l_:
    pl.plot(mu)
pl.title('mu_l')
    
#%%#

## Batch 3 first 5 bands testing mu_s with mu_l = 0 
daytime = []
daytime.append('2018-02-14 09:18:14.057160') #  
daytime.append('2018-02-14 09:17:31.803409') #  
daytime.append('2018-02-14 09:16:52.473327') #  
daytime.append('2018-02-14 09:14:18.128510') #  

mu_s_ = []
psnrsure_ = []

for day in daytime :
    drctry = os.path.join(os.getcwd(),'output/'+day)
    os.chdir(drctry)
    mu_s_.append(np.load('mu_s_tst.npy'))
    psnrsure_.append(np.load('psnrsure.npy'))
    os.chdir('../..')

pl.figure()
for mu in mu_s_:
    pl.plot(mu)
pl.title('mu_s')
    
pl.figure()
for psnr in psnrsure_:
    pl.plot(psnr)
pl.title('psnrsure')
    
#%% Batch 4 last 5 bands testing mu_s with mu_l = 0 
daytime = []
#daytime.append('2018-02-14 12:56:11.232973') # 
daytime.append('2018-02-14 11:24:27.026967') #  
daytime.append('2018-02-14 11:24:55.045829') #  
daytime.append('2018-02-14 11:24:48.018588') #  
daytime.append('2018-02-14 11:24:37.169880') #  

mu_s_ = []
psnrsure_ = []
snr_ = []
sugar0_ = []

for day in daytime :
    drctry = os.path.join(os.getcwd(),'output/'+day)
    os.chdir(drctry)
    mu_s_.append(np.load('mu_s_tst.npy'))
    psnrsure_.append(np.load('psnrsure.npy'))
    snr_.append(np.load('snr_tst.npy'))
    sugar0_.append(np.load('sugar0.npy'))
    os.chdir('../..')

pl.figure()
for mu in mu_s_:
    pl.plot(mu)
pl.title('mu_s')
    
pl.figure()
for psnr in psnrsure_:
    pl.plot(psnr)
pl.title('psnrsure')
    
pl.figure()
for snr in snr_:
    pl.plot(snr)
pl.title('snr')

pl.figure()
for sugar in sugar0_:
    pl.plot(sugar)
pl.plot(0*sugar)
pl.title('sugar0')

#%% Batch 50 bands testing mu_l with mu_s = 0.7 
daytime = []
daytime.append('2018-02-19 18:10:17.775849') #   
daytime.append('2018-02-19 18:10:34.171932') #  

mu_l_ = []
psnrsure_ = []
snr_ = []
sugar1_ = []

for day in daytime :
    drctry = os.path.join(os.getcwd(),'output/'+day)
    os.chdir(drctry)
    mu_l_.append(np.load('mu_l_tst.npy'))
    psnrsure_.append(np.load('psnrsure.npy'))
    snr_.append(np.load('snr_tst.npy'))
    sugar1_.append(np.load('sugar1.npy'))
    os.chdir('../..')

pl.figure()
for mu in mu_l_:
    pl.plot(mu)
pl.title('mu_l')
    
pl.figure()
for psnr in psnrsure_:
    pl.plot(psnr)
pl.title('psnrsure')
    
pl.figure()
for snr in snr_:
    pl.plot(snr)
pl.title('snr')

pl.figure()
for sugar in sugar1_:
    pl.plot(sugar[1:])
pl.plot(0*sugar[1:])
pl.title('sugar1')


#%%

daytime = []
daytime.append('2018-02-20 17:48:32.102744') #   
daytime.append('2018-02-20 18:48:08.794495') #  
daytime.append('2018-02-20 18:46:15.482854') #  

mu_l_ = []
mu_s_ = []
psnrsure_ = []
sugar1_ = []
sugar0_ = []

for day in daytime :
    drctry = os.path.join(os.getcwd(),'output/'+day)
    os.chdir(drctry)
    mu_l_.append(np.load('mu_l_tst.npy'))
    mu_s_.append(np.load('mu_s_tst.npy'))
    psnrsure_.append(np.load('psnrsure.npy'))
    sugar0_.append(np.load('sugar0.npy'))
    sugar1_.append(np.load('sugar1.npy'))
    os.chdir('../..')

pl.figure()
for mu in mu_l_:
    pl.plot(mu)
pl.title('mu_l')
    
pl.figure()
for mu in mu_s_:
    pl.plot(mu)
pl.title('mu_s')

pl.figure()
for psnr in psnrsure_:
    pl.plot(psnr)
pl.title('psnrsure')
    
pl.figure()
for sugar in sugar0_:
    pl.plot(sugar[1:])
pl.plot(0*sugar[1:])
pl.title('sugar0')

pl.figure()
for sugar in sugar1_:
    pl.plot(sugar[1:])
pl.plot(0*sugar[1:])
pl.title('sugar1')

