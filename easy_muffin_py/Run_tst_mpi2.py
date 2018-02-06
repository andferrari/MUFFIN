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
folder = 'data256Eusipco'
file_in = data_suffix
folder = os.path.join(os.getcwd(), folder)
genname = os.path.join(folder, file_in)
psfname = genname+'_psf.fits'
drtname = genname+'_dirty.fits'

CubePSF = checkdim(fits.getdata(psfname, ext=0))[:,:,0:L]
CubeDirty = checkdim(fits.getdata(drtname, ext=0))[:,:,0:L]
skyname = genname+'_sky.fits'

if os.path.isfile(skyname):
    if rank==0:
        print('')
        print('estimating variance')
        
    #sky = checkdim(fits.getdata(skyname, ext=0))[:,:,0:L]
    sky = checkdim(fits.getdata(skyname, ext=0))
    #sky = np.transpose(sky)[:,:,0:L]
    sky = sky[:,:,0:L]
    
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
    else:
        np.save('x0_tst.npy',EM.xf)
        np.save('wmses_tst.npy',EM.wmselistsure)
        np.save('mu_s_tst.npy',mu_s)
        np.save('mu_l_tst.npy',mu_l)


## ==============================================================================
## Print Figures
#%% ==============================================================================
import matplotlib.pyplot as pl 
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

## step 10 100 250 bands 1 3 
daytime = '2018-02-01 16:31:29.877066'

daytime = '2018-02-02 13:31:31.861911' # 0.4 4 10 100
daytime = '2018-02-02 13:29:26.649440' # 0.1 3 1 10
daytime = '2018-02-02 13:28:51.909670' # 1 3 1 100 

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

#pl.figure()
#pl.imshow(x0_tst[:,:,0])
#pl.colorbar()

#pl.figure()
#pl.plot(wmse_tst,label='wmse_tst')
#pl.plot(wmses_tst,'-*',label='wmses_tst')
#pl.plot(wmsesfdmc_tst,'-^',label='wmses_fdmctst')
#pl.legend()
#
pl.figure()
pl.plot(snr_tst,label='snr_tst')
pl.legend()

pl.figure()
pl.plot(sugar0[1::],'*',label='sugar0')
pl.plot(0*sugar0)
pl.legend()

pl.figure()
pl.plot(sugar1[1::],label='sugar1')
pl.plot(0*sugar1)
pl.legend()

#pl.figure()
#pl.plot(cost,label='cost')
#pl.legend()

pl.figure()
pl.plot(mu_s_tst,label='mu_s')
pl.legend()

pl.figure()
pl.plot(mu_l_tst,label='mu_l')
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
    