#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:56:44 2016

@author: antonyschutz
"""
import numpy as np
import SuperNiceSpectraDeconv as SNSD
import matplotlib.pylab as pl


file_in = 'm31_3d_conv_10db'
mu_s = .5
mu_l = 0.1
nb = 8
nbitermax = 10

xt, snr = SNSD.easy_muffin(mu_s,mu_l,nb,nbitermax,file_in,folder='data/')