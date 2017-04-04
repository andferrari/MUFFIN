#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:51:20 2017

@author: rammanouil
"""
 
import time
__time_tic_toc=time.time()

def tic():
    """ Python implementation of Matlab tic() function """
    global __time_tic_toc
    __time_tic_toc=time.time()

def toc(message='Elapsed time : {} s'):
    """ Python implementation of Matlab toc() function """
    t=time.time()
    print('')
    print(message.format(t-__time_tic_toc))
    return t-__time_tic_toc

def toq():
    """ Python implementation of Julia toc() function """
    t=time.time()
    return t-__time_tic_toc