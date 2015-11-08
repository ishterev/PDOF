# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 09:25:23 2015

@author: shterev
"""
import os
import h5py
import scipy.io as sio

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))


N_EV = 100

problem = 'valley_filling'
strategy = 'home'
gamma=0                       
V2G = True

mpi = True
                
            
name= DATA_DIR + '/results/' + problem + '/' + str(N_EV) + 'EVs_' + strategy
                        
if V2G:
   name += '_V2G'
                        
name +='_gamma_' + str(gamma)

if(mpi):
    name += '_mpi'

name += '.mat'
                        
data = sio.loadmat(name)
#data = h5py.File(name, 'r') # open read-only

for key,val in data.items() :
       
    if(key == 'r_norm'):       
       r_norm = data[key][()]
          
    if(key == 's_norm'):
       s_norm = data[key][()]
                 
    if(key == 'eps_pri'):
       eps_pri = data[key][()] 
                 
    if(key == 'eps_dual'):
       eps_dual = data[key][()]   
                 
    if(key == 'meminfo'): 
       meminfo = data[key][()]
                 
    if(key == 'cost'): 
       cost = data[key][()]
       
    if(key == 'costAggr'): 
       costAggr = data[key][()]

    if(key == 'costEVs'): 
       costEVs = data[key][()]
       
    if(key == 'rho'): 
       rho = data[key][()]
                 
    if(key == 'x'): 
       x = data[key][()]
       
    if(key == 'y'): 
       y = data[key][()]

    if(key == 'z'): 
       z = data[key][()]
       
    if(key == 'time'): 
       time = data[key][()]
       
n = [range(np.size(time))]       
plt.figure(1)

#plt.subplot(211)
plt.plot(n, time, 'bo') #, n, meminfo, 'ro')

plt.figure(2)
plt.plot(n, meminfo/(1024*1024), 'ro') 

#plt.subplot(212, projection = '3d')
#plt.plot(x, y, z)

'''plt.subplot(211) 
plt.plot(n, r_norm, 'bo')#, n, eps_pri, 'ro')

plt.subplot(212)
plt.plot(n, eps_pri, 'ro')
#plt.plot(n, s_norm, 'bo', n , eps_dual, 'ro')

plt.show()

plt.figure(2)
plt.subplot(211)
plt.plot(n, s_norm, 'bo')#, n , eps_dual, 'ro')

plt.subplot(212)
plt.plot(n , eps_dual, 'ro')'''



plt.show()
              

'''       
plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')



plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()

'''


                        
                    