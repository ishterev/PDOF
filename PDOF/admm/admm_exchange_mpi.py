# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015
@author: shterev
"""
# mpiexec -n 4 python admm_exchange_mpi.py 100

import numpy as np
from scipy.linalg.blas import ddot, dnrm2
#from numpy import linalg as LA
import scipy.io as sio

import sys
from mpi4py import MPI

from opt_problem import *

import time
import psutil

VmPeakStart = psutil.virtual_memory()[3] # in B for MB / 1000 000

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

MAXITER  = 500#1000# int(1e4);   # Maximal amount of iterations
ABSTOL   = 1e-4
RELTOL   = 1e-2# 1e-2;1e-3;1e-4;

DISP = True         # Display reults iteratively (0)yes / (1)no
HISTORY = True      # Save the produced values on each iteration

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  

chargeStrategy = 'home'
V2G = True
gamma = 0

N_EV = 4 # Number of EVs
ID = '0' # id

if len(sys.argv) > 1:
    N_EV = int(sys.argv[1])
    
    if len(sys.argv) > 2 :
        ID = sys.argv[2]
        

N = N_EV + 1        # Number of agents N=N_EV+1

deltaT=15*60;                   # Time slot duration [sec]
T= 24*3600/deltaT ;             # Number of time slots

lamb=1e-1;
mu=1e-1;

# divide the EVs in "size" chunks of lenght n = N / size (+ 1)
n = N / size
startidx = n * rank  
mod = N % size  

if( mod != 0):
      if(rank < mod):
         n += 1
         startidx += rank
      else:
         startidx += mod
         
        
opt_probs = np.empty((n,), dtype=np.object)        


if rank == 0:
      
        problem = OptProblem_Aggregator()  
        
        D = problem.D        
        # Empirical [price/demand^2]     
        delta =  np.mean(problem.price)/(np.mean(problem.D) * (3600*1000)  *15*60 ) ;     
        OptProblem_ValleyFilling_Home.delta = delta;
        OptProblem_ValleyFilling_Home.gamma = gamma;
        OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta;
        
        opt_probs[0] = problem
        
        for i in xrange(1, n):
              problem = OptProblem_ValleyFilling_Home(i, V2G)
              opt_probs[i] = problem
else:
        for i in xrange(n):
              problem = OptProblem_ValleyFilling_Home(startidx + i, V2G)
              opt_probs[i] = problem

eps_pri = np.sqrt(N)  # Primal stoping criteria  Convergence
eps_dual = np.sqrt(N)  # Dual stoping criteria

rho=0.5            # Augmented penalty parameter ADMM Step size
v=0               # parameter for changing rho

#if(rank == 0 and DISP):
#           print ("%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10\n" %
#                  ('iter', 'step_size', 'r_norm', 'eps_pri', 's_norm','eps_dual', 'objective'))

# save history
if (rank == 0 and HISTORY):
    
    # save results and exit
    historyFileName= DATA_DIR + '/results/' + str(N_EV) + 'EVs_' +  chargeStrategy
    if V2G :
       historyFileName +='_V2G';

    historyFileName +='_gamma_' + str(gamma) ###
    historyFileName +='.mat'
        
    history = {}    
    history["time"] = np.zeros((MAXITER,), dtype='float64')
    history["meminfo"] = np.zeros((MAXITER,), dtype='float64')
    history["cost"] = np.zeros((MAXITER,), dtype='float64')   
    history["costEVs"] = np.zeros((MAXITER,), dtype='float64')
    history["costAgr"] = np.zeros((MAXITER,), dtype='float64')
    history["r_norm"] = np.zeros((MAXITER,), dtype='float64')
    history["s_norm"] = np.zeros((MAXITER,), dtype='float64')
    history["eps_pri"] = np.zeros((MAXITER,), dtype='float64')
    history["eps_dual"] = np.zeros((MAXITER,), dtype='float64')
    history["rho"] = np.zeros((MAXITER,), dtype='float64')
    
    
xi = np.zeros((n,T,1)) # np.zeros((T, N))
ui = np.zeros((n,T,1))
zi = np.zeros((n,T,1))  # np.zeros((T, N))
x_mean = np.zeros((T,1))
#ri = np.zeros((T,1))


send = np.zeros(5)
recv = np.zeros(5)
         

if(rank == 0):                  
   
   #Cost of iteration
   cost = 0
   costEVs = 0
   costAgr = 0
   xAggr = np.zeros((T,1)) 
   
   tic = time.time()


# ADMM loop.
for k in xrange(MAXITER):#50

        send = np.zeros(5)
        xsum = np.zeros((T,1))

        for j in range(n):
            
            zi_old = zi[j]
            zi[j] = xi[j] - x_mean            
            
            ui[j] = ui[j] + x_mean 
            
            problem = opt_probs[j]
            problem.setParameters(rho, xi[j] - x_mean - ui[j])
            xi[j], ci = problem.solve()
            
            #send[0] += ddot(ri, ri)
            #send[0] += xi[j]
            send[0] += ddot(xi[j], xi[j])
            send[1] += ddot(ui[j], ui[j])
            send[2] += ddot(zi[j], zi[j])
            
            zdiff = zi[j] - zi_old
            send[3] += ddot(zdiff, zdiff)
            
            if rank == 0 and j == 0:
                # aggregator
                xAggr = xi[j]
                costAgr = ci
            else:    
               send[4] += ci #costEVs
               
            xsum += xi[j]
            
            
            
      
        comm.Allreduce(send, recv, op=MPI.SUM)
        
        comm.Allreduce(xsum, x_mean, op=MPI.SUM)    
        
        if rank == 0:
           # x_mean := xsum
           x_sum = D + x_mean - xAggr#D+x_sum
           cost = ddot(x_sum, x_sum) #+ delta*costEVs; % real cost of the EVs        
        
        # up until now this was only the sum
        x_mean /=  N
        
        ####
        ## 
        ## x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp
        ##
        ## For exchange A = 1 and B = -1
        ##
        ## e_pri = sqrt(p)+ e_abs + e_rel * max{Axk2,Bzk2,c2},
        ## e_dual = sqrt(n) + e_abs + e_rel * A.T x yk2
        ##
        ## n = p = T
        #######

        #x_mean = recv[0] / N
        
        nxstack = np.sqrt(recv[0])  # sqrt(sum ||x_i||_2^2) 
        nystack = rho * np.sqrt(recv[1])  # sqrt(sum ||y_i||_2^2) 
        nzstack = np.sqrt(recv[2]) # dnrm2(-1 * zi)
        nzdiffstack = np.sqrt(recv[3])
        
        if(rank == 0):
            costEVs = recv[4]
        
        #zi_old = zi
        #zi = xi - x_mean
        
        # rk+1 = Axk+1 + Bzk+1 − c
        r_norm  =  np.sqrt(N) * dnrm2(x_mean)  # r_i = x_mean ; sqrt(sum ||r_i||_2^2);  
       
        # |ρ * A.TxB * (zk+1 − zk)|2
        s_norm = rho * nzdiffstack
       
        eps_pri = np.sqrt(T) * ABSTOL + RELTOL * max(nxstack, nzstack) 
        eps_dual = np.sqrt(T) * ABSTOL + RELTOL * nystack    
        
        if(rank == 0 and DISP):
           print ("\n%3d\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" %
                  (k, rho, r_norm, eps_pri, s_norm, eps_dual,cost))
                  
        #  save history
        if (rank == 0 and HISTORY):
       
           #performance memory
           history['time'][k]= time.time() - tic #toc
       
           VmPeakIteration = psutil.virtual_memory()[3] #used
           meminfo=VmPeakIteration-VmPeakStart;
           history['meminfo'][k]=meminfo;
           
           history['cost'][k]= cost 
           history['costEVs'][k]= costEVs 
           history['costAgr'][k]=costAgr
           history['r_norm'][k]=r_norm
           history['s_norm'][k]=s_norm
           history['eps_pri'][k]=eps_pri
           history['eps_dual'][k]=eps_dual
           history['rho'][k]=rho
       
        # stopping criteria
        if (r_norm <= eps_pri and s_norm <= eps_dual):
          #if (r_LA.norm < eps_pri):   
            if (rank == 0):
                print "Finished at step ", i
            break
       
        #if (rank == 0): print "ADMM iteration ", i

        # update rho 
        # According to Boyd et. al
          
        if(rho < 10000):
          
          if s_norm != 0 :
             rho_old = rho
             v_old = v
             v = rho * r_norm/s_norm - 1   
             rho = rho* np.exp(lamb * v + mu * (v - v_old))
             
             if(rho == 2):
                rho += ABSTOL #dirty hack, because of the aggregator function
                #continue
       
             # rescale u
             ui = (rho_old/rho) * ui
           
          else: # rarely, but could happen s_norm = 0
             # alternatively
             t_incr = 2
             t_decr = 2
             m = 10
       
             rho_old = rho
             if(r_norm > m * s_norm):
                rho = t_incr * rho        
             elif(s_norm > m * r_norm):
                rho = rho / t_decr
                
             if(rho == 2):
                rho += ABSTOL # dirty hack, because of the aggregator function
        
             ui = (rho_old/rho) * ui
        
        #ri = x_mean


ui = rho * ui


comm.Barrier()

'''
# alternatively
if rank == 0:
        x = np.zeros((T,N))         # Agents profile 
        u = np.zeros((T,N))         # Help variable
        z = np.zeros((T,N))         # Scaled price 
        
else:
        x = None
        u = None
        z = None
        
# gatherv
comm.Gather(xi, x , root = 0)
comm.Gather(ui, u , root = 0)
comm.Gather(zi, z , root = 0)       
'''

if rank == 0:
        x = np.zeros((T,N))         # Agents profile 
        u = np.zeros((T,N))         # Help variable
        z = np.zeros((T,N))         # Scaled price 
        
        xlist = []
        ulist = []
        zlist = []
        
else:
        x = None
        u = None
        z = None
        
        xlist = None
        ulist = None
        zlist = None
        
if rank == 0 and HISTORY:
    sio.savemat(historyFileName, history)
    
xlist = comm.gather(xi, root = 0)
ulist = comm.gather(ui, root = 0)
zlist = comm.gather(zi, root = 0)

# root process prints results
if rank == 0:

   x = np.concatenate(xlist, axis=0).reshape((T,N))#np.array(xlist).reshape((T,N))
   u = np.concatenate(ulist, axis=0).reshape((T,N))#np.array(ulist).reshape((T,N))
   z = np.concatenate(zlist, axis=0).reshape((T,N))#np.array(zlist).reshape((T,N))
   
   print "x", x
   print "u", u
   print "z", z
        