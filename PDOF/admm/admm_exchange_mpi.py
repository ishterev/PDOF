# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015
@author: shterev
"""
# mpiexec -n 4 python admm_exchange_mpi.py 100

import numpy as np
from scipy.linalg.blas import ddot, dnrm2
#from numpy import linalg as LA

import sys
from mpi4py import MPI

from opt_problem import *


MAXITER  = 50#1000# int(1e4);   # Maximal amount of iterations
ABSTOL   = 1e-4
RELTOL   = 1e-2# 1e-2;1e-3;1e-4;

DISP = True         # Display reults iteratively (0)yes / (1)no
HISTORY = False      # Save the produced values on each iteration

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  

chargeStrategy = 'home'
V2G = True
gamma = 0

N = int(sys.argv[1]) + 1        # Number of agents N=N_EV+1
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
    
    
xi = np.zeros((n,T,1)) # np.zeros((T, N))
ui = np.zeros((n,T,1))
zi = np.zeros((n,T,1))  # np.zeros((T, N))
x_mean = np.zeros((T,1))
#ri = np.zeros((T,1))


send = np.zeros(4)
recv = np.zeros(4)


    # ADMM loop.
for i in xrange(MAXITER):#50

        send = np.zeros(4)
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
            
            xsum += xi[j]
            
      
        comm.Allreduce(send, recv, op=MPI.SUM)
        
        comm.Allreduce(xsum, x_mean, op=MPI.SUM)        
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
        
        #zi_old = zi
        #zi = xi - x_mean
        
        # rk+1 = Axk+1 + Bzk+1 − c
        r_norm  =  np.sqrt(N) * dnrm2(x_mean)  # r_i = x_mean ; sqrt(sum ||r_i||_2^2);  
       
        # |ρ * A.TxB * (zk+1 − zk)|2
        s_norm = rho * nzdiffstack
       
        eps_pri = np.sqrt(T) * ABSTOL + RELTOL * max(nxstack, nzstack) 
        eps_dual = np.sqrt(T) * ABSTOL + RELTOL * nystack    
        
        if(rank == 0 and DISP):
           cost = 0
           print ("\n%3d\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" %
                  (i, rho, r_norm, eps_pri, s_norm, eps_dual,cost))
       
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
                rho = 2.1 #dirty hack, because of aggregator function
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
                rho = 2.1 # dity hack, because of aggregator function
        
             ui = (rho_old/rho) * ui
        
        #ri = x_mean


ui = rho * ui


comm.Barrier()
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


# root process prints results
if comm.rank == 0:
        print "x", x
        print "u", u
        print "z", z