# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015
@author: shterev
"""
# mpiexec -n 4 python admm_exchange_mpi.py 10000

import numpy as np
from numpy import linalg as LA

import sys
from mpi4py import MPI

from opt_problem import *


MAXITER  = 1000# int(1e4);   # Maximal amount of iterations
ABSTOL   = 1e-4
RELTOL   = 1e-2# 1e-2;1e-3;1e-4;

DISP = False         # Display reults iteratively (0)yes / (1)no
HISTORY = False     # Save the produced values on each iteration

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  

chargeStrategy = 'home'
V2G = True
gamma = 0

N = int(sys.argv[1]) + 1   # Number of agents N=N_EV+1
deltaT=15*60;                   # Time slot duration [sec]
T= 24*3600/deltaT ;             # Number of time slots

lamb=1e-1;
mu=1e-1;

if rank == 0:
      
        problem = OptProblem_Aggregator()       
        
        # Empirical [price/demand^2]     
        delta =  np.mean(problem.price)/(np.mean(problem.D) * (3600*1000)  *15*60 ) ;     
        OptProblem_ValleyFilling_Home.delta = delta;
        OptProblem_ValleyFilling_Home.gamma = gamma;
        OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta;
else:
        problem = OptProblem_ValleyFilling_Home(rank, V2G)

eps_pri = np.sqrt(N)  # Primal stoping criteria  Convergence
eps_dual = np.sqrt(N)  # Dual stoping criteria

rho=0.5            # Augmented penalty parameter ADMM Step size
v=0               # parameter for changing rho
    
    
xi = np.zeros(T) # np.zeros((T, N))
ui = np.zeros(T)
zi = np.zeros(T)  # np.zeros((T, N))
x_mean = np.zeros(T)
ri = np.zeros(T)

send = np.zeros(3)
recv = np.zeros(3)


    # ADMM loop.
for i in xrange(MAXITER):#50
    
        ui = ui + x_mean       
        problem.setParameters(rho, xi - x_mean - ui)
        xi, ci = problem.solve()
        
        send[0] = np.dot(ri, ri.T) 
        send[1] = np.dot(xi, xi.T)
        send[2] = np.dot(ui, ui.T)
        #send[2] *= np.power(rho, 2)

        comm.Allreduce(send, recv, op=MPI.SUM)
        
        comm.Allreduce(xi, x_mean, op=MPI.SUM)        
        x_mean /=  N

        r_norm  = np.sqrt(recv[0])  # sqrt(sum ||r_i||_2^2) 
        nxstack = np.sqrt(recv[1])  # sqrt(sum ||x_i||_2^2) 
        nystack = rho * np.sqrt(recv[2])  # sqrt(sum ||y_i||_2^2) 
        
        zi_old = zi
        zi = xi - x_mean
       
        s_norm = np.sqrt(N) * rho * LA.norm(zi - zi_old)
       
        eps_pri = np.sqrt(T*N) * ABSTOL + RELTOL * max(nxstack, LA.norm(-1 * zi))
        eps_dual = np.sqrt(T*N) * ABSTOL + RELTOL * nystack    
       
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
       
             # rescale u
             ui = (rho_old/rho) * ui
           
          else:
             # alternatively
             t_incr = 2
             t_decr = 2
             m = 10
       
             rho_old = rho
             if(r_norm > m * s_norm):
                rho = t_incr * rho        
             elif(s_norm > m * r_norm):
                rho = rho / t_decr
        
             ui = (rho_old/rho) * ui
        
        ri = x_mean


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
        
        
comm.Gather(xi, x , root = 0)
comm.Gather(ui, u , root = 0)
comm.Gather(zi, z , root = 0)


# root process prints results
if comm.rank == 0:
        print "x", x
        print "u", u
        print "z", z