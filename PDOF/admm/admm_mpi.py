# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015
@author: shterev
"""
###################################################################################################
# To run execute the follwoing command. The console should point to this file's working directory
#
# mpiexec -n 4 python admm_mpi.py 100
#
# where 100 is an example for the number of (EVs + Aggregator)
###################################################################################################
"""

The aggregator in a hierarchical control structure influences EV charging behavior
through an incentive signal to reach certain goals. While the aggregator wants to use EVs 
to minimize their operational costs, the individual EVs want to minimize their charging 
and battery depreciation costs.

To adderess this trade off, the problem can be formulated as a standard exchange optimization problem
and consider the EVs and the aggregator as its agents. The exchange problem considers N agents 
exchanging a common good under an equilibrium constraint [Rivera et. al].

We use MPI(Message Passing Interface) for our distributed-memory implementation. It is based on
the works of Boyd et. al. In MPI the processes are coordinating each other by explicitly sending and 
receiving messages. Our concrete approach is single-program multiple-data (SPMD) i.e. there is only
one file of code and that the code is written to assign certain lines to certain processors.
In this case each processor or subsystem executes the same code but uses its own local variables 
and can read in its own portion of the data.


"""
import numpy as np
from scipy.linalg.blas import ddot, dnrm2
#from numpy import linalg as LA
import scipy.io as sio

import sys
import os
from mpi4py import MPI
   
from opt_problem_loader import *

import time
import psutil


# Statistics in B about system memory usage, for MB / (1024 * 1024)
VmPeakStart = psutil.virtual_memory()[3] # memory in use up until now

# The direcory containg all EV data and a place for results etc.
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

MAXITER  = int(1e3);#int(1e4);   # Maximal amount of iterations
ABSTOL   = 1e-4# absolute and relative tolernce
RELTOL   = 1e-2# 1e-2;1e-3;1e-4;

DISP = True         # Display results iteratively
HISTORY = True      # Save the produced values on each iteration

# The communicator represents a system of computers or processors which can communicate
# with each other via MPI commands
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # rank of the(this) calling process in the communicator
size = comm.Get_size() # number of processes in the communicator

problem_type = "test_valley_filling"
N = 5 # Number of optimization problems
ID = '0' # number of test run
if len(sys.argv) > 1:
    N = int(sys.argv[1])
    
    if len(sys.argv) > 2 :
        T = sys.argv[2] 
        
        if len(sys.argv) > 3 :
           ID = sys.argv[3] 

# auxiliary variables for calculating the step size of rho
lamb=1e-1
mu=1e-1

# divide the EVs in "size" chunks of lenght n = N / size (+ 1)
cs = N / size # chunk size
startidx = cs * rank  
mod = N % size  

if( mod != 0):
      if(rank < mod):
         cs += 1
         startidx += rank
      else:
         startidx += mod
         
# every EV and the aggregator has its own optimization problem which is solved independantly
# but each are subject to a common equilibrium constraint
if(rank == 0 and DISP):
           print 'Reading in data ...'            

# Reading in the data
loader = OptProblemLoaderFactory._get(problem_type)
if rank == 0:       
       # Reading in the data   
       opt_probs = loader.load(0, cs)
              
else:
       # Reading in the data   
       opt_probs = loader.load(startidx, startidx + cs)
       
# global synchronisation: halt until all other tasks of the communicator 
# have posted the same call
comm.Barrier()
       
n,m,p = loader.getProblemDimensions()  # x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp   
                 

eps_pri = np.sqrt(p)  # Primal stopping criteria for convergence
eps_dual = np.sqrt(n)  # Dual stopping criteria

rho=0.5            # Augmented penalty parameter ADMM Step size
v=0               # parameter for changing rho

# a snapshot table of every iteration
if(rank == 0 and DISP):
           print ("%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n" %
                    ('iter', 'step_size', 'r_norm', 'eps_pri', 's_norm','eps_dual', 'objective'))

# save history

if (rank == 0):
    if (HISTORY):
    
        # save results in this file
        historyFileName= DATA_DIR + '/results/admm/' + str(N)
        historyFileName +='_mpi.mat'
    
        # a dictionary containing snapshots of every iteration
        history = {}    
        history["time"] = np.zeros((MAXITER,), dtype='float64') # elapsed time
        history["meminfo"] = np.zeros((MAXITER,), dtype='float64') # memory in use for this programm
        history["cost"] = np.zeros((MAXITER,), dtype='float64') # objective 
        history["r_norm"] = np.zeros((MAXITER,), dtype='float64') # primal residual
        history["s_norm"] = np.zeros((MAXITER,), dtype='float64') # dual residual
        history["eps_pri"] = np.zeros((MAXITER,), dtype='float64') # primal feasability tolerance
        history["eps_dual"] = np.zeros((MAXITER,), dtype='float64') # dual feasability tolerance
        history["rho"] = np.zeros((MAXITER,), dtype='float64') # penalty parameter
    
    
# the x, u and z chunks for this process    
xi = np.zeros((cs,n,1)) 
ui = np.zeros((cs,p,1)) 
zi = np.zeros((cs,m,1)) 

# Used to send and recieve data over MPI 
send = np.zeros(7)
recv = np.zeros(7)         

if(rank == 0):                  
   
   #Cost of iteration
   cost = 0
   # start timing
   tic = time.time() 


# ADMM loop.
for k in xrange(MAXITER):

        send = np.zeros(7) # reinitialize

        for i in xrange(cs):
            
            problem = opt_probs[i]
            
            # xk+1 = argmin x   f(x) + (ρ/2)Ax + Bzk − c + uk2
            problem.setParametersObjX(rho, zi[i], ui[i])             
            xi[i], cost = problem.solveX()
            
            # zk+1 = argmin z   g(z) + (ρ/2)Axk+1 + Bz − c + uk2
            problem.setParametersObjZ(rho, xi[i], ui[i])
            zi_old = np.copy(zi[i])
            zi[i] = problem.solveZ()
            
            # uk+1 = uk + Axk+1 + Bzk+1 − c
            ui[i] = problem.solveU(xi[i], zi[i], ui[i])
            
            # Axi + Bzi - c
            ri = problem.getPrimalResidual(xi[i],zi[i])
            # rho * A.T * B * (zi-zi_old)
            si = problem.getDualResidual(zi[i], zi_old)

            # (Axi, Bzi, c)
            eps_pri = problem.getPrimalFeasability()
            # A.T * ui
            eps_dual = problem.getDualFeasability()
                        
            # used for calculating convergence
            send[0] += ddot(ri, ri)# (ri)^2
            send[1] += ddot(si, si)# (si)^2
            
            # used for calculating convergence
            send[2] += ddot(eps_pri[0], eps_pri[0])# (Axi)^2
            send[3] += ddot(eps_pri[1], eps_pri[1])# (Bzi)^2
            send[4] += ddot(eps_pri[2], eps_pri[2])# c^2
            
            send[5] += ddot(eps_dual, eps_dual)# (A.T * ui)^2
            
            
            send[6] += cost
            
        # Reduces values on all processes to a single value onto all processes:
        # Here the operation computes the global sum over all processors of the contents of the vector
        # "send", and stores the result on every processor in "recv".
        comm.Allreduce(send, recv, op=MPI.SUM)
        
        ####
        ##
        ## ADMM convergence criteria
        ##
        ## x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp
        ##
        ## e_pri = sqrt(p)+ e_abs + e_rel * max{Axk2,Bzk2,c2},
        ## e_dual = sqrt(n) + e_abs + e_rel * A.T x yk2
        ##
        #######
        
        # Temporary results for the convergence test
        r_norm = np.sqrt(recv[0])# sqrt(sum ||ri||_2^2) 
        s_norm = rho * np.sqrt(recv[1])# sqrt(sum ||si||_2^2) 
        
        # Temporary results
        nxstack = np.sqrt(recv[2]) # sqrt(sum ||Axi||_2^2) 
        nzstack = np.sqrt(recv[3]) # sqrt(sum ||Bzi||_2^2) 
        ncstack = np.sqrt(recv[4]) # sqrt(sum ||ci||_2^2)         
        nystack = rho * np.sqrt(recv[5]) # sqrt(sum ||Ak yk||_2^2) ; rescaling y := rho * u
        
        
        # as described above        
        eps_pri = np.sqrt(p) * ABSTOL + RELTOL * max(nxstack, nzstack, ncstack) 
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * nystack    
        
        # current snapshot
        if(rank == 0 and DISP):
           print ("\n%3d\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" %
                  (k, rho, r_norm, eps_pri, s_norm, eps_dual,cost))
                  
        if(rank == 0):
            cost = recv[6]
                  
        #  save history
        if (rank == 0):            
           if (HISTORY):
       
               # the elapsed time
               history['time'][k]= time.time() - tic #toc
           
               #performance memory
               VmPeakIteration = psutil.virtual_memory()[3]
               meminfo=VmPeakIteration-VmPeakStart;
               history['meminfo'][k]=meminfo;
           
               history['cost'][k]= cost 
               history['r_norm'][k]=r_norm
               history['s_norm'][k]=s_norm
               history['eps_pri'][k]=eps_pri
               history['eps_dual'][k]=eps_dual
               history['rho'][k]=rho
               
           
          
       
        # stopping criteria
        if (r_norm <= eps_pri and s_norm <= eps_dual):  
            if (rank == 0):
                print "Finished ADMM at step " + str(k) + " after " + str(time.time() - tic) + " seconds" 
                break

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
           
          else: # alternatively and rarely, but could happen when s_norm = 0
             t_incr = 2
             t_decr = 2
             m = 10
       
             rho_old = rho
             if(r_norm > m * s_norm):
                rho = t_incr * rho        
             elif(s_norm > m * r_norm):
                rho = rho / t_decr
        
             ui = (rho_old/rho) * ui
        
        #ri = x_mean

# rescale u to get y (y = rho * u)
ui = rho * ui

# global synchronisation: halt until all other tasks of the communicator 
# have posted the same call
comm.Barrier()


if rank == 0:
        x = np.empty((n,N))          
        y = np.empty((p,N))         
        z = np.empty((m,N))         
        
        xlist = []
        ylist = []
        zlist = []
        
else:
        x = None
        y = None
        z = None
        
        xlist = None
        ylist = None
        zlist = None

# collect data from all tasks and deliver it to the root task
xlist = comm.gather(xi, root = 0)# a list of comm.size numpy arrays
xi = None

ylist = comm.gather(ui, root = 0)# y := u (y = rho * u)
ui = None

zlist = comm.gather(zi, root = 0)
zi = None

# root process prints results
if rank == 0:

   # create numpy arrays
   x = np.concatenate(xlist, axis=0)#.reshape((T,N))
   xlist = None
   
   y = np.concatenate(ylist, axis=0)#.reshape((T,N))
   ylist = None
   
   z = np.concatenate(zlist, axis=0)#.reshape((T,N))
   zlist = None
  
   if DISP:
       
      print "x:\n", x
      print "y:\n", y
      print "z:\n", z
   
   # Save history into a file       
   if HISTORY:
       
      history["x"] = x
      history["y"] = y
      history["z"] = z
      sio.savemat(historyFileName, history)