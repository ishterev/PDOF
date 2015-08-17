# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015
@author: shterev
"""
###################################################################################################
# To run execute the follwoing command. The console should point to this file's working directory
#
# mpiexec -n 4 python admm_exchange_valley_filling_mpi.py 100
#
# where 100 is an example for the number of EVs
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
from mpi4py import MPI

from opt_problem_evs import *

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

chargeStrategy = 'home'
V2G = True
gamma = 0 #trade off parameter

N_EV = 0 # Number of EVs
ID = '0' # number of test run
if len(sys.argv) > 1:
    N_EV = int(sys.argv[1])
    
    if len(sys.argv) > 2 :
        ID = sys.argv[2]       

N = N_EV + 1 # Number of agents in the exchange N=N_EV+1

deltaT=15*60 # Time slot duration [sec]
T= 24*3600/deltaT # Number of time slots

# auxiliary variables for calculating the step size of rho
lamb=1e-1
mu=1e-1

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
         
# every EV and the aggregator has its own optimization problem which is solved independantly
# but each are subject to a common equilibrium constraint
opt_probs = np.empty((n,), dtype=np.object)      

if(rank == 0 and DISP):
           print 'Reading in data ...' 
           

# Reading in the data
if rank == 0:
       
        # w.l.o.g. and for convenience, the aggregator is the 0th element in the 0th process
        problem = OptProblem_Aggregator_ValleyFilling()  
        
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
                 

eps_pri = np.sqrt(N)  # Primal stopping criteria for convergence
eps_dual = np.sqrt(N)  # Dual stopping criteria

rho=0.5            # Augmented penalty parameter ADMM Step size
v=0               # parameter for changing rho

# a snapshot table of every iteration
if(rank == 0 and DISP):
           print ("%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n" %
                    ('iter', 'step_size', 'r_norm', 'eps_pri', 's_norm','eps_dual', 'objective'))

# save history
if (rank == 0 and HISTORY):
    
    # save results in this file
    historyFileName= DATA_DIR + '/results/' + str(N_EV) + 'EVs_' +  chargeStrategy
    if V2G :
       historyFileName +='_V2G';

    historyFileName +='_gamma_' + str(gamma) ###
    historyFileName +='_mpi.mat'
    
    # a dictionary containing snapshots of every iteration
    history = {}    
    history["time"] = np.zeros((MAXITER,), dtype='float64') # elapsed time
    history["meminfo"] = np.zeros((MAXITER,), dtype='float64') # memory in use for this programm
    history["cost"] = np.zeros((MAXITER,), dtype='float64') # + delta*costEVs; real cost of the EVs 
    history["costEVs"] = np.zeros((MAXITER,), dtype='float64') # sum of EVs costs
    history["costAggr"] = np.zeros((MAXITER,), dtype='float64')# aggregator's cost
    history["r_norm"] = np.zeros((MAXITER,), dtype='float64') # primal residual
    history["s_norm"] = np.zeros((MAXITER,), dtype='float64') # dual residual
    history["eps_pri"] = np.zeros((MAXITER,), dtype='float64') # primal feasability tolerance
    history["eps_dual"] = np.zeros((MAXITER,), dtype='float64') # dual feasability tolerance
    history["rho"] = np.zeros((MAXITER,), dtype='float64') # penalty parameter
    
    
# the x, u and z chunks for this process    
xi = np.zeros((n,T,1)) # Profile of a single EV for all time slots (for aggregator,respectively, aggregated EV profile for all time slots)
ui = np.zeros((n,T,1)) # scaled price vector (dual variable)
zi = np.zeros((n,T,1)) # only needed for convergence checks
x_mean = np.zeros((T,1)) # the mean of all xi

# Used to send and recieve data over MPI 
send = np.zeros(5)
recv = np.zeros(5)
         

if(rank == 0):                  
   
   #Cost of iteration
   cost = 0 # + delta*costEVs; real cost of the EVs  
   costEVs = 0 # sum of EVs costs
   costAggr = 0 # aggregator's cost
   xAggr = np.zeros((T,1)) # aggregated EV profile for all time slots
   
   tic = time.time() # start timing


# ADMM loop.
for k in xrange(MAXITER):

        send = np.zeros(5) # reinitialize
        xsum = np.zeros((T,1)) # sum of all n xi in the process

        for j in xrange(n):
            
            # ADMM iteration step (simplified for optimal exchage)
            zi_old = zi[j]
            zi[j] = xi[j] - x_mean            
            
            ui[j] = ui[j] + x_mean 
            
            # optimization for the current EV
            problem = opt_probs[j]
            problem.setParameters(rho, xi[j] - x_mean - ui[j])
            # optimal profile, cost
            xi[j], ci = problem.solve()
            
            # used for calculating convergence
            send[0] += ddot(xi[j], xi[j])# xi[j]^2
            send[1] += ddot(ui[j], ui[j])
            send[2] += ddot(zi[j], zi[j])
            
            zdiff = zi[j] - zi_old
            send[3] += ddot(zdiff, zdiff)
            
            # read in aggregator
            if rank == 0 and j == 0:               
                xAggr = xi[j]
                costAggr = ci
            # or accumulate EV costs 
            else:
               send[4] += ci #costEVs
            
            # sum all profiles
            xsum += xi[j]
            
            
            
        # Reduces values on all processes to a single value onto all processes:
        # Here the operation computes the global sum over all processors of the contents of the vector
        # "send", and stores the result on every processor in "recv".
        comm.Allreduce(send, recv, op=MPI.SUM)
        # similarly, sums all local "xsum" into "x_mean"
        comm.Allreduce(xsum, x_mean, op=MPI.SUM)    
        
        # at this point x_mean represents the sum of all agents profiles (EVs + aggregator)        
        if rank == 0:
           # To calculate the real cost of EVs we need only the EVs profiles sum (=> - aggregator)
           x_sum = x_mean - xAggr
           cost = ddot(D + x_sum, D + x_sum) #+ delta*costEVs; real cost of the EVs        
        
        # now, it is really the mean
        x_mean /=  N
        
        ####
        ##
        ## ADMM convergence criteria
        ##
        ## x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp
        ##
        ## For exchange A = 1 and B = -1
        ##
        ## e_pri = sqrt(p)+ e_abs + e_rel * max{Axk2,Bzk2,c2},
        ## e_dual = sqrt(n) + e_abs + e_rel * A.T x yk2
        ##
        ## This time n = p = T
        #######

        # Temporary results for the convergence test
        nxstack = np.sqrt(recv[0])  # sqrt(sum ||x_i||_2^2) 
        nystack = rho * np.sqrt(recv[1])  # sqrt(sum ||y_i||_2^2) ; rescaling y := rho * u
        nzstack = np.sqrt(recv[2]) # dnrm2(-1 * zi)
        nzdiffstack = np.sqrt(recv[3])
        
        if(rank == 0):
            costEVs = recv[4]

        
        # rk+1 = Axk+1 + Bzk+1 − c
        r_norm  =  np.sqrt(N) * dnrm2(x_mean)# r_i = x_mean => sqrt(sum ||r_i||_2^2) = sqrt(N * ||x_mean||_2^2)
       
        # |ρ * A.TxB * (zk+1 − zk)|2
        s_norm = rho * nzdiffstack # |ρ * -1 * (zk+1 − zk)|2 = ρ * |zk+1 − zk|2
       
        # as described above        
        eps_pri = np.sqrt(T) * ABSTOL + RELTOL * max(nxstack, nzstack) 
        eps_dual = np.sqrt(T) * ABSTOL + RELTOL * nystack    
        
        # current snapshot
        if(rank == 0 and DISP):
           print ("\n%3d\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" %
                  (k, rho, r_norm, eps_pri, s_norm, eps_dual,cost))
                  
        #  save history
        if (rank == 0 and HISTORY):
       
           # the elapsed time
           history['time'][k]= time.time() - tic #toc
           
           #performance memory
           VmPeakIteration = psutil.virtual_memory()[3]
           meminfo=VmPeakIteration-VmPeakStart;
           history['meminfo'][k]=meminfo;
           
           history['cost'][k]= cost 
           history['costEVs'][k]= costEVs 
           history['costAggr'][k]=costAggr
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
             
             if(rho == 2):
                # a dirty hack, because of the analytical solution for x in 
                # OptProblem_Aggregator_ValleyFilling:
                # x = rho/(rho-2)*K - 2/(rho-2) * D
                rho += ABSTOL
                #continue
       
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
                
             if(rho == 2):
                # a dirty hack, because of the analytical solution for x in 
                # OptProblem_Aggregator_ValleyFilling:
                # x = rho/(rho-2)*K - 2/(rho-2) * D
                rho += ABSTOL 
                
               
        
             ui = (rho_old/rho) * ui
        
        #ri = x_mean

# rescale u to get y (y = rho * u)
ui = rho * ui

# global synchronisation: halt until all other tasks of the communicator 
# have posted the same call
comm.Barrier()


if rank == 0:
        x = np.empty((T,N))         # Agents profile 
        y = np.empty((T,N))         # Price vector
        z = np.empty((T,N))         # Help variable 
        
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
        