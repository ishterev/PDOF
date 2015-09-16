# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015
@author: shterev
"""
###################################################################################################
# To run execute the follwoing command. The console should point to this file's working directory
#
# python admm_exchange_valley_filling.py 100
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

The implementation is based on the works of Boyd et. al. 


"""
import numpy as np
from scipy.linalg.blas import ddot, dnrm2
#from numpy import linalg as LA
import scipy.io as sio

CVXPY = False
if(CVXPY):
   from opt_problem_evs_cvxpy import *
else:
   from opt_problem_evs_gurobi import *

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

chargeStrategy = 'home'
V2G = True
gamma = 0 #trade off parameter

N_EV = 4 # Number of EVs
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
         
# every EV and the aggregator has its own optimization problem which is solved independantly
# but each are subject to a common equilibrium constraint
opt_probs = np.empty((N,), dtype=np.object)      

if(DISP):
   print 'Reading in data ...' 
           

# Reading in the data
# w.l.o.g. and for convenience, the aggregator is the 0th element
problem = OptProblem_Aggregator_ValleyFilling()  
        
D = problem.D        
# Empirical [price/demand^2]     
delta =  np.mean(problem.price)/(np.mean(problem.D) * (3600*1000)  *15*60 ) ;     
OptProblem_ValleyFilling_Home.delta = delta;
OptProblem_ValleyFilling_Home.gamma = gamma;
OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta;
        
opt_probs[0] = problem
        
for i in xrange(1, N):
    problem = OptProblem_ValleyFilling_Home(i, V2G)
    opt_probs[i] = problem
              
              
eps_pri = np.sqrt(N)  # Primal stopping criteria for convergence
eps_dual = np.sqrt(N)  # Dual stopping criteria

rho=0.5            # Augmented penalty parameter ADMM Step size
v=0               # parameter for changing rho

# a snapshot table of every iteration
if(DISP):
  print ("%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n" %
        ('iter', 'step_size', 'r_norm', 'eps_pri', 's_norm','eps_dual', 'objective'))

# save history
if (HISTORY):
    
    # save results in this file
    historyFileName= DATA_DIR + '/results/valley_filling/' + str(N_EV) + 'EVs_' +  chargeStrategy
    if V2G :
       historyFileName +='_V2G';

    historyFileName +='_gamma_' + str(gamma) ###
    historyFileName +='.mat'
    
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
x = np.zeros((N,T,1)) # Profile of a single EV for all time slots (for aggregator,respectively, aggregated EV profile for all time slots)
u = np.zeros((N,T,1)) # scaled price vector (dual variable)
z = np.zeros((N,T,1)) # only needed for convergence checks
x_mean = np.zeros((T,1)) # the mean of all agents profiles (EVs + aggregator) 

  
#Cost of iteration
cost = 0 # + delta*costEVs; real cost of the EVs  
costEVs = 0 # sum of EVs costs
costAggr = 0 # aggregator's cost
xAggr = np.zeros((T,1)) # aggregated EV profile for all time slots
   
tic = time.time() # start timing


# ADMM loop.
for k in xrange(MAXITER):
    
        costEVs = 0 # reinitialize
        # Temporary results for the convergence test
        nxstack = 0  # sqrt(sum ||x_i||_2^2) 
        nystack = 0  # sqrt(sum ||y_i||_2^2) 
        nzstack = 0 # dnrm2(-1 * z)
        nzdiffstack = 0
        xsum = np.zeros((T,1)) # sum of all x

        for j in xrange(N):
            
            # ADMM iteration step (simplified for optimal exchage)            
            zi_old = z[j]
            z[j] = x[j] - x_mean 
                                
            u[j] = u[j] + x_mean 
            
            # optimization for the current EV
            problem = opt_probs[j]
            problem.setParameters(rho, x[j] - x_mean - u[j])
            # optimal profile, cost
            x[j], ci = problem.solve()   
            
            # used for calculating convergence
            nxstack += ddot(x[j], x[j])# x[j]^2
            nystack += ddot(u[j], u[j])
            nzstack += ddot(z[j], z[j])
            
            zdiff = z[j] - zi_old
            nzdiffstack += ddot(zdiff, zdiff)
            
            # read in aggregator
            if j == 0:               
                xAggr = x[j]
                costAggr = ci
            # or accumulate EV costs 
            else:
               costEVs += ci #costEVs
            
            # sum of all agents profiles (EVs + aggregator)  
            xsum += x[j]
            
        # Temporary results for the convergence test
        nxstack = np.sqrt(nxstack)  # sqrt(sum ||x_i||_2^2) 
        nystack = rho * np.sqrt(nystack)  # sqrt(sum ||y_i||_2^2); rescaling y := rho * u
        nzstack = np.sqrt(nzstack) # dnrm2(-1 * zi)
        nzdiffstack = np.sqrt(nzdiffstack)
       
        # To calculate the real cost of EVs we need only the EVs profiles sum (=> - aggregator)
        x_sum = xsum - xAggr
        cost = ddot(D + x_sum, D + x_sum) #+ delta*costEVs; real cost of the EVs        
        
        # the mean of all agents profiles (EVs + aggregator) 
        x_mean = xsum / N
        
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

        
        # rk+1 = Axk+1 + Bzk+1 − c
        r_norm  =  np.sqrt(N) * dnrm2(x_mean)# r_i = x_mean => sqrt(sum ||r_i||_2^2) = sqrt(N * ||x_mean||_2^2)
       
        # |ρ * A.TxB * (zk+1 − zk)|2
        s_norm = rho * nzdiffstack # |ρ * -1 * (zk+1 − zk)|2 = ρ * |zk+1 − zk|2
       
        # as described above        
        eps_pri = np.sqrt(T) * ABSTOL + RELTOL * max(nxstack, nzstack) 
        eps_dual = np.sqrt(T) * ABSTOL + RELTOL * nystack    
        
        # current snapshot
        if(DISP):
           print ("\n%3d\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" %
                  (k, rho, r_norm, eps_pri, s_norm, eps_dual,cost))
                  
        #  save history
        if (HISTORY):
       
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
             u = (rho_old/rho) * u
           
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
                
               
        
             u = (rho_old/rho) * u
        
        #ri = x_mean

# rescale u to get y (y = rho * u)
y = rho * u


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
        