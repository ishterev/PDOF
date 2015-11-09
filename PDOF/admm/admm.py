# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015
@author: shterev
"""
###################################################################################################
# To run execute the follwoing command. The console should point to this file's working directory
#
# python admm.py 100
#
# where 100 is an example for the number of problems (EVs + Aggregator)
###################################################################################################
"""

The aggregator in a hierarchical control structure influences EV charging behavior
through an incentive signal to reach certain goals. While the aggregator wants to use EVs 
to minimize their operational costs, the individual EVs want to minimize their charging 
and battery depreciation costs.

To adderess this trade off, the problem can be formulated as a standard exchange optimization problem
and consider the EVs and the aggregator as its agents. The exchange problem considers N agents 
exchanging a common good under an equilibrium constraint [Rivera et. al].

"""
import numpy as np
from scipy.linalg.blas import ddot, dnrm2
#from numpy import linalg as LA
import scipy.io as sio

import sys
import os
   
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
         
# every EV and the aggregator has its own optimization problem which is solved independantly
# but each are subject to a common equilibrium constraint
if(DISP):
   print 'Reading in data ...'            

# Reading in the data
loader = OptProblemLoaderFactory._get(problem_type)
opt_probs = loader.load(0, N)
       
n,m,p = loader.getProblemDimensions()  # x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp                    
idf_constr = loader.getIdentificatorFunctionConstraint()

eps_pri = np.sqrt(p)  # Primal stopping criteria for convergence
eps_dual = np.sqrt(n)  # Dual stopping criteria

rho=0.5            # Augmented penalty parameter ADMM Step size
v=0               # parameter for changing rho

# a snapshot table of every iteration
if(DISP):
    print ("%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n" %
          ('iter', 'step_size', 'r_norm', 'eps_pri', 's_norm','eps_dual', 'objective'))

# save history
if (HISTORY):    
    # save results in this file
    historyFileName= DATA_DIR + '/results/admm/' + str(N)
    historyFileName +='.mat'
    
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
    
    
x = np.zeros((N,n,1)) 
u = np.zeros((N,p,1)) 
z = np.zeros((N,m,1)) 
z_old = np.zeros((N,m,1)) 

#Cost of iteration
cost = 0
# start timing
tic = time.time() 

# ADMM loop.
for k in xrange(MAXITER):
    
        # reinitialization
        # Temporary results for the convergence test
        cost = 0
        r_norm = 0 # sqrt(sum ||ri||_2^2) 
        s_norm = 0 # sqrt(sum ||si||_2^2) 
        
        nxstack = 0 # sqrt(sum ||Axi||_2^2) 
        nzstack = 0 # sqrt(sum ||Bzi||_2^2) 
        ncstack = 0 # sqrt(sum ||ci||_2^2)         
        nystack = 0 # sqrt(sum ||Ak yk||_2^2)

        for i in xrange(N):
            
            problem = opt_probs[i]
            
            # xk+1 = argmin x   f(x) + (ρ/2)Ax + Bzk − c + uk2
            problem.setParametersObjX(rho, z[i], u[i])             
            x[i], ci = problem.solveX()
            cost += ci
            
            # zk+1 = argmin z   g(z) + (ρ/2)Axk+1 + Bz − c + uk2
            problem.setParametersObjZ(rho, x[i], u[i])
            z_old[i] = z[i]
            z[i] = problem.solveZ()            
            
            
        if(idf_constr is not None):
           z = idf_constr.project(z)
            
            
        for i in xrange(N):
            
            # uk+1 = uk + Axk+1 + Bzk+1 − c
            u[i] = problem.solveU(x[i], z[i], u[i])
            
            # Axi + Bzi - c
            ri = problem.getPrimalResidual(x[i],z[i])
            # rho * A.T * B * (zi-zi_old)
            si = problem.getDualResidual(z[i], z_old[i])

            # (Axi, Bzi, c)
            eps_pri = problem.getPrimalFeasability()
            # A.T * ui
            eps_dual = problem.getDualFeasability()
                        
            # used for calculating convergence
            r_norm += ddot(ri, ri)# (ri)^2
            s_norm += ddot(si, si)# (si)^2
            
            # used for calculating convergence
            nxstack += ddot(eps_pri[0], eps_pri[0])# (Axi)^2
            nzstack += ddot(eps_pri[1], eps_pri[1])# (Bzi)^2
            ncstack += ddot(eps_pri[2], eps_pri[2])# c^2
            
            nystack += ddot(eps_dual, eps_dual)# (A.T * ui)^2
        
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
        r_norm = np.sqrt(r_norm)# sqrt(sum ||ri||_2^2) 
        s_norm = rho * np.sqrt(s_norm)# sqrt(sum ||si||_2^2) 
        
        # Temporary results
        nxstack = np.sqrt(nxstack) # sqrt(sum ||Axi||_2^2) 
        nzstack = np.sqrt(nzstack) # sqrt(sum ||Bzi||_2^2) 
        ncstack = np.sqrt(ncstack) # sqrt(sum ||ci||_2^2)         
        nystack = rho * np.sqrt(nystack) # sqrt(sum ||Ak yk||_2^2) ; rescaling y := rho * u
        
        # as described above        
        eps_pri = np.sqrt(p) * ABSTOL + RELTOL * max(nxstack, nzstack, ncstack) 
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * nystack    
        
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

             u = (rho_old/rho) * u
        
        #ri = x_mean

# rescale u to get y (y = rho * u)
y = rho * u

# create numpy arrays

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