# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015

@author: shterev
"""

from cvxpy import *
import numpy as np
from numpy import linalg as LA
import itertools as it
import operator as op

from util import math
from functools import partial

from multiprocessing import Pool, cpu_count

import time

def local_update(rho, x_var, x_mean, f_xi_ui):
    
    f, xi, ui = f_xi_ui
    ui = ui + x_mean    
    xi = math.prox(f, x_var, rho, xi - x_mean - ui)
    #xi_pri = (LA.norm(xi)**2) # for eps primal |Ax|2
    #ui_dual = (LA.norm(rho * ui)**2) # for eps dual |A.T y|2
        
    return (xi, ui)
    

NUM_PROCS = cpu_count() - 1 or 1

MAXITER  = int(1e4);   # Maximal amount of iterations
ABSTOL   = 1e-4;
RELTOL   = 1e-2;# 1e-2;1e-3;1e-4;

# Problem data.
m = 4 # 100
N =  3 # 75
np.random.seed(1)
A = np.random.randn(m, N)
b = np.random.randn(m, 1)



# Setup problem.
sqrt_N = np.sqrt(N)
eps_pri = sqrt_N  # Primal stoping criteria  Convergence
eps_dual= sqrt_N  # Dual stoping criteria
rho=0.5            # Augmented penalty parameter ADMM Step size
v=0;               # parameter for changing rho
lamb=1e-1;
mu=1e-1;

x = Variable(N)
gamma = 0.1

funcs = [sum_squares(A*x - b),
         gamma*norm(x, 1)]
         
n = len(funcs)
         
xi = np.zeros((n, N, 1))
ui = np.zeros((n, N, 1))
z = np.zeros((n, N, 1))
x_mean = np.zeros((N, 1))


if __name__ == "__main__":
  
   pool = Pool(NUM_PROCS)
   t0 = time.time()
   # ADMM loop.
   for i in range(MAXITER):#50
   
       update = partial(local_update, rho, x, x_mean)       
       xi_ui = pool.map(update, it.izip(funcs, xi, ui))
       
       xi = np.array(map(op.itemgetter(0), xi_ui))
       ui = np.array(map(op.itemgetter(1), xi_ui))
       
       x_mean = np.divide(np.sum(xi, axis=0), N)
       z_old = z
       z = xi - x_mean
       
       r_norm = sqrt_N * LA.norm(x_mean)
       s_norm = sqrt_N * rho * LA.norm(z - z_old)
       
       eps_pri = sqrt_N * ABSTOL + RELTOL * max(LA.norm(xi), LA.norm(-1 * z))
       eps_dual = sqrt_N * ABSTOL + RELTOL * LA.norm(rho * ui)    
       
       # stopping criteria
       if (r_norm <= eps_pri and s_norm <= eps_dual):
         #if (r_LA.norm < eps_pri):   
           print "Finished at step ", i
           break
       
       print "ADMM iteration ", i

       # update rho 

       # According to Boyd et. al
          
       rho_old = rho
       v_old = v
       v = rho * r_norm/s_norm - 1   
       rho = rho* np.exp(lamb * v + mu * (v - v_old))
       
       # rescale u
       ui = (rho_old/rho) * ui
       
       # alternatively
       '''t_incr = 2
       t_decr = 2
       mu = 10
       
       rho_old = rho
       if(r_norm > mu * s_norm):
         rho = t_incr * rho
         
       elif(s_norm > mu * r_norm):
         rho = rho / t_decr
       else:
           continue
       
       ui = (rho_old/rho) * ui'''
         
         
       


   print time.time() - t0, "seconds ADMM time"
   # Compare ADMM with standard solver.

   pool.close()
   pool.join() #waits for all the processes to finish
   
   y = Variable(N)
   z = Variable(N)   
   min_sum = sum_squares(A*y - b) + gamma*norm(z, 1)
   
   t0 = time.time()
   prob = Problem(Minimize(min_sum), [y + z == 0])
   result = prob.solve()
   print time.time() - t0, "seconds ECOS time"
   
   print "ADMM best", (sum_squares(np.dot(A, xi[0]) - b) + gamma*norm(xi[1], 1)).value
   print "ECOS best", result