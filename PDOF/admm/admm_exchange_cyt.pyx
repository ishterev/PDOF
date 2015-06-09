# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:01:32 2015
@author: shterev
"""
cimport cython

from cvxpy import *

import numpy as np
cimport numpy as np
from numpy import linalg as LA

DTYPE = np.float64
#ctypedef np.float64_t DTYPE_t

import itertools as it
from opt_problem import *


from util import math
from functools import partial

from multiprocessing import Pool, freeze_support

import time

cdef unsigned int NUM_PROCS = cpu_count() - 1 or 1

cdef unsigned int MAXITER  = 1000; #int(1e4);   # Maximal amount of iterations
cdef DTYPE ABSTOL   = 1e-4;
cdef DTYPE RELTOL   = 1e-2;# 1e-2;1e-3;1e-4;

cdef DTYPE lamb=1e-1;
cdef DTYPE mu=1e-1;

def local_update(rho, x_mean, p_xi_ui):
    
    prob, xi, ui = p_xi_ui
    ui = ui + x_mean       
    prob.setParameters(rho, xi - x_mean - ui)
    xi = prob.solve()
    #xi = math.prox(f, x_var, rho, xi - x_mean - ui)
    #xi_pri = (LA.norm(xi)**2) # for eps primal |Ax|2
    #ui_dual = (LA.norm(rho * ui)**2) # for eps dual |A.T y|2
        
    return (xi, ui)
    

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def main(prob_list):
    
    if(len(prob_list) == 0):
        return  
        
    p = prob_list[0]
    
    cdef Py_ssize_t N = p.getN()# unsigned int
    cdef Py_ssize_t n = len(prob_list) #unsigned int
    
    # Setup problem
    cdef DTYPE sqrt_N = np.sqrt(N)
    #cdef DTYPE eps_pri = sqrt_N  # Primal stoping criteria  Convergence
    #cdef DTYPE eps_dual = sqrt_N  # Dual stoping criteria
    cdef DTYPE rho = 0.5      # Augmented penalty parameter ADMM Step size
    cdef DTYPE v=0               # parameter for changing rho
       
    cdef np.ndarray[DTYPE, ndim=3, mode="c"] xi = np.zeros((n, N, 1), dtype=DTYPE)
    cdef np.ndarray[DTYPE, ndim=3, mode="c"] ui = np.zeros((n, N, 1), dtype=DTYPE)
    cdef np.ndarray[DTYPE, ndim=3, mode="c"] z = np.zeros((n, N, 1), dtype=DTYPE)
    cdef np.ndarray[DTYPE, ndim=2, mode="c"] x_mean = np.zeros((N, 1), dtype=DTYPE)
    
    pool = Pool(math.NUM_PROCS)
    cdef unsigned long t_start = time.time()
    # ADMM loop.
    cdef unsigned int i
    for i in range(math.MAXITER):#50
   
        update = partial(local_update, rho, x_mean)       
        xi_ui = pool.map(update, it.izip(plist, xi, ui))
        
        #for j in range(n):
        #    xi[j] = xi_ui[j][0]
        #    ui[j] = xi_ui[j][1]
          
        cdef Py_ssize_t idx = 0 # unsigned int idx
        while(len(xi_ui) > 0):
            xi[idx], ui[idx] = xi_ui.pop(0)
            idx += 1
            
            
        #xi = np.array(map(op.itemgetter(0), xi_ui))
        #ui = np.array(map(op.itemgetter(1), xi_ui))
       
        x_mean = np.divide(np.sum(xi, axis=0), N)
        cdef np.ndarray[DTYPE, ndim=3] z_old = z
        z = xi - x_mean
       
        cdef DTYPE r_norm = sqrt_N * LA.norm(x_mean)
        cdef DTYPE s_norm = sqrt_N * rho * LA.norm(z - z_old)
       
        cdef DTYPE eps_pri = sqrt_N * math.ABSTOL + math.RELTOL * max(LA.norm(xi), LA.norm(-1 * z))
        cdef DTYPE eps_dual = sqrt_N * math.ABSTOL + math.RELTOL * LA.norm(rho * ui)    
       
        # stopping criteria
        if (r_norm <= eps_pri and s_norm <= eps_dual):
          #if (r_LA.norm < eps_pri):   
            print "Finished at step ", i
            break
       
        print "ADMM iteration ", i

        # update rho 
        # According to Boyd et. al
          
        cdef DTYPE rho_old = rho
        cdef DTYPE v_old = v
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


    print time.time() - t_start, "seconds ADMM time"

    pool.close()
    pool.join() #waits for all the processes to finish
    
    ui = rho * ui

    return [xi, z, ui] # x, z, y TODO: in np.array
    
    


if __name__ == "__main__":

   freeze_support()
   
   # Problem data.
   m = 4 # 100
   N =  3 # 75
   np.random.seed(1)
   A = np.random.randn(m, N)
   b = np.random.randn(m, 1)

   x1 = Variable(N)
   func1 = sum_squares(A*x1 - b);
   p1 = OptimizationProblem(func1, [-100000 <= x1, x1 <= 100000])
   
   
   gamma = 0.1
   x2 = Variable(N)
   func2 = gamma*norm(x2, 1)
   p2 = OptimizationProblem(func2, [-100000 <= x2, x2 <= 100000])

   plist = [p1, p2]
         
   x_z_y = main(plist)
  
   # Compare ADMM with standard solver.
   
   y = Variable(N)
   z = Variable(N)   
   min_sum = sum_squares(A*y - b) + gamma*norm(z, 1)
   
   t0 = time.time()
   prob = Problem(Minimize(min_sum), [y + z == 0])
   result = prob.solve()
   print time.time() - t0, "seconds ECOS time"
   
   print "ADMM best", (sum_squares(np.dot(A, x_z_y[0][0]) - b) + gamma*norm(x_z_y[0][1], 1)).value
   print "ECOS best", result