# -*- coding: utf-8 -*-
"""
Created on Fri May 01 20:00:46 2015
@author: shterev
"""

from cvxpy import *
import numpy as np

from multiprocessing import cpu_count

NUM_PROCS = cpu_count() - 1 or 1

MAXITER  = 1000; #int(1e4);   # Maximal amount of iterations
ABSTOL   = 1e-4;
RELTOL   = 1e-2;# 1e-2;1e-3;1e-4;

rho_init=0.5            # Augmented penalty parameter ADMM Step size
lamb=1e-1;
mu=1e-1;

def prox(f, x, rho, v):           
    f += (rho/2)*sum_squares(x - v)
    Problem(Minimize(f)).solve(solver=CVXOPT)
    return np.array(x.value)


''' evtl. make these calls here, r_ and s_norms are varying 
def r_norm(xi, zi):
    return norm_2(np.subtract(xi, zi));
    
def s_norm(zi, zi_old):
    return norm_2(-rho*(np.subtract(zi, zi_old)));
    
def eps_pri(xi, zi):
    return (sqrt(N) * ABSTOL + RELTOL * max(norm_2(xi), norm_2(np.dot(-1, zi)))).value;
    
def eps_dual(ui):
    return  (sqrt(N) * ABSTOL + RELTOL* norm_2(np.dot(rho, ui))).value; '''