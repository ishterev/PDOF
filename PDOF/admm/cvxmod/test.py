# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:11:17 2015

@author: shterev
"""
from cvxmod import *
from cvxmod.atoms import quadform

import numpy as np

m = 4 # 100
N =  3 # 75

A = param('A', m, N)
b = param('b', m, 1)

x = optvar('x', N, 1)

objv = quadform(A*x -b) #sum_squares(A*x1 - b);

gamma = 0.1
x2 = optvar('x2', N, 1)
objv += gamma*norm1(x2)

constr = [x + x2 == 0]
prob = problem(minimize(objv), constr)

#np.random.seed(1)
#A.value = matrix(np.random.randn(m, N))
#b.value = matrix(np.random.randn(m, 1))


#prob.solve()

#print x.value

prob.cgen('test/')

print "Code generated"