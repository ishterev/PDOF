# -*- coding: utf-8 -*-
"""
Created on Sat May 09 00:06:45 2015

@author: shterev
"""
from cvxpy import *
import numpy as np

class OptimizationProblem:
        
    def __init__(self, f, constraints): 
        self.f = f
        # initilize the problem
        self.problem = Problem(Minimize(f), constraints)   
        # read in the x variable
        self.x = self.problem.variables()[0]
        # declare parameters
        self.v = Parameter(self.getN(), value = np.zeros(self.getN()))
        self.rho = Parameter(sign="positive", value = 0.5)
        # add proximal term to objective
        self.problem.objective = Minimize(f + 
                      (self.rho/2)*sum_squares(self.x - self.v))
        
        
    def solve(self):
        self.problem.solve(solver=CVXOPT) #solver=CVXOPT
        return np.array(self.x.value) 
        
    def setParameters(self, rho, v):
        self.rho.value = rho
        self.v.value = v
        
    def getN(self):
        return self.x.size[0]
        
        
        
if __name__ == "__main__":

   #freeze_support()
   
   # Problem data.
   m = 4 # 100
   N =  3 # 75
   np.random.seed(1)
   A = np.random.randn(m, N)
   b = np.random.randn(m, 1)

   x1 = Variable(N)
   func1 = sum_squares(A*x1 - b);
   p1 = OptimizationProblem(func1, [-100000 <= x1, x1 <= 100000])
   p1.setParameters(0.5, np.zeros(N))
   p1.solve()
   
   print x1.value
        
    
        
        

        