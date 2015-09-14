# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:03:36 2015

@author: shterev
"""

import numpy as np
from numpy import linalg as LA
from scipy.linalg.blas import ddot, dnrm2
from cvxpy import *
from gurobipy import *


# Common interface
class OptimizationProblem:
           
    def solve(self):
        pass
        
    def setParameters(self, params):
        pass
    

class OptimizationProblemCvxpy(OptimizationProblem):
    
    # objective function has only one variable that is x
    def setX(self, shape): # (96,1)
        self.x = Variable(shape)
        return self.x
        
    def getX(self):
        return self.x
        
    def addConstraint(self, constraint):
        if self.constraints:
           self.constraints.append(constraint)
        else:
          self.constraints = [constraint]
          
    def setObjective(self, f, sense = 'min'):
        
        x = self.getX()
        v = Parameter(x.size, value = np.zeros(x.size))
        rho = Parameter(sign="positive", value = 0.5)
        
        self.v = v
        self.rho = rho
        # add proximal term to objective
        if(sense == 'min'):
           self.objective = Minimize(f + (rho/2)*sum_squares(x - v))
        elif(sense == 'max'):
            self.objective = Maximize(f + (rho/2)*sum_squares(x - v))
        else:
            raise ValueError('Objective has no specified sense')
            
    def setV(self, v):  
        self.v.value = v
        return self.v

        
    def setParameters(self, rho, K):       
        # commented out for tweaking
        # assert self.rho
        # assert self.K
        
        self.rho.value = rho
        self.K.value = K
            
            
    def setModel(self):
        # assert self.objective
        self.model = Problem(self.objective, self.constraints)
        
    
    def solve(self):
        # assert self.model
        self.model.solve(solver=GUROBI) #solver=CVXOPT
        return np.array(self.x.value) 
        
        
        
class OptimizationProblemGurobi(OptimizationProblem):
    
    # objective function has only one variable that is x
    def setX(self, shape): 
    
        #assert self.model
         
        # Add variables to model
        for i in xrange(shape[0]):# shape is either of the form (n,1) or (n)
            self.model.addVar()
        self.model.update()
        self.x = self.model.getVars()         

        return self.x
        
    def getX(self):
        return self.x
        

    def addConstraint(self, constraint):
         #assert self.model        
         assert len(constraint) >= 3

         ###########################################################################
         #
         # [left quad, left linear, <=, b] ([A, B, '<=' , b] <=> x.T A x + Bx <= b)
         #
         ###########################################################################
         if (len(constraint) == 4):
             
             A = constraint[0]
             B = constraint[1]
             op = constraint[2]
             b = constraint[3]
             
             if(op == '==' or op == '='):
                op = GRB.EQUAL
             elif(op == '<=' or op == '<'):
                op = GRB.LESS_EQUAL
             elif(op == '>=' or op == '>'):
                op = GRB.GREATER_EQUAL
             else:
                op = GRB.EQUAL
                
             ############################################################################################
             #
             # (opt. 1/2*) x.T A x + Bx <= b, because of the quadratic term it can be only one line 
             #
             # i.e. (1,n) x (n x n) x (n,1) + (1, n) x (n,1) = (1,1)
             #
             ############################################################################################
             
             # Input check and normalization
             
             # normalization (n) -> (1,n), 
             if(len(B.shape) == 1):
                B.reshape((1, B.shape[0]))
             # (1) -> (1,1)
             if(len(b.shape) == 1):# should not happen, (n) numpy arrays are always read in as (n,1)
                b.reshape((b.shape[0], 1))  
                
             n = self.getX().size[0]             
             assert A.shape == (n,n) 
             assert B.shape == (1,n)
             assert b.shape == (1,1)
             
             x = self.getX()            
             expr = QuadExpr()
             for i in xrange(n):
                
                 for j in xrange(n):                     
                     expr += x[j] * A[i][j] * x[i]
                          
             for i in xrange(n):
                 if B[0][i] != 0:
                     expr += B[0][i] * x[i]
       
             self.model.addConstr(expr,  op, b[0][0])
             self.model.update()          
                    
         
         #########################################################
         #   
         # [left, operator, right] ([A, '<=' , b] <=> Ax <= b)  
         #
         ##########################################################
         elif (len(constraint) == 3):
                
             A = constraint[0]
             op = constraint[1]
             b = constraint[2]
             
             if(op == '==' or op == '='):
                op = GRB.EQUAL
             elif(op == '<=' or op == '<'):
                op = GRB.LESS_EQUAL
             elif(op == '>=' or op == '>'):
                op = GRB.GREATER_EQUAL
             else:
                op = GRB.EQUAL
             
                
             #####################################################################
             #
             # we have m times Bx <= b 
             #
             # i.e. (m, n) x (n,1) = (m,1)
             #
             ######################################################################
             
             # Input check and normalization
             
             # normalization (n) -> (1,n) 
             if(len(A.shape) == 1):
                A.reshape((1, A.shape[0]))
             # (n) -> (n,1)
             if(len(b.shape) == 1):# should not happen, (n) numpy arrays are always read in as (n,1)
                b.reshape((b.shape[0], 1))  
                
             n = len(self.getX())            
             # (m,n) 
             assert len(A.shape) == 2 and A.shape[1] == n
             m = A.shape[0]
             # (m,n) x (n,1) = (m,1)
             assert b.shape == (m,1)
                
             x = self.getX()  
             for i in xrange(m):
                 
                 expr = LinExpr()                
                 for j in xrange(n):
                     if A[i][j] != 0:
                        expr += A[i][j] * x[j]
                        
                 self.model.addConstr(expr,  op, b[i][0])
                     
             self.model.update()   
                
         else:
             raise ValueError("Unspecified constraint argument")
          
          
    def setObjective(self, f, sense = 'min'):
        
        #assert self.model        
        assert len(f) >= 1
        
        x = self.getX() 
        n = len(x)
        
        fexpr = None

        ###########################################################################
        #
        # [A, B], 'min' <=> minimize (opt. 1/2*) x.T A x + Bx 
        #
        ###########################################################################
        if (len(f) == 2):
             
            A = f[0]
            B = f[1]
             
            assert A.shape == (n,n) 
            assert B.shape == (1,n) 
                       
            fexpr = QuadExpr()
            for i in xrange(n):
                
                for j in xrange(n): 
                    if A[i][j] != 0:                    
                       fexpr += x[i] * A[i][j] * x[i]
                          
            for i in xrange(n):
                if B[0][i] != 0:
                   fexpr += B[0][i] * x[i]
                     
         
        ###########################################################################
        #
        # [A], 'min' <=> minimize Ax 
        #
        ###########################################################################   
        if (len(f) == 1):
             
            A = f[0]             
            assert A.shape == (1,n)             
                       
            fexpr = LinExpr()            
            for i in xrange(n):
                 if A[0][i] != 0:
                     fexpr += A[0][i] * x[i]
                     
                     
        ########################################################
        #             
        # Populate objective
        #
        ########################################################                    
                     
        if(sense == 'min'):
            self.model.modelSense = GRB.MINIMIZE
        elif(sense == 'max'):
            self.model.modelSense = GRB.MAXIMIZE
        else:
            raise ValueError('Objective has no specified sense')  
        
        v = self.getV()
        # (n) -> (n,1)
        if(len(v.shape) == 1):# should not happen, (n) numpy arrays are always read in as (n,1)
           v.reshape((v.shape[0], 1))             
        assert v.shape == (n,1) 
           
        rho = self.rho
        
        obj = QuadExpr()
        for i in xrange(n):
            tmp = x[i] - v[i][0] 
            obj += tmp * tmp
        
        # Minimize(f + (rho/2)*sum_squares(x - v))
        obj =  1/2 * (obj) # fexpr + rho * 1/2 * (obj)        
        
        self.model.setObjective(obj)
        self.model.update()
        
        
    def setV(self, v):  
        self.v = v
        return self.v
        
    def getV(self):  
        return self.v
        
    def setParameters(self, rho, K):       
        # commented out for tweaking
        # assert self.rho
        # assert self.K
        
        self.rho = rho
        self.K = K
            
            
    def setModel(self):
        # assert self.objective
        self.model = Model() 
        self.model.params.OutputFlag = 1 # verbose = 1
        
        #self.model = Problem(self.objective, self.constraints)
        
    # Optimizes the concrete problem
    def solve(self):
        # assert self.model
        # Solve
        self.model.optimize()         
        #print self.model.status == GRB.status.OPTIMAL 
        n = len(self.getX())  
        x = np.zeros((n, 1)) 
        if self.model.status == GRB.status.OPTIMAL:
           for i in xrange(n):
               x[i][0] = self.getX()[i].x
                          
        return x 
    
    
         
            
 