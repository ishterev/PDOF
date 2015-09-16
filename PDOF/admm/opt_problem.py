# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:03:36 2015

@author: shterev
"""

import numpy as np
from cvxpy import *
from gurobipy import *


# Common interface
class OptimizationProblem:
           
    def solve(self):
        pass
        
    def setParameters(self, rho, K):
        pass
    

class OptimizationProblemCvxpy(OptimizationProblem):
    
    constraints = []
    
    # objective function has only one variable that is x
    def setX(self, shape): # (96)
        self.x = Variable(shape) # -> (96,1)
        return self.x
        
    def getX(self):
        return self.x
        
    def addConstraint(self, constraint):
           self.constraints.append(constraint)
          
    def setObjective(self, f, sense = 'min'):
        
        x = self.getX()
        self.K = Parameter(x.size[0], value = np.zeros(x.size))
        self.rho = Parameter(sign="positive", value = 0.5)
        
        # add proximal term to objective
        prox = sum_squares(x - self.K)
        prox *= self.rho 
        prox *= 1/2 # N.B.!!! *= rho/2 is syntactically correct but not semantically; *= 0.5 leads to the same incorrect result
        f += prox
        
        if(sense == 'min'):
           self.objective = Minimize(f)
        elif(sense == 'max'):
            self.objective = Maximize(f)
        else:
            raise ValueError('Objective has no specified sense')

        
    def setParameters(self, rho, K):       
        # commented out for tweaking
        # assert self.rho
        # assert self.K
        
        self.rho.value = rho
        self.K.value = K
            
            
    def setModel(self):
        # assert self.objective
        self.model = Problem(self.objective, self.constraints)
        
    
    def optimize(self):
        # assert self.model
        self.model.solve(solver=GUROBI) #solver=CVXOPT
        return np.array(self.x.value) 
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solve(self):
        return self.optimize()
        
        
        
class OptimizationProblemGurobi(OptimizationProblem):
    
    # objective function has only one variable that is x
    def setX(self, shape): 
    
        #assert self.model
        #assert len(shape) == 1 or len(shape) == 2
         
        # Add variables to model
        for i in xrange(shape[0]):# shape is either of the form (n,1) or (n)
            self.model.addVar(lb = -GRB.INFINITY) # N.B.!!! otherwise lb is set on default to 0
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
       
             self.model.addQConstr(expr,  op, b[0][0])
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
             
            if(A is not None): 
               assert A.shape == (n,n) 
            if(B is not None): 
               assert B.shape == (1,n) 
                       
            fexpr = QuadExpr()
            if(A is not None): 
               for i in xrange(n):
                
                   for j in xrange(n): 
                       if A[i][j] != 0:                    
                          fexpr += x[i] * A[i][j] * x[i]
                          
            if(B is not None):              
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
            if(A is not None):              
               assert A.shape == (1,n)             
                       
            fexpr = LinExpr() 
            if(A is not None): 
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
        
        K = self.K
        # (n) -> (n,1)
        if(len(K.shape) == 1):# should not happen, (n) numpy arrays are always read in as (n,1)
           K.reshape((K.shape[0], 1))             
        assert K.shape == (n,1) 
        
        obj = QuadExpr()
        for i in xrange(n):
            tmp = x[i] - K[i][0] 
            obj += tmp * tmp
            
        obj *= self.rho
        # rho * 1/2 * obj (two times *) is syntactically correct but not semantically 
        # and delivers wrong results
        obj *= 1/2 
        
        # Minimize(f + (rho/2)*sum_squares(x - v))
        obj +=  fexpr     # 1/2 * (obj)    
        
        self.model.setObjective(obj)
        self.model.update()
        
        
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
    def optimize(self):
        # assert self.model
        # Solve
        self.model.optimize()         
        n = len(self.getX())  
        x = np.zeros((n, 1)) 
        if self.model.status == GRB.status.OPTIMAL:
           for i in xrange(n):
               x[i][0] = self.getX()[i].x
                          
        return x 
    
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solve(self):
        return self.optimize()
    
    
         
            
 