# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:03:36 2015

@author: shterev
"""

import numpy as np
from cvxpy import *
from gurobipy import *


# A common interface for a single ADMM optimization problem
class OptimizationProblem:
           
    def solve(self):
        pass
      
    def setParameters(self, rho, *args):
        pass
    
    def getProblemDimensions(self):
        pass
    
# A single ADMM optimization problem in its general form:
# Minimize f(x) + g(x)
# s.t. Ax + Bz = c
class OptimizationProblem_Cvxpy(OptimizationProblem):
    # s.t. Ax + Bz = c
    A = None
    B = None
    c = None
    
    x = None
    z = None
    xk = None
    zk = None
    zk_old = None
    uk = None
    
    constraints_x = []
    constraints_z = []
    
    rho = Parameter(sign="positive", value = 0.5)
    
    objective_x = None
    objective_z = None
    
    m = 0
    n = 0
    p = 0
    
    def setX(self, shape): # (96)
        self.x = Variable(shape)# -> (96,1)
        # auxiliary parameters for the k+1 th z step
        self.xk = Parameter(shape, value = np.zeros((shape, 1))) 
        #self.uk = Parameter(value = 0) 
        return self.x
        
    def getX(self):
        return self.x
        
    def setZ(self, shape): # (96)
        self.z = Variable(shape)# -> (96,1)
        # auxiliary parameter for the k+1 th x step
        self.zk = Parameter(shape, value = np.zeros((shape, 1))) 
        self.zk_old = Parameter(shape, value = np.zeros((shape, 1))) 
        return self.z
        
    def getZ(self):
        return self.z

    def addConstraint(self, constraint):
           self.addConstraintX(constraint)
           
    def addConstraintX(self, constraint):
           self.constraints_x.append(constraint)
           
    def addConstraintZ(self, constraint):
           self.constraints_z.append(constraint)
    
    ####################         
    #         
    #   Ax + Bz = c   
    #
    #  This actual constraint is NOT simply "used" during optimization like any other one,    
    #  moreover it is itself embedded in the ADMM algorithm in the Langrangian terms
    #
    #########################
    def addMainConstraint(self, A = None, B = None, c = None):
        
        assert self.x
        assert self.z
                
        # consistency checks x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp
        m = -1
        n = -1
        p = -1
        if(A is not None):
           # normalization (n) -> (1,n) 
           if(len(A.shape) == 1):
              A.reshape((1, A.shape[0]))
           p,n = A.shape
           assert(self.getX().size[0] == n)
           
        if(B is not None):
           # normalization (m) -> (1,m) 
           if(len(B.shape) == 1):
              B.reshape((1, B.shape[0]))
           m = B.shape[1]
           assert(self.getZ().size[0] == m)
           
           if(p >= 0):
              assert(p == B.shape[0])
           else:
              p = B.shape[0]
           
        if(c is not None):
           if(not isinstance(c, np.ndarray)):
               c = c * np.ones((p,1))
           # (p) -> (p,1)
           if(len(c.shape) == 1):
              c.reshape((c.shape[0], 1)) 
           if(p >= 0):
              assert(p == c.shape[0])
           else:
               p = c.shape[0]
               
           assert(c.shape[1] == 1)
        
        # problem matrices
        self.A = A
        self.B = B
        self.c = c 
        
        # problem dimension
        self.m = m
        self.n = n
        self.p = p
    
        # uk dimensions are now clear, declare it
        self.uk = Parameter(p, value = np.zeros((p, 1))) 
        
    def setObjective(self, f, sense = 'min'):
        self.setObjectiveX(f, sense)
          
    def setObjectiveX(self, f, sense = 'min'):
              
        # min f(x) + 1/2 * rho * ||A * x + B * zk - c + uk||2
        # expr = A * x + B * zk - c + uk 
        if(self.c is not None):
           expr = -self.c
        else:
           expr = 0          
        # eventually cosnistency checks x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp         
        if(self.A is not None):
           expr += self.A * self.x 
        if(self.B is not None):
           expr += self.B * self.zk
        if(self.uk is not None):
           expr += self.uk
        
        # add proximal term to objective
        prox = sum_squares(expr)
        prox *= self.rho 
        prox *= 1/2 # N.B.!!! *= rho/2 is syntactically correct but not semantically; *= 0.5 leads to the same incorrect result
        f += prox
        
        if(sense == 'min'):
           self.objective_x = Minimize(f)
        elif(sense == 'max'):
            self.objective_x = Maximize(f)
        else:
            raise ValueError('Objective has no specified sense')
            
        
    def setParameters(self, rho, z, u):  
        self.setParametersObjX(rho, z, u)  

        
    def setParametersObjX(self, rho, zk, uk):       
        # commented out for tweaking
        # assert self.rho
        # assert self.K
        
        if(rho is not None):
           self.rho = rho #self.rho.value
        if(self.zk is not None):
           self.zk = zk #self.zk.value
        if(self.uk is not None):
           self.uk = uk #self.uk.value
        
        
    def setObjectiveZ(self, g, sense = 'min'):
              
        # argmin g(z) + 1/2 * rho * ||A * xk + B * z - c + uk||2
        # expr = A * xk + B * z - c + uk 
        if(self.c is not None):
           expr = -self.c
        else:
           expr = 0          
        # eventually cosnistency checks x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp         
        if(self.A is not None):
           expr += self.A * self.xk 
        if(self.B is not None):
           expr += self.B * self.z
        if(self.uk is not None):
           expr += self.uk
        
        # add proximal term to objective
        prox = sum_squares(expr)
        prox *= self.rho 
        prox *= 1/2 # N.B.!!! *= rho/2 is syntactically correct but not semantically; *= 0.5 leads to the same incorrect result
        g += prox
        
        if(sense == 'min'):
           self.objective_z = Minimize(g)
        elif(sense == 'max'):
            self.objective_z = Maximize(g)
        else:
            raise ValueError('Objective has no specified sense')
            
           
    def setParametersObjZ(self, rho, xk, uk = None):       
        # commented out for tweaking
        # assert self.rho
        # assert self.K
        
        if(rho is not None):
           self.rho = rho #self.rho.value     
        if(uk is not None):
           self.uk = uk #self.uk.value
           
        self.xk = xk   
            
            
    def setModel(self):
        if(self.objective_x is not None):
           self.model_x = Problem(self.objective_x, self.constraints_x)
        if(self.objective_z is not None):
           self.model_z = Problem(self.objective_z, self.constraints_z)
        
    
    def optimize(self):
        return self.optimizeX()
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solve(self):
        return self.solveX()
        
    def optimizeX(self):
        # assert self.model
        self.model_x.solve(solver=GUROBI) #solver=CVXOPT
        self.xk = np.array(self.x.value)
        return (self.xk, 0) 
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solveX(self):
        return self.optimizeX()
        
    def optimizeZ(self):
        # assert self.model
        self.model_z.solve(solver=GUROBI) #solver=CVXOPT
        self.zk_old = self.zk
        self.zk = np.array(self.z.value)
        return self.zk
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solveZ(self):
        return self.optimizeZ()
        
    def solveU(self, xk = None, zk = None, uk = None):
        
        if(xk is not None):
            self.xk = xk
        if(zk is not None):
            self.zk = zk
        if(uk is not None):
            self.uk = uk
            
        if(self.A is not None):
            self.uk += np.dot(self.A, self.xk)
        
        if(self.B is not None):
            self.uk += np.dot(self.B, self.zk)
            
        if(self.c is not None):
            self.uk -= self.c
                 
        return self.uk
        
    def getPrimalResidual(self, xk = None, zk = None):
        
        if(xk is not None):
            self.xk = xk
        if(zk is not None):
            self.zk = zk
        
        rk = 0
        if(self.A is not None):
            rk += np.dot(self.A, self.xk)
        
        if(self.B is not None):
            rk += np.dot(self.B, self.zk)
            
        if(self.c is not None):
            rk -= self.c
            
        return rk
        
    
    def getDualResidual(self, zk = None, zk_old = None):
        
        if(self.A is None or self.B is None):
            return 0
        
        if(zk is not None):
           #self.zk_old = self.zk
           self.zk = zk
        if(zk_old is not None):
           self.zk_old = zk_old
           
        sk = self.zk - self.zk_old
        tmp = np.dot(self.A.T, self.B)
        sk = np.dot(tmp, sk)
        #sk *= self.rho # multiplied afterward in the ADMM algorithm
        
        return sk
        
        
    def getPrimalFeasability(self):
        a = 0
        if(self.A is not None):
            a = np.dot(self.A, self.xk)
        b = 0   
        if(self.B is not None):
            b = np.dot(self.B, self.zk)
        c = 0   
        if(self.c is not None):
           c = self.c
           
        return (a,b,c)
        
    
    def getDualFeasability(self):
        a = 0
        if(self.A is not None):
            a = np.dot(self.A.T, self.uk)
               
        return a
        
    def getProblemDimensions(self):
        return (self.n,self.m,self.p) # A (pxn) , B (pxm), c (p,1)
           
    
# A single ADMM optimization problem in its general form:
# Minimize f(x) + g(x)
# s.t. Ax + Bz = c        
class OptimizationProblem_Gurobi(OptimizationProblem):
    
    # s.t. Ax + Bz = c
    A = None
    B = None
    c = None
    
    x = None
    z = None
    xk = None
    zk = None
    zk_old = None
    uk = None
    
    rho = 0.5
    
    constraints_x = []
    constraints_z = []
    
    f = None
    g = None   
    sense_f = None
    sense_g = None
    objective_x = None
    objective_z = None
    
    m = 0
    n = 0
    p = 0
    
    
    def setX(self, shape): 
    
        #assert self.model
        #assert len(shape) == 1 or len(shape) == 2
         
        # Add variables to model
        for i in xrange(shape):# n <=> (n,1)
            self.model_x.addVar(lb = -GRB.INFINITY) # N.B.!!! otherwise lb is set on default to 0
        self.model_x.update()
        self.x = self.model_x.getVars()     
        
        # auxiliary parameters for the k+1 th z step
        self.xk = np.zeros((shape, 1))
        #self.uk = 0

        return self.x
        
    def getX(self):
        return self.x
        
    def setZ(self, shape): 
    
        #assert self.model
        #assert len(shape) == 1 or len(shape) == 2
         
        # Add variables to model
        for i in xrange(shape):# shape is either of the form (n,1) or (n)
            self.model_z.addVar(lb = -GRB.INFINITY) # N.B.!!! otherwise lb is set on default to 0
        self.model_z.update()
        self.z = self.model_z.getVars()     
        
        # auxiliary parameter for the k+1 th x step
        self.zk = np.zeros((shape, 1))
        self.zk_old = np.zeros((shape, 1))

        return self.x
        
    def getZ(self):
        return self.z
        
    def addConstraint(self, constraint):
           self.addConstraintX(constraint)
           
           
    ###################################################################################################
    #
    # A constraint takes the form of a list: left hand side, an operator, and a right hand side.
    # The quadratic version x.T A x + Bx <= b: 
    # 
    # [left quad, left linear, <=, b] ([A, B, '<=' , b] <=> x.T A x + Bx <= b)
    #       
    # and the linear one Ax <= b:
    #
    # [left, operator, right] ([A, '<=' , b] <=> Ax <= b) 
    #
    ###################################################################################################    
    def addConstraintX(self, constraint):
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

             if(not isinstance(b, np.ndarray)):
                b = b * np.ones((1,1))                                
             # (1) -> (1,1)
             if(len(b.shape) == 1):
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
       
             self.model_x.addQConstr(expr,  op, b[0][0])
             self.model_x.update()          
                    
         
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
                
             if(not isinstance(b, np.ndarray)):
                b = b * np.ones((1,1)) 
             # (n) -> (n,1)
             if(len(b.shape) == 1):
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
                        
                 self.model_x.addConstr(expr,  op, b[i][0])
                     
             self.model_x.update()   
                
         else:
             raise ValueError("Unspecified constraint argument")
             
    
    ###################################################################################################
    #
    # A constraint takes the form of a list: left hand side, an operator, and a right hand side.
    # The quadratic version z.T A z + Bz <= b: 
    # 
    # [left quad, left linear, <=, b] ([A, B, '<=' , b] <=> z.T A z + Bz <= b)
    #       
    # and the linear one Az <= b:
    #
    # [left, operator, right] ([A, '<=' , b] <=> Az <= b) 
    #
    ###################################################################################################          
    def addConstraintZ(self, constraint):
         #assert self.model        
         assert len(constraint) >= 3

         ###########################################################################
         #
         # [left quad, left linear, <=, b] ([A, B, '<=' , b] <=> z.T A z + Bz <= b)
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
             # (opt. 1/2*) z.T A z + Bz <= b, because of the quadratic term it can be only one line 
             #
             # i.e. (1,n) x (n x n) x (n,1) + (1, n) x (n,1) = (1,1)
             #
             ############################################################################################
             
             # Input check and normalization
             
             # normalization (n) -> (1,n), 
             if(len(B.shape) == 1):
                B.reshape((1, B.shape[0]))
                
             if(not isinstance(b, np.ndarray)):
                b = b * np.ones((1,1))   
             # (1) -> (1,1)
             if(len(b.shape) == 1):
                b.reshape((b.shape[0], 1))  
                
             n = self.getZ().size[0]             
             assert A.shape == (n,n) 
             assert B.shape == (1,n)
             assert b.shape == (1,1)
             
             z = self.getZ()            
             expr = QuadExpr()
             for i in xrange(n):
                
                 for j in xrange(n):                     
                     expr += z[j] * A[i][j] * z[i]
                          
             for i in xrange(n):
                 if B[0][i] != 0:
                     expr += B[0][i] * z[i]
       
             self.model_z.addQConstr(expr,  op, b[0][0])
             self.model_z.update()          
                    
         
         #########################################################
         #   
         # [left, operator, right] ([A, '<=' , b] <=> Az <= b)  
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
             # we have m times B_i_j * z_i <= b_i 
             #
             # i.e. (m, n) x (n,1) = (m,1)
             #
             ######################################################################
             
             # Input check and normalization
             
             # normalization (n) -> (1,n) 
             if(len(A.shape) == 1):
                A.reshape((1, A.shape[0]))
                
             if(not isinstance(b, np.ndarray)):
                b = b * np.ones((1,1))   
             # (n) -> (n,1)
             if(len(b.shape) == 1):
                b.reshape((b.shape[0], 1))  
                
             n = len(self.getZ())            
             # (m,n) 
             assert len(A.shape) == 2 and A.shape[1] == n
             m = A.shape[0]
             # (m,n) x (n,1) = (m,1)
             assert b.shape == (m,1)
                
             z = self.getZ()  
             for i in xrange(m):
                 
                 expr = LinExpr()                
                 for j in xrange(n):
                     if A[i][j] != 0:
                        expr += A[i][j] * z[j]
                        
                 self.model_z.addConstr(expr,  op, b[i][0])
                     
             self.model_z.update()   
                
         else:
             raise ValueError("Unspecified constraint argument")
             
             
    #########################################################################################         
    #         
    #  Ax + Bz = c   
    #
    #  The actual constraint is NOT simply "used" during optimization like any other one,    
    #  moreover it is itself embedded in the ADMM algorithm in the Langrangian terms
    #
    #########################################################################################
    def addMainConstraint(self, A = None, B = None, c = None):
        
        assert self.x
        assert self.z
                
        # consistency checks x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp
        m = -1
        n = -1
        p = -1
        if(A is not None):
           # normalization (n) -> (1,n) 
           if(len(A.shape) == 1):
              A.reshape((1, A.shape[0]))
           p,n = A.shape
           assert(len(self.getX()) == n)
           
        if(B is not None):
           # normalization (m) -> (1,m) 
           if(len(B.shape) == 1):
              B.reshape((1, B.shape[0]))
           m = B.shape[1]
           assert(len(self.getZ()) == m)
           
           if(p >= 0):
              assert(p == B.shape[0])
           else:
              p = B.shape[0]
           
        if(c is not None):
           if(not isinstance(c, np.ndarray)):
              c = c * np.ones((p,1))
           # (p) -> (p,1)
           if(len(c.shape) == 1):
              c.reshape((c.shape[0], 1)) 
           if(p >= 0):
              assert(p == c.shape[0])
           else:
               p = c.shape[0]
               
           assert(c.shape[1] == 1)
           
        # problem matrices
        self.A = A
        self.B = B
        self.c = c 
        
        # problem dimension
        self.m = m
        self.n = n
        self.p = p
    
        # uk dimensions are now known, declare it
        self.uk = np.zeros((p, 1))
        
    def setObjective(self, f, sense = 'min'):
        self.setObjectiveX(f, sense)
        
     
    ######################################################################################
    #
    # The target function f takes the form of a list. 
    # The quadratic optimization Minimize x.T A x + Bx is as follows:
    #
    # [quad term, linear term] ([A, B], 'min' <=> minimize x.T A x + Bx)
    #
    # and the linear one Minimize Ax :
    #
    # [linear term] ([A], 'min' <=> minimize Ax)  
    #
    ######################################################################################
    def setObjectiveX(self, f, sense = None):
        
        #assert self.model        
        assert len(f) >= 1
        
        self.f = f
        
        x = self.getX() 
        n = len(x)
        
        fexpr = None

        ###########################################################################
        #
        # Optimize a quadratic function 
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
            # The default value for the target function. A constant is good e.g. to represent 
            # the indicator function (with the appropriate constraint sum(xi) == 0) etc.
            fexpr += 0
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
            # The default value for the target function. A constant is good e.g. to represent 
            # the indicator function (with the appropriate constraint sum(xi) == 0) etc.
            fexpr += 0
            if(A is not None): 
               for i in xrange(n):
                   if A[0][i] != 0:
                      fexpr += A[0][i] * x[i]
                     
                     
        ########################################################
        #             
        # Populate objective
        #
        ########################################################                    
        if(sense is not None):
           self.sense_f = sense
        else:
           sense = self.sense_f            
        if(sense == 'min'):
            self.model_x.modelSense = GRB.MINIMIZE
        elif(sense == 'max'):
            self.model_x.modelSense = GRB.MAXIMIZE
        else:
            raise ValueError('Objective has no specified sense')  
            
                  
        x = self.getX()  
        zk = self.zk
        uk = self.uk
        
        A = self.A
        B = self.B
        c = self.c
        
        # A is of the form (p,n) , B - (p,m) and c - (p,1)
        # assert uk is (p,1) and zk is (m,1)     
        m = self.m
        n = self.n
        p = self.p       
                
        if(B is not None):
           assert(zk.shape == (m,1))      
        
        assert(uk is not None)
        assert(uk.shape == (p,1))
        
        # min f(x) + 1/2 * rho * ||A * x + B * zk - c + uk||2
        obj = QuadExpr()        
        
        for i in xrange(p):
            
            # A * x + B * zk - c + uk 
            p_expr = LinExpr() 
            if(A is not None):                         
               for j in xrange(n):
                   if A[i][j] != 0:
                      p_expr += A[i][j] * x[j]
                      
            if(B is not None):        
               for j in xrange(m):
                   if B[i][j] != 0:
                      p_expr += B[i][j] * zk[j]
                      
            if(c is not None):
               p_expr -= c[i]                
            
            p_expr += uk[i]
            
            # 2nd norm 
            obj += (p_expr) * (p_expr)            
            
        obj *= self.rho
        # rho * 1/2 * obj (two times *) is syntactically correct but not semantically 
        # and delivers wrong results
        obj *= 1/2 
        obj +=  fexpr     # 1/2 * (obj)    
        
        self.model_x.setObjective(obj)
        self.model_x.update()
        
        
    def setParameters(self, rho, z, u):  
        self.setParametersObjX(rho, z, u)  

        
    def setParametersObjX(self, rho, zk, uk):       
        
        if(rho is not None):
           self.rho = rho
           
        # (n) -> (n,1)
        if(len(zk.shape) == 1):
           zk.reshape((zk.shape[0], 1))  
        
        # if(self.B is not None):
        assert(self.B.shape[0] == zk.shape[0])
           
        if(len(uk.shape) == 1):
           uk.reshape((uk.shape[0], 1))   
           
        if(self.c is not None):
           assert(self.B.shape[0] == zk.shape[0])
           
        self.zk = zk
        self.uk = uk
        
        self.setObjectiveX(self.f, self.sense_f)
        
    
    
    ######################################################################################
    #
    # The target function g takes the form of a list. 
    # The quadratic optimization Minimize z.T A z + Bz is as follows:
    #
    # [quad term, linear term] ([A, B], 'min' <=> minimize z.T A z + Bz)
    #
    # and the linear one Minimize Az :
    #
    # [linear term] ([A], 'min' <=> minimize Az)  
    #
    ######################################################################################
    def setObjectiveZ(self, g, sense = None):
        
        #assert self.model        
        #assert len(g) >= 1
        
        self.g = g
        
        z = self.getZ() 
        m = len(z)
        
        gexpr = None

        ###########################################################################
        #
        # Optimize a quadratic function 
        # [A, B], 'min' <=> minimize (opt. 1/2*) z.T A z + Bz
        #
        ###########################################################################
        if (len(g) == 2):
             
            A = g[0]
            B = g[1]
             
            if(A is not None): 
               assert A.shape == (m,m) 
            if(B is not None): 
               assert B.shape == (1,m) 
                       
            gexpr = QuadExpr()
            # The default value for the target function. A constant is good e.g. to represent 
            # the indicator function (with the appropriate constraint sum(xi) == 0) etc.
            gexpr += 0
            if(A is not None): 
               for i in xrange(m):
                
                   for j in xrange(m): 
                       if A[i][j] != 0:                    
                          gexpr += z[i] * A[i][j] * z[i]
                          
            if(B is not None):              
               for i in xrange(m):
                   if B[0][i] != 0:
                      gexpr += B[0][i] * z[i]
                     
         
        ###########################################################################
        #
        # [A], 'min' <=> minimize Ax 
        #
        ###########################################################################   
        if (len(g) == 1):
             
            A = g[0]
            if(A is not None):              
               assert A.shape == (1,m)             
                       
            gexpr = LinExpr() 
            # The default value for the target function. A constant is good e.g. to represent 
            # the indicator function (with the appropriate constraint sum(xi) == 0) etc.
            gexpr += 0
            if(A is not None): 
               for i in xrange(m):
                   if A[0][i] != 0:
                      gexpr += A[0][i] * z[i]
                     
                     
        ########################################################
        #             
        # Populate objective
        #
        ########################################################                             
        if(sense is not None):
           self.sense_g = sense
        else:
           sense = self.sense_g
            
        if(sense == 'min'):
            self.model_z.modelSense = GRB.MINIMIZE
        elif(sense == 'max'):
            self.model_z.modelSense = GRB.MAXIMIZE
        else:
            raise ValueError('Objective has no specified sense')  
            
                  
        xk = self.xk  
        z = self.z
        uk = self.uk
        
        A = self.A
        B = self.B
        c = self.c
        
        # A is of the form (p,n) , B - (p,m) and c - (p,1)
        # assert xk is (n,1) and uk is (p,1)   
        m = self.m
        n = self.n
        p = self.p   
        
        if(A is not None):
            assert(xk.shape == (n,1)) 
            
        assert(uk is not None) 
        assert(uk.shape == (p,1))
        
        # min g(x) + 1/2 * rho * ||A * xk + B * z - c + uk||2
        obj = QuadExpr()        
        
        for i in xrange(p):
            
            # A * x + B * zk - c + uk 
            p_expr = LinExpr() 
            if(A is not None):                         
               for j in xrange(n):
                   if A[i][j] != 0:
                      p_expr += A[i][j] * xk[j]
                      
            if(B is not None):        
               for j in xrange(m):
                   if B[i][j] != 0:
                      p_expr += B[i][j] * z[j]
                      
            if(c is not None):
               p_expr -= c[i]                
            
            p_expr += uk[i]
            
            # 2nd norm 
            obj += (p_expr) * (p_expr)            
            
        obj *= self.rho
        # rho * 1/2 * obj (two times *) is syntactically correct but not semantically 
        # and delivers wrong results
        obj *= 1/2 
        obj +=  gexpr     # 1/2 * (obj)    
        
        self.model_z.setObjective(obj)
        self.model_z.update()
        
   
    def setParametersObjZ(self, rho, xk, uk = None):       
        # commented out for tweaking
        # assert self.rho
        # assert self.K
        
        if(rho is None):
           self.rho = rho        
        if(uk is not None):
           self.uk = uk
           
        self.xk = xk   
        
        self.setObjectiveZ(self.g, self.sense_g)
        
        
    def setModel(self):
        
        self.model_x = Model() 
        self.model_x.params.OutputFlag = 0 # verbose = 1    
        
        self.model_z = Model() 
        self.model_z.params.OutputFlag = 0 # verbose = 1    
        
    
    def optimize(self):
        return self.optimizeX()
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solve(self):
        return self.solveX()
        
    def optimizeX(self):
        # assert self.model
        # Solve
        self.model_x.optimize()         
        x = np.zeros((self.n, 1)) 
        if self.model_x.status == GRB.status.OPTIMAL:
           for i in xrange(self.n):
               x[i][0] = self.getX()[i].x
                          
        return (x,0) 
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solveX(self):
        return self.optimizeX()
        
    def optimizeZ(self):
        # assert self.model
        # Solve
        self.model_z.optimize()         
        z = np.zeros((self.m, 1)) 
        if self.model_z.status == GRB.status.OPTIMAL:
           for i in xrange(self.m):
               z[i][0] = self.getZ()[i].x
                          
        return z 
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solveZ(self):
        return self.optimizeZ()
        
    def solveU(self, xk, zk, uk):
        
        if(xk is not None):
            self.xk = xk
        if(zk is not None):
            self.zk = zk
        if(uk is not None):
            self.uk = uk
            
        if(self.A is not None):
            self.uk += np.dot(self.A, self.xk)
        
        if(self.B is not None):
            self.uk += np.dot(self.B, self.zk)
            
        if(self.c is not None):
            self.uk -= self.c
                 
        return self.uk
        
    
    def getPrimalResidual(self, xk = None, zk = None):
        
        if(xk is None):
            self.xk = xk
        if(zk is None):
            self.zk = zk
        
        rk = 0
        if(self.A is not None):
            rk += np.dot(self.A, self.xk)
        
        if(self.B is not None):
            rk += np.dot(self.B, self.zk)
            
        if(self.c is not None):
            rk -= self.c
            
        return rk
        
    
    def getDualResidual(self, zk = None, zk_old = None):
        
        if(self.A is None or self.B is None):
            return 0
        
        if(zk is not None):
           #self.zk_old = self.zk
           self.zk = zk
        if(zk_old is not None):
           self.zk_old = zk_old
           
        # rho * A.T * B * (zi-zi_old) 
        tmp = np.dot(self.A.T, self.B) 
        sk = self.zk - self.zk_old        
        sk = np.dot(tmp, sk)
        #sk *= self.rho # multiplied afterward in the ADMM algorithm
        
        return sk
        
        
    def getPrimalFeasability(self):
        a = 0
        if(self.A is not None):
            a = np.dot(self.A, self.xk)
        b = 0   
        if(self.B is not None):
            b = np.dot(self.B, self.zk)
        c = 0   
        if(self.c is not None):
           c = self.c
           
        return (a,b,c)
        
    
    def getDualFeasability(self):
        a=0
        if(self.A is not None):
            a = np.dot(self.A.T, self.uk)
               
        return a
        
        
    def getProblemDimensions(self):
        return (self.n,self.m,self.p) # A (pxn) , B (pxm), c (p,1)
 
######################################################################################################
#
# An ADMM exchange problem:
# 
# Minimize Sum (f(xi))
# s.t. Sum (xi) = 0
#
# i.e. Ax + Bz = c gets x - z = 0  (A = (n x n) identity, B = (n x n) -identity and c = (n x 1) 0) 
# and g(z) of the general ADMM form (min f(x) + g(z)) is the indicator function of the set {0} => Sum (xi) = 0
# this leads to a majority of simplifications of the algorithm and computations
#
######################################################################################################      
class OptimizationProblem_Exchange_Cvxpy(OptimizationProblem):
    
    constraints = []
    n = 0
    
    # objective function has only one variable that is x
    def setX(self, n):
        self.x = Variable(n)
        self.n = n
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
        
    def getProblemDimensions(self):
        return (n, n, n)
        
        
######################################################################################################
#
# An ADMM exchange problem:
# 
# Minimize Sum (f(xi))
# s.t. Sum (xi) = 0
#
# i.e. Ax + Bz = c gets x - z = 0  (A = (n x n) identity, B = (n x n) -identity and c = (n x 1) 0) 
# and g(z) of the general ADMM form (min f(x) + g(z)) is the indicator function of the set {0} => Sum (xi) = 0
# this leads to a majority of simplifications of the algorithm and computations
#
######################################################################################################            
class OptimizationProblem_Exchange_Gurobi(OptimizationProblem):
    
    n = 0
    
    # objective function has only one variable that is x 
    def setX(self, n): 
         
        # Add variables to model
        for i in xrange(n): 
            self.model.addVar(lb = -GRB.INFINITY) # N.B.!!! otherwise lb is set on default to 0
        self.model.update()
        self.x = self.model.getVars()          
        self.n = n

        return self.x
        
    def getX(self):
        return self.x
        
    ###################################################################################################
    #
    # A constraint takes the form of a list: left hand side, an operator, and a right hand side.
    # The quadratic version x.T A x + Bx <= b: 
    # 
    # [left quad, left linear, <=, b] ([A, B, '<=' , b] <=> x.T A x + Bx <= b)
    #       
    # and the linear one Ax <= b:
    #
    # [left, operator, right] ([A, '<=' , b] <=> Ax <= b) 
    #
    ################################################################################################### 
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
                
             if(not isinstance(b, np.ndarray)):
                b = b * np.ones((1,1))
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
        self.model.params.OutputFlag = 0 # verbose = 1
        
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
    
    
    def getProblemDimensions(self):
        return (n,n,n)
            
 