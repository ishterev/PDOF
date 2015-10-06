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
    
# A single ADMM optimization problem in its general form:
# Minimize f(x) + g(x)
# s.t. Ax + Bz = c
class OptimizationProblem_Cvxpy(OptimizationProblem):
    # s.t. Ax + Bz = c
    A = None
    B = None
    c = None
    
    constraints_x = []
    constraints_z = []
    rho = Parameter(sign="positive", value = 0.5)
    
    def setX(self, shape): # (96)
        self.x = Variable(shape)# -> (96,1)
        # auxiliary parameters for the k+1 th z step
        self.xk = Parameter(shape[0], value = np.zeros(shape)) 
        self.uk = Parameter(value = 0) 
        return self.x
        
    def getX(self):
        return self.x
        
    def setZ(self, shape): # (96)
        self.z = Variable(shape)# -> (96,1)
        # auxiliary parameter for the k+1 th x step
        self.zk = Parameter(shape[0], value = np.zeros(shape))  
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
                
        # consistency checks x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp
        m,n,p = -1
        if(A):
           # normalization (n) -> (1,n) 
           if(len(A.shape) == 1):
              A.reshape((1, A.shape[0]))
           p,n = A.shape
           assert(self.getX().size[0] == n)
           
        if(B):
           # normalization (m) -> (1,m) 
           if(len(B.shape) == 1):
              B.reshape((1, B.shape[0]))
           m = B.shape[1]
           assert(self.getZ().size[0] == m)
           
           if(p >= 0):
              assert(p == B.shape[0])
           else:
              p = B.shape[0]
           
        if(c):
           # (p) -> (p,1)
           if(len(c.shape) == 1):
              c.reshape((c.shape[0], 1)) 
           if(p >= 0):
              assert(p == c.shape[0])
           else:
               p = c.shape[0]
               
           assert(c.shape[1] == 1)
           
        self.A = A
        self.B = B
        self.c = c
        
    def setObjective(self, f, sense = 'min'):
        self.setObjectiveX(f, sense)
          
    def setObjectiveX(self, f, sense = 'min'):
              
        # min f(x) + 1/2 * rho * ||A * x + B * zk - c + uk||2
        # expr = A * x + B * zk - c + uk 
        if(self.c):
           expr = -self.c
        else:
           expr = 0          
        # eventually cosnistency checks x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp         
        if(self.A):
           expr += self.A * self.x 
        if(self.B):
           expr += self.B * self.zk
        if(self.uk):
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
        
        if(rho):
           self.rho.value = rho
        self.zk.value = zk
        self.uk.value = uk
        
        
    def setObjectiveZ(self, g, sense = 'min'):
              
        # argmin g(z) + 1/2 * rho * ||A * xk + B * z - c + uk||2
        # expr = A * xk + B * z - c + uk 
        if(self.c):
           expr = -self.c
        else:
           expr = 0          
        # eventually cosnistency checks x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp         
        if(self.A):
           expr += self.A * self.xk 
        if(self.B):
           expr += self.B * self.z
        if(self.uk):
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
        
        if(rho):
           self.rho.value = rho        
        if(uk):
           self.uk.value = uk
           
        self.xk.value = xk   
            
            
    def setModel(self):
        # assert self.objective
        self.model_x = Problem(self.objective_x, self.constraints_x)
        if(self.objective_z):
           self.model_z = Problem(self.objective_z, self.constraints_z)
        
    
    def optimize(self):
        return self.optimizeX()
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solve(self):
        return self.solveX()
        
    def optimizeX(self):
        # assert self.model
        self.model_x.solve(solver=GUROBI) #solver=CVXOPT
        return np.array(self.x.value) 
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solveX(self):
        return self.optimizeX()
        
    def optimizeZ(self):
        # assert self.model
        self.model_z.solve(solver=GUROBI) #solver=CVXOPT
        return np.array(self.z.value) 
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solveZ(self):
        return self.optimizeZ()
        
        
# A single ADMM optimization problem in its general form:
# Minimize f(x) + g(x)
# s.t. Ax + Bz = c        
class OptimizationProblem_Gurobi(OptimizationProblem):
    
    # s.t. Ax + Bz = c
    A = None
    B = None
    c = None
    
    constraints_x = []
    constraints_z = []
    rho = 0.5
    
    def setX(self, shape): 
    
        #assert self.model
        #assert len(shape) == 1 or len(shape) == 2
         
        # Add variables to model
        for i in xrange(shape[0]):# shape is either of the form (n,1) or (n)
            self.model_x.addVar(lb = -GRB.INFINITY) # N.B.!!! otherwise lb is set on default to 0
        self.model_x.update()
        self.x = self.model_x.getVars()     
        
        # auxiliary parameters for the k+1 th z step
        self.xk = np.zeros(shape)
        self.uk = 0

        return self.x
        
    def getX(self):
        return self.x
        
    def setZ(self, shape): 
    
        #assert self.model
        #assert len(shape) == 1 or len(shape) == 2
         
        # Add variables to model
        for i in xrange(shape[0]):# shape is either of the form (n,1) or (n)
            self.model_z.addVar(lb = -GRB.INFINITY) # N.B.!!! otherwise lb is set on default to 0
        self.model_z.update()
        self.z = self.model_z.getVars()     
        
        # auxiliary parameter for the k+1 th x step
        self.zk = np.zeros(shape)

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
             # (1) -> (1,1)
             if(len(b.shape) == 1):# should not happen, (n) numpy arrays are always read in as (n,1)
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
             # (n) -> (n,1)
             if(len(b.shape) == 1):# should not happen, (n) numpy arrays are always read in as (n,1)
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
                
        # consistency checks x ∈ Rn and z ∈ Rm, where A ∈ Rp×n, B ∈ Rp×m, and c ∈ Rp
        m,n,p = -1
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
           # (p) -> (p,1)
           if(len(c.shape) == 1):
              c.reshape((c.shape[0], 1)) 
           if(p >= 0):
              assert(p == c.shape[0])
           else:
               p = c.shape[0]
               
           assert(c.shape[1] == 1)
           
        self.A = A
        self.B = B
        self.c = c
        
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
    def setObjectiveX(self, f, sense = 'min'):
        
        #assert self.model        
        assert len(f) >= 1
        
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
        m,n,p = -1        
        if(A is not None):
            n = A.shape[1]
            p = A.shape[0] 
        
        if(B is not None):
           m = B.shape[1]
           if(p < 0):
              p = B.shape[0] 
            
           assert(zk.shape == (m,1))      
           
        assert(uk.shape == (p,1))
        
        # min f(x) + 1/2 * rho * ||A * x + B * zk - c + uk||2
        obj = QuadExpr()        
        
        for i in xrange(p):
            
            # A * x + B * zk - c + uk 
            p_expr = LinExpr() 
            if(self.A is not None):                         
               for j in xrange(n):
                   if A[i][j] != 0:
                      p_expr += A[i][j] * x[j]
                      
            if(self.B is not None):        
               for j in xrange(m):
                   if B[i][j] != 0:
                      p_expr += B[i][j] * zk[j]
                      
            if(self.c is not None):
               p_expr -= self.c[i]                
            
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
        
        if(rho):
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
    def setObjectiveZ(self, g, sense = 'min'):
        
        #assert self.model        
        assert len(g) >= 1
        
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
            if(A is not None): 
               for i in xrange(m):
                   if A[0][i] != 0:
                      gexpr += A[0][i] * z[i]
                     
                     
        ########################################################
        #             
        # Populate objective
        #
        ########################################################                    
                     
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
        m,n,p = -1        
        if(A is not None):
            n = A.shape[1]
            p = A.shape[0] 
            
            assert(xk.shape == (n,1)) 
        
        if(B is not None):
           m = B.shape[1]
           if(p < 0):
              p = B.shape[0] 
            
        assert(uk.shape == (p,1))
        
        # min g(x) + 1/2 * rho * ||A * xk + B * z - c + uk||2
        obj = QuadExpr()        
        
        for i in xrange(p):
            
            # A * x + B * zk - c + uk 
            p_expr = LinExpr() 
            if(self.A is not None):                         
               for j in xrange(n):
                   if A[i][j] != 0:
                      p_expr += A[i][j] * xk[j]
                      
            if(self.B is not None):        
               for j in xrange(m):
                   if B[i][j] != 0:
                      p_expr += B[i][j] * z[j]
                      
            if(self.c is not None):
               p_expr -= self.c[i]                
            
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
        
        if(rho):
           self.rho = rho        
        if(uk):
           self.uk = uk
           
        self.xk = xk   
        
        
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
        n = len(self.getX())  
        x = np.zeros((n, 1)) 
        if self.model.status == GRB.status.OPTIMAL:
           for i in xrange(n):
               x[i][0] = self.getX()[i].x
                          
        return x 
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solveX(self):
        return self.optimizeX()
        
    def optimizeZ(self):
        # assert self.model
        # Solve
        self.model_z.optimize()         
        m = len(self.getZ())  
        z = np.zeros((m, 1)) 
        if self.model_z.status == GRB.status.OPTIMAL:
           for i in xrange(n):
               z[i][0] = self.getZ()[i].x
                          
        return z 
        
    # on default same as optimize, could contain pre and postprocessing in overriding subclasses    
    def solveZ(self):
        return self.optimizeZ()
 
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
    
    
         
            
 