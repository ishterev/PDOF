# -*- coding: utf-8 -*-
"""
Created on Sat May 09 00:06:45 2015

@author: shterev
"""
import numpy as np
from scipy.linalg.blas import ddot
from gurobipy import *

import h5py

from opt_problem import *

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

deltaT = 900 #=15*60  Time slot duration [sec]
T = 96 #= 24*3600/deltaT # Number of time slots
I = np.identity(T)
O = np.zeros((T,1))
        
class OptProblem_Aggregator_PriceBased(OptimizationProblem_Gurobi):
    
      p = None                # regularization
      re = None 
      xamin = 0
        
      def __init__(self): 
          
          data = loadAggr()
   
          for key,val in data.items() :
          
              if(key == 'price'):
                 self.price = data['price'][()]# Base demand profile 
          
          data.close()
          
         
          self.p=np.tile(self.price,(4,1))
          self.p=self.p / (3600*1000) * deltaT  # scaling of price in EUR/kW

          self.re=  100e3 *np.ones((T,1))  # maximal aviliable load 1GW 
          self.xamin=-100e3*np.ones((T,1))
          self.xmax=self.re
          
          self.setModel()
         
          self.setX(T)
          self.setZ(T)
          
          # x - z = 0
          self.addMainConstraint(I, -I, O)
          
          # g(z) - indicator function of the set {0} => Sum (zi) = 0 (<=>  Sum (xi) = 0)
          self.setObjectiveZ([None], 'min')# [None] = 0x^2 + 0x + 0
         
          
          
          
      def setParametersObjX(self, rho, zk, uk):                  
          self.rho = rho         #augement cost parameter
          self.zk = zk
          self.uk = uk
          self.K = zk - uk # xold - xmean - u  Normalization parameter
          
          
      def solveX(self):
          
          x= self.K + self.p/self.rho          
 
          # box constraints
          indx = np.where(x<-self.re)
          if indx:
              x[indx]=-self.re[indx]
             
          indx = np.where(x>-self.xamin)
          if indx:
              x[indx]=-self.xamin[indx]
              
           
          self.x = x
          self.xk = x
 
          cost = -np.dot(self.p.T, x) # -p'*x
          
          return (x,cost)
          
          
          

class OptProblem_PriceBased_Home(OptimizationProblem_Gurobi):
    
      gamma = 0 # Trade-off parameter
      alpha = (0.05 * 15 * 60)/3600 #Battery depresiation cost [EUR/kWh] and transformed to [EUR/kW]
     
      def __init__(self, idx, discharge = False): 
          
          self.idx = idx
          self.discharge = discharge
          
          self.xmax = 4   # Max charging power for greedy need to add some 1e-3 or something for it to be feasible

          if self.discharge: 
              self.xmin = -4 # yes
          else:
              self.xmin = 0  # no
                
          data = loadEV('home', self.idx)              
              
          for key,val in data.items() :
       
              if(key == 'A'):       
                 self.A_in = data[key][()].T 
          
              if(key == 'R'):
                 self.R_in = data[key][()][0][0]
                 
              if(key == 'd'):
                 self.d_in = data[key][()] 
                 
              if(key == 'B'): # and self.discharge
                 self.B_in = data[key][()].T   
                 
              if(key == 'S_max'): # and self.discharge
                 self.Smax_in = data[key][()].T
                 
              if(key == 'S_min'): # and self.discharge
                 self.Smin_in = data[key][()].T
          
          data.close()          
          
          self.setModel()
          
          self.setX(T)
          self.setZ(T)
          
          # x - z = 0
          self.addMainConstraint(I, -I, O)
          
          self.addConstraintX([I, '>=', self.d * self.xmin]) # lb
          self.addConstraintX([I, '<=', self.d * self.xmax]) # ub
          
          self.addConstraintX([self.A, '==', self.R]) 
              
          # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G 
          
             self.addConstraintX([self.B, '>=', self.Smin - 1e-4])
             self.addConstraintX([self.B, '<=', self.Smax + 1e-4])
             
             
          if(self.gamma == 0):
              obj = [None]
          else:# f = gamma * alpha * x^2
              obj = [self.gamma * self.alpha * I, None]
              
          self.setObjectiveX(obj, 'min')
             
             
          # g(z) - indicator function of the set {0} => Sum (zi) = 0 (<=>  Sum (xi) = 0)
          self.setObjectiveZ([None], 'min')# [None] = 0x^2 + 0x + 0
                           
      '''     
      def solveX(self):          
          
          if(self.gamma == 0):
              obj = [None]
          else:# f = gamma * alpha * x^2
              obj = [self.gamma * self.alpha * I, None]
              
          self.setObjectiveX(obj, 'min')
      
          x, cost = self.optimizeX()                 
          cost = self.gamma*self.alpha*ddot(x,x)    
          
          #self.x = x
          #self.xk = x

          return (x , cost)
          '''
        
        

class OptProblem_Aggregator_ValleyFilling(OptimizationProblem_Gurobi):
        
      def __init__(self): 
          
          data = loadAggr()
   
          for key,val in data.items() :
               
              if(key == 'D'):       
                 self.D = data[key][()].T
          
              if(key == 'price'):
                 self.price = data['price'][()].T
          
          data.close()
          
          self.setModel()
          
          self.setX(T)
          self.setZ(T)
          
          # x - z = 0
          self.addMainConstraint(I, -I, O)
          
          # g(z) - indicator function of the set {0} => Sum (zi) = 0 (<=>  Sum (xi) = 0)
          self.setObjectiveZ([None], 'min')# [None] = 0x^2 + 0x + 0

          
          
                
      def setParametersObjX(self, rho, zk, uk):                  
          self.rho = rho         #augement cost parameter
          self.zk = zk
          self.uk = uk
          self.K = zk - uk # xold - xmean - u  Normalization parameter
          
          
      def solveX(self):
          
          if(self.rho == 2):
              self.rho += 1e-9
          
          x = self.rho/(self.rho-2)* self.K - 2/(self.rho-2) * self.D

          cost = ddot(self.D-x,self.D-x) #LA.norm(self.D-x)^2
          
          self.x = x
          self.xk = x
                    
          return (x,cost)
          
          
        
        
        
        
class OptProblem_ValleyFilling_Home(OptimizationProblem_Gurobi):
    
      delta = 1 # demand electricity price relationship  (ONLY FOR VALLEY FILLING)
      gamma = 0 # Trade-off parameter
      alpha = (0.05 * 15 * 60)/3600 #/ delta     #Battery depresiation cost [EUR/kWh] and transformed to [EUR/kW]
     
      def __init__(self, idx, discharge = False): 
          
          self.idx = idx
          self.discharge = discharge
          
          self.xmax = 4   # Max charging power for greedy need to add some 1e-3 or something for it to be feasible

          if self.discharge: 
              self.xmin = -4 # yes
          else:
              self.xmin = 0  # no
                
          data = loadEV('home', self.idx)              
              
          for key,val in data.items() :
       
              if(key == 'A'):       
                 self.A_in = data[key][()].T 
          
              if(key == 'R'):
                 self.R_in = data[key][()][0][0]
                 
              if(key == 'd'):
                 self.d_in = data[key][()] 
                 
              if(key == 'B'): # and self.discharge
                 self.B_in = data[key][()].T   
                 
              if(key == 'S_max'): # and self.discharge
                 self.Smax_in = data[key][()].T
                 
              if(key == 'S_min'): # and self.discharge
                 self.Smin_in = data[key][()].T
          
          data.close()  
          
          self.setModel()  
          
          self.setX(T)
          self.setZ(T)
          
          # x - z = 0
          self.addMainConstraint(I, -I, O)
          
          self.addConstraintX([I, '>=', self.d_in * self.xmin]) # lb
          self.addConstraintX([I, '<=', self.d_in * self.xmax]) # ub
          
          self.addConstraintX([self.A_in, '==', self.R_in]) 
              
          # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G 
          
             self.addConstraintX([self.B_in, '>=', self.Smin_in - 1e-4])
             self.addConstraintX([self.B_in, '<=', self.Smax_in + 1e-4])
             
          
          if(self.gamma == 0):
              obj = [None]
          else:# f = gamma * alpha * x^2
              obj = [self.gamma * self.alpha * I, None]
              
          self.setObjectiveX(obj, 'min')
             
          
          # g(z) - indicator function of the set {0} => Sum (zi) = 0 (<=>  Sum (xi) = 0)
          self.setObjectiveZ([None], 'min')# [None] = 0x^2 + 0x + 0
           
      '''     
      def solveX(self):
          
          if(self.gamma == 0):
              obj = [None]
          else:# f = gamma * alpha * x^2
              obj = [self.gamma * self.alpha * I, None]
              
          self.setObjectiveX(obj, 'min')
      
          x, cost = self.optimizeX()   
          
          #self.x = x
          #self.xk = x                         
          
          cost = self.gamma*self.delta*self.alpha * ddot(x,x)
              
          return (x , cost)
      '''
               
                
  
def loadEV(strategy, idx):
    
    file_base = DATA_DIR + '/EVs/' + strategy + '/'

    #print 'Path: ', os.path.abspath(file_base)      
    
    #os.chdir(file_base)
    file_name = file_base + str(idx) + '.mat' #tr(idx).encode('utf-8')
    return h5py.File(file_name, 'r') # open read-only
    # f.close()
    
def loadAggr():
    
    #file_base = '../data/Aggregator/'    
    #os.chdir(file_base)    
    file_name = DATA_DIR + '/Aggregator/aggregator.mat'
    return h5py.File(file_name, 'r') # open read-only
    # f.close()   
        
        
if __name__ == "__main__":
    
   #reload(sys)  
   #sys.setdefaultencoding('utf8')
    
   '''
   
   aggr = loadAggr()
   #D,price = np.empty
   
   for key,val in aggr.items() :
       
       if(key == 'D'):       
          D = aggr[key][()].T
          
       if(key == 'price'):
          price = aggr['price'][()].T
          
   aggr.close()
   
   a = OptProblem_Aggregator_ValleyFilling()
   
   a.setParametersObjX(0.5, np.zeros((96, 1)), np.zeros((96, 1)))
   x, c = a.solveX()
   z = a.solveZ()
   
   #print x
   
   
   #D = aggr['D'][()]
   #price = aggr['price'][()]  # Energy price
 
   delta=  np.mean(price)/(np.mean(D) * (3600*1000)  *15*60 )       # Empirical [price/demand^2]
   
   OptProblem_ValleyFilling_Home.delta = delta
   OptProblem_ValleyFilling_Home.gamma = 0
   OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta
   
   op = OptProblem_ValleyFilling_Home(1, True)   
   op.setParametersObjX(0.5, np.zeros((96, 1)), np.zeros((96, 1)))
   x, c = op.solveX()
   #op.setParametersObjZ(0.5, x, np.zeros((96, 1)))
   #z = op.solveZ() 
   
   print x
   #print z
   
   '''
   
   model = Model() 
   model.params.OutputFlag = 0 # verbose = 1            
   model.modelSense = GRB.MINIMIZE
      
   # Add variables to model
   for i in xrange(96):# n <=> (n,1)
        model.addVar(lb = -GRB.INFINITY) # N.B.!!! otherwise lb is set on default to 0
   model.update()
   x = model.getVars()     
        
   # auxiliary parameters for the k+1 th z step
   uk = 0.5 * np.ones((96, 1))
   zk = 0.3 * np.ones((96, 1))
   #self.uk = 0   

   for i in xrange(96):
                                         
       model.addConstr(x[i],  '<', 0.1) 
       model.addConstr(x[i],  '>', -0.1)         
                       
   fexpr = LinExpr() 
   # The default value for the target function. A constant is good e.g. to represent 
   # the indicator function (with the appropriate constraint sum(xi) == 0) etc.
   fexpr += 0
   
   A = np.identity(96)
   B = -np.identity(96)
   c = np.zeros((96,1))
   
   rho = 0.5
   
   obj = QuadExpr()    
   for i in xrange(96):
            
       # A * x + B * zk - c + uk 
       p_expr = LinExpr() 
       if(A is not None):                         
          for j in xrange(96):
              if A[i][j] != 0:
                 p_expr += A[i][j] * x[j]
                      
       if(B is not None):        
          for j in xrange(96):
              if B[i][j] != 0:
                 p_expr += B[i][j] * zk[j]
                      
       if(c is not None):
          p_expr -= c[i]                
            
       p_expr += uk[i]
            
       # 2nd norm 
       obj += (p_expr) * (p_expr)            
            
   #obj *= rho
   # rho * 1/2 * obj (two times *) is syntactically correct but not semantically 
   # and delivers wrong results
   #obj *= 0.5 
   obj +=  fexpr     # 1/2 * (obj)    
        
   model.setObjective(obj)
   model.update()
       
   model.optimize()         
   xRslt = np.zeros((96, 1)) 
   if model.status == GRB.status.OPTIMAL:
      for i in xrange(96):
          xRslt[i][0] = x[i].x
              
   print xRslt
                          
       
   
   
   
   
   
   
   
   