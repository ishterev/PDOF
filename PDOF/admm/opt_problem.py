# -*- coding: utf-8 -*-
"""
Created on Sat May 09 00:06:45 2015

@author: shterev
"""
#from cvxpy import *
import numpy as np
from numpy import linalg as LA
from gurobipy import *

import h5py

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

class OptimizationProblem:
    
    T = 96
        
    def solve(self):
        pass
        
    def setParameters(self, params):
        pass
        
    def getT(self):
        return OptimizationProblem.T
        
        

class OptProblem_Aggregator(OptimizationProblem):
        
      def __init__(self): 
          
          data = loadAggr()
   
          for key,val in data.items() :
               
              if(key == 'D'):       
                 self.D = data[key][()]
          
              if(key == 'price'):
                 self.price = data['price'][()]
          
          data.close()
          
          
      def setParameters(self, rho, K):
                  
          self.rho = rho;         #augement cost parameter
          self.K = K # xold - xmean - u  Normalization parameter
          
          
      def solve(self):
          
          x = self.rho/(self.rho-2)* self.K - 2/(self.rho-2) * self.D

          cost = LA.norm(self.D-x)  
          
          return (x,cost)
          
          
        
        
        
        
class OptProblem_ValleyFilling_Home(OptimizationProblem):
    
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
                 self.A = data[key][()]
          
              if(key == 'R'):
                 self.R = data[key][()][0][0]
                 
              if(key == 'd'):
                 self.d = data[key][()]
                 
              if(key == 'B'): # and self.discharge
                 self.B = data[key][()]                 
                 
              if(key == 'S_max'): # and self.discharge
                 self.Smax = data[key][()]
                 
              if(key == 'S_min'): # and self.discharge
                 self.Smin = data[key][()]
          
          data.close()

 
          self.model = Model() 
          self.model.params.OutputFlag = 0 # verbose = 1
        
          # Add variables to model
          for j in xrange(OptimizationProblem.T):
              self.model.addVar(lb= (self.d[j][0] * self.xmin) , ub = (self.d[j][0] * self.xmax))
          self.model.update()
          self.vars = self.model.getVars()
        
                
          # Aeq * x = beq 
          expr = LinExpr()
          for i in xrange(OptimizationProblem.T):   
              if self.A[i][0] != 0:
                 expr += (self.A[i][0])*(self.vars[i]) 
               
          self.model.addConstr(expr,  GRB.EQUAL, self.R)
                
                
          '''if self.discharge:  # yes V2G 
              # Smin <= B * x <= Smax
              for i in xrange(OptimizationProblem.T):
                  expr = LinExpr()
                  expr += self.B[i]*self.vars[i]                
                  self.model.addConstr(expr,  GRB.GREATER_EQUAL, self.Smin[i][0] - 1e-4)
                  self.model.addConstr(expr,  GRB.LESS_EQUAL, self.Smax[i][0] + 1e-4)'''
              
          # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G 
             expr = LinExpr()
             for i in xrange(self.B.shape[0]):#rows 96
                 
                 for j in range(self.B.shape[1]):#cols 61
                     if self.B[i][j] != 0:
                        expr += self.B[i][j]*self.vars[j]
                        
                        self.model.addConstr(expr,  GRB.GREATER_EQUAL, self.Smin[0][j] - 1e-4)
                        self.model.addConstr(expr,  GRB.LESS_EQUAL, self.Smax[0][j] + 1e-4)        
          
          
          
          
          
          self.model.update()
           
                      
                      
      def setParameters(self, rho, K):
                  
           
          #self.idx = params.idx;         # EV index
          #selfchargeStrategy=params.chargeStrategy   # Charging strategy
           
          self.rho = rho;         #augement cost parameter
          self.K = K # xold - xmean - u  Normalization parameter
           
          #self.OptimizationProblem.T=length(params.xold);                         # Number of time slots
          #self.discharge=params.discharge ; # discharge allowed
           
      def solve(self):
          
          dd = (self.rho / (2*self.gamma * self.alpha + self.rho)) * self.K;   
                   
          # Populate objective
          obj = QuadExpr() # Cost= 1/2 * norm(C*xtemp -dd)^2;
          for i in xrange(OptimizationProblem.T):
              tmp = self.vars[i] - dd[i][0]
              obj += tmp * tmp
            
          obj = 1/2 * (obj)
          
          self.model.modelSense = GRB.MINIMIZE
          self.model.setObjective(obj)
          self.model.update()
            
            
          # Solve
          self.model.optimize()
          
          #print self.model.status == GRB.status.OPTIMAL

          '''if self.model.status == GRB.status.OPTIMAL:
               x = self.model.getAttr('x', vars)              
               return np.asarray(x, order = 'F')
          else:
               return np.zeros((OptimizationProblem.T, 1))'''
               
              
          x = np.zeros(OptimizationProblem.T) #np.zeros((OptimizationProblem.T, 1))
          if self.model.status == GRB.status.OPTIMAL:
             for i in xrange(OptimizationProblem.T):
                 x[i] = self.vars[i].x #x[i][0] = self.vars[i].x
              

          return (x , self.model.getObjective().getValue())       
           
          #print self.model.getAttr("x", self.model.getVars())
               
          #return np.asarray(self.model.getAttr(GRB.attr.x, self.vars), order = 'F')
                
  
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
   
   aggr = loadAggr()
   #D,price = np.empty
   
   for key,val in aggr.items() :
       
       if(key == 'D'):       
          D = aggr[key][()]
          
       if(key == 'price'):
          price = aggr['price'][()]
          
   aggr.close()
   
   a = OptProblem_Aggregator()
   
   a.setParameters(0.5, np.zeros((96, 1)))
   x, c = a.solve()
   
   print x
   
   
   #D = aggr['D'][()]
   #price = aggr['price'][()]  # Energy price
 
   delta=  np.mean(price)/(np.mean(D) * (3600*1000)  *15*60 ) ;      # Empirical [price/demand^2]
   
   op = OptProblem_ValleyFilling_Home(1, True)
   
   OptProblem_ValleyFilling_Home.delta = delta;
   OptProblem_ValleyFilling_Home.gamma = 0;
   OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta;
   
   op.setParameters(0.5, np.zeros((96, 1)))
   x, c = op.solve()
   
   print x
   
   
   