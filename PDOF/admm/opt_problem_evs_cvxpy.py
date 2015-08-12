# -*- coding: utf-8 -*-
"""
Created on Sat May 09 00:06:45 2015

@author: shterev
"""
#from cvxpy import *
import numpy as np
from numpy import linalg as LA
from cvxpy import *

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
        
        

class OptProblem_Aggregator_ValleyFilling(OptimizationProblem):
        
      def __init__(self): 
          
          data = loadAggr()
   
          for key,val in data.items() :
               
              if(key == 'D'):       
                 self.D = data[key][()].T
          
              if(key == 'price'):
                 self.price = data['price'][()].T
          
          data.close()
          
          
      def setParameters(self, rho, K):
                  
          self.rho = rho;         #augement cost parameter
          self.K = K # xold - xmean - u  Normalization parameter
          
          
      def solve(self):
          
          x = self.rho/(self.rho-2)* self.K - 2/(self.rho-2) * self.D

          cost = LA.norm(self.D-x) 
          cost *=cost
          
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
                 self.A = data[key][()].T 
                 A = self.A
          
              if(key == 'R'):
                 self.R = data[key][()][0][0]
                 R = self.R
                 
              if(key == 'd'):
                 self.d = data[key][()] 
                 d = self.d
                 
              if(key == 'B'): # and self.discharge
                 self.B = data[key][()].T   
                 B = self.B
                 
              if(key == 'S_max'): # and self.discharge
                 self.Smax = data[key][()].T
                 Smax = self.Smax
                 
              if(key == 'S_min'): # and self.discharge
                 self.Smin = data[key][()].T
                 Smin = self.Smin
          
          data.close()          
          
          self.x = Variable(OptimizationProblem.T, 1)
          
          self.lb = self.d * self.xmin <= self.x
          self.ub = self.x <= self.d * self.xmax 
         
          # Aeq * x = beq 
          self.Aeq = self.A * self.x == self.R         
          
           # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G 
             self.Aineq1 = self.Smin - 1e-4 <= self.B*self.x
             self.Aineq2 = self.B*self.x <= self.Smax + 1e-4
             
          self.K = Parameter(OptimizationProblem.T, value = np.zeros((OptimizationProblem.T, 1)))
          self.rho = Parameter(sign="positive", value = 0.5)
         
          self.dd = (self.rho / (2*self.gamma * self.alpha + self.rho)) * self.K
         
          #C = np.eye(OptimizationProblem.T)

          self.Cost = 1/2 * sum_squares(self.x - self.dd)# 1/2 * (norm(self.x - self.dd) ** 2) C*self.x - self.dd
          self.Constraints=[self.lb, self.ub, self.Aeq]
         
          if self.discharge:
             self.Constraints.append(self.Aineq1)
             self.Constraints.append(self.Aineq2)
         
          self.problem = Problem(Minimize(self.Cost), self.Constraints)   
         
                      
      def setParameters(self, rho, K):
                  
           
          #self.idx = params.idx;         # EV index
          #selfchargeStrategy=params.chargeStrategy   # Charging strategy
           
          self.rho.value = rho;         #augement cost parameter
          self.K.value = K # xold - xmean - u  Normalization parameter
           
          #self.OptimizationProblem.T=length(params.xold);                         # Number of time slots
          #self.discharge=params.discharge ; # discharge allowed
           
      def solve(self):
          
        self.problem.solve(solver=GUROBI) #solver=CVXOPT
        return (np.array(self.x.value), self.problem.value) 
          
          
                
  
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
          D = aggr[key][()].T
          
       if(key == 'price'):
          price = aggr['price'][()].T
          
   aggr.close()
   
   a = OptProblem_Aggregator_ValleyFilling()
   
   a.setParameters(0.5, np.zeros((96, 1)))
   x, c = a.solve()
   
   #print x
   
   
   #D = aggr['D'][()]
   #price = aggr['price'][()]  # Energy price
 
   delta=  np.mean(price)/(np.mean(D) * (3600*1000)  *15*60 ) ;      # Empirical [price/demand^2]
   
   OptProblem_ValleyFilling_Home.delta = delta;
   OptProblem_ValleyFilling_Home.gamma = 0;
   OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta;
   
   op = OptProblem_ValleyFilling_Home(1, True)   
   op.setParameters(0.5, np.zeros((96, 1)))
   x, c = op.solve()
   
   print x
   
   
   