# -*- coding: utf-8 -*-
"""
Created on Sat May 09 00:06:45 2015

@author: shterev
"""
#from cvxpy import *
import numpy as np
from cvxpy import *

import os
import h5py

from opt_problem import *

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

deltaT = 900 #=15*60  Time slot duration [sec]
T = 96 #= 24*3600/deltaT # Number of time slots


class OptProblem_Aggregator_PriceBased(OptimizationProblem):
    
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
          
          
      def setParameters(self, rho, K):
                  
          self.rho = rho         #augement cost parameter
          self.K = K # xold - xmean - u  Normalization parameter
          
          
      def solve(self):
          
          x= self.K + self.p/self.rho          
 
          # box constraints
          indx = np.where(x<-self.re)
          if indx:
              x[indx]=-self.re[indx]
             
          indx = np.where(x>-self.xamin)
          if indx:
              x[indx]=-self.xamin[indx]
 

          cost = -np.dot(self.p.T, x) # -p'*x
          
          return (x,cost)
          
          
          
class OptProblem_PriceBased_Home(OptimizationProblemCvxpy):
    
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
                 self.A = data[key][()].T 
          
              if(key == 'R'):
                 self.R = data[key][()][0][0]
                 
              if(key == 'd'):
                 self.d = data[key][()] 
                 
              if(key == 'B'): # and self.discharge
                 self.B = data[key][()].T   
                 
              if(key == 'S_max'): # and self.discharge
                 self.Smax = data[key][()].T
                 
              if(key == 'S_min'): # and self.discharge
                 self.Smin = data[key][()].T
          
          data.close()          
          
          self.setX(T)
          
          self.addConstraint(self.d * self.xmin <= self.x) #lb
          self.addConstraint(self.x <= self.d * self.xmax ) #ub
          # Aeq * x = beq 
          self.addConstraint(self.A * self.x == self.R)
          
           # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G           
             self.addConstraint(self.Smin - 1e-4 <= self.B*self.x)
             self.addConstraint(self.B*self.x <= self.Smax + 1e-4)
             
           
          f = self.gamma * self.alpha * sum_squares(self.x)
          self.setObjective(f, 'min')
         
          self.setModel()
         
           
      def solve(self):
          
        xRslt =  self.optimize()
        costRslt=self.gamma*self.alpha*ddot(xRslt, xRslt)
        
        return (xRslt, costRslt) 
        
        

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
                  
          self.rho = rho         #augement cost parameter
          self.K = K # xold - xmean - u  Normalization parameter
          
          
      def solve(self):
          
          x = self.rho/(self.rho-2)* self.K - 2/(self.rho-2) * self.D

          cost = ddot(self.D-x,self.D-x) #LA.norm(self.D-x)^2
          
          return (x,cost)
          
          
        
        
        
        
class OptProblem_ValleyFilling_Home(OptimizationProblemCvxpy):
    
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
          
              if(key == 'R'):
                 self.R = data[key][()][0][0]
                 
              if(key == 'd'):
                 self.d = data[key][()] 
                 
              if(key == 'B'): # and self.discharge
                 self.B = data[key][()].T   
                 
              if(key == 'S_max'): # and self.discharge
                 self.Smax = data[key][()].T
                 
              if(key == 'S_min'): # and self.discharge
                 self.Smin = data[key][()].T
          
          data.close()          
          
          self.setX(T)
          
          self.addConstraint(self.d * self.xmin <= self.x) #lb
          self.addConstraint(self.x <= self.d * self.xmax ) #ub
          # Aeq * x = beq 
          self.addConstraint(self.A * self.x == self.R)
          
           # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G           
             self.addConstraint(self.Smin - 1e-4 <= self.B*self.x)
             self.addConstraint(self.B*self.x <= self.Smax + 1e-4)
             
           
          f = self.gamma * self.alpha * sum_squares(self.x)
          self.setObjective(f, 'min')
         
          self.setModel()  
         
                      
      def solve(self):

        xRslt = xRslt =  self.optimize()
        costRslt=self.gamma*self.delta*self.alpha*ddot(xRslt, xRslt) #self.problem.value    
        
        return (xRslt, costRslt) 
          
          
                
  
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
 
   delta=  np.mean(price)/(np.mean(D) * (3600*1000)  *15*60 )       # Empirical [price/demand^2]
   
   OptProblem_ValleyFilling_Home.delta = delta
   OptProblem_ValleyFilling_Home.gamma = 0
   OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta
   
   op = OptProblem_ValleyFilling_Home(1, True)   
   op.setParameters(0.5, np.zeros((96, 1)))
   x, c = op.solve()
   
   print x
   
   
   