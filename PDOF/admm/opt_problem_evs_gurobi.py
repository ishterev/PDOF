# -*- coding: utf-8 -*-
"""
Created on Sat May 09 00:06:45 2015

@author: shterev
"""
import numpy as np
from numpy import linalg as LA
from scipy.linalg.blas import ddot, dnrm2
from gurobipy import *

import h5py

from opt_problem import *

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

T = 96
        
class OptProblem_Aggregator_PriceBased(OptimizationProblem):
          
      p = None                # regularization
      re = None 
      xamin = 0
        
      def __init__(self): 
          
          data = loadAggr()
   
          for key,val in data.items() :
          
              if(key == 'price'):
                 self.price = data['price'][()]#.T
          
          data.close()
          
          
      def setParameters(self, rho, K):
                  
          self.rho = rho;         #augement cost parameter
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
 

          cost = -np.dot(self.p.T, x) # -p'*x;
          
          return (x,cost)
          
          
          

class OptProblem_PriceBased_Home(OptimizationProblemGurobi):
    
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
          
          # check the arrays, because after reading in there is no difference between a and a.T
          for key,val in data.items() :
       
              if(key == 'A'):       
                 self.A = data[key][()].T
                 #A = self.A # for debug
          
              if(key == 'R'):
                 self.R = data[key][()]
                 
              if(key == 'd'):
                 self.d = data[key][()]
                 #d = self.d
                 
              if(key == 'B'): # and self.discharge
                 self.B = data[key][()].T 
                 #B = self.B
                 
              if(key == 'S_max'): # and self.discharge
                 self.Smax = data[key][()].T
                 #Smax = self.Smax
                 
              if(key == 'S_min'): # and self.discharge
                 self.Smin = data[key][()].T
                # Smin = self.Smin
          
          data.close()

          self.setModel()
          self.setX((T,1))
          
          self.addConstraint([np.identity(T), '>=', self.d * self.xmin]) # lb
          self.addConstraint([np.identity(T), '<=', self.d * self.xmax]) # ub
          
          self.addConstraint([self.A, '==', self.R]) 
              
          # Smin <= B * x <= Smax
          if False and self.discharge:  # yes V2G 
          
             self.addConstraint([self.B, '>=', self.Smin - 1e-4])
             self.addConstraint([self.B, '<=', self.Smax + 1e-4])
                           
           
      def solve(self):          
          # rho         #augement cost parameter
          # K = xold - xmean - u  Normalization parameter
          self.setV((self.rho / (2*self.gamma * self.alpha + self.rho)) * self.K)
          
          self.setObjective([np.zeros((1,T))], 'min')
      
          x = OptimizationProblemGurobi.solve(self)                 
          cost=self.gamma*self.alpha*ddot(x,x)             

          return (x , cost) #self.model.getObjective().getValue()      
           
          #print self.model.getAttr("x", self.model.getVars())
               
          #return np.asarray(self.model.getAttr(GRB.attr.x, self.vars), order = 'F')
               
        
        

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

          cost = ddot(self.D-x,self.D-x) #LA.norm(self.D-x)^2
                    
          return (x,cost)
          
          
        
        
        
        
class OptProblem_ValleyFilling_Home(OptimizationProblemGurobi):
    
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
                 #A = self.A # for debug
          
              if(key == 'R'):
                 self.R = data[key][()]
                 
              if(key == 'd'):
                 self.d = data[key][()]
                 #d = self.d
                 
              if(key == 'B'): # and self.discharge
                 self.B = data[key][()].T 
                 #B = self.B
                 
              if(key == 'S_max'): # and self.discharge
                 self.Smax = data[key][()].T
                 #Smax = self.Smax
                 
              if(key == 'S_min'): # and self.discharge
                 self.Smin = data[key][()].T
                # Smin = self.Smin
          
          self.setModel()
          self.setX((T,1))
          
          self.addConstraint([np.identity(T), '>=', self.d * self.xmin]) # lb
          self.addConstraint([np.identity(T), '<=', self.d * self.xmax]) # ub
          
          self.addConstraint([self.A, '==', self.R]) 
              
          # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G 
          
             self.addConstraint([self.B, '>=', self.Smin - 1e-4])
             self.addConstraint([self.B, '<=', self.Smax + 1e-4])
           
           
      def solve(self):
          
          # rho         #augement cost parameter
          # K = xold - xmean - u  Normalization parameter
          self.setV((self.rho / (2*self.gamma * self.alpha + self.rho)) * self.K)
          
          self.setObjective([np.zeros((1,T))], 'min')
      
          x = OptimizationProblemGurobi.solve(self)                           
          
          cost=self.gamma*self.delta*self.alpha*ddot(x,x);
              
          return (x , cost) #self.model.getObjective().getValue()      
           
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
   
   a = OptProblem_Aggregator_PriceBased()
   
   price= a.price # Base demand profile 
   p=np.tile(price,(4,1))
   p=p / (3600*1000) * 15*60  # scaling of price in EUR/kW
   a.p=p
   
   a.re=  100e3 * np.zeros((96,1))  # maximal aviliable load 1GW 
   a.xamin=-100e3 * np.zeros((96,1))
   
   a.setParameters(0.5, np.zeros((96,1)))
   x, c = a.solve()
   
   #print x
   
   
   #D = aggr['D'][()]
   #price = aggr['price'][()]  # Energy price
 
   #delta=  np.mean(a.price)/(np.mean(a.D) * (3600*1000)  *15*60 ) ;      # Empirical [price/demand^2]
   
   op = OptProblem_PriceBased_Home(1, True)
   
   #OptProblem_ValleyFilling_Home.delta = delta;
   OptProblem_PriceBased_Home.gamma = 0;
   #OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta;
   
   op.setParameters(0.5, np.zeros((96, 1)))
   x, c = op.solve()
   
   print x
   
   
   