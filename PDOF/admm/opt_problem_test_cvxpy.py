# -*- coding: utf-8 -*-
"""
Created on Sat May 09 00:06:45 2015

@author: shterev
"""
#from cvxpy import *
import numpy as np
from scipy.linalg.blas import ddot
from cvxpy import *

import os
import scipy.io as sio
#import h5py

from opt_problem import *

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

deltaT = 900 #=15*60  Time slot duration [sec]
T = 96 #= 24*3600/deltaT # Number of time slots

try:
    inf = float('inf')
except:  # check for a particular exception here?
    inf = 1e30000


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

 Test : Exchange ADMM problems in the general form   

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class OptProblem_Aggregator_PriceBased(OptimizationProblem_Cvxpy):
    
      p = None                # regularization
      re = None 
      xamin = 0
        
      def __init__(self): 
          
          data = loadAggr()
   
          for key,val in data.items() :
          
              if(key == 'price'):
                 self.price = data['price'][()]# Base demand profile 
          
          #data.close()
          
         
          self.p=np.tile(self.price,(4,1))
          self.p=self.p / (3600*1000) * deltaT  # scaling of price in EUR/kW

          self.re=  100e3 *np.ones((T,1))  # maximal aviliable load 1GW 
          self.xamin=-100e3*np.ones((T,1))
          self.xmax=self.re
          
         
          self.setX(T)
          self.setZ(T)
          
          # x - z = 0
          self.addMainConstraint(np.identity(T), -np.identity(T), 0)
          
          # g(z) - indicator function of the set {0} => Sum (zi) = 0 (<=>  Sum (xi) = 0)          
          # g = abs(sum_entries(self.getZ())) * inf
          g = 0
          self.setObjectiveZ(g, 'min')
          self.addConstraintZ(sum_entries(self.getZ()) == 0)
         
          self.setModel()
          
          
      def setParametersObjX(self, rho, zk, uk):                  
          if(rho is not None):
             self.rho.value = rho #self.rho.value
          if(self.zk is not None):
             self.zk.value = zk #self.zk.value
          if(self.uk is not None):
             self.uk.value = uk #self.uk.value
          self.K = zk - uk # xold - xmean - u  Normalization parameter
          
          
      def solveX(self):
          
          x= (self.K + self.p/self.rho).value          
 
          # box constraints
          indx = np.where(x<-self.re)
          if indx:
              x[indx]=-self.re[indx]
             
          indx = np.where(x>-self.xamin)
          if indx:
              x[indx]=-self.xamin[indx]
              
           
          self.x.value = x
          self.xk = x
 
          cost = -np.dot(self.p.T, x) # -p'*x
          
          return (x,cost)
          
          
          
class OptProblem_PriceBased_Home(OptimizationProblem_Cvxpy):
    
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
          
          #data.close()          
          
          self.setX(T)
          self.setZ(T)
          
          # x - z = 0
          self.addMainConstraint(np.identity(T), -np.identity(T), 0)
          
          self.addConstraintX(self.d_in * self.xmin <= self.x) #lb
          self.addConstraintX(self.x <= self.d_in * self.xmax ) #ub
          # Aeq * x = beq 
          self.addConstraintX(self.A_in * self.x == self.R_in)
          
           # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G           
             self.addConstraintX(self.Smin_in - 1e-4 <= self.B_in*self.x)
             self.addConstraintX(self.B_in*self.x <= self.Smax_in + 1e-4)
             
           
          f = self.gamma * self.alpha * sum_squares(self.x)
          self.setObjectiveX(f, 'min')
          
          # g(z) - indicator function of the set {0} => Sum (zi) = 0 (<=>  Sum (xi) = 0)
          g = 0
          self.setObjectiveZ(g, 'min')
          self.addConstraintZ(sum_entries(self.getZ()) == 0)
         
          self.setModel()
         
         
           
      def solveX(self):
          
        xRslt, costRslt =  self.optimizeX()
        costRslt=self.gamma*self.alpha*ddot(xRslt, xRslt)        
                   
        #self.x.value = xRslt
        #self.xk = xRslt
        
        return (xRslt, costRslt) 
        
        

class OptProblem_Aggregator_ValleyFilling(OptimizationProblem_Cvxpy):
        
      def __init__(self): 
          
          data = loadAggr()
   
          for key,val in data.items() :
               
              if(key == 'D'):       
                 self.D = data[key][()].T
          
              if(key == 'price'):
                 self.price = data['price'][()].T
          
          #data.close()
          
          self.setX(T)
          self.setZ(T)
          
          # x - z = 0
          self.addMainConstraint(np.identity(T), -np.identity(T), 0)
          
          # g(z) - indicator function of the set {0} => Sum (zi) = 0 (<=>  Sum (xi) = 0)
          g = 0
          self.setObjectiveZ(g, 'min')
          self.addConstraintZ(sum_entries(self.getZ()) == 0)
         
          self.setModel()
          
          
      def setParametersObjX(self, rho, zk, uk):                  
          if(rho is not None):
             self.rho.value = rho #self.rho.value
          if(self.zk is not None):
             self.zk.value = zk #self.zk.value
          if(self.uk is not None):
             self.uk.value = uk #self.uk.value
          self.K = zk - uk # xold - xmean - u  Normalization parameter
          
          
      def solveX(self):
          
          if(self.rho.value == 2):
              self.rho.value += 1e-9
          
          x = (self.rho/(self.rho-2)* self.K - 2/(self.rho-2) * self.D).value
                     
          self.x.value = x
          self.xk = x

          cost = ddot(self.D-x,self.D-x) #LA.norm(self.D-x)^2
          
          return (x,cost)
          
          
        
        
        
        
class OptProblem_ValleyFilling_Home(OptimizationProblem_Cvxpy):
    
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
          
          #data.close()          
          
          self.setX(T)
          self.setZ(T)
          
          # x - z = 0
          self.addMainConstraint(np.identity(T), -np.identity(T), 0)
          
          self.addConstraintX(self.d_in * self.xmin <= self.x) #lb
          self.addConstraintX(self.x <= self.d_in * self.xmax ) #ub
          # Aeq * x = beq 
          self.addConstraintX(self.A_in * self.x == self.R_in)
          
           # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G           
             self.addConstraintX(self.Smin_in - 1e-4 <= self.B_in * self.x)
             self.addConstraintX(self.B_in*self.x <= self.Smax_in + 1e-4)
             
           
          f = self.gamma * self.alpha * sum_squares(self.x)
          self.setObjectiveX(f, 'min')
          
          # g(z) - indicator function of the set {0} => Sum (zi) = 0 (<=>  Sum (xi) = 0)
          g = 0
          self.setObjectiveZ(g, 'min')
          self.addConstraintZ(sum_entries(self.getZ()) == 0)
         
          self.setModel()  
         
                      
      def solveX(self):

        x, c = self.optimizeX()
        c=self.gamma*self.delta*self.alpha*ddot(x, x) #self.problem.value    
                           
        #self.x.value = xRslt
        #self.xk = xRslt
        
        return (x, c) 
          
          
                
  
def loadEV(strategy, idx):
    
    file_base = DATA_DIR + '/EVs/' + strategy + '/'

    #print 'Path: ', os.path.abspath(file_base)      
    
    #os.chdir(file_base)
    file_name = file_base + str(idx) + '.mat' #tr(idx).encode('utf-8')
    return sio.loadmat(file_name)# h5py.File(file_name, 'r') # open read-only
    # f.close()
    
def loadAggr():
    
    #file_base = '../data/Aggregator/'    
    #os.chdir(file_base)    
    file_name = DATA_DIR + '/Aggregator/aggregator.mat'
    return sio.loadmat(file_name) # h5py.File(file_name, 'r') # open read-only
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
   
   a.setParameters(0.5, np.zeros((96, 1)), np.zeros((96, 1)))
   x, c = a.solveX()
   a.setParametersObjZ(0.5, x, np.ones((96, 1)))
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
   op.setParametersObjZ(0.5, x, np.zeros((96, 1)))
   z = op.solveZ() 
   
   print x
   print z
   
