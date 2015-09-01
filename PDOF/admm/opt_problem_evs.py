# -*- coding: utf-8 -*-
"""
Created on Sat May 09 00:06:45 2015

@author: shterev
"""
#from cvxpy import *
import numpy as np
from numpy import linalg as LA
from scipy.linalg.blas import ddot, dnrm2
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
        
        
class OptProblem_Aggregator_PriceBased(OptimizationProblem):
        
      def __init__(self): 
          
          data = loadAggr()
   
          for key,val in data.items() :
          
              if(key == 'price'):
                 self.price = data['price'][()].T
          
          data.close()
          
          
      def setParameters(self, rho, K):
                  
          self.rho = rho;         #augement cost parameter
          self.K = K # xold - xmean - u  Normalization parameter
          
          
      def solve(self):
          
          x= self.K + self.p/self.rho          
 
          # box constraints
          indx = where(x<-self.re)
          if indx:
              x[indx]=-self.re[indx]
             
          indx = where(x>-self.xamin)
          if indx:
              x[indx]=-self.xamin[indx]
 

          cost = -np.dot(self.p, x) # -p'*x;
          
          return (x,cost)
          
          
          

class OptProblem_PriceBased_Home(OptimizationProblem):
    
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
                 #A = self.A # for debug
          
              if(key == 'R'):
                 self.R = data[key][()][0][0]
                 
              if(key == 'd'):
                 self.d = data[key][()].T
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

 
          self.model = Model() 
          self.model.params.OutputFlag = 0 # verbose = 1
        
          # Add variables to model
          for i in xrange(OptimizationProblem.T):
              self.model.addVar(lb= (self.d[0][i] * self.xmin) , ub = (self.d[0][i] * self.xmax))
          self.model.update()
          self.vars = self.model.getVars()
        
                
          # Aeq * x = beq 
          expr = LinExpr()
          for i in xrange(OptimizationProblem.T):   
              if self.A[0][i] != 0:
                 expr += (self.A[0][i])*(self.vars[i]) 
               
          self.model.addConstr(expr,  GRB.EQUAL, self.R)
              
          # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G 
             
             for i in xrange(self.B.shape[0]):#rows 61
                 expr = LinExpr()
                 
                 for j in range(self.B.shape[1]):#cols 96
                     if self.B[i][j] != 0:
                        expr += self.B[i][j]*self.vars[j]
                        
                 self.model.addConstr(expr,  GRB.GREATER_EQUAL, self.Smin[i][0] - 1e-4)
                 self.model.addConstr(expr,  GRB.LESS_EQUAL, self.Smax[i][0] + 1e-4)        
          
          
          
          
          
          self.model.update()
           
                      
                      
      def setParameters(self, rho, K):
                  
           
          #self.idx = params.idx;         # EV index
          #selfchargeStrategy=params.chargeStrategy   # Charging strategy
           
          self.rho = rho;         #augement cost parameter
          self.K = K # xold - xmean - u  Normalization parameter
           
          #self.OptimizationProblem.T=length(params.xold);                         # Number of time slots
          #self.discharge=params.discharge ; # discharge allowed
           
      def solve(self):
          
          #print self.rho, self.gamma, self.alpha, self.K
          
          dd = (self.rho / (2*self.gamma * self.alpha + self.rho)) * self.K;   
                   
          # Populate objective
          obj = QuadExpr() # Cost= 1/2 * norm(C*xtemp -dd)^2;
          for i in xrange(OptimizationProblem.T):
              tmp = self.vars[i] - dd[i][0] #dd[i][0]
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
               
              
          x = np.zeros((OptimizationProblem.T, 1)) #np.zeros((OptimizationProblem.T, 1))
          if self.model.status == GRB.status.OPTIMAL:
             for i in xrange(OptimizationProblem.T):
                 x[i][0] = self.vars[i].x #x[i][0] = self.vars[i].x
                 
          cost=self.gamma*self.alpha*ddot(x,x);
              

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
                 #A = self.A # for debug
          
              if(key == 'R'):
                 self.R = data[key][()][0][0]
                 
              if(key == 'd'):
                 self.d = data[key][()].T
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

 
          self.model = Model() 
          self.model.params.OutputFlag = 0 # verbose = 1
        
          # Add variables to model
          for i in xrange(OptimizationProblem.T):
              self.model.addVar(lb= (self.d[0][i] * self.xmin) , ub = (self.d[0][i] * self.xmax))
          self.model.update()
          self.vars = self.model.getVars()
        
                
          # Aeq * x = beq 
          expr = LinExpr()
          for i in xrange(OptimizationProblem.T):   
              if self.A[0][i] != 0:
                 expr += (self.A[0][i])*(self.vars[i]) 
               
          self.model.addConstr(expr,  GRB.EQUAL, self.R)
              
          # Smin <= B * x <= Smax
          if self.discharge:  # yes V2G 
             
             for i in xrange(self.B.shape[0]):#rows 61
                 expr = LinExpr()
                 
                 for j in range(self.B.shape[1]):#cols 96
                     if self.B[i][j] != 0:
                        expr += self.B[i][j]*self.vars[j]
                        
                 self.model.addConstr(expr,  GRB.GREATER_EQUAL, self.Smin[i][0] - 1e-4)
                 self.model.addConstr(expr,  GRB.LESS_EQUAL, self.Smax[i][0] + 1e-4)        
          
          
          
          
          
          self.model.update()
           
                      
                      
      def setParameters(self, rho, K):
                  
           
          #self.idx = params.idx;         # EV index
          #selfchargeStrategy=params.chargeStrategy   # Charging strategy
           
          self.rho = rho;         #augement cost parameter
          self.K = K # xold - xmean - u  Normalization parameter
           
          #self.OptimizationProblem.T=length(params.xold);                         # Number of time slots
          #self.discharge=params.discharge ; # discharge allowed
           
      def solve(self):
          
          #print self.rho, self.gamma, self.alpha, self.K
          
          dd = (self.rho / (2*self.gamma * self.alpha + self.rho)) * self.K;   
                   
          # Populate objective
          obj = QuadExpr() # Cost= 1/2 * norm(C*xtemp -dd)^2;
          for i in xrange(OptimizationProblem.T):
              tmp = self.vars[i] - dd[i][0] #dd[i][0]
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
               
              
          x = np.zeros((OptimizationProblem.T, 1)) #np.zeros((OptimizationProblem.T, 1))
          if self.model.status == GRB.status.OPTIMAL:
             for i in xrange(OptimizationProblem.T):
                 x[i][0] = self.vars[i].x #x[i][0] = self.vars[i].x
                 
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
   
   a = OptProblem_Aggregator_ValleyFilling()
   
   a.setParameters(0.5, np.zeros((96,1)))
   x, c = a.solve()
   
   #print x
   
   
   #D = aggr['D'][()]
   #price = aggr['price'][()]  # Energy price
 
   delta=  np.mean(a.price)/(np.mean(a.D) * (3600*1000)  *15*60 ) ;      # Empirical [price/demand^2]
   
   op = OptProblem_ValleyFilling_Home(1, True)
   
   OptProblem_ValleyFilling_Home.delta = delta;
   OptProblem_ValleyFilling_Home.gamma = 0;
   OptProblem_ValleyFilling_Home.alpha /= OptProblem_ValleyFilling_Home.delta;
   
   op.setParameters(0.5, np.zeros((96, 1)))
   x, c = op.solve()
   
   print x
   
   
   