# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 01:40:51 2015

@author: shterev
"""
import numpy as np

CVXPY = False
if(CVXPY):
   from opt_problem_evs_cvxpy import *
else:
   from opt_problem_evs_gurobi import *
   
deltaT = 900 #=15*60  Time slot duration [sec]
T = 96 #= 24*3600/deltaT # Number of time slots


class  OptProblemLoaderFactory:
    
    @staticmethod
    def _get(problem = "valley_filling", *args):
        
        if(problem == "valley_filling"):
          
          return OptProblemLoader_ValleyFilling(args)
          
        elif(problem == "price_based"):
            
            return OptProblemLoader_PriceBased(args)
            
        else:# default
            return OptProblemLoader_ValleyFilling(args)
           

class OptProblemLoader:    
    
      def load(self, startIdx, endIdx):
          pass
      
      
class OptProblemLoader_PriceBased(OptProblemLoader):   
    
      def __init__(self, strategy = "home", gamma = 0, V2G = True): 
          self.strategy = strategy 
          self.gamma = gamma
          self.V2G = V2G
          
      # load optimization problem from startIdx to endIdx - 1 including
      def load(self, startIdx, endIdx):
    
          opt_probs = np.empty((endIdx,), dtype=np.object)
    
          if(self.strategy == "home"):
              probClass = OptProblem_PriceBased_Home
          else:# opt. TODO: minEnergy, greedy
              probClass = OptProblem_PriceBased_Home
    
          if(startIdx == 0):
              # w.l.o.g. and for convenience, the aggregator is the 0th element
              problem = OptProblem_Aggregator_PriceBased()  

              price= problem.price # Base demand profile 
              p=np.tile(price,(4,1))
              p=p / (3600*1000) * deltaT  # scaling of price in EUR/kW
              problem.p=p

              problem.re=  100e3 *np.ones((T,1))  # maximal aviliable load 1GW 
              problem.xamin=-100e3*np.ones((T,1))
        
              opt_probs[0] = problem
        
              probClass.gamma = self.gamma
              for i in xrange(1, endIdx):
                    problem = probClass(i, self.V2G)
                    opt_probs[i] = problem
              
          else:
              for i in xrange(endIdx):
                    problem = probClass(startidx + i, self.V2G)
                    opt_probs[i] = problem
        
          return opt_probs
          
          
class OptProblemLoader_ValleyFilling(OptProblemLoader):   
    
      def __init__(self, strategy = "home", gamma = 0, delta = 1, V2G = True): 
          self.strategy = strategy 
          self.gamma = gamma 
          self.delta = delta
          self.V2G = V2G
    
    
      # load optimization problem from stratIdx to endIdx - 1 including
      def load(self, startIdx, endIdx):
    
          opt_probs = np.empty((endIdx,), dtype=np.object)
          
          if(self.strategy == "home"):
              probClass = OptProblem_ValleyFilling_Home
          else:# minEnergy, greedy
              probClass = OptProblem_ValleyFilling_Home
    
          if(startIdx == 0):
              
              # w.l.o.g. and for convenience, the aggregator is the 0th element
              problem = OptProblem_Aggregator_ValleyFilling()  
              opt_probs[0] = problem
        
              # Empirical [price/demand^2]     
              delta =  np.mean(problem.price)/(np.mean(problem.D) * (3600*1000) * deltaT )     
              probClass.delta = delta
              probClass.gamma = self.gamma
              probClass.alpha /= probClass.delta
        
              for i in xrange(1, endIdx):
                    problem = probClass(i, self.V2G)
                    opt_probs[i] = problem
              
          else:
              for i in xrange(endIdx):
                    problem = probClass(startidx + i, self.V2G)
                    opt_probs[i] = problem
              
          return opt_probs