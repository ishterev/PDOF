# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 04:39:14 2015

@author: shterev
"""
'''
When f is the indicator function I and C is a closed nonempty convex set so that
IC(x) =  0 , if x ∈ C
      =  +∞ , if x ∈ C
      
the proximal operator of f reduces to Euclidean projection onto C.

In a canonical ADMM formulation we can move a constraint into the objective using an indicator
function. For example 

minimize Sum fi(xi) , subject to x1 = x2 = · · · = xN (global consensus)

becomes

minimize Sum fi(xi) + IC(x1, . . . , xN), 

where C is the consensus set C = {(x1, . . . , xN) | x1 = · · · = xN}.

In the notation of the general ADMM problem minimize f(x) + g(x), f is the
sum of the terms fi, while g is the indicator function of the consistency
constraint.
      
'''
import numpy as np

class IdentificatorFunctionConstraint:
    
    def project(self, z):
        pass
    
# subject to x1 = x2 = · · · = xN    
class IdentificatorFunctionConsensusConstraint(IdentificatorFunctionConstraint):
    
    def project(self, z):
        
        # assert z is not None
        # assert isinstance(z, np.ndarray)
        
        # projecting onto the consensus set is simple: we replace each zi with its average z_    
        z_ = np.mean(z, axis = 0) 
        shape = z.shape
        z = None # clear memory
        z = z_ * np.ones(shape)
        return z
        
        
# subject to x1 + · · · + xN = 0   
class IdentificatorFunctionEquilibriumConstraint(IdentificatorFunctionConstraint):
    
    def project(self, z):
    
        # assert z is not None
        # assert isinstance(z, np.ndarray)
    
        # assert len(z.shape) == 3 and z.shape[2] == 1 # == (N,m,1)
    
        # projecting onto C is simple de-meaning

        N = z.shape[0] 
        n = z.shape[2]
        
        z_ = np.zeros((n,1))       
        z_ = np.mean(z, axis = 0)
        z -= z_
        
        ''' alternatively
        for i in xrange(N): # z_ = np.mean(z, axis = 0)  
            z_ += z[i]  
            
        for i in xrange(n): # z_ = np.mean(z, axis = 0)  
            z_[i][0] /= N
        
        for i in xrange(N): # z_ = np.mean(z, axis = 0)  
            z[i] -= z_ 
        '''
                
        return z
    
    
    
    
    
    
    
    
    
    

