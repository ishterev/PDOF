cimport admm.tools.params
from admm.tools.params cimport *

cimport numpy as np
import numpy as np

cdef class Solver:
    
    cdef double [:] A # 96x1
    cdef double [:] R # 1x1
    cdef double [:] d # 96x1
    
    # In this function, load all problem instance data.
    #          self.params.A[i] = ...;
    cdef void load_data(self, double [:, :] xold, double [:, :] u, double [:] xmean, params p) 

    cdef double[:] solve(self, double [:, :] xold, double [:, :] u, double [:] xmean, params p) 
        
    cpdef double[:] test(self)

    cdef void load_default_data(self)