# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:56:11 2015

@author: shterev
"""

from csolve_home cimport *
from admm.tools.params cimport *

from cython cimport view
cimport numpy as np
import numpy as np

cdef class TestSolver:
    
    cpdef test_ext(self)
    
    cdef void test(self)
        
    '''cdef double[:, :] use_solution(self, cs.Vars vars)'''
    
    cpdef printResult(self)
        
    cdef void load_default_data(self)