# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:56:11 2015

@author: shterev
"""

from csolve_home cimport *
from csolve_home cimport set_defaults

from admm.tools.params cimport *

from cython cimport view
cimport numpy as np
import numpy as np

cdef class Test:
    
    cdef Vars vars
    cdef Params params
    cdef Workspace work
    cdef Settings settings
   
    def say_hello_to(name):
       print("Hello")
       
       
    cpdef test(self):  
        #self.params.d[0] = 0.203191610298302;
       set_defaults()