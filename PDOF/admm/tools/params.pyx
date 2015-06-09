# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:18:51 2015

@author: shterev
"""

cdef class params:

    
   def __cinit__(self, double xmax = 4, double xmin = -4, int discharge = 0):
       
       self.xmax = xmax
       
       if discharge > 0: 
            self.xmin=0
       else:
            self.xmin = xmin
        