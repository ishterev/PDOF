# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:18:51 2015

@author: shterev
"""

cdef class params:
    
   cdef double delta    # demand electricity price relationship  (ONLY FOR VALLEY FILLING)
   cdef int idx         # EV index
   cdef int chargeStrategy # Charging strategy
   cdef double gamma    #Trade-off parameter
   cdef double rho      #augement cost parameter
   cdef double alpha   # 0.05/3600 * 15*60 / delta#Battery depresiation cost [EUR/kWh] and transformed to [EUR/kW]
   cdef int discharge  # discharge allowed
   cdef double xmax   # Max charging power for greedy need to add some 1e-3 or someting for it to be feasible
   cdef double xmin   # Min charging power
