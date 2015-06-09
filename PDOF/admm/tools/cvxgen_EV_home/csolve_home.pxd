# -*- coding: utf-8 -*-
"""
Created on Fri May 22 04:21:26 2015

@author: shterev
"""

cdef extern from "solver.h":
    
   cdef struct Params_t:      
        double d[96]
        double Aeq[96]
        double beq[1]
        double lb[96]
        double ub[96]
   ctypedef Params_t Params
   
   
   cdef struct Vars_t:
        double *x # 96 rows.       
   ctypedef Vars_t Vars


   cdef struct Workspace_t:    
        double h[192]        
        double s_inv[192]        
        double s_inv_z[192]        
        double b[1]        
        double q[96]        
        double rhs[481]        
        double x[481]        
        double *s        
        double *z        
        double *y        
        double lhs_aff[481]        
        double lhs_cc[481]        
        double buffer[481]        
        double buffer2[481]        

        double KKT[960]        
        double L[480]        
        double d[481]        
        double v[481]        
        double d_inv[481]        

        double gap        
        double optval        

        double ineq_resid_squared        
        double eq_resid_squared        

        double block_33[1]        

        # Pre-op symbols.
        double quad_253366050816[1]        

        int converged         
   ctypedef  Workspace_t Workspace

   cdef struct Settings_t:
        double resid_tol        
        double eps        
        int max_iters        
        int refine_steps        

        int better_start        
        # Better start obviates the need for s_init and z_init.
        double s_init        
        double z_init        

        int verbose        
        #Show extra details of the iterative refinement steps.
        int verbose_refinement        
        int debug        

        #For regularization. Minimum value of abs(D_ii) in the kkt D factor.
        double kkt_reg        
   ctypedef Settings_t Settings
   
   Vars vars;
   Params params;
   Workspace work;
   Settings settings;
      
   
   #Vars vars
   #Params params
   #Workspace work
   #Settings settings
   
   long solve()
   void set_defaults()
   void setup_indexing()