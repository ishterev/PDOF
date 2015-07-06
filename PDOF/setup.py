# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 02:55:32 2015

@author: shterev
"""

# To build, run:
# python setup.py build

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.command.build_clib import build_clib
import numpy

''''' to run:

first: to compile the C lib call python setup.py build
then: python setup.py build_ext --inplace

'''''

lib_solver_home = ('csolve_home', {'sources': ['admm/tools/cvxgen_EV_home/solver.c',
                                               'admm/tools/cvxgen_EV_home/matrix_support.c',
                                               'admm/tools/cvxgen_EV_home/ldl.c',
                                               'admm/tools/cvxgen_EV_home/testsolver.c',
                                               'admm/tools/cvxgen_EV_home/util.c']})

ext_modules = [
    
     
    
    Extension("admm.tools.params", ["admm/tools/params.pyx"]
    ,
        #include_dirs = [ev_home],
        #libraries = [...],
        #library_dirs = [ev_home]
        ),
    Extension("admm.tools.cvxgen_EV_home.test", ["admm/tools/cvxgen_EV_home/test.pyx"]
    ,
        #include_dirs = [ev_home],
        #libraries = ["solver"],
        #library_dirs = [ev_home]
    ),
        
    Extension("admm.tools.cvxgen_EV_home.test_home", ["admm/tools/cvxgen_EV_home/test_home.pyx"]
    ,
        #include_dirs = [ev_home],
        #libraries = ["solver"],
        #library_dirs = [ev_home]
        ), 
        
    Extension("admm.tools.cvxgen_EV_home.solve_home", ["admm/tools/cvxgen_EV_home/solve_home.pyx"]
    ,
        #include_dirs = [ev_home],
        #libraries = ["solver"],
        #library_dirs = [ev_home]
        ), 
        
    Extension("admm.EV_loader", ["admm/EV_loader.pyx"]
    ,
        #include_dirs = [ev_home],
        #libraries = ["solver"],
        #library_dirs = [ev_home]
        ), 
     #Everything but primes.pyx is included here.
     #Extension("*", [ev_home + "*.pyx"],
        #include_dirs = [...],
        #libraries = [...],
        #library_dirs = [...]),
    #)
]

setup(
name = 'PDOF',
libraries = [lib_solver_home],
cmdclass = {'build_clib': build_clib, 'build_ext': build_ext},
ext_modules = ext_modules,
include_dirs=[numpy.get_include()]
)

#setup(
 #   name = "Test app",
  #  ext_modules = cythonize(ext_modules),
    #libraries = [lib_solver_home],
    #include_dirs=[numpy.get_include()]
#)
