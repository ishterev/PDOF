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
from Cython.Build import cythonize
import os

ev_home = os.path.abspath(".")
print ev_home

libhello = ('cprog', {'sources': ['cprog.c']})

ext_modules = [
    
     #Extension("csolve_home", ["admm/tools/cvxgen_EV_home/solver.c"]
     #,
        #include_dirs = ["admm/tools/cvxgen_EV_home/"],
        #libraries = ["admm/tools/cvxgen_EV_home/solver.h"],
        #library_dirs = [ev_home]
       #),
    
    #Extension("cython_prog2", ["cython_prog2.pyx"]
    #    ,
     #   include_dirs = [ev_home],
        #libraries = ["solver"],
        #library_dirs = [ev_home]
      #  )
    #,
    
    #Extension("admm.tools.params", ["admm/tools/params.pyx"]
    #,
        #include_dirs = [ev_home],
        #libraries = [...],
        #library_dirs = [ev_home]
    #    ),
    Extension("admm.tools.cvxgen_EV_home.test", ["admm/tools/cvxgen_EV_home/test.pyx"]
    ,
        #include_dirs = [ev_home],
        #libraries = ["solver"],
        #library_dirs = [ev_home]
        ),
        
    ''' Extension("admm.tools.cvxgen_EV_home.test_home", 
              ["admm/tools/cvxgen_EV_home/test_home.pyx"]
    ,
        #include_dirs = [ev_home],
        #libraries = ["solver"],
        #library_dirs = [ev_home]
        ), '''
     #Everything but primes.pyx is included here.
     #Extension("*", [ev_home + "*.pyx"],
        #include_dirs = [...],
        #libraries = [...],
        #library_dirs = [...]),
    #)
]

setup(
name = 'Test app',
libraries = [libhello],
cmdclass = {'build_clib': build_clib, 'build_ext': build_ext},
ext_modules = ext_modules
include_dirs=[numpy.get_include()]
)

#setup(
 #   name = "Test app",
  #  ext_modules = cythonize(ext_modules),
    #libraries = [lib_solver_home],
    #include_dirs=[numpy.get_include()]
#)
