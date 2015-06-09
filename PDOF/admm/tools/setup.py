# -*- coding: utf-8 -*-
"""
Created on Fri May 29 06:33:54 2015

@author: shterev
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

''''' to run

python setup.py build_ext --inplace

'''''

ev_home = "admm/tools/cvxgen_EV_home/"

extensions = [
    Extension("params", ["params.pyx"]
    #,
        #include_dirs = [ev_home],
        #libraries = [...],
        #library_dirs = [ev_home]
        ),
    # Everything but primes.pyx is included here.
    # Extension("*", [ev_home + "*.pyx"],
        #include_dirs = [...],
        #libraries = [...],
        #library_dirs = [...]),
    #)
]
setup(
    name = "PDOF",
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
)

