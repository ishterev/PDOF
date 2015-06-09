import sys
from distutils.core import setup
from distutils.command.build_clib import build_clib
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

''''' to run

python setup.py build_ext --inplace

'''''

ext_modules=[
    Extension("test_home", ["test_home.pyx"])
]


def main():
    setup(
        name = 'test_home',
        cmdclass = {'build_ext': build_ext},
        ext_modules = ext_modules,
        include_dirs=[numpy.get_include()]
    )

if __name__ == '__main__':
    main()