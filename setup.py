from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_args = dict(
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[]
    )

ext_modules = [
    Extension(name="dummy._code", sources=["src/dummy/_code.pyx"], **ext_args),
    ]

setup(ext_modules=cythonize(ext_modules))
