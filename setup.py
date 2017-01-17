#!/usr/bin/env python
if __name__ == '__main__':
  from numpy.distutils.core import setup
  from numpy.distutils.extension import Extension
  from Cython.Build import cythonize
  ext = []
  ext += [Extension(name='rbf.halton',sources=['rbf/halton.pyx'])]
  ext += [Extension(name='rbf.misc.bspline',sources=['rbf/misc/bspline.pyx'])]
  ext += [Extension(name='rbf.geometry',sources=['rbf/geometry.pyx'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'])]
  ext += [Extension(name='rbf.poly',sources=['rbf/poly.pyx'])]
  setup(name='RBF',
        version='1.0',
        description='Package containing the tools necessary for radial basis function (RBF) applications',
        author='Trever Hines',
        author_email='treverhines@gmail.com',
        url='www.github.com/treverhines/RBF',
        packages=['rbf','rbf.misc'],
        ext_modules=cythonize(ext),
        license='MIT')


