#!/usr/bin/env python
if __name__ == '__main__':
  from numpy.distutils.core import setup
  from numpy.distutils.extension import Extension
  from Cython.Build import cythonize
  ext = []
  ext += [Extension(name='rbf.halton',sources=['rbf/halton.pyx'])]
  ext += [Extension(name='rbf.bspline',sources=['rbf/bspline.pyx'])]
  ext += [Extension(name='rbf.geometry',sources=['rbf/geometry.pyx'])]
  setup(name='RBF',
        version='0.1',
        description='package developed for a course on spectral methods',
        author='Trever Hines',
        author_email='treverhines@gmail.com',
        url='www.github.com/treverhines/RBF',
        packages=['rbf'],
        ext_modules=cythonize(ext),
        license='MIT')


