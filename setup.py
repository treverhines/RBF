#!/usr/bin/env python
if __name__ == '__main__':
  from setuptools import setup
  from setuptools.extension import Extension
  from Cython.Build import cythonize
  import versioneer
  ext = []
  ext += [Extension(name='rbf.poly',
                    sources=['rbf/poly.pyx'])]
  ext += [Extension(name='rbf.sputils',
                    sources=['rbf/sputils.pyx'])]
  ext += [Extension(name='rbf.pde.halton',
                    sources=['rbf/pde/halton.pyx'])]
  ext += [Extension(name='rbf.pde.geometry',
                    sources=['rbf/pde/geometry.pyx'])]
  ext += [Extension(name='rbf.pde.sampling',
                    sources=['rbf/pde/sampling.pyx'])]
  setup(name='RBF',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='Package containing the tools necessary for radial basis '
                    'function (RBF) applications',
        author='Trever Hines',
        author_email='treverhines@gmail.com',
        url='www.github.com/treverhines/RBF',
        packages=['rbf', 'rbf.pde'],
        ext_modules=cythonize(ext),
        license='MIT')


