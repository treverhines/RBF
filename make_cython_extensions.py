#!/usr/bin/env python
if __name__ == '__main__':
    from setuptools.extension import Extension
    from Cython.Build import cythonize

    cy_ext = []
    cy_ext += [Extension(name='rbf.poly', sources=['rbf/poly.pyx'])]
    cy_ext += [Extension(name='rbf.sputils', sources=['rbf/sputils.pyx'])]
    cy_ext += [Extension(name='rbf.pde.halton', sources=['rbf/pde/halton.pyx'])]
    cy_ext += [Extension(name='rbf.pde.geometry', sources=['rbf/pde/geometry.pyx'])]
    cy_ext += [Extension(name='rbf.pde.sampling', sources=['rbf/pde/sampling.pyx'])]
    ext = cythonize(cy_ext)
    for itm in ext:
        print('name=%s, sources=%s' % (itm.name, itm.sources))
