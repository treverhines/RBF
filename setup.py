#!/usr/bin/env python
if __name__ == '__main__':
    from setuptools import setup
    from setuptools.extension import Extension
    from pathlib import Path
    import subprocess as sp
    import numpy as np
    import json
    import re

    # This package may not be distributed with `make_version.py` and
    # `make_cython_extensions.py`. In that case, the version file and
    # extensions should already exist.
    if Path('make_version.py').exists():
        sp.call(['python', 'make_version.py'])

    if Path('make_cython_extensions.py').exists():
        sp.call(['python', 'make_cython_extensions.py'])

    version_file = Path('rbf/_version.py')
    version_text = version_file.read_text()
    version_info = dict(
        re.findall('(__[A-Za-z_]+__)\s*=\s*"([^"]+)"', version_text)
        )

    ext = []
    ext += [Extension(name='rbf.poly', sources=['rbf/poly.c'])]
    ext += [Extension(name='rbf.sputils', sources=['rbf/sputils.c'])]
    ext += [Extension(name='rbf.pde.halton', sources=['rbf/pde/halton.c'])]
    ext += [Extension(name='rbf.pde.geometry', sources=['rbf/pde/geometry.c'])]
    ext += [Extension(name='rbf.pde.sampling', sources=['rbf/pde/sampling.c'])]

    with open('rbf/_rbf_ufuncs/metadata.json', 'r') as f:
        rbf_ufunc_metadata = json.load(f)

    for itm in rbf_ufunc_metadata:
        ext += [Extension(name=itm['module'], sources=itm['sources'], include_dirs=[np.get_include()])]

    setup(
        name='treverhines-rbf',
        version=version_info['__version__'],
        description='Package containing the tools necessary for radial basis '
                    'function (RBF) applications',
        author='Trever Hines',
        author_email='treverhines@gmail.com',
        url="https://www.github.com/treverhines/RBF",
        packages=['rbf', 'rbf.pde', 'rbf._rbf_ufuncs'],
        ext_modules=ext,
        include_package_data=True,
        licence='MIT',
        install_requires=[
            'numpy>=1.10',
            'scipy',
            'sympy',
            'cython',
            'rtree'
            ]
        )
