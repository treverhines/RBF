Installation
============
RBF requires the following packages: numpy, scipy, sympy, cython, and rtree.
These dependencies should all be installable with conda or pip.

Download the RBF package

.. code-block:: bash

  $ git clone http://github.com/treverhines/RBF.git

Compile and install

.. code-block:: bash

  $ cd RBF
  $ python setup.py install

Test that everything works

.. code-block:: bash

  $ cd test
  $ python -m unittest discover
