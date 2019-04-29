Installation
============
`RBF` requires the following python packages: `numpy`, `scipy`,
`sympy`, `cython`, and `rtree`.  It should be straight-forward to find
and install these packages with either `pip` or `conda`.

download the RBF package

.. code-block:: bash

  $ git clone http://github.com/treverhines/RBF.git

compile and install

.. code-block:: bash

  $ cd RBF
  $ python setup.py install
  
test that everything works

.. code-block:: bash

  $ cd test
  $ python -m unittest discover

