RBF
***
Python package containing the tools necessary for radial basis
function (RBF) applications.  Applications include
interpolating/smoothing scattered data and solving PDEs over
complicated domains.  The complete documentation for this package 
can be found `here <http://rbf.readthedocs.io>`_.

Features
--------
* Efficient evaluation of RBFs and their analytically derived derivatives
* Regularized RBF interpolation for noisy, scattered, data
* Generation of radial basis function finite difference (RBF-FD)
  weights
* RBF-FD Filtering for denoising **BIG**, scattered data
* Node and stencil generation algorithms for solving PDEs over
  complicated domains
* Computational geometry functions for 1, 2, and 3 spatial dimensions

Quick Demo
----------

Smoothing Scattered Data
++++++++++++++++++++++++
.. literalinclude:: /docs/scripts/interpolate.a.py

The above code will produce this plot, which shows the observations as
scatter points and the smoothed interpolant as the color field.

.. image:: /docs/figures/interpolate.a.png

Solving PDEs
++++++++++++
.. literalinclude:: /docs/scripts/basis.a.py

The above code will produce this plot, which shows the collocation
nodes as black points and the interpolated solution as the color field.

.. image:: /docs/scripts/basis.a.png


