.. RBF documentation master file, created by
   sphinx-quickstart on Tue Oct 18 17:39:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RBF
***
Python package containing the tools necessary for radial basis 
function (RBF) applications.  Applications include 
interpolating/smoothing scattered data and solving PDEs over irregular 
domains. RBF is developed by Trever Hines (hinest@umich.edu), 
University of Michigan, and the source code for this project can be 
found `here <http://www.github.com/treverhines/RBF>`_

Features
--------
* Efficient functions to evaluate RBFs and their analytically derived 
  derivatives
* Regularized RBF interpolants (including smoothing splines) for
  noisy, scattered, data
* An algorithm for generating Radial Basis Function Finite Difference 
  (RBF-FD) weights
* RBF-FD Filtering for denoising **BIG**, scattered data
* Node and stencil generation algorithms for solving PDEs over
  irregular domains
* Halton sequence generator
* Computational geometry functions for 1, 2, and 3 spatial dimensions

Table of Contents
-----------------
.. toctree::
  :maxdepth: 2

  installation
  basis
  interpolate
  fd
  filter
  nodes
  stencil
  geometry
  halton

Quick Demo
----------

Smoothing Scattered Data
++++++++++++++++++++++++
.. literalinclude:: ../scripts/interpolate.a.py

The above code will produce this plot, which shows the observations as 
scatter points and the smoothed interpolant as the color field.

.. image:: ../figures/interpolate.a.png

Solving PDEs
++++++++++++
.. literalinclude:: ../scripts/basis.a.py

The above code will produce this plot, which shows the collocation 
nodes as black points and the interpolated solution as the color field.

.. image:: ../figures/basis.a.png

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
