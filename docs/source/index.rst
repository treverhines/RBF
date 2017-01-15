.. RBF documentation master file, created by
   sphinx-quickstart on Tue Oct 18 17:39:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RBF
+++
Python package containing tools for radial basis function (RBF) 
applications.  Applications include interpolating/smoothing scattered 
data and solving PDEs over irregular domains. RBF is developed by 
Trever Hines (hinest@umich.edu), University of Michigan, and the 
source code for this project can be found `here 
<http://www.github.com/treverhines/RBF>`_.

Features
========
* Functions for evaluating RBFs and their exact derivatives.
* A class for RBF interpolation, which is used for interpolating and 
  smoothing scattered, noisy, N-dimensional data.
* An abstraction for Gaussian processes. Gaussian processes are 
  primarily used here for Gaussian process regression (GPR), which is 
  a nonparametric Bayesian interpolation/smoothing method.
* RBF-FD Filtering for denoising large, scattered data sets.
* An algorithm for generating Radial Basis Function Finite Difference
  (RBF-FD) weights. This is used for solving large scale PDEs over 
  irregular domains.
* A Node generation algorithm which can be used for solving PDEs with 
  the spectral RBF method or the RBF-FD method.
* Halton sequence generator.
* Computational geometry functions (e.g. point in polygon testing) for
  1, 2, and 3 spatial dimensions.

Table of Contents
=================
.. toctree::
  :maxdepth: 2

  installation
  basis
  interpolate
  gpr
  fd
  filter
  nodes
  stencil
  geometry
  halton

Quick Demo
==========

Smoothing Scattered Data
------------------------
.. literalinclude:: ../scripts/interpolate.a.py

The above code will produce this plot, which shows the observations as 
scatter points and the smoothed interpolant as the color field.

.. image:: ../figures/interpolate.a.png

Solving PDEs
------------
.. literalinclude:: ../scripts/basis.a.py

The above code will produce this plot, which shows the collocation 
nodes as black points and the interpolated solution as the color field.

.. image:: ../figures/basis.a.png

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
