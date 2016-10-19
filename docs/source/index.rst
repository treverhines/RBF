.. RBF documentation master file, created by
   sphinx-quickstart on Tue Oct 18 17:39:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RBF
***
Python package containing the tools necessary for radial basis 
function (RBF) applications.  Applications include 
interpolating/smoothing scattered data and solving PDEs over 
complicated domains. 


Features
--------
* Efficient evaluation of RBFs and their analytically derived derivatives
* Regularized RBF interpolation for noisy, scattered, data
* Generation of radial basis function finite difference (RBF-FD) 
  weights
* RBF-FD Filtering for denoising **LARGE**, scattered data
* Node and stencil generation algorithms for solving PDEs over 
  complicated domains
* Computational geometry functions for 1, 2, and 3 spatial dimensions

Table of Contents
-----------------
.. toctree::
   :maxdepth: 2

   basis
   fd
   geometry

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
