PDE
===
The package `rbf.pde` contains tools for solving partial differential
equations (PDEs) using the RBF and RBF-FD method. This includes
methods for generating RBF-FD weight matrices (`rbf.pde.fd`),
generating nodes within the domain (`rbf.pde.nodes`), and some basic
computational geometry functions for two and three-dimensions
(`rbf.pde.geometry`). This package was primarily inspired by [1].

.. toctree::
  :maxdepth: 2

  fd
  nodes
  knn
  geometry
  halton

References
----------
  [1] Fornberg, B. and N. Flyer. A Primer on Radial Basis 
  Functions with Applications to the Geosciences. SIAM, 2015.

