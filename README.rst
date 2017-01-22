RBF
+++
Python package containing tools for radial basis function (RBF) 
applications.  Applications include interpolating/smoothing scattered 
data and solving PDEs over irregular domains.  The complete 
documentation for this package can be found `here 
<http://rbf.readthedocs.io>`_.

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
* A node generation algorithm which can be used for solving PDEs with 
  the spectral RBF method or the RBF-FD method.
* Halton sequence generator.
* Computational geometry functions (e.g. point in polygon testing) for
  1, 2, and 3 spatial dimensions.

Quick Demo
==========
Smoothing Scattered Data
------------------------
.. code-block:: python

  ''' 
  In this example we generate synthetic scattered data with added noise 
  and then fit it with a smoothed interpolant. See rbf.filter for 
  smoothing large data sets.
  '''
  import numpy as np
  from rbf.interpolate import RBFInterpolant
  import matplotlib.pyplot as plt
  np.random.seed(1)

  # create noisy data
  x_obs = np.random.random((100,2)) # observation points
  u_obs = np.sin(2*np.pi*x_obs[:,0])*np.cos(2*np.pi*x_obs[:,1])
  u_obs += np.random.normal(0.0,0.2,100)

  # create smoothed interpolant
  I = RBFInterpolant(x_obs,u_obs,penalty=0.001)

  # create interpolation points
  x_itp = np.random.random((10000,2))
  u_itp = I(x_itp)

  plt.tripcolor(x_itp[:,0],x_itp[:,1],u_itp,vmin=-1.1,vmax=1.1,cmap='viridis')
  plt.scatter(x_obs[:,0],x_obs[:,1],s=100,c=u_obs,vmin=-1.1,vmax=1.1,cmap='viridis')
  plt.xlim((0.05,0.95))
  plt.ylim((0.05,0.95))
  plt.colorbar()
  plt.tight_layout()
  plt.savefig('../figures/interpolate.a.png')
  plt.show()


The above code will produce this plot, which shows the observations as
scatter points and the smoothed interpolant as the color field.

.. image:: docs/figures/interpolate.a.png

Solving PDEs
------------
.. code-block:: python

  ''' 
  In this example we solve the Poisson equation with fix boundary 
  conditions on an irregular domain.
  '''
  import numpy as np
  from rbf.basis import phs3
  from rbf.domain import logo
  from rbf.geometry import contains
  from rbf.nodes import menodes
  import matplotlib.pyplot as plt
  # Define the problem domain. This is done by specifying the vertices of the
  # domain, *vert*, and the vertex indices making up each segment, *smp*.
  vert,smp = logo()
  N = 500 # total number of nodes
  nodes,smpid = menodes(N,vert,smp) # generate nodes
  edge_idx, = ((smpid>=0) & (smpid<12)).nonzero() # identify edge nodes
  eye1_idx, = ((smpid>=12) & (smpid<44)).nonzero() # identify top eye nodes
  eye2_idx, = (smpid>=44).nonzero() # identify bottom eye nodes
  interior_idx, = (smpid==-1).nonzero() # identify interior nodes
  # create left hand side" matrix
  A = np.empty((N,N))
  A[interior_idx]  = phs3(nodes[interior_idx],nodes,diff=[2,0])
  A[interior_idx] += phs3(nodes[interior_idx],nodes,diff=[0,2])
  A[edge_idx] = phs3(nodes[edge_idx],nodes)
  A[eye1_idx] = phs3(nodes[eye1_idx],nodes)
  A[eye2_idx] = phs3(nodes[eye2_idx],nodes)
  # set "right hand side" boundary conditions
  d = np.zeros(N)
  d[eye1_idx] = -1.0
  d[eye2_idx] = 1.0
  # Solve the PDE
  coeff = np.linalg.solve(A,d) # solve for the RBF coefficients
  # interpolate the solution on a grid
  xg,yg = np.meshgrid(np.linspace(-0.6,1.6,500),np.linspace(-0.6,1.6,500))
  points = np.array([xg.flatten(),yg.flatten()]).T
  u = phs3(points,nodes).dot(coeff) # evaluate at the interp points
  u[~contains(points,vert,smp)] = np.nan # mask outside points
  ug = u.reshape((500,500)) # fold back into a grid
  # make a contour plot of the solution
  fig,ax = plt.subplots()
  p = ax.contourf(xg,yg,ug,cmap='viridis',vmin=-0.5,vmax=0.5)
  ax.plot(nodes[:,0],nodes[:,1],'ko',markersize=4)
  for s in smp:
    ax.plot(vert[s,0],vert[s,1],'k-',lw=2)

  ax.set_aspect('equal')
  ax.set_xlim((-0.6,1.6))
  ax.set_ylim((-0.6,1.6))
  fig.colorbar(p,ax=ax)
  fig.tight_layout()
  plt.savefig('../figures/basis.a.png')
  plt.show()


The above code will produce this plot, which shows the collocation
nodes as black points and the interpolated solution as the color field.

.. image:: docs/figures/basis.a.png


