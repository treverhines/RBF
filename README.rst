RBF
***
Python package containing the tools necessary for radial basis 
function (RBF) applications.  Applications include 
interpolating/smoothing scattered data and solving PDEs over irregular
domains.  The complete documentation for this package can be found 
`here <http://rbf.readthedocs.io>`_.

Features
--------
* Efficient functions to evaluate RBFs and their analytically derived 
  derivatives
* Regularized RBF interpolants (including smoothing splines) for 
  noisy, scattered, data sets
* An abstraction for Gaussian processes. Gaussian processes are
  primarily used here for Gaussian process regression (GPR), which is 
  a nonparametric Bayesian interpolation/smoothing method.
* An algorithm for generating Radial Basis Function Finite Difference 
  (RBF-FD) weights
* RBF-FD Filtering for denoising large, scattered data sets
* Node and stencil generation algorithms for solving PDEs over
  irregular domains
* Halton sequence generator
* Computational geometry functions for 1, 2, and 3 spatial dimensions

Quick Demo
----------

Smoothing Scattered Data
++++++++++++++++++++++++
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
++++++++++++
.. code-block:: python

  ''' 
  In this example we solve the Poisson equation with a constant forcing 
  term using the spectral RBF method.
  '''
  import numpy as np
  from rbf.basis import phs3
  from rbf.domain import circle
  from rbf.nodes import menodes
  import matplotlib.pyplot as plt

  # define the problem domain
  vert = np.array([[0.762,0.057],[0.492,0.247],[0.225,0.06 ],[0.206,0.056],
                   [0.204,0.075],[0.292,0.398],[0.043,0.609],[0.036,0.624],
                   [0.052,0.629],[0.373,0.63 ],[0.479,0.953],[0.49 ,0.966],
                   [0.503,0.952],[0.611,0.629],[0.934,0.628],[0.95 ,0.622],
                   [0.941,0.607],[0.692,0.397],[0.781,0.072],[0.779,0.055]])

  smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],
                  [9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
                  [16,17],[17,18],[18,19],[19,0]])

  N = 500 # total number of nodes
  nodes,smpid = menodes(N,vert,smp) # generate nodes
  boundary, = (smpid>=0).nonzero() # identify boundary nodes
  interior, = (smpid==-1).nonzero() # identify interior nodes

  # create left-hand-side matrix and right-hand-side vector
  A = np.empty((N,N))
  A[interior]  = phs3(nodes[interior],nodes,diff=[2,0])
  A[interior] += phs3(nodes[interior],nodes,diff=[0,2])
  A[boundary,:] = phs3(nodes[boundary],nodes)
  d = np.empty(N)
  d[interior] = -100.0
  d[boundary] = 0.0

  # Solve the PDE
  coeff = np.linalg.solve(A,d) # solve for the RBF coefficients
  itp = menodes(10000,vert,smp)[0] # interpolation points
  soln = phs3(itp,nodes).dot(coeff) # evaluate at the interp points

  fig,ax = plt.subplots()
  p = ax.scatter(itp[:,0],itp[:,1],s=20,c=soln,edgecolor='none',cmap='viridis')
  ax.set_aspect('equal')
  ax.plot(nodes[:,0],nodes[:,1],'ko',markersize=4)
  ax.set_xlim((0.025,0.975))
  ax.set_ylim((0.03,0.98))
  plt.colorbar(p,ax=ax)
  plt.tight_layout()
  plt.savefig('../figures/basis.a.png')
  plt.show()


The above code will produce this plot, which shows the collocation
nodes as black points and the interpolated solution as the color field.

.. image:: docs/figures/basis.a.png


