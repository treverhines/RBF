# RBF
Python package containing the tools necessary for radial basis 
function (RBF) applications.  Applications include 
interpolating/smoothing scattered data and solving PDEs over 
complicated domains.


## Features
* Efficient evaluation of RBFs and their analytically derived spatial 
derivatives.  This package allows for unlimited spatial dimensions and 
arbitrary spatial derivatives. 
* Regularized RBF interpolation, which can fit a smoothed interpolant 
to noisy data
* Generation of radial basis function finite difference (RBF-FD) 
weights, which are used to estimate derivatives of scattered data
* Efficient generation of RBF-FD stencils which can be given 
constraints to not cross a user defined boundary. This is useful if 
the user does not want to estimate a derivative over a known 
discontinuity.
* computational geometry functions for 1, 2, and 3 spatial dimensions. 
Among these functions is a point in polygon/polyhedra test
* Halton sequence generator
* Node generation with a minimum energy algorithm.  This is used for 
solving PDEs with the spectral RBF method or the RBF-FD method


## Table of Contents
1. [Installation](#installation)
2. [Logging](#logging)
3. [Usage](#usage)
  1. [Basis](#basis)
  2. [Interpolation](#interpolation)
  3. [Solving PDEs: Spectral Method](#solving-pdes-spectral-method)
  3. [Solving PDEs: RBF-FD Method](#solving-pdes-rbf-fd-method)
4. [To Do](#to-do)

## Installation
RBF requires the following python packages: numpy, scipy, sympy, 
matplotlib, and cython.  These dependencies should be satisfied with 
just the base Anaconda python package 
(https://www.continuum.io/downloads)

download the RBF package
```
$ git clone http://github.com/treverhines/RBF.git 
```
compile and install
```
$ cd RBF
$ python setup.py install
```
test that everything works
```
$ cd test
$ python test_all.py
```


## Logging
This package uses loggers for some of the more time intensive 
processes.  To print the logged content to stdout, start your python 
script with
```python
import logging
logging.basicConfig(level=logging.INFO)
```


## Usage
The following is a quick introduction to some of the features of this 
package.  More examples can be found in the `demo` directory.

### Basis
The linchpin of this module is the RBF class, which is used to 
evaluate an RBF and its derivatives.  An RBF is instantiated using a 
symbolic sympy expression.  Evaluating the RBFs is done by calling the 
RBF instance where the user supplies the evaluation points, the RBF 
centers, and the desired derivate (if any).  When called, an 
analytical derivative of the symbolic expression is evaluated and then 
the function is compiled into cython code.  This compiled code is 
saved and reused when the RBF is called again with the same derivative 
specification.
  
Here is an example where an RBF is instantiated and then the RBF and 
its first derivative are evaluated. See the help documentation for 
`rbf.basis.RBF` for more information on the arguments
```python
import numpy as np
import matplotlib.pyplot as plt
import rbf.basis

R = rbf.basis.get_R()
expr = 1/(1 + R**2) # inverse quadratic
iq = rbf.basis.RBF(expr)

# create RBF centers
c = np.array([-2.0,0.0,2.0])[:,None]
# create evaluation points
x = np.linspace(-5.0,5.0,1000)[:,None]

# evaluate each RBF at x
soln = iq(x,c)
# evaluate the first derivative of each RBF at x
soln_diff = iq(x,c,diff=(1,))
```
plotting the results
```python
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln)
ax.set_xlim((-5.0,5.0))
ax.set_title('inverse quadratic')
ax.grid()

fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln_diff)
ax.set_xlim((-5.0,5.0))
ax.set_title('inverse quadratic first derivative')
ax.grid()
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/basis/figures/basis_1.png "basis_1")
![alt text](https://github.com/treverhines/RBF/blob/master/demo/basis/figures/basis_2.png "basis_2")

The user does not need to worry about instantiation of an RBF class 
because many of the commonly used RBFs are already instantiated and 
can be called using function in the `rbf.basis` module. The available 
functions are
* ga : gaussian, exp(-(EPS\*R)^2)
* iq : inverse quadratic, 1/(1+(EPS\*R^2)
* mq : multiquadratic, sqrt(1 + (EPS\*R)^2)
* imq : inverse multiquadratic, 1/sqrt(1 + (EPS\*R)^2)
* phs{1,3,5,7} : odd order polyharmonic splines, (EPS\*R)^{1,3,5,7}
* phs{2,4,6,8} : even order polyharmonic splines, log(EPS\*R)(EPS\*R)^{2,4,6,8}  

EPS is a scaling factor which can be obtained for defining your own 
RBFs by calling `rbf.basis.get_EPS()`. When evaluating the RBF, you 
can set the scaling factor with the `eps` key word argument.  For 
interpolation problems or when trying to solve a PDE, EPS is often 
treated as a free parameter that needs to be optimized. This can 
become an intractible burden for large problems. When using odd order 
polyharmonic splines, which are scale-invariant, the shape parameter 
does not need to be optimized. Odd order polyharmonic splines 
generally perform well for interpolation and solving PDEs.

### Interpolation
Radial basis functions are most commonly used for interpolating 
scattered data in multidimensional space, but for simplicity we start 
with a one-dimensional demonstration.  Creating a simple RBF 
interpolant is straight forward with an RBF instance
```python
from rbf.basis import phs3

x = np.linspace(-np.pi,np.pi,5)[:,None] # observation points
u = np.sin(x[:,0]) # values at the observation points
xitp = np.linspace(-4.0,4.0,1000)[:,None] # interpolation points
A = phs3(x,x) # coefficient matrix
coeff = np.linalg.solve(A,u) # find the coefficients for each RBF

# Evaluates each of the RBFs at the interpolation points
uitp = phs3(xitp,x).dot(coeff) 
```
Alternatively, we can arrive at the same solution with the 
`RBFInterpolant` class
```python
from rbf.interpolate import RBFInterpolant

I = RBFInterpolant(x,u,order=-1)
uitp = I(xitp)
```
The `order` key word argument specifies the order of the polynomial 
which is added to the interpolant for improved accuracy.  By setting 
it to -1, we indicate that we do not want to add any polynomial to our 
interpolant. By default, the RBFInterpolant adds a constant and linear 
term (i.e. order=1). The default RBF used by `RBFInterpolant` is 
`phs3`.  Using the default arguments we see that our interpolant is a 
better prediction of the true signal, sin(x).
```python
I = RBFInterpolant(x,u)
uitp2 = I(xitp)
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/interpolate/figures/interp1d.png "demo_interpolate_1d")

In the next example we fit a smoothed interpolant to 100 noisy samples 
of a two-dimensional function. The smoothness is controlled with the 
`penalty` argument. To further show off the features of 
`RBFInterpolant` we show that we can easily differentiate the smoothed 
interpolant. 
```
x = np.random.random((100,2)) # observation points

# find the function value at the observation points
u = np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
u += np.random.normal(0.0,0.1,100)

# interpolation points
a = np.linspace(0,1,100)
x1itp,x2itp = np.meshgrid(a,a)
xitp = np.array([x1itp.ravel(),x2itp.ravel()]).T

# form interpolant
I = RBFInterpolant(x,u,penalty=0.001)

# evaluate the interpolant
uitp = I(xitp)

# evaluate the x derivative of the interpolant
dxitp = I(xitp,diff=(1,0))
```
In the below figure we compare the smoothed interpolant and its 
derivative the the true underlying signal.

![alt text](https://github.com/treverhines/RBF/blob/master/demo/interpolate/figures/interp2d.png "interp2d")


### Solving PDEs: Spectral Method
We can numerically solve PDEs over an arbitrary N-dimensional domain 
with RBFs.  Unlike finite element methods or traditional finite 
difference methods which require a mesh (nodes with known 
connectivity), the RBF method just needs to know the nodes. This makes 
it easier to discretize a complicated domain and gives the user more 
control over how that discretization is done.

#### Node generation
The `rbf.nodes` module provides a function for node generation over an 
arbitary 1, 2, or 3 dimensional closed domain and also allows for 
variable node density.  Throughout this package domains are defined 
using simplicial complexes, which are a collection of simplices 
(points, line segments, and triangles).  A simplicial complex is 
defined with two arrays, one specificing the locations of vertices and 
the other specifying the vertex indices which make up each simplex.  
For example a unit square can be described as
```python
vert = [[0.0,0.0],
        [1.0,0.0],
        [1.0,1.0]
        [0.0,1.0]]
smp = [[0,1],
       [1,2],
       [2,3],
       [3,0]]
```             
where each row in smp defines the vertices in a simplex making up the 
unit square. The vertices and simplices for some simple domains can be 
generated from the functions in `rbf.domain`.

We now generate 1000 nodes which are quasi-uniformly spaced within the 
unit square. This is done with a minimum energy algorithm. See 
`rbf.nodes.menodes` for a detailed description of the arguments and 
the algorithm.
```python
from rbf.nodes import menodes

# number of nodes
N = 1000

# generate nodes. *nodes* is a (N,2) array and *smpid* is a (N,) 
# identifying the simplex, if any, that each node is attached to
nodes,smpid = menodes(N,vert,smp)

boundary, = np.nonzero(smpid>=0)
interior, = np.nonzero(smpid==-1)
```
plotting `interior` nodes as black and `boundary` nodes as blue we get 
this figure

![alt text](https://github.com/treverhines/RBF/blob/master/demo/nodes/figures/square.png "square")

In this next example, we create a more complicated domain and have a 
node density that corresponds with the image Lenna.png (located in 
`rbf/demo`)
```python
from PIL import Image

# define the domain
t = np.linspace(0,2*np.pi,201)
t = t[:-1]
radius = 0.45*(0.1*np.sin(10*t) + 1.0)
vert = np.array([0.5+radius*np.cos(t),0.5+radius*np.sin(t)]).T
smp = np.array([np.arange(200),np.roll(np.arange(200),-1)]).T
                 
N = 30000

# make gray scale image
img = Image.open('Lenna.png')
imga = np.array(img,dtype=float)/256.0
gray = np.linalg.norm(imga,axis=-1)
# normalize so that the max value is 1
gray = gray/gray.max()

# define the node density function
def rho(p):
  # x and y are mapped to integers between 0 and 512
  p = p*512
  p = np.array(p,dtype=int)
  return 1.0001 - gray[511-p[:,1],p[:,0]]


nodes,smpid = menodes(N,vert,smp,rho=rho)
interior, = np.nonzero(smpid==-1)
boundary, = np.nonzero(smpid>=0)
```
again plotting the `interior` nodes as black and `boundary` nodes as 
blue

![alt text](https://github.com/treverhines/RBF/blob/master/demo/nodes/figures/lenna.png "lenna")

#### Laplace's equation on a circle
Here we are solving Laplace's equation over a unit circle, where the 
boundaries are fixed at zero and there is an applied forcing term. The 
solution to this problem is (1-r)\*sin(x)\*cos(y). The forcing 
term needed to produce this solution is defined below. This 
demonstration can also be found in demo/pde/spectral/2d/laplacian.py.
```python
import rbf.domain

def forcing(x,y):
  return ((2*x**2*np.sin(x)*np.cos(y) - 
           2*x*np.cos(x)*np.cos(y) + 
           2*y**2*np.sin(x)*np.cos(y) + 
           2*y*np.sin(x)*np.sin(y) - 
           2*np.sqrt(x**2 + y**2)*np.sin(x)*np.cos(y) - 
           np.sin(x)*np.cos(y))/np.sqrt(x**2 + y**2))

# total number of nodes
N = 100

# define a circular domain
vert,smp = rbf.domain.circle()

nodes,smpid = menodes(N,vert,smp)
boundary, = (smpid>=0).nonzero()
interior, = (smpid==-1).nonzero()

# create the left-hand-side matrix which is the Laplacian of the basis 
# function for interior nodes and the undifferentiated basis functions 
# for the boundary nodes
A = np.zeros((N,N))
A[interior]  = phs3(nodes[interior],nodes,diff=[2,0]) 
A[interior] += phs3(nodes[interior],nodes,diff=[0,2]) 
A[boundary]  = phs3(nodes[boundary],nodes,diff=[0,0])

# create the right-hand-side vector, consisting of the forcing term 
# for the interior nodes and zeros for the boundary nodes
d = np.zeros(N)
d[interior] = forcing(nodes[interior,0],nodes[interior,1]) 

# find the RBF coefficients that solve the PDE
coeff = np.linalg.solve(A,d)
```
Now that we have the coefficients for the basis functions we can 
evaluate the interpolant and view the solution. 

![alt text](https://github.com/treverhines/RBF/blob/master/demo/pde/spectral/2d/figures/demo_spectral_laplacian.png "demo_spectral_laplacian")


### Solving PDEs: RBF-FD Method
The radial basis function generated finite difference (RBF-FD) method 
is a relatively new method for solving PDEs.  The RBF-FD method allows 
one to approximate a derivative as a weighted sum of function 
realizations at N neighboring locations, where the locations can be 
randomly distributed.  Once the weights have been computed, the method 
is effectively identical to solving a PDE with a traditional finite 
difference method.  This package offers two functions for computing 
the RBF-FD weights, `rbf.fd.weights` and `rbf.fd.weight_matrix`. The 
latter function allows the user to solve a PDE with almost the exact 
same procedure as for the spectral RBF method (see 
`rbf/demo/pde/fd/2d/laplacian.py`).

For the function `rbf.fd.weight_matrix`, the stencil generation is 
done under the hood. By default the stencils are just a collection of 
nearest neighbors which are efficiently found with 
`scipy.spatial.cKDTree`. However, a nearest neighbor stencil may not 
be appropriate for some problems.  For example you may have a domain 
with edges that *nearly* touch and you do not want the PDE to be 
enforced across that boundary. The function 
`rbf.stencil.stencil_network` creates nearest neighbor stencils but it 
does not allow stencils to reach across boundaries. This is 
effectively done by redefining the distance norm so that if a line 
segment connecting two points intersects a boundary then they are 
infinitely far away.  This function then makes it possible to solve 
problems like this

![alt text](https://github.com/treverhines/RBF/blob/master/demo/pde/fd/2d/figures/annulus.png "demo_fd_annulus")

The above plot is showing the solution to Laplace's equation on a slit 
annulus. The edges are free surfaces except for the top and bottom of 
the slit, which are fixed at 1 and -1.  The code which generated the 
above script can be found in `rbf/demo/pde/fd/2d/annulus.py`.

RBFs seem to have a hard time handling free surface boundary 
conditions. In order to get a stable solution it is often necessary to 
add ghost nodes. Ghost nodes are additional nodes placed outside the 
boundary. Rather than enforcing the PDE at the ghost nodes, the added 
rows in the stiffness matrix are used to enforce the PDE at the 
boundary nodes.  A ghost node demonstration can be found in 
`rbf/demo/pde/fd/2d/ghosts.py`. The below figure shows the solution to 
the same PDE as above but with the addition of ghost nodes

![alt text](https://github.com/treverhines/RBF/blob/master/demo/pde/fd/2d/figures/ghosts.png "demo_fd_annulus")


### To Do
This package contains more features but they have not yet been 
included in this help documentation. They include
* generation of RBF-FD stencils (module: rbf.stencil)
* generation of RBF-FD weights (module: rbf.fd)
* computational geometry (module: rbf.geometry)
* generation of halton sequences (module: rbf.halton)
* Monte-Carlo integration (module: rbf.integrate)
* generation of B-spline basis functions (module: rbf.bspline) 
* Smoothing and differentiating large, sparse, data sets (My latest 
project!)

See the documentation within the modules for help on using these 
features

