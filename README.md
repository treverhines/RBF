# RBF
Package containing the tools necessary for radial basis function (RBF) applications

## Features
* Efficient evaluation of RBFs and their analytically derived spatial derivatives.  This package allows for unlimited spatial dimensions and arbitrarily spatial derivatives   
* Regularized RBF interpolation, which can fit a smoothed interpolant to noisy data   
* Generation of radial basis function finite difference (RBF-FD) weights, which are used to estimate derivatives of scattered data
* Efficient generation of RBF-FD stencils which can be given constraints to not cross a user defined boundary. This is useful if the user does not want to estimate a derivative over a known discontinuity.  
* computational geometry functions for 1, 2, and 3 spatial dimensions. Among these functions is a point in polygon/polyhedra test
* Halton sequence generator
* Node generation with a minmimum energy algorithm.  This is used for solving PDEs with the spectral RBF method or the RBF-FD method
* functions for Monte-Carlo integration and recursive Monte-Carlo integration over an polygonal/polyhedral domain

## Usage
The following walks through the examples in `RBF/demo`

## Basis
The linchpin of this module is the RBF class, which is used to evaluate an RBF and its derivatives.  An RBF is instantiated using a symbolic sympy expression.  Evaluating the RBFs is done by calling the RBF instance where the user supplies the evaluation points, the RBF centers, and the desired derivate (if any).  When called, an analytical derivative of the symbolic expression is evaluated and then the function is compiled into cython code.  This compiled code is saved and reused when the RBF is called again with the same derivative specification.   
  
Here is an example where an RBF is instantiated and then the RBF and its first derivative are evaluated. See the help documentation for rbf.basis.RBF for more information on the arguments
```
import numpy as np
import matplotlib.pyplot as plt
import rbf.basis

# demonstrate instantiation and evaluation of RBF
R = rbf.basis.get_R()
expr = 1/(1 + R**2) # inverse quadratic
iq = rbf.basis.RBF(expr)

# create RBF centers
# indexing with 'None' changes it from a (N,) array to a (N,1) array
c = np.array([-2.0,0.0,2.0])[:,None]
# create evaluation points
x = np.linspace(-5.0,5.0,1000)[:,None]

# evaluate each RBF at x
soln = iq(x,c)
# evaluate the first derivative of each RBF at x
soln_diff = iq(x,c,diff=(1,))
```
plot the resuts
```
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln)
ax.set_xlim((-5.0,5.0))
ax.set_title('inverse quadratic')
ax.grid()
fig.tight_layout()
plt.savefig('figures/demo_basis_1.png')

fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln_diff)
ax.set_xlim((-5.0,5.0))
ax.set_title('inverse quadratic first derivative')
ax.grid()
fig.tight_layout()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_basis_1.png "demo_basis_1")
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_basis_2.png "demo_basis_2")

The user does not need to worry about instantiation of an RBF class because many of the commonly used RBFs are already instantiated and can be called using function in the rbf.basis module.  The available functions are
* ga : gaussian, exp(-(EPS\*R)^2)
* iq : inverse quadratic, 1/(1+(EPS\*R)^2)
* mq : multiquadratic, sqrt(1 + (EPS\*R)^2)
* imq : inverse multiquadratic, 1/sqrt(1 + (EPS\*R)^2)
* phs{1,3,5,7} : odd order polyharmonic splines, (EPS\*R)^{1,3,5,7}
* phs{2,4,6,8} : even order polyharmonic splines, log(EPS\*R)(EPS\*R)^{2,4,6,8}  

EPS is a scaling factor which can be obtained for defining your own RBFs by calling `rbf.basis.get_EPS()`. When evaluating the RBF, you can set the scaling factor with the `eps` key word argument.  For interpolation problems or when trying to solve a PDE, EPS is often treated as a free parameter that needs to be optimized. This can become an intractible burden for large problems.  When using odd order polyharmonic splines, which are scale-invariant, the shape parameter does not need to be optimized. Odd order polyharmonic splines generally perform well for interpolation and solving PDEs.     

## Interpolation
### 1-D interpolation
Creating a simple RBF interpolant is straight forward with an RBF instance
```
# import a prebuilt RBF function
from rbf.basis import phs3

# create 5 observation points
x = np.linspace(-np.pi,np.pi,5)[:,None]

# find the function value at the observation points
u = np.sin(x[:,0])

# create interpolation points
xitp = np.linspace(-4.0,4.0,1000)[:,None]

# create the coefficient matrix, where each observation point point is 
# an RBF center and each RBF is evaluated at the observation points
A = phs3(x,x)

# find the coefficients for each RBF
coeff = np.linalg.solve(A,u)

# create the interpolation matrix which evaluates each of the 5 RBFs 
# at the interpolation points
Aitp = rbf.basis.phs3(xitp,x)

# evaluate the interpolant at the interpolation points
uitp1 = Aitp.dot(coeff)
```
You can also interpolate using the RBFInterpolant class
```
from rbf.interpolate import RBFInterpolant

# This command will produce an identical interpolant to the one 
# created above 
# I = RBFInterpolant(x,u,basis=phs3,order=-1)

# The default values for basis and order are phs3 and 0, where the 
# latter means that a constant term is added to the interpolation 
# function. This tends to produce much better results
I = RBFInterpolant(x,u)
uitp2 = I(xitp)
```
plot the results from the above two code blocks
```
# plot the results
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x[:,0],u,'ko')
ax.plot(xitp[:,0],uitp1,'r-')
ax.plot(xitp[:,0],uitp2,'b-')
ax.plot(xitp[:,0],np.sin(xitp[:,0]),'k--')
ax.legend(['observation points','interpolant 1','interpolant 2','true solution'$
ax.set_title('third-order polyharmonic spline interpolation')
fig.tight_layout()
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_interpolate_1d.png "demo_interpolate_1d")

### 2-D interpolation
Here I provide an example for 2-D interpolation and also I demonstrate how to differentiate the interpolant
```np.random.seed(1)

# create 20 2-D observation points
x = np.random.random((100,2))

# find the function value at the observation points
u = np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])

# create interpolation points
a = np.linspace(0,1,100)
x1itp,x2itp = np.meshgrid(a,a)
xitp = np.array([x1itp.ravel(),x2itp.ravel()]).T

# form interpolant
I = RBFInterpolant(x,u)

# evaluate the interpolant
uitp = I(xitp)
# evaluate the x derivative of the interpolant
dxitp = I(xitp,diff=(1,0))

# find the true values
utrue = np.sin(2*np.pi*xitp[:,0])*np.cos(2*np.pi*xitp[:,1])
dxtrue = 2*np.pi*np.cos(2*np.pi*xitp[:,0])*np.cos(2*np.pi*xitp[:,1])
```
plot the results
```
fig,ax = plt.subplots(2,2)
p = ax[0,0].tripcolor(xitp[:,0],xitp[:,1],uitp)
ax[0,0].scatter(x[:,0],x[:,1],c=u,s=100,clim=p.get_clim())
fig.colorbar(p,ax=ax[0,0])
p = ax[0,1].tripcolor(xitp[:,0],xitp[:,1],dxitp)
fig.colorbar(p,ax=ax[0,1])
p = ax[1,0].tripcolor(xitp[:,0],xitp[:,1],utrue)
fig.colorbar(p,ax=ax[1,0])
p = ax[1,1].tripcolor(xitp[:,0],xitp[:,1],dxtrue)
fig.colorbar(p,ax=ax[1,1])

ax[0,0].set_aspect('equal')
ax[1,0].set_aspect('equal')
ax[0,1].set_aspect('equal')
ax[1,1].set_aspect('equal')
ax[0][0].set_xlim((0,1))
ax[0][0].set_ylim((0,1))
ax[0][0].set_title('RBF interpolant')
ax[1][0].set_title('true solution')
ax[0][1].set_title('interpolant x derivative')
ax[1][1].set_title('true x derivative')
fig.tight_layout()
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_interpolate_2d.png "demo_interpolate_2d")

## Solving PDEs with the spectral RBF method
### Node generation
We can numerically solve PDEs over an arbitrary N-dimensional domain with RBFs.  Unlike finite element methods or traditional finite difference methods which require a mesh (nodes with known connectivity), the RBF method just needs to know the nodes. This makes it easier to discretize a complicated domain and gives the user more control over how that discretization is done.

The `rbf.nodes` module provides a function for node generation over an arbitary 1, 2, or 3 dimensional closed domain and also allows for variable node density.  Throughout this package domains are defined using simplicial complexes, which are a collection of simplices (points, line segments, and triangles).  A simplicial complex is defined with two arrays, one specificing the locations of vertices and the other specifying the vertex indices which make up each simplex.  For example a unit square can be described as
```
vert = [[0.0,0.0],
        [1.0,0.0],
        [1.0,1.0]
        [0.0,1.0]]
smp = [[0,1],
       [1,2],
       [2,3],
       [3,0]]
```             
where each row in smp defines the vertices in a simplex making up the unit square.

We now generate 1000 nodes which are quasi-uniformly spaced within the unit square. This is done with a minimum energy algorithm. See `rbf.nodes.make_nodes` for a detailed description of the arguments and the algorithm.
```
from rbf.nodes import make_nodes

# number of nodes
N = 1000

# number of iterations for the node generation algorithm
itr = 100

# step size scaling factor. default is 0.1. smaller values create more 
# uniform spacing after sufficiently many iterations
delta = 0.1

# generate nodes. nodes1 is a (N,2) array and smpid is a (N,) 
# identifying the simplex, if any, that each node is attached to
nodes1,smpid1 = make_nodes(N,vert,smp,itr=itr,delta=delta)
```
plot the results
```
fig,ax = plt.subplots()
# plot interior nodes
ax.plot(nodes1[smpid1==-1,0],nodes1[smpid1==-1,1],'ko')
# plot boundary nodes
ax.plot(nodes1[smpid1>=0,0],nodes1[smpid1>=0,1],'bo')
ax.set_aspect('equal')
ax.set_xlim((-0.1,1.1))
ax.set_ylim((-0.1,1.1))
fig.tight_layout()
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_nodes_1.png "demo_nodes_1")

In this next example, we create a more complicated domain and have a node density that corresponds with the image Lenna.png (located in `rbf/demo`)
```
from PIL import Image

# define more complicated domain
t = np.linspace(0,2*np.pi,201)
t = t[:-1]
radius = 0.45*(0.1*np.sin(10*t) + 1.0)
vert = np.array([0.5+radius*np.cos(t),
                 0.5+radius*np.sin(t)]).T
smp = np.array([np.arange(200),np.roll(np.arange(200),-1)]).T

N = 30000
itr = 20
delta = 0.1

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

nodes2,smpid2 = make_nodes(N,vert,smp,rho=rho,itr=itr,delta=delta)
```
plot the results
```
fig,ax = plt.subplots()
# plot interior nodes
ax.plot(nodes2[smpid2==-1,0],nodes2[smpid2==-1,1],'k.',markersize=2.0)
# plot boundary nodes
ax.plot(nodes2[smpid2>=0,0],nodes2[smpid2>=0,1],'b.',markersize=2.0)
ax.set_aspect('equal')
fig.tight_layout()
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_nodes_2.png "demo_nodes_2")

### Laplacian on a circle
Here we are solving the the Laplacian equation over a unit circle, where the boundaries are fixed at zero and there is an applied forcing term.  The solution to this problem is (1-r)\*sin(x)\*cos(y)

```
def true_soln(pnts):
  # true solution with has zeros on the unit circle
  r = np.sqrt(pnts[:,0]**2 + pnts[:,1]**2)
  soln = (1 - r)*np.sin(pnts[:,0])*np.cos(pnts[:,1])
  return soln

def forcing(pnts):
  # laplacian of the true solution (forcing term)
  x = pnts[:,0]
  y = pnts[:,1]
  out = ((2*x**2*np.sin(x)*np.cos(y) -
          2*x*np.cos(x)*np.cos(y) +
          2*y**2*np.sin(x)*np.cos(y) +
          2*y*np.sin(x)*np.sin(y) -
          2*np.sqrt(x**2 + y**2)*np.sin(x)*np.cos(y) -
          np.sin(x)*np.cos(y))/np.sqrt(x**2 + y**2))

  return out

# define a circular domain
t = np.linspace(0.0,2*np.pi,100)
vert = np.array([np.cos(t),np.sin(t)]).T
smp = np.array([np.arange(100),np.roll(np.arange(100),-1)]).T

# create the nodes
N = 100
nodes,smpid = make_nodes(N,vert,smp)
boundary, = (smpid>=0).nonzero()

# basis function used to solve this PDE
basis = rbf.basis.phs3

# create the left-hand-side matrix which is the Laplacian of the basis 
# function for interior nodes and the undifferentiated basis functions 
# for the boundary nodes
A  = basis(nodes,nodes,diff=(2,0))
A += basis(nodes,nodes,diff=(0,2))
A[boundary,:] = basis(nodes[boundary],nodes)

# create the right-hand-side vector, consisting of the forcing term 
# for the interior nodes and zeros for the boundary nodes
d = forcing(nodes)
d[boundary] = 0.0

# find the RBF coefficients that solve the PDE
coeff = np.linalg.solve(A,d)

# create a collection of interpolation points to evaluate the 
# solution. It is easiest to just call make_nodes again
itp,dummy = make_nodes(10000,vert,smp,itr=0)

# solution at the interpolation points
soln = basis(itp,nodes).dot(coeff)
```
plot the results
```
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].set_title('RBF solution')
p = ax[0].tripcolor(itp[:,0],itp[:,1],soln)
ax[0].plot(nodes[:,0],nodes[:,1],'ko')
# plot the boundary
for s in smp:
  ax[0].plot(vert[s,0],vert[s,1],'k-',lw=2)

fig.colorbar(p,ax=ax[0])

ax[1].set_title('error')
p = ax[1].tripcolor(itp[:,0],itp[:,1],soln - true_soln(itp))
for s in smp:
  ax[1].plot(vert[s,0],vert[s,1],'k-',lw=2)

fig.colorbar(p,ax=ax[1])
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_xlim((-1.05,1.05))
ax[0].set_ylim((-1.05,1.05))
ax[1].set_xlim((-1.05,1.05))
ax[1].set_ylim((-1.05,1.05))
fig.tight_layout()
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_spectral_laplacian.png "demo_spectral_laplacian")

## To Do
This package contains more features but they have not yet been included in this help documentation. They include
* generation of RBF-FD stencils (module: rbf.stencil)
* generation of RBF-FD weights (module: rbf.fd)
* computational geometry (module: rbf.geometry)
* generation of halton sequences (module: rbf.halton)
* Monte-Carlo integration (module: rbf.integrate)
* generation of B-spline basis functions (module: rbf.bspline) 

See the documentation within the modules for help on using these features 
