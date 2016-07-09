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

# Usage
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
plt.savefig('figures/demo_interpolate_1d.png')
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_interpolate_1d.png "demo_interpolate_1d")



