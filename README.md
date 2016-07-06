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
See the accompanying examples in `RBF/demo`. Most of these demonstrations are one-dimensional for the sake of visualization. The extension to multiple dimensions should be straight forward when you read the help documentation for each function.

## Basis
The linchpin of this module is the RBF class, which is used to evaluate an RBF and its derivatives.  An RBF is instantiated using a symbolic sympy expression.  Evaluating the RBFs is done by calling the RBF instance where the user supplies the evaluation points, the RBF centers, and the desired derivate (if any).  When called, an analytical derivative of the symbolic expression is evaluated and then the function is compiled into cython code.  This compiled code is saved and reused when the RBF is called again with the same derivative specification.   
  
Here is an example where an RBF is instantiated and then evaluated. See the help documentation for rbf.basis.RBF for more information on the arguments
```
import numpy as np
import rbf.basis
import matplotlib.pyplot as plt

# demonstrate instantiation and evaluation of RBF
R = rbf.basis.get_R()
expr = 1/(1 + R**2) # inverse quadratic
iq = rbf.basis.RBF(expr)

c = np.array([[-2.0],[0.0],[2.0]]) # RBF centers
x = np.linspace(-5.0,5.0,1000)[:,None] # evaluate at these points

soln = iq(x,c) # evaluate the RBFs
soln_diff = iq(x,c,diff=(1,)) # evaluate the first derivative
```
plot the resuts
```
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln)
ax.set_xlim((-5.0,5.0))
ax.set_title('inverse quadratic')
ax.grid()
fig.tight_layout()
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_basis_1.png "demo_basis_1")
```
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln_diff)
ax.set_xlim((-5.0,5.0))
ax.set_title('inverse quadratic first derivative')
ax.grid()
fig.tight_layout()
plt.show()
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_basis_2.png "demo_basis_2")

The user does not need to worry about instantiation of an RBF class because many of the commonly used RBFs are already instantiated and can be called using function in the rbf.basis module.  The available functions are
* ga : gaussian, exp(-(EPS\*R)^2)
* iq : inverse quadratic, 1/(1+(EPS\*R)^2)
* mq : multiquadratic, sqrt(1 + (EPS\*R)^2)
* imq : inverse multiquadratic, 1/sqrt(1 + (EPS\*R)^2)
* phs{1,3,5,7} : odd order polyharmonic splines, (EPS\*R)^{1,3,5,7}
* phs{2,4,6,8} : even order polyharmonic splines, log(EPS\*R)(EPS\*R)^{2,4,6,8}  

EPS is a scaling factor which can be obtained for defining your own RBFs by calling `rbf.basis.get_EPS()`. When evaluating the RBF, you can set the scaling factor with the `eps` key word argument. 

## Interpolation
Creating a simple RBF interpolant is straight forward with an RBF instance
```
x = np.linspace(-np.pi,np.pi,5)[:,None] # observation points
u = np.sin(x[:,0]) # values
xitp = np.linspace(-4.0,4.0,1000)[:,None] # interpolation points

A = rbf.basis.phs3(x,x)
coeff = np.linalg.solve(A,u) # estimate coefficients for each RBF
Aitp = rbf.basis.phs3(xitp,x) # interpolation matrix
uitp = Aitp.dot(coeff)

fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x[:,0],u,'ko')
ax.plot(xitp[:,0],uitp,'k-')
ax.plot(xitp[:,0],np.sin(xitp[:,0]),'k--')
ax.legend(['observation points','interpolant','true solution'],loc=2,frameon=False)
ax.set_title('third-order polyharmonic spline interpolation')
fig.tight_layout()
plt.savefig('figures/demo_interpolate.png')
```
![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_interpolate.png "demo_basis_2")

The RBFInterpolant class, which is in the rbf.interpolate module, is provided for more complicated interpolation problems (e.g. smoothed interpolation).

