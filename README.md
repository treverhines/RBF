# RBF
Package containing the tools necessary for radial basis function (RBF) applications

## Features
 * Efficient evaluation of RBFs and their analytically derived spatial derivatives.  This package allows for unlimitted spatial dimensions and arbitrarily spatial derivatives   
 * Regularized RBF interpolation, which can fit a smoothed interpolant to noisy data   
 * Generation of radial basis function finite difference (RBF-FD) weights, which are used to estimate derivatives of scattered data
 * Efficient generation of RBF-FD stencils which can be given constraints to not cross a user defined boundary. This is useful if the user does not want to estimate a derivative over a known discontinuity.  
 * computational geometry functions for 1, 2, and 3 spatial dimensions. Among these functions is a point in polygon/polyhedra test
 * Halton sequence generator
 * Node generation with a minmimum energy algorithm.  This is used for solving PDEs with the spectral RBF method or the RBF-FD method
 * functions for Monte-Carlo integration and recursive Monte-Carlo integration over an polygonal/polyhedral domain

## Basis
  The linchpin of this module is the RBF class, which is used to evaluate an RBF and its derivatives.  An RBF is instantiated using a symbolic sympy expression.  Evaluating the RBFs is done by calling the RBF instance where the user supplies the positions where the RBFs are to be evaluated, the centers of the RBFs and the desired derivate (if any).  When called, the an analytical derivative of the symbolic expression is evalauted and then the function is compiled into cython code. This compiled code is saved and reused when the RBF is called again with the same derivative specification.   
  
  The user does not need to worry about instantiation of an RBF class because many of the commonly used RBFs are already instantiated and can be called using function in the rbf.basis module.     

![alt text](https://github.com/treverhines/RBF/blob/master/demo/figures/demo_basis.png "demo_basis")
