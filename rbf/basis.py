#!/usr/bin/env python
#
# This module defines the gaussian, multiquadratic, and inverse
# quadratic radial basis functions.  This module makes use of the
# class, RBF, which takes a symbolic expression of an RBF and converts
# it and its derivatives into a numerical function. So you can
# evaluate any arbitrary derivative of an RBF even though the
# derivatives are not explicitly written anywhere in this module.
#
# For example, consider the Gaussian RBF which is saved as 'ga'.
# If we want to evaluate the first derivative of ga with respect to
# the second spatial dimension then the command would be.
#
# >> ga(x,centers,eps,diff=(0,1)). 
#
# See the help documentation for ga for more information about its
# arguments

from __future__ import division
import sympy 
from sympy.utilities.autowrap import ufuncify
from sympy.printing.theanocode import theano_function
import numpy as np

_R = sympy.symbols('R')
_EPS = sympy.symbols('EPS')

class RBF(object):
  def __init__(self,expr):    
    '''
    Parameters
    ----------
      expr: Symbolic expression of the RBF with respect to _R and _EPS
    '''
    assert expr.has(_R), (
      'RBF expression does not contain _R')
    
    assert expr.has(_EPS), (
      'RBF expression does not contain _EPS')
    self.R_expr = expr
    self.cache = {}

  def __call__(self,x,c,eps=None,diff=None,dtype=np.float64):
    '''
    evaluates M radial basis functions (RBFs) with arbitary dimension
    at N points.

    Parameters                                       
    ----------                                         
      x: ((N,) or (N,D) array) locations to evaluate the RBF
                                                                          
      centers: ((M,) or (M,D) array) centers of each RBF
                                                                 
      eps: ((M,) array, default=np.ones(M)) Scale parameter for each RBF
                                                                           
      diff: ((D,) tuple, default=(0,)*dim) a tuple whos length is
        equal to the number of spatial dimensions.  Each value in the
        tuple must be an integer indicating the order of the
        derivative in that spatial dimension.  For example, if the the
        spatial dimensions of the problem are 3 then diff=(2,0,1)
        would compute the second derivative in the first dimension and
        the first derivative in the third dimension.

    Returns
    -------
      out: (N,M) array for each M RBF evaluated at the N points

    Note 
    ---- 
      the derivatives are computed symbolically in Sympy and then
      lambdified to evaluate the expression with the provided values.
      The lambdified functions are cached in the scope of the radial
      module and will be recalled if a value for diff is used more
      than once in the Python session.

    ''' 
    x = np.asarray(x,dtype=dtype)
    c = np.asarray(c,dtype=dtype)
    if eps is not None:
      eps = np.asarray(eps,dtype=dtype)
   
    assert (len(x.shape) == 1) | (len(x.shape) == 2), (
      'x must be a 1-D or 2-D array')
    assert (len(c.shape) == 1) | (len(c.shape) == 2), (
      'c must be a 1-D or 2-D array')

    if len(x.shape) == 1:
      x = x[:,None,None]

    if len(c.shape) == 1:
      c = c[None,:,None]

    if len(x.shape) == 2:
      x = x[:,None,:]

    if len(c.shape) == 2:
      c = c[None,:,:]

    N = x.shape[0]
    M = c.shape[1]
    assert x.shape[2] == c.shape[2], (
      'if x and c are 2-D arrays then their second dimensions must '
      'have the same length')
    dim = x.shape[2]
    if eps is None:
      eps = np.ones(M,dtype=dtype)

    assert len(eps.shape) == 1, (
      'eps must be a 1D array')

    assert len(eps) == M, (
      'length of eps must be equal to the number of centers')

    #x = x.swapaxes(0,2)
    #c = c.swapaxes(0,2)
    x = np.einsum('ijk->kij',x)
    c = np.einsum('ijk->kij',c)

    if diff is None:
      diff = (0,)*dim

    while len(diff) < dim:
      diff += (0,)

    assert len(diff) == dim, (
      'cannot specify derivatives for dimensions that are higher than '
      'the dimensions of x and center')

    # add function to cache if not already
    if diff not in self.cache:
      dim = len(diff)
      c_sym = sympy.symbols('c:%s' % dim)
      x_sym = sympy.symbols('x:%s' % dim)    
      r_sym = sympy.sqrt(sum((x_sym[i]-c_sym[i])**2 for i in range(dim)))
      expr = self.R_expr.subs(_R,r_sym)            
      for direction,order in enumerate(diff):
        if order == 0:
          continue
        expr = expr.diff(*(x_sym[direction],)*order)
        #expr = expr.expand().simplify()

      #self.cache[diff] = sympy.lambdify(x_sym+c_sym+(_EPS,),expr,'numpy')
      self.cache[diff] = ufuncify(x_sym+c_sym+(_EPS,),expr)
 
    args = (tuple(x)+tuple(c)+(eps,))    
    return self.cache[diff](*args)


class RBFInterpolant(object):
  '''
  A callable RBF interpolant
  '''
  def __init__(self,
               x,
               eps, 
               value=None,
               alpha=None,
               rbf=None,
               dtype=np.float64):
    '''
    Initiates the RBF interpolant

    Parameters
    ----------
      x: ((N,) or (N,D) array) x coordinate of the interpolation
        points which also make up the centers of the RBFs

      eps: ((N,) array) shape parameters for each RBF  
 
      value: ((N,) or (N,R) array) Values at the x coordinates. If this 
        is not provided then alpha must be given

      alpha: ((N,) or (N,R) array) Coefficients for each RBFs. If this 
       is not provided then value must be given  

      rbf: type of rbf to use. either mq, ga, or iq

    '''
    x = np.asarray(x,dtype=dtype)
    eps = np.asarray(eps,dtype=dtype)
    if value is not None:
      value = np.asarray(value,dtype=dtype)

    if alpha is not None:
      alpha = np.asarray(alpha,dtype=dtype)

    if rbf is None:
      rbf = mq

    assert (value is not None) != (alpha is not None), (
      'either val or alpha must be given')

    x_shape = np.shape(x)
    N = x_shape[0]
    assert len(x_shape) <= 2
    assert np.shape(eps) == (N,)

    if len(x_shape) == 1:
      x = x[:,None]

    if alpha is not None:
      alpha_shape = np.shape(alpha)
      assert len(alpha_shape) <= 2
      assert alpha_shape[0] == N
      if len(alpha_shape) == 1:
        alpha = alpha[:,None]

      alpha_shape = np.shape(alpha)
      R = alpha_shape[1]
   
    if value is not None:
      value_shape = np.shape(value)
      assert len(value_shape) <= 2
      assert value_shape[0] == N
      if len(value_shape) == 1:
        value = value[:,None]

      value_shape = np.shape(value)
      R = value_shape[1]

    if alpha is None:
      alpha = np.zeros((N,R),dtype=dtype)
      G = rbf(x,x,eps,dtype=dtype)
      for r in range(R):
        alpha[:,r] = np.linalg.solve(G,value[:,r])

    self.dtype = dtype
    self.x = x
    self.eps = eps
    self.alpha = alpha
    self.R = R
    self.rbf = rbf

  def __call__(self,xitp,diff=None):
    '''
    Returns the interpolant evaluated at xitp

    Parameters 
    ---------- 
      xitp: ((N,) or (N,D) array) points where the interpolant is to 
        be evaluated

      diff: ((D,) tuple, default=(0,)*dim) a tuple whos length is
        equal to the number of spatial dimensions.  Each value in the
        tuple must be an integer indicating the order of the
        derivative in that spatial dimension.  For example, if the the
        spatial dimensions of the problem are 3 then diff=(2,0,1)
        would compute the second derivative in the first dimension and
        the first derivative in the third dimension.

    '''
    out = np.zeros((len(xitp),self.R),dtype=self.dtype)
    xitp = np.asarray(xitp,dtype=self.dtype)
    for r in range(self.R):
      out[:,r] = np.sum(self.rbf(xitp,
                                 self.x,
                                 self.eps,
                                 diff=diff,
                                 dtype=self.dtype)*self.alpha[:,r],1)

    return out 


_FUNCTION_DOC = '''
  evaluates M radial basis functions (RBFs) with arbitary dimension at N points.

  Parameters                                       
  ----------                                         
    x: ((N,) or (N,D) array) locations to evaluate the RBF
                                                                          
    centers: ((M,) or (M,D) array) centers of each RBF
                                                                 
    eps: ((M,) array, default=np.ones(M)) Scale parameter for each RBF
                                                                           
    diff: ((D,) tuple, default=(0,)*dim) a tuple whos length is equal to the number 
      of spatial dimensions.  Each value in the tuple must be an integer
      indicating the order of the derivative in that spatial dimension.  For 
      example, if the the spatial dimensions of the problem are 3 then 
      diff=(2,0,1) would compute the second derivative in the first dimension
      and the first derivative in the third dimension.

  Returns
  -------
    out: (N,M) array for each M RBF evaluated at the N points

  Note
  ----
    the derivatives are computed symbolically in Sympy and then lambdified to 
    evaluate the expression with the provided values.  The lambdified functions
    are cached in the scope of the radial module and will be recalled if 
    a value for diff is used more than once in the Python session.        
'''

_IQ = RBF(1/(1+(_EPS*_R)**2))
def iq(*args,**kwargs):
  '''                             
  Inverse Quadratic
  '''                                                             
  return _IQ(*args,**kwargs)

iq.__doc__ += _FUNCTION_DOC

_GA = RBF(sympy.exp(-(_EPS*_R)**2))
def ga(*args,**kwargs):
  '''                        
  Gaussian
  '''
  return _GA(*args,**kwargs)

ga.__doc__ += _FUNCTION_DOC

_MQ = RBF(sympy.sqrt(1 + (_EPS*_R)**2))
def mq(*args,**kwargs):
  '''                     
  multiquadratic
  '''
  return _MQ(*args,**kwargs)

mq.__doc__ += _FUNCTION_DOC


