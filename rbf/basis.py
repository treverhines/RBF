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
import numpy as np
import warnings

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

  def __call__(self,x,c,eps=None,diff=None):
    '''
    evaluates M radial basis functions (RBFs) with arbitary dimension
    at N points.

    Parameters                                       
    ----------                                         
      x: (N,D) array of locations to evaluate the RBF
                                                                          
      centers: (M,D) array of centers for each RBF
                                                                 
      eps: (default=None) (M,) array of shape parameters for each RBF.
        If not given then the shape parameter for each RBF is 1.0
                                                                           
      diff: (default=None) a tuple whos length is equal to the number
        of spatial dimensions.  Each value in the tuple must be an
        integer indicating the order of the derivative in that spatial
        dimension.  For example, if the the spatial dimensions of the
        problem are 3 then diff=(2,0,1) would compute the second
        derivative in the first dimension and the first derivative in
        the third dimension.

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
    # Ensure that arguments have proper dimensions
    x = np.asarray(x)
    c = np.asarray(c)

    assert (x.ndim == 2), (
      'x must be a 2-D array')
    assert (c.ndim == 2), (
      'c must be a 2-D array')
    assert x.shape[1] == c.shape[1], (
      'the spatial dimensions of x and c must be equal')

    if eps is None:
      eps = np.ones(c.shape[0])   

    eps = np.asarray(eps)

    assert eps.ndim == 1, (
      'eps must be a 1D array')

    assert eps.shape[0] == c.shape[0], (
      'length of eps must be equal to the number of centers')

    if diff is None:
      diff = (0,)*x.shape[1]

    assert len(diff) == x.shape[1], (
      'length of derivative specification must be equal to the '
      'spatial dimensions of x and c')

    # expand to allow for broadcasting
    x = x[:,None,:]
    c = c[None,:,:]

    x = np.einsum('ijk->kij',x)
    c = np.einsum('ijk->kij',c)

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

      #self.cache[diff] = sympy.lambdify(x_sym+c_sym+(_EPS,),expr,'numpy')
      self.cache[diff] = ufuncify(x_sym+c_sym+(_EPS,),expr)
 
    args = (tuple(x)+tuple(c)+(eps,))    
    return self.cache[diff](*args)


# DEPRICATED
class RBFInterpolant(object):
  '''
  A callable RBF interpolant
  '''
  def __init__(self,
               x,
               eps, 
               value=None,
               alpha=None,
               rbf=None):
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
    x = np.asarray(x)
    eps = np.asarray(eps)
    if value is not None:
      value = np.asarray(value)

    if alpha is not None:
      alpha = np.asarray(alpha)

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
      alpha = np.zeros((N,R))
      G = rbf(x,x,eps)
      for r in range(R):
        alpha[:,r] = np.linalg.solve(G,value[:,r])

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
    out = np.zeros((len(xitp),self.R))
    xitp = np.asarray(xitp)
    for r in range(self.R):
      out[:,r] = np.sum(self.rbf(xitp,
                                 self.x,
                                 self.eps,
                                 diff=diff)*self.alpha[:,r],1)

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

def replace_nan(x):
  '''
  this is orders of magnitude faster than np.nan_to_num
  '''
  x[np.isnan(x)] = 0.0
  return x

_PHS8 = RBF((_EPS*_R)**8*sympy.log(_EPS*_R))
def phs8(*args,**kwargs):
  '''                             
  eighth order polyharmonic spline:
    phi(r) = (eps*r)^8*log(eps*r)
  
  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''                                                             
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return replace_nan(_PHS8(*args,**kwargs))

phs8.__doc__ += _FUNCTION_DOC


_PHS7 = RBF((_EPS*_R)**7)
def phs7(*args,**kwargs):
  '''                             
  seventh order polyharmonic spline:
    phi(r) = (eps*r)^7

  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''                                                             
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return replace_nan(_PHS7(*args,**kwargs))

phs7.__doc__ += _FUNCTION_DOC


_PHS6 = RBF((_EPS*_R)**6*sympy.log(_EPS*_R))
def phs6(*args,**kwargs):
  '''                             
  sixth order polyharmonic spline:
    phi(r) = (eps*r)^6*log(eps*r)
  
  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''                                                             
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return replace_nan(_PHS6(*args,**kwargs))

phs6.__doc__ += _FUNCTION_DOC


_PHS5 = RBF((_EPS*_R)**5)
def phs5(*args,**kwargs):
  '''                             
  fifth order polyharmonic spline:
    phi(r) = (eps*r)^5

  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''                                                             
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return replace_nan(_PHS5(*args,**kwargs))

phs5.__doc__ += _FUNCTION_DOC


_PHS4 = RBF((_EPS*_R)**4*sympy.log(_EPS*_R))
def phs4(*args,**kwargs):
  '''                             
  fourth order polyharmonic spline:
    phi(r) = (eps*r)^4*log(eps*r)
  
  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''                                                             
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return replace_nan(_PHS4(*args,**kwargs))

phs4.__doc__ += _FUNCTION_DOC


_PHS3 = RBF((_EPS*_R)**3)
def phs3(*args,**kwargs):
  '''                             
  third order polyharmonic spline:
    phi(r) = (eps*r)^3

  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''                                                             
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return replace_nan(_PHS3(*args,**kwargs))

phs3.__doc__ += _FUNCTION_DOC


_PHS2 = RBF((_EPS*_R)**2*sympy.log(_EPS*_R))
def phs2(*args,**kwargs):
  '''                             
  second order polyharmonic spline:
    phi(r) = (eps*r)^2*log(eps*r)
  
  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''                                                             
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return replace_nan(_PHS2(*args,**kwargs))

phs2.__doc__ += _FUNCTION_DOC


_PHS1 = RBF(_EPS*_R)
def phs1(*args,**kwargs):
  '''                             
  first order polyharmonic spline:
    phi(r) = eps*r

  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''                                                             
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return replace_nan(_PHS1(*args,**kwargs))

phs1.__doc__ += _FUNCTION_DOC


_IMQ = RBF(1/sympy.sqrt(1+(_EPS*_R)**2))
def imq(*args,**kwargs):
  '''                             
  inverse multiquadratic:
    phi(r) = 1/sqrt(1 + (eps*r)^2)
  '''                                                             
  return _IMQ(*args,**kwargs)

imq.__doc__ += _FUNCTION_DOC


_IQ = RBF(1/(1+(_EPS*_R)**2))
def iq(*args,**kwargs):
  '''                             
  inverse quadratic:
    phi(r) = 1/(1 + (eps*r)^2)
  '''                                                             
  return _IQ(*args,**kwargs)

iq.__doc__ += _FUNCTION_DOC


_GA = RBF(sympy.exp(-(_EPS*_R)**2))
def ga(*args,**kwargs):
  '''                        
  Gaussian:
    phi(r) = e^(-(eps*r)^2)
  '''
  return _GA(*args,**kwargs)

ga.__doc__ += _FUNCTION_DOC


_MQ = RBF(sympy.sqrt(1 + (_EPS*_R)**2))
def mq(*args,**kwargs):
  '''                     
  multiquadratic:
    phi(r) = sqrt(1 + (eps*r)^2)
  '''
  return _MQ(*args,**kwargs)

mq.__doc__ += _FUNCTION_DOC


