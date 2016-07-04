#!/usr/bin/env python
''' 
This module defines some of the commonly used radial basis functions. 
It makes use of the class, RBF, which takes a symbolic expression of 
an RBF and converts it and its derivatives into a numerical function.  
This allows for the evaluation of any arbitrary derivative of an RBF 
even though the derivatives are not explicitly written anywhere in 
this module.
''' 
from __future__ import division 
import sympy 
from sympy.utilities.autowrap import ufuncify 
import numpy as np 
import warnings 
import copy

# define global symbolic variables
_R = sympy.symbols('R')
_EPS = sympy.symbols('EPS')
_SYM_TO_NUM = 'cython'


def _check_lambdified_output(fin):
  ''' 
  when lambdifying a sympy expression, the output is a scalar if the 
  expression is independent of R. This function checks the output of a 
  lambdified function and if the output is a scalar then it expands 
  the output to the proper output size. The proper output size is 
  (N,M) where N is the number of collocation points and M is the 
  number of basis functions
  '''
  def fout(*args,**kwargs):
    out = fin(*args,**kwargs)
    x = args[0]
    eps = args[-1]
    if np.isscalar(out):
      arr = np.empty((x.shape[0],eps.shape[0]))
      arr[...] = out
      out = arr

    return out

  return fout  


def _replace_nan(x):
  ''' 
  this is orders of magnitude faster than np.nan_to_num
  '''
  x[np.isnan(x)] = 0.0
  return x


def get_R():
  ''' 
  returns the symbolic variable R that can be used in RBF expressions
  '''
  return copy.deepcopy(_R)


def get_EPS():
  ''' 
  returns the symbolic variable EPS that can be used in RBF 
  expressions
  '''
  return copy.deepcopy(_EPS)


def set_sym_to_num(package):
  ''' 
  controls how the RBF class converts the symbolic expressions to 
  numerical expressions
  
  Parameters
  ----------
    package : str
      either 'numpy' or 'cython'. If 'numpy' then the symbolic 
      expression is converted using sympy.lambdify. If 'cython' then 
      the expression if converted using 
      sympy.utilities.autowrap.ufuncify, which converts the expression 
      to cython code and then compiles it. Note that there is a ~1 
      second overhead to compile the cython code
      
  '''
  global _SYM_TO_NUM 
  if package in ['cython','numpy']:
    _SYM_TO_NUM = package
  else:
    raise ValueError('package must either be "cython" or "numpy" ')  
  

class RBF(object):
  ''' 
  Stores a symbolic expression of an RBF and evaluates the expression 
  numerically when called. The symbolic expression must be a function 
  of the global variable R, where R is the radial distance to the RBF 
  center.  The expression may optionally be a function of the global 
  variable EPS, where EPS is a shape parameter.  If EPS is not given 
  then R is substituded with EPS*R upon instantiation
  
  Usage
  -----
    # create an inverse multiquadratic rbf, evaluate the rbf 
    # centered at [0.0] at 10 random points
    >>> R = get_R()
    >>> EPS = get_EPS()
    >>> iq_expr = 1/(1 + (EPS*R)**2)
    >>> iq = RBF(iq)
    >>> x = np.random.random(10,1)
    >>> center = np.array([[0.0]])
    >>> values = iq(x,center)
    
  '''
  def __init__(self,expr):    
    ''' 
    Parameters
    ----------
      expr : sympy expression
        symbolic expression of the RBF. This must contain the symbolic 
        variable R, which can be obtained by calling get_R. It may 
        optionally contain the shape parameter EPS, which can be 
        obtained by calling get_EPS. If EPS is not in the symbolic 
        expression then R is substituted with EPS*R

    '''
    if not expr.has(_R):
      raise ValueError('RBF expression must be a function of rbf.basis.R')
    
    if not expr.has(_EPS):
      # if EPS is not in the expression then substitute EPS*R for R
      expr = expr.subs(_R,_EPS*_R)
      
    self.expr = expr
    self.cache = {}

  def __call__(self,x,c,eps=None,diff=None,check_input=True):
    ''' 
    evaluates M radial basis functions (RBFs) at N points.

    Parameters                                       
    ----------                                         
      x : (N,D) array 
        evaluate the RBFs at these positions 
                                                                          
      c : (M,D) array 
        centers for each RBF
                                                                 
      eps : (M,) array, optional
        shape parameters for each RBF. Defaults to 1.0
                                                                           
      diff : (D,) int array, optional
        a tuple whos length is equal to the number of spatial 
        dimensions.  Each value in the tuple must be an integer 
        indicating the order of the derivative in that spatial 
        dimension.  For example, if the the spatial dimensions of the 
        problem are 3 then diff=(2,0,1) would compute the second 
        derivative in the first dimension then the first derivative in 
        the third dimension. In other words, it would compute the 
        d^3u/dx^2*dz, where x and z are the first and third 
        spatial dimension and u is the RBF
        
      check_input : bool, optional
        indicate whether or not to check the size and data type for 
        the input arguments. If False then this function can speed up 
        significantly but errors will be less comprehensible. Also, 
        eps and diff must be provided if this is False

    Returns
    -------
      out : (N,M) array
        alternant matrix consisting of each RBF evaluated at x

    Note 
    ---- 
      the derivatives are computed symbolically in Sympy and then
      lambdified to evaluate the expression with the provided values.
      The lambdified functions are cached in the scope of the radial
      module and will be recalled if a value for diff is used more
      than once in the Python session.

    '''
    # if check is True then run the following code to ensure proper 
    # types and sizes
    if check_input:    
      x = np.asarray(x,dtype=float)
      c = np.asarray(c,dtype=float)
      if eps is None:
        eps = np.ones(c.shape[0])   
      else:  
        eps = np.asarray(eps,dtype=float)

      if diff is None:
        diff = (0,)*x.shape[1]
      else:
        # make sure diff is immutable
        diff = tuple(diff)

      # make sure the input arguments have the proper dimensions
      if not ((x.ndim == 2) & (c.ndim == 2)):
        raise ValueError(
          'x and c must be two-dimensional arrays')

      if not (x.shape[1] == c.shape[1]):
        raise ValueError(
          'x and c must have the same number of spatial dimensions')

      if x.shape[1] == 0:
        raise ValueError(
          'spatial dimensions of x and c must be at least one')

      if not ((eps.ndim == 1) & (eps.shape[0] == c.shape[0])):
        raise ValueError(
          'eps must be a one-dimensional array with length equal to '
          'the number of rows in c')
    
      if not (len(diff) == x.shape[1]):
        raise ValueError(
          'diff must have the same length as the number of spatial '
          'dimensions  in x and c')

    # expand to allow for broadcasting
    x = x[:,None,:]
    c = c[None,:,:]

    # this does the same thing as np.rollaxis(x,-1) but is much faster
    x = np.einsum('ijk->kij',x)
    c = np.einsum('ijk->kij',c)

    # add function to cache if not already
    if diff not in self.cache:
      dim = len(diff)
      c_sym = sympy.symbols('c:%s' % dim)
      x_sym = sympy.symbols('x:%s' % dim)    
      r_sym = sympy.sqrt(sum((x_sym[i]-c_sym[i])**2 for i in range(dim)))
      expr = self.expr.subs(_R,r_sym)            
      for direction,order in enumerate(diff):
        if order == 0:
          continue
        expr = expr.diff(*(x_sym[direction],)*order)

      if _SYM_TO_NUM == 'numpy':
        func = sympy.lambdify(x_sym+c_sym+(_EPS,),expr,'numpy')
        func = _check_lambdified_output(func)
        self.cache[diff] = func

      elif _SYM_TO_NUM == 'cython':        
        func = ufuncify(x_sym+c_sym+(_EPS,),expr)
        self.cache[diff] = func
 
    args = (tuple(x)+tuple(c)+(eps,))    
    return self.cache[diff](*args)
    
_FUNCTION_DOC = ''' 
  evaluates M radial basis functions (RBFs) at N points.

  Parameters                                       
  ----------                                         
    x : (N,D) array 
      evaluate the RBFs at these positions 
                                                                       
    c : (M,D) array 
      centers for each RBF
        
    eps : (M,) array, optional
      shape parameters for each RBF. Defaults to 1.0
                                                                           
    diff : (D,) int array, optional
      a tuple whos length is equal to the number of spatial 
      dimensions.  Each value in the tuple must be an integer 
      indicating the order of the derivative in that spatial 
      dimension.  For example, if the the spatial dimensions of the 
      problem are 3 then diff=(2,0,1) would compute the second 
      derivative in the first dimension then the first derivative in 
      the third dimension. In other words, it would compute the 
      d^3u/dx^2*dz, where x and z are the first and third 
      spatial dimension and u is the RBF

  Returns
  -------
    out : (N,M) array
      alternant matrix consisting of each RBF evaluated at x

  Note 
  ---- 
    the derivatives are computed symbolically in Sympy and then
    lambdified to evaluate the expression with the provided values.
    The lambdified functions are cached in the scope of the radial
    module and will be recalled if a value for diff is used more
    than once in the Python session.
'''


_PHS8 = RBF((_EPS*_R)**8*sympy.log(_EPS*_R))
def phs8(*args,**kwargs):
  ''' 
  eighth-order polyharmonic spline:

    (EPS*R)^8*log(EPS*R)
  
  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _replace_nan(_PHS8(*args,**kwargs))

phs8.__doc__ += _FUNCTION_DOC


_PHS7 = RBF((_EPS*_R)**7)
def phs7(*args,**kwargs):
  ''' 
  seventh-order polyharmonic spline:

    (EPS*R)^7

  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _replace_nan(_PHS7(*args,**kwargs))

phs7.__doc__ += _FUNCTION_DOC


_PHS6 = RBF((_EPS*_R)**6*sympy.log(_EPS*_R))

def phs6(*args,**kwargs):
  ''' 
  sixth-order polyharmonic spline:

    (EPS*R)^6*log(EPS*R)
  
  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's
  '''
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _replace_nan(_PHS6(*args,**kwargs))

phs6.__doc__ += _FUNCTION_DOC


_PHS5 = RBF((_EPS*_R)**5)
def phs5(*args,**kwargs):
  '''                             
  fifth-order polyharmonic spline:

    (EPS*R)^5

  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's

  '''
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _replace_nan(_PHS5(*args,**kwargs))

phs5.__doc__ += _FUNCTION_DOC


_PHS4 = RBF((_EPS*_R)**4*sympy.log(_EPS*_R))
def phs4(*args,**kwargs):
  ''' 
  fourth-order polyharmonic spline:

    (EPS*R)^4*log(EPS*R)
  
  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's

  '''
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _replace_nan(_PHS4(*args,**kwargs))

phs4.__doc__ += _FUNCTION_DOC


_PHS3 = RBF((_EPS*_R)**3)
def phs3(*args,**kwargs):
  ''' 
  third-order polyharmonic spline:

    (EPS*R)^3

  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's

  '''
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _replace_nan(_PHS3(*args,**kwargs))

phs3.__doc__ += _FUNCTION_DOC


_PHS2 = RBF((_EPS*_R)**2*sympy.log(_EPS*_R))
def phs2(*args,**kwargs):
  ''' 
  second-order polyharmonic spline:

    (EPS*R)^2*log(EPS*R)
  
  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's

  '''
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _replace_nan(_PHS2(*args,**kwargs))

phs2.__doc__ += _FUNCTION_DOC


_PHS1 = RBF(_EPS*_R)
def phs1(*args,**kwargs):
  ''' 
  first-order polyharmonic spline:

    EPS*R

  NOTE 
  ----
    This RBF usually does not include a shape parameter. It is 
    included here for the sake of consistency with the other RBF's

  '''
  # division by zero errors may occur for R=0. Ignore warnings and
  # replace nan's with zeros
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _replace_nan(_PHS1(*args,**kwargs))

phs1.__doc__ += _FUNCTION_DOC


_IMQ = RBF(1/sympy.sqrt(1+(_EPS*_R)**2))
def imq(*args,**kwargs):
  ''' 
  inverse multiquadratic:

    1/sqrt(1 + (EPS*R)^2)

  '''
  return _IMQ(*args,**kwargs)

imq.__doc__ += _FUNCTION_DOC


_IQ = RBF(1/(1+(_EPS*_R)**2))
def iq(*args,**kwargs):
  '''                             
  inverse quadratic:

    1/(1 + (EPS*R)^2)

  '''                                                             
  return _IQ(*args,**kwargs)

iq.__doc__ += _FUNCTION_DOC


_GA = RBF(sympy.exp(-(_EPS*_R)**2))
def ga(*args,**kwargs):
  '''                        
  Gaussian:

    exp(-(EPS*R)^2)

  '''
  return _GA(*args,**kwargs)

ga.__doc__ += _FUNCTION_DOC


_MQ = RBF(sympy.sqrt(1 + (_EPS*_R)**2))
def mq(*args,**kwargs):
  '''                     
  multiquadratic:

    sqrt(1 + (EPS*R)^2)

  '''
  return _MQ(*args,**kwargs)

mq.__doc__ += _FUNCTION_DOC


