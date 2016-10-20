''' 
This module defines some of the commonly used radial basis functions. 
It makes use of the class, *RBF*, which takes a symbolic expression of 
an RBF and converts it and its derivatives into a numerical function.  
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
  returns the symbolic variable for :math:`r` which is used to 
  instantiate an *RBF*
  '''
  return copy.deepcopy(_R)


def get_EPS():
  ''' 
  returns the symbolic variable for :math:`\epsilon` which is used to 
  instantiate an *RBF*
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
  Stores a symbolic expression of a Radial Basis Function (RBF) and 
  evaluates the expression numerically when called. 
  
  Parameters
  ----------
  expr : sympy expression
    Symbolic expression of the RBF. This must be a function of the 
    symbolic variable *R*, which is returned by the function *get_R*. 
    *R* is the radial distance to the RBF center.  The expression may 
    optionally be a function of *EPS*, which is a shape parameter 
    obtained by the function *get_EPS*.  If *EPS* is not provided then 
    *R* is substituted with *R* * *EPS* .
  
  Examples
  --------
  Instantiate an inverse quadratic RBF

  >>> R = get_R()
  >>> EPS = get_EPS()
  >>> iq_expr = 1/(1 + (EPS*R)**2)
  >>> iq = RBF(iq_expr)
  
  Evaluate an inverse quadratic at 10 points ranging from -5 to 5. 
  Note that the evaluation points and centers are two dimensional 
  arrays

  >>> x = np.linspace(-5.0,5.0,10)[:,None]
  >>> center = np.array([[0.0]])
  >>> values = iq(x,center)
    
  '''
  def __init__(self,expr):    
    if not expr.has(_R):
      raise ValueError('RBF expression must be a function of rbf.basis.R')
    
    if not expr.has(_EPS):
      # if EPS is not in the expression then substitute EPS*R for R
      expr = expr.subs(_R,_EPS*_R)
      
    self.expr = expr
    self.cache = {}

  def __call__(self,x,c,eps=None,diff=None):
    ''' 
    Evaluates the RBF
    
    Parameters                                       
    ----------                                         
    x : (N,D) array 
      evaluation points
                                                                       
    c : (M,D) array 
      RBF centers 
        
    eps : (M,) array, optional
      shape parameters for each RBF. Defaults to 1.0
                                                                           
    diff : (D,) int array, optional
      Tuple indicating the derivative order for each spatial dimension. 
      For example, if there are three spatial dimensions then providing 
      (2,0,1) would return the RBF after differentiating it twice along 
      the first axis and once along the third axis.

    Returns
    -------
    out : (N,M) array
      Returns the RBFs with centers *c* evaluated at *x*

    Notes
    -----
    This function evaluates the RBF and its derivatives symbolically 
    using sympy and then the symbolic expression is converted to a 
    numerical function. The numerical function is cached and then reused 
    when this function is called multiple times with the same derivative 
    specification.

    '''
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
        'dimensions in x and c')

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

  where :math:`r = ||x - c||_2`, and :math:`\epsilon` is a shape 
  parameter. :math:`x` and :math:`c` are the evaluation points and RBF 
  centers, respectively.

  Parameters                                       
  ----------                                         
  x : (N,D) array 
    evaluation points
                                                                       
  c : (M,D) array 
    RBF centers 
        
  eps : (M,) array, optional
    shape parameters for each RBF. Defaults to 1.0
                                                                           
  diff : (D,) int array, optional
    Tuple indicating the derivative order for each spatial dimension. 
    For example, if there are three spatial dimensions then providing 
    (2,0,1) would return the RBF after differentiating it twice along 
    the first axis and once along the third axis.

  Returns
  -------
  out : (N,M) array
    Returns the RBFs with centers *c* evaluated at *x*

  Notes
  -----
  This function evaluates the RBF and its derivatives symbolically 
  using sympy and then the symbolic expression is converted to a 
  numerical function. The numerical function is cached and then reused 
  when this function is called multiple times with the same derivative 
  specification.

'''


_PHS8 = RBF((_EPS*_R)**8*sympy.log(_EPS*_R))
def phs8(*args,**kwargs):
  ''' 
  Eighth-order polyharmonic spline (phs8), which is defined as

  .. math:: 
    (\epsilon r)^8\log(\epsilon r),

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
  Seventh-order polyharmonic spline (phs7), which is defined as

  .. math::
    (\epsilon r)^7

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
  Sixth-order polyharmonic spline (phs6), which is defined as
  
  .. math::
    (\epsilon r)^6\log(\epsilon r)
  
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
  Fifth-order polyharmonic spline (phs5), which is defined as
  
  .. math::
    (\epsilon r)^5

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
  Fourth-order polyharmonic spline (phs4), which is defined as

  .. math::
    (\epsilon r)^4\log(\epsilon r)

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
  Third-order polyharmonic spline (phs3), which is defined as

  .. math::
    (\epsilon r)^3

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
  Second-order polyharmonic spline (phs2), which is defined as

  .. math::
    (\epsilon r)^2\log(\epsilon r)
  
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
  First-order polyharmonic spline (phs1), which is defined as

  .. math::
    \epsilon r

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
  Inverse multiquadratic (imq), which is defined as

  .. math::
    (1 + (\epsilon r)^2)^{-1/2}

  '''
  return _IMQ(*args,**kwargs)

imq.__doc__ += _FUNCTION_DOC


_IQ = RBF(1/(1+(_EPS*_R)**2))
def iq(*args,**kwargs):
  '''                             
  Inverse quadratic (iq), which is defined as

  .. math::
    (1 + (\epsilon r)^2)^{-1}

  '''                                                             
  return _IQ(*args,**kwargs)

iq.__doc__ += _FUNCTION_DOC


_GA = RBF(sympy.exp(-(_EPS*_R)**2))
def ga(*args,**kwargs):
  '''                        
  Gaussian, which is defined as

  .. math::
    \exp(-(\epsilon r)^2)

  '''
  return _GA(*args,**kwargs)

ga.__doc__ += _FUNCTION_DOC


_MQ = RBF(sympy.sqrt(1 + (_EPS*_R)**2))
def mq(*args,**kwargs):
  '''                     
  Multiquadratic, which is defined as

  .. math::
    (1 + (\epsilon r)^2)^{1/2}

  '''
  return _MQ(*args,**kwargs)

mq.__doc__ += _FUNCTION_DOC


