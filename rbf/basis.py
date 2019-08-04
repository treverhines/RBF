''' 
This module contains the `RBF` class, which is used to symbolically define and
numerically evaluate a radial basis function. `RBF` instances have been
predefined in this module for some of the commonly used radial basis functions.
The predefined radial basis functions are shown in the table below. For each
expression in the table, :math:`r = ||x - c||_2` and :math:`\epsilon` is a
shape parameter. :math:`x` and :math:`c` are the evaluation points and radial
basis function centers, respectively. The names of the predefined `RBF`
instances are given in the "Abbreviation" column. The "Positive Definite"
column identifies whether the RBFs are always positive definite and, if not,
under what conditions they are positive definite. RBFs identified as being
"Conditional (order i)" are conditionally positive definite with order i as
defined in Section 7.1 of [1]. The Wendland class of RBFs are only positive
definite for the indicated number of spatial dimensions.

=================================  ============  =====================  ======================================
Name                               Abbreviation  Positive Definite      Expression
=================================  ============  =====================  ======================================
Eighth-order polyharmonic spline   phs8          Conditional (order 5)  :math:`(\epsilon r)^8\log(\epsilon r)`
Seventh-order polyharmonic spline  phs7          Conditional (order 4)  :math:`(\epsilon r)^7`
Sixth-order polyharmonic spline    phs6          Conditional (order 4)  :math:`(\epsilon r)^6\log(\epsilon r)`
Fifth-order polyharmonic spline    phs5          Conditional (order 3)  :math:`(\epsilon r)^5`
Fourth-order polyharmonic spline   phs4          Conditional (order 3)  :math:`(\epsilon r)^4\log(\epsilon r)`
Third-order polyharmonic spline    phs3          Conditional (order 2)  :math:`(\epsilon r)^3`
Second-order polyharmonic spline   phs2          Conditional (order 2)  :math:`(\epsilon r)^2\log(\epsilon r)`
First-order polyharmonic spline    phs1          Conditional (order 1)  :math:`\epsilon r`
Multiquadratic                     mq            Conditional (order 1)  :math:`(1 + (\epsilon r)^2)^{1/2}`
Inverse multiquadratic             imq           Yes                    :math:`(1 + (\epsilon r)^2)^{-1/2}`
Inverse quadratic                  iq            Yes                    :math:`(1 + (\epsilon r)^2)^{-1}`
Gaussian                           ga            Yes                    :math:`\exp(-(\epsilon r)^2)`
Exponential                        exp           Yes                    :math:`\exp(-r/\epsilon)`
Squared Exponential                se            Yes                    :math:`\exp(-r^2/(2\epsilon^2))`
Matern (v = 3/2)                   mat32         Yes                    :math:`(1 + \sqrt{3} r/\epsilon)\exp(-\sqrt{3} r/\epsilon)`
Matern (v = 5/2)                   mat52         Yes                    :math:`(1 + \sqrt{5} r/\epsilon + 5r^2/(3\epsilon^2))\exp(-\sqrt{5} r/\epsilon)`
Wendland (d=1, k=0)                wen10         Yes (1-D only)         :math:`(1 - r/\epsilon)_+`
Wendland (d=1, k=1)                wen11         Yes (1-D only)         :math:`(1 - r/\epsilon)_+^3(3r/\epsilon + 1)`
Wendland (d=1, k=2)                wen12         Yes (1-D only)         :math:`(1 - r/\epsilon)_+^5(8r^2/\epsilon^2 + 5r/\epsilon + 1)`
Wendland (d=3, k=0)                wen30         Yes (1, 2, and 3-D)    :math:`(1 - r/\epsilon)_+^2`
Wendland (d=3, k=1)                wen31         Yes (1, 2, and 3-D)    :math:`(1 - r/\epsilon)_+^4(4r/\epsilon + 1)`
Wendland (d=3, k=2)                wen32         Yes (1, 2, and 3-D)    :math:`(1 - r/\epsilon)_+^6(35r^2/\epsilon^2 + 18r/\epsilon + 3)/3`
=================================  ============  =====================  ======================================

References
----------
[1] Fasshauer, G., Meshfree Approximation Methods with Matlab. World Scientific
Publishing Co, 2007.

''' 
from __future__ import division 
import logging
import weakref

import sympy 
import numpy as np 
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from sympy.utilities.autowrap import ufuncify
from sympy import lambdify

from rbf.poly import powers
from rbf.utils import assert_shape

logger = logging.getLogger(__name__)


# the method used to convert sympy expressions to numeric functions
_SYMBOLIC_TO_NUMERIC_METHOD = 'ufuncify'


def get_r():
  ''' 
  returns the symbolic variable for :math:`r` which is used to instantiate an
  `RBF`
  '''
  return sympy.symbols('r')


def get_eps():
  ''' 
  returns the symbolic variable for :math:`\epsilon` which is used to
  instantiate an `RBF`
  '''
  return sympy.symbols('eps')


_EPS = get_eps()


_R = get_r()


class RBF(object):
  ''' 
  Stores a symbolic expression of a Radial Basis Function (RBF) and evaluates
  the expression numerically when called.
  
  Parameters
  ----------
  expr : sympy expression
    Sympy expression for the RBF. This must be a function of the symbolic
    variable `r`, which can be obtained by calling `get_r()` or
    `sympy.symbols('r')`. `r` is the radial distance to the RBF center.  The
    expression may optionally be a function of `eps`, which is a shape
    parameter obtained by calling `get_eps()` or `sympy.symbols('eps')`.  If
    `eps` is not provided then `r` is substituted with `r*eps`.
  
  tol : float or sympy expression, optional  
    This is for when an RBF or its derivatives contain a removable singularity
    at the center. If `tol` is specified, then the limiting value of the RBF at
    its center will be evaluated symbolically, and that limit will be returned
    for all evaluation points, `x`, that are within `tol` of the RBF center,
    `c`. If the limit of the RBF at `x = c` is known, then it can be manually
    specified with the `limits` arguments. `tol` can be a float or a sympy
    expression containing `eps`.

  limits : dict, optional
    Contains the values of the RBF or its derivatives at the center. For
    example, `{(0,1):2*eps}` indicates that the derivative with respect to the
    second spatial dimension is `2*eps` at `x = c`. If this dictionary is
    provided and `tol` is not `None`, then it will be searched before
    estimating the limit with the method describe above.

  Examples
  --------
  Instantiate an inverse quadratic RBF

  >>> from rbf.basis import *
  >>> r = get_r()
  >>> eps = get_eps()
  >>> iq_expr = 1/(1 + (eps*r)**2)
  >>> iq = RBF(iq_expr)
  
  Evaluate an inverse quadratic at 10 points ranging from -5 to 5. Note that
  the evaluation points and centers are two dimensional arrays

  >>> x = np.linspace(-5.0, 5.0, 10)[:, None]
  >>> center = np.array([[0.0]])
  >>> values = iq(x, center)
    
  Instantiate a sinc RBF. This has a singularity at the RBF center and it must
  be handled separately by specifying a number for `tol`.
  
  >>> import sympy
  >>> sinc_expr = sympy.sin(r)/r
  >>> sinc = RBF(sinc_expr) # instantiate WITHOUT specifying `tol`
  >>> x = np.array([[-1.0], [0.0], [1.0]])
  >>> c = np.array([[0.0]])
  >>> sinc(x, c) # this incorrectly evaluates to nan at the center
  array([[ 0.84147098],
         [        nan],
         [ 0.84147098]])

  >>> sinc = RBF(sinc_expr, tol=1e-10) # instantiate specifying `tol`
  >>> sinc(x, c) # this now correctly evaluates to 1.0 at the center
  array([[ 0.84147098],
         [ 1.        ],
         [ 0.84147098]])
  
  '''
  _INSTANCES = []

  @property
  def expr(self):
    # `expr` is read-only. 
    return self._expr

  @property
  def tol(self):
    # `tol` is read-only
    return self._tol

  @property
  def limits(self):
    # `limits` is read-only
    return self._limits

  def __new__(cls, *args, **kwargs):
    # this keeps track of RBF and RBF subclass instances 
    instance = object.__new__(cls)    
    cls._INSTANCES += [weakref.ref(instance)]
    return instance
       
  def __init__(self, expr, tol=None, limits=None):
    ## SANITIZE `EXPR`
    # make sure `expr` is a sympy expression
    if not issubclass(type(expr), sympy.Expr):
      raise ValueError(
        '`expr` must be a sympy expression')
        
    # make sure that `expr` does not contain any symbols other than 
    # `r` and `eps`
    other_symbols = expr.free_symbols.difference({_R, _EPS})
    if len(other_symbols) != 0:
      raise ValueError(
        '`expr` cannot contain any symbols other than `r` and `eps`')
        
    # make sure that `expr` at least has `r`
    if not expr.has(_R):
      raise ValueError(
        '`expr` must contain the symbol `r`')
    
    if not expr.has(_EPS):
      # if `eps` is not in the expression then substitute `eps*r` for
      # `r`
      expr = expr.subs(_R, _EPS*_R)
      
    self._expr = expr
    
    ## SANITIZE `TOL`
    if tol is not None:
      # make sure `tol` is a scalar or a sympy expression of `eps`
      tol = sympy.sympify(tol)
      other_symbols = tol.free_symbols.difference({_EPS})
      if len(other_symbols) != 0:
        raise ValueError(
          '`tol` cannot contain any symbols other than `eps`')
  
    self._tol = tol

    ## SANITIZE `LIMITS`
    if limits is None:
      limits = {}
      
    self._limits = limits
    
    ## create the cache for numerical functions    
    self._cache = {}


  def __call__(self, x, c, eps=1.0, diff=None):
    ''' 
    Numerically evaluates the RBF or its derivatives.
    
    Parameters                                       
    ----------                                         
    x : (N, D) float array 
      Evaluation points
                                                                       
    c : (M, D) float array 
      RBF centers 
        
    eps : float or (M,) float array, optional
      Shape parameters for each RBF. Defaults to 1.0
                                                                           
    diff : (D,) int array, optional
      Specifies the derivative order for each spatial dimension. For example,
      if there are three spatial dimensions then providing (2, 0, 1) would
      cause this function to return the RBF after differentiating it twice
      along the first dimension and once along the third dimension.

    Returns
    -------
    (N, M) float array
      The RBFs with centers `c` evaluated at `x`

    Notes
    -----
    * The default method for converting the symbolic RBF to a numeric function
      limits the number of spatial dimensions `D` to 15. There is no such
      limitation when the conversion method is set to "lambdify". Set the
      conversion method using the function `set_symbolic_to_numeric_method`.

    * The derivative order can be arbitrarily high, but some RBFs, such as
      Wendland and Matern, become numerically unstable when the derivative
      order exceeds 2.

    '''
    x = np.asarray(x, dtype=float)
    assert_shape(x, (None, None), 'x')

    c = np.asarray(c, dtype=float)
    assert_shape(c, (None, x.shape[1]), 'c')

    # makes `eps` an array of constant values if it is a scalar
    if np.isscalar(eps):
      eps = np.full(c.shape[0], eps, dtype=float)

    else:  
      eps = np.asarray(eps, dtype=float)
      assert_shape(eps, (c.shape[0],), 'eps')

    # if `diff` is not given then take no derivatives
    if diff is None:
      diff = (0,)*x.shape[1]

    else:
      # make sure diff is immutable
      diff = tuple(diff)
      assert_shape(diff, (x.shape[1],), 'diff')

    # add numerical function to cache if not already
    if diff not in self._cache:
      self._add_diff_to_cache(diff)

    # expand to allow for broadcasting
    x = x.T[:, :, None] 
    c = c.T[:, None, :]
    args = (tuple(x) + tuple(c) + (eps,))
    # evaluate the cached function for the given `x`, `c`, and `eps`
    out = self._cache[diff](*args)
    return out

  def __repr__(self):
    out = '<RBF : %s>' % str(self.expr)
    return out
     
  def _add_diff_to_cache(self, diff):
    '''     
    Symbolically differentiates the RBF and then converts the expression to a
    function which can be evaluated numerically.
    '''   
    logger.debug('Creating a numerical function for the RBF %s with '
                 'the derivative %s ...' % (self, str(diff)))
    dim = len(diff)
    c_sym = sympy.symbols('c:%s' % dim)
    x_sym = sympy.symbols('x:%s' % dim)    
    r_sym = sympy.sqrt(sum((xi-ci)**2 for xi, ci in zip(x_sym, c_sym)))

    # substitute 'r' in the RBF expression with the cartesian spatial variables
    # and differentiate the RBF with respect to them
    expr = self.expr.subs(_R, r_sym)            
    for xi, order in zip(x_sym, diff):
      if order == 0:
        continue

      expr = expr.diff(*(xi,)*order)

    # if `tol` is given, form a separate expression for the RBF near its center
    if self.tol is not None:
      if diff in self.limits:
        # use a user-specified limit if available      
        lim = self.limits[diff]

      else: 
        logger.debug('Symbolically evaluating the RBF at its center ...')
        # evaluate the limit of the RBF at (x0=tol+c0, x1=c1, x2=c2, ...) as
        # tol goes to zero.
        lim = expr.subs(zip(x_sym[1:], c_sym[1:]))
        lim = lim.simplify()
        lim = lim.limit(x_sym[0], c_sym[0])
        logger.debug('Value of the RBF at its center: %s' % lim)

      # create a piecewise symbolic function which is `lim` when `r_sym < tol`
      # and `expr` otherwise
      expr = sympy.Piecewise((lim, r_sym < self.tol), (expr, True)) 

    if _SYMBOLIC_TO_NUMERIC_METHOD == 'ufuncify':      
      func = ufuncify(x_sym + c_sym + (_EPS,), expr, backend='numpy')

    elif _SYMBOLIC_TO_NUMERIC_METHOD == 'lambdify':
      func = lambdify(x_sym + c_sym + (_EPS,), expr, modules=['numpy'])

    else:
      raise ValueError()          
        
    self._cache[diff] = func
    logger.debug('The numeric function has been created and cached')
    
  def clear_cache(self):
    ''' 
    Clears the cache of numeric functions. Makes a cache dictionary if it does
    not already exist
    '''
    self._cache = {}
    
  def __getstate__(self):
    # This method is needed for RBF instances to be picklable. The cached
    # numerical functions are not picklable and so we need to remove them from
    # the state dictionary.

    # make a shallow copy of the instances __dict__ so that we do not mess with
    # it
    state = dict(self.__dict__)
    state['_cache'] = {}
    return state
  

class SparseRBF(RBF):
  ''' 
  Stores a symbolic expression of a compact Radial Basis Function (RBF) and
  evaluates the expression numerically when called. Calling a `SparseRBF`
  instance will return a csc sparse matrix.
  
  Parameters
  ----------
  expr : sympy expression
    Sympy expression for the RBF. This must be a function of the symbolic
    variable `r`, which can be obtained by calling `get_r()` or
    `sympy.symbols('r')`. `r` is the radial distance to the RBF center.  The
    expression may optionally be a function of `eps`, which is a shape
    parameter obtained by calling `get_eps()` or `sympy.symbols('eps')`.  If
    `eps` is not provided then `r` is substituted with `r*eps`.
  
  support : float or sympy expression
    Indicates the support of the RBF. The RBF is set to zero for radial
    distances greater than `support`, regardless of what `expr` evaluates to.
    This can be a float or a sympy expression containing `eps`.
    
  tol : float or sympy expression, optional  
    This is for when an RBF or its derivatives contain a removable singularity
    at the center. If `tol` is specified, then the limiting value of the RBF at
    its center will be evaluated symbolically, and that limit will be returned
    for all evaluation points, `x`, that are within `tol` of the RBF center,
    `c`. If the limit of the RBF at `x = c` is known, then it can be manually
    specified with the `limits` arguments. `tol` can be a float or a sympy
    expression containing `eps`.

  limits : dict, optional
    Contains the values of the RBF or its derivatives at the center. For
    example, `{(0, 1):2*eps}` indicates that the derivative with respect to the
    second spatial dimension is `2*eps` at `x = c`. If this dictionary is
    provided and `tol` is not `None`, then it will be searched before
    estimating the limit with the method describe above.

  ''' 
  @property
  def supp(self):
    return self._supp
  
  def __init__(self, expr, supp, **kwargs):
    RBF.__init__(self, expr, **kwargs)
    ## SANITIZE `SUPP`
    # make sure `supp` is a scalar or a sympy expression of `eps`
    supp = sympy.sympify(supp)
    other_symbols = supp.free_symbols.difference({_EPS})
    if len(other_symbols) != 0:
      raise ValueError(
        '`supp` cannot contain any symbols other than `eps`')
  
    self._supp = supp

  def __call__(self, x, c, eps=1.0, diff=None):
    ''' 
    Numerically evaluates the RBF or its derivatives.
    
    Parameters                                       
    ----------                                         
    x : (N, D) float array 
      Evaluation points
                                                                       
    c : (M, D) float array 
      RBF centers 
        
    eps : float, optional
      Shape parameter
                                                                           
    diff : (D,) int array, optional
      Specifies the derivative order for each Cartesian direction. For example,
      if there are three spatial dimensions then providing (2, 0, 1) would
      cause this function to return the RBF after differentiating it twice
      along the first axis and once along the third axis.

    Returns
    -------
    out : (N, M) csc sparse matrix
      The RBFs with centers `c` evaluated at `x`
      
    '''
    x = np.asarray(x, dtype=float)
    assert_shape(x, (None, None), 'x')

    c = np.asarray(c, dtype=float)
    assert_shape(c, (None, x.shape[1]), 'c')

    if not np.isscalar(eps):
      raise NotImplementedError(
        '`eps` must be a scalar for `SparseRBF` instances')

    # convert scalar to (1,) array
    eps = np.array([eps], dtype=float)

    if diff is None:
      diff = (0,)*x.shape[1]

    else:
      # make sure diff is immutable
      diff = tuple(diff)
      assert_shape(diff, (x.shape[1],), 'diff')

    # add numerical function to cache if not already
    if diff not in self._cache:
      self._add_diff_to_cache(diff)

    # convert self.supp from a sympy expression to a float
    supp = float(self.supp.subs(_EPS, eps[0]))

    # find the nonzero entries based on distances between `x` and `c`
    nx, nc = x.shape[0], c.shape[0]
    xtree = cKDTree(x)
    ctree = cKDTree(c)
    # `idx` contains the indices of `x` which are within `supp` of each node in
    # `c`
    idx = ctree.query_ball_tree(xtree, supp)

    # total nonzero entries in the output array
    nnz = sum(len(i) for i in idx)
    # allocate sparse matrix data
    data = np.zeros(nnz, dtype=float)
    rows = np.zeros(nnz, dtype=int)
    cols = np.zeros(nnz, dtype=int)
    # `n` is the total number of data entries thus far
    n = 0
    for i, idxi in enumerate(idx):
      # `m` is the number of nodes in `x` close to `c[[i]]`
      m = len(idxi)
      # properly shape `x` and `c` for broadcasting
      xi = x.T[:, idxi, None]
      ci = c.T[:, None, i][:, :, None]
      args = (tuple(xi) + tuple(ci) + (eps,))
      data[n:n + m] = self._cache[diff](*args)[:, 0]
      rows[n:n + m] = idxi
      cols[n:n + m] = i
      n += m

    # convert to a csc_matrix
    out = csc_matrix((data, (rows, cols)), (nx, nc))
    return out

  def __repr__(self):
    out = '<SparseRBF : %s (support = %s)>' % (str(self.expr), str(self.supp))
    return out

  
def clear_rbf_caches():
    '''
    Clear the caches of numerical functions for all the RBF instances
    '''
    for inst in RBF._INSTANCES:
        if inst() is not None:
            inst().clear_cache()


def get_rbf(val):
    '''
    Returns the `RBF` corresponding to `val`. If `val` is a string, then this
    return the correspondingly named predefined `RBF`. If `val` is an RBF
    instance then this returns `val`.
    '''
    if issubclass(type(val), RBF):
        return val

    elif val in _PREDEFINED:
        return _PREDEFINED[val]

    else:        
        raise ValueError(
            "Cannot interpret '%s' as an RBF. Use one of %s"
            % (val, set(_PREDEFINED.keys())))
        

def set_symbolic_to_numeric_method(method): 
  ''' 
  Sets the method that all RBF instances will use for converting sympy
  expressions to numeric functions. This can be either "ufuncify" or
  "lambdify". "ufuncify" will write and compile C code for a numpy universal
  function, and "lambdify" will evaluate the sympy expression using
  python-level numpy functions. Calling this function will cause all caches of
  numeric functions to be cleared.
  '''
  global _SYMBOLIC_TO_NUMERIC_METHOD
  if method not in {'lambdify', 'ufuncify'}:
    raise ValueError(
      '`method` must be either "lambdify" or "ufuncify"')
            
  _SYMBOLIC_TO_NUMERIC_METHOD = method
  clear_rbf_caches()


## Instantiate some common RBFs
#####################################################################
_phs8_limits = {}
_phs8_limits.update((tuple(i), 0.0) for i in powers(7, 1))
_phs8_limits.update((tuple(i), 0.0) for i in powers(7, 2))
_phs8_limits.update((tuple(i), 0.0) for i in powers(7, 3))
phs8 = RBF((_EPS*_R)**8*sympy.log(_EPS*_R), tol=1e-10, limits=_phs8_limits)

_phs7_limits = {}
_phs7_limits.update((tuple(i), 0.0) for i in powers(6, 1))
_phs7_limits.update((tuple(i), 0.0) for i in powers(6, 2))
_phs7_limits.update((tuple(i), 0.0) for i in powers(6, 3))
phs7 = RBF((_EPS*_R)**7, tol=1e-10, limits=_phs7_limits)

_phs6_limits = {}
_phs6_limits.update((tuple(i), 0.0) for i in powers(5, 1))
_phs6_limits.update((tuple(i), 0.0) for i in powers(5, 2))
_phs6_limits.update((tuple(i), 0.0) for i in powers(5, 3))
phs6 = RBF((_EPS*_R)**6*sympy.log(_EPS*_R), tol=1e-10, limits=_phs6_limits)

_phs5_limits = {}
_phs5_limits.update((tuple(i), 0.0) for i in powers(4, 1))
_phs5_limits.update((tuple(i), 0.0) for i in powers(4, 2))
_phs5_limits.update((tuple(i), 0.0) for i in powers(4, 3))
phs5 = RBF((_EPS*_R)**5, tol=1e-10, limits=_phs5_limits)

_phs4_limits = {}
_phs4_limits.update((tuple(i), 0.0) for i in powers(3, 1))
_phs4_limits.update((tuple(i), 0.0) for i in powers(3, 2))
_phs4_limits.update((tuple(i), 0.0) for i in powers(3, 3))
phs4 = RBF((_EPS*_R)**4*sympy.log(_EPS*_R), tol=1e-10, limits=_phs4_limits)

_phs3_limits = {}
_phs3_limits.update((tuple(i), 0.0) for i in powers(2, 1))
_phs3_limits.update((tuple(i), 0.0) for i in powers(2, 2))
_phs3_limits.update((tuple(i), 0.0) for i in powers(2, 3))
phs3 = RBF((_EPS*_R)**3, tol=1e-10, limits=_phs3_limits)

_phs2_limits = {}
_phs2_limits.update((tuple(i), 0.0) for i in powers(1, 1))
_phs2_limits.update((tuple(i), 0.0) for i in powers(1, 2))
_phs2_limits.update((tuple(i), 0.0) for i in powers(1, 3))
phs2 = RBF((_EPS*_R)**2*sympy.log(_EPS*_R), tol=1e-10, limits=_phs2_limits)

_phs1_limits = {}
_phs1_limits.update((tuple(i), 0.0) for i in powers(0, 1))
_phs1_limits.update((tuple(i), 0.0) for i in powers(0, 2))
_phs1_limits.update((tuple(i), 0.0) for i in powers(0, 3))
phs1 = RBF(_EPS*_R, tol=1e-10, limits=_phs1_limits)

# inverse multiquadratic
imq = RBF(1/sympy.sqrt(1 + (_EPS*_R)**2))

# inverse quadratic
iq = RBF(1/(1 + (_EPS*_R)**2))

# Gaussian
ga = RBF(sympy.exp(-(_EPS*_R)**2))

# multiquadratic
mq = RBF(sympy.sqrt(1 + (_EPS*_R)**2))

# exponential
exp = RBF(sympy.exp(-_R/_EPS))

# squared exponential
se = RBF(sympy.exp(-_R**2/(2*_EPS**2)))

# Matern
_mat32_limits = {(0,): 1.0, 
                 (0, 0): 1.0, 
                 (0, 0, 0): 1.0, 
                 (1,): 0.0, 
                 (1, 0): 0.0, 
                 (0, 1): 0.0, 
                 (1, 0, 0): 0.0, 
                 (0, 1, 0): 0.0, 
                 (0, 0, 1): 0.0, 
                 (2,): -3.0/_EPS**2,
                 (2, 0): -3.0/_EPS**2, 
                 (0, 2): -3.0/_EPS**2, 
                 (2, 0, 0): -3.0/_EPS**2, 
                 (0, 2, 0): -3.0/_EPS**2, 
                 (0, 0, 2): -3.0/_EPS**2, 
                 (1, 1): 0.0, 
                 (1, 1, 0): 0.0, 
                 (1, 0, 1): 0.0, 
                 (0, 1, 1): 0.0}

_mat52_limits = {(0,): 1.0, 
                 (0, 0): 1.0, 
                 (0, 0, 0): 1.0, 
                 (1,): 0.0, 
                 (1, 0): 0.0, 
                 (0, 1): 0.0, 
                 (1, 0, 0): 0.0, 
                 (0, 1, 0): 0.0, 
                 (0, 0, 1): 0.0, 
                 (2,): -5.0/(3.0*_EPS**2), 
                 (2, 0): -5.0/(3.0*_EPS**2), 
                 (0, 2): -5.0/(3.0*_EPS**2), 
                 (2, 0, 0): -5.0/(3.0*_EPS**2), 
                 (0, 2, 0): -5.0/(3.0*_EPS**2), 
                 (0, 0, 2): -5.0/(3.0*_EPS**2), 
                 (1, 1): 0.0,
                 (1, 1, 0): 0.0,
                 (1, 0, 1): 0.0,
                 (0, 1, 1): 0.0}

mat32 = RBF((1 + sympy.sqrt(3)*_R/_EPS)                       * sympy.exp(-sympy.sqrt(3)*_R/_EPS), tol=1e-8*_EPS, limits=_mat32_limits)

mat52 = RBF((1 + sympy.sqrt(5)*_R/_EPS + 5*_R**2/(3*_EPS**2)) * sympy.exp(-sympy.sqrt(5)*_R/_EPS), tol=1e-4*_EPS, limits=_mat52_limits)

# Wendland 
_wen10_limits = {(0,): 1.0}

_wen11_limits = {(0,): 1.0,
                 (1,): 0.0,
                 (2,): -12.0/_EPS**2}

_wen12_limits = {(0,): 1.0,
                 (1,): 0.0,
                 (2,): -14.0/_EPS**2}

_wen30_limits = {(0,): 1.0,
                 (0, 0): 1.0,
                 (0, 0, 0): 1.0}

_wen31_limits = {(0,): 1.0,
                 (0, 0): 1.0,
                 (0, 0, 0): 1.0,
                 (1,): 0.0,
                 (1, 0): 0.0,
                 (0, 1): 0.0,
                 (1, 0, 0): 0.0,
                 (0, 1, 0): 0.0,
                 (0, 0, 1): 0.0,
                 (2,): -20.0/_EPS**2,
                 (2, 0): -20.0/_EPS**2,
                 (0, 2): -20.0/_EPS**2,
                 (2, 0, 0): -20.0/_EPS**2,
                 (0, 2, 0): -20.0/_EPS**2,
                 (0, 0, 2): -20.0/_EPS**2,
                 (1, 1): 0.0,
                 (1, 1, 0): 0.0,
                 (1, 0, 1): 0.0,
                 (0, 1, 1): 0.0}

_wen32_limits = {(0,): 1.0,
                 (0, 0): 1.0,
                 (0, 0, 0): 1.0,
                 (1,): 0,
                 (1, 0): 0.0,
                 (0, 1): 0.0,
                 (1, 0, 0): 0.0,
                 (0, 1, 0): 0.0,
                 (0, 0, 1): 0.0,
                 (2,): -56.0/(3.0*_EPS**2),
                 (2, 0): -56.0/(3.0*_EPS**2),
                 (0, 2): -56.0/(3.0*_EPS**2),
                 (2, 0, 0): -56.0/(3.0*_EPS**2),
                 (0, 2, 0): -56.0/(3.0*_EPS**2),
                 (0, 0, 2): -56.0/(3.0*_EPS**2),
                 (1, 1): 0.0,
                 (1, 1, 0): 0.0,
                 (1, 0, 1): 0.0,
                 (0, 1, 1): 0.0}

wen10 = RBF(sympy.Piecewise(((1 - _R/_EPS)                                         , _R < _EPS), (0.0, True)), tol=1e-8*_EPS, limits=_wen10_limits)

wen11 = RBF(sympy.Piecewise(((1 - _R/_EPS)**3*(3*_R/_EPS + 1)                      , _R < _EPS), (0.0, True)), tol=1e-8*_EPS, limits=_wen11_limits) 

wen12 = RBF(sympy.Piecewise(((1 - _R/_EPS)**5*(8*_R**2/_EPS**2 + 5*_R/_EPS + 1)    , _R < _EPS), (0.0, True)), tol=1e-8*_EPS, limits=_wen12_limits) 

wen30 = RBF(sympy.Piecewise(((1 - _R/_EPS)**2                                      , _R < _EPS), (0.0, True)), tol=1e-8*_EPS, limits=_wen30_limits)  

wen31 = RBF(sympy.Piecewise(((1 - _R/_EPS)**4*(4*_R/_EPS + 1)                      , _R < _EPS), (0.0, True)), tol=1e-8*_EPS, limits=_wen31_limits) 

wen32 = RBF(sympy.Piecewise(((1 - _R/_EPS)**6*(35*_R**2/_EPS**2 + 18*_R/_EPS + 3)/3, _R < _EPS), (0.0, True)), tol=1e-8*_EPS, limits=_wen32_limits) 

# sparse Wendland 
spwen10 = SparseRBF(         (1 - _R/_EPS)                                         , _EPS, tol=1e-8*_EPS, limits=_wen10_limits)

spwen11 = SparseRBF(         (1 - _R/_EPS)**3*(3*_R/_EPS + 1)                      , _EPS, tol=1e-8*_EPS, limits=_wen11_limits)

spwen12 = SparseRBF(         (1 - _R/_EPS)**5*(8*_R**2/_EPS**2 + 5*_R/_EPS + 1)    , _EPS, tol=1e-8*_EPS, limits=_wen12_limits)

spwen30 = SparseRBF(         (1 - _R/_EPS)**2                                      , _EPS, tol=1e-8*_EPS, limits=_wen30_limits)

spwen31 = SparseRBF(         (1 - _R/_EPS)**4*(4*_R/_EPS + 1)                      , _EPS, tol=1e-8*_EPS, limits=_wen31_limits)

spwen32 = SparseRBF(         (1 - _R/_EPS)**6*(35*_R**2/_EPS**2 + 18*_R/_EPS + 3)/3, _EPS, tol=1e-8*_EPS, limits=_wen32_limits)

_PREDEFINED = {'phs8':phs8, 'phs7':phs7, 'phs6':phs6, 'phs5':phs5,
               'phs4':phs4, 'phs3':phs3, 'phs2':phs2, 'phs1':phs1,
               'mq':mq, 'imq':imq, 'iq':iq, 'ga':ga, 'exp':exp,
               'se':se, 'mat32':mat32, 'mat52':mat52, 
               'wen10':wen10, 'wen11':wen11, 'wen12':wen12,
               'wen30':wen30, 'wen31':wen31, 'wen32':wen32,
               'spwen10':spwen10, 'spwen11':spwen11, 
               'spwen12':spwen12, 'spwen30':spwen30, 
               'spwen31':spwen31, 'spwen32':spwen32}

