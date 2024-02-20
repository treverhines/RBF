'''
This module contains the `RBF` class, which is used to symbolically define and
numerically evaluate a radial basis function. `RBF` instances have been
predefined in this module for some of the commonly used radial basis functions.
The predefined radial basis functions are shown in the table below. For each
expression in the table, :math:`r = ||x - c||_2` and :math:`\epsilon` is a
shape parameter. :math:`x` and :math:`c` are the evaluation points and radial
basis function centers, respectively.

The below table also shows the conditionally positive definite (CPD) order for
each RBF. The definition of the CPD order can be found in Section 7.1. of [1].
For practical purposes, the CPD order is used to determine the degree of the
polynomial that should accompany the RBF so that the interpolation problem or
PDE being solved is well-posed. If solving an interpolation problem/PDE with an
RBF that has CPD order `m`, then the functional form of the solution should
include a polynomial with degree `m-1`. If the CPD order is 0, the RBF is
positive definite and requires no additional polynomial.

====================================  ================================================================================  =========
Name                                  Expression                                                                        CPD Order
====================================  ================================================================================  =========
phs8 (8th-order polyharmonic spline)  :math:`-(\epsilon r)^8\log(\epsilon r)`                                           5
phs7 (7th-order polyharmonic spline)  :math:`(\epsilon r)^7`                                                            4
phs6 (6th-order polyharmonic spline)  :math:`(\epsilon r)^6\log(\epsilon r)`                                            4
phs5 (5th-order polyharmonic spline)  :math:`-(\epsilon r)^5`                                                           3
phs4 (4th-order polyharmonic spline)  :math:`-(\epsilon r)^4\log(\epsilon r)`                                           3
phs3 (3rd-order polyharmonic spline)  :math:`(\epsilon r)^3`                                                            2
phs2 (2nd-order polyharmonic spline)  :math:`(\epsilon r)^2\log(\epsilon r)`                                            2
phs1 (1st-order polyharmonic spline)  :math:`-\epsilon r`                                                               1
mq (multiquadric)                     :math:`-(1 + (\epsilon r)^2)^{1/2}`                                               1
imq (inverse multiquadric)            :math:`(1 + (\epsilon r)^2)^{-1/2}`                                               0
iq (inverse quadratic)                :math:`(1 + (\epsilon r)^2)^{-1}`                                                 0
ga (Gaussian)                         :math:`\exp(-(\epsilon r)^2)`                                                     0
exp (exponential)                     :math:`\exp(-r/\epsilon)`                                                         0
se (squared exponential)              :math:`\exp(-r^2/(2\epsilon^2))`                                                  0
mat32 (Matern, v = 3/2)               :math:`(1 + \sqrt{3} r/\epsilon)\exp(-\sqrt{3} r/\epsilon)`                       0
mat52 (Matern, v = 5/2)               :math:`(1 + \sqrt{5} r/\epsilon + 5r^2/(3\epsilon^2))\exp(-\sqrt{5} r/\epsilon)`  0
wen10 (Wendland, dim=1, k=0)          :math:`(1 - r/\epsilon)_+`                                                        0
wen11 (Wendland, dim=1, k=1)          :math:`(1 - r/\epsilon)_+^3(3r/\epsilon + 1)`                                     0
wen12 (Wendland, dim=1, k=2)          :math:`(1 - r/\epsilon)_+^5(8r^2/\epsilon^2 + 5r/\epsilon + 1)`                   0
wen30 (Wendland, dim=3, k=0)          :math:`(1 - r/\epsilon)_+^2`                                                      0
wen31 (Wendland, dim=3, k=1)          :math:`(1 - r/\epsilon)_+^4(4r/\epsilon + 1)`                                     0
wen32 (Wendland, dim=3, k=2)          :math:`(1 - r/\epsilon)_+^6(35r^2/\epsilon^2 + 18r/\epsilon + 3)/3`               0
====================================  ================================================================================  =========

References
----------
[1] Fasshauer, G., Meshfree Approximation Methods with Matlab. World Scientific
Publishing Co, 2007.

'''
from __future__ import division
import logging
import weakref
import json
from pathlib import Path
from importlib import import_module

import sympy
import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from sympy.utilities.autowrap import ufuncify
from sympy import lambdify


from rbf.utils import assert_shape


logger = logging.getLogger(__name__)

class InplaceError(Exception):
    pass


R, EPS = sympy.symbols('r, eps')


class RBF(object):
    '''
    Stores a symbolic expression of a Radial Basis Function (RBF) and evaluates
    the expression numerically when called.

    Parameters
    ----------
    expr : sympy expression
        Sympy expression for the RBF. This must be a function of the symbolic
        variable `r`. `r` is the Euclidean distance to the RBF center. The
        expression may optionally be a function of the shape parameter `eps`.
        If `eps` is not provided then `r` is substituted with `r*eps`. The
        symbolic variables `r` and `eps` can be created with
        `r, eps = sympy.symbols('r, eps')` or can be retrieved through the
        module-level attributes `R` and `EPS`.

    tol : float or sympy expression, optional
        Distance from the RBF center within which the RBF expression or its
        derivatives are not numerically stable. The symbolically evaluated
        limit at the center is returned when evaluating points where
        `r <= tol`. This can be a float or a sympy expression containing `eps`.

        If the limit of the RBF at its center is known, then it can be manually
        specified with the `limits` arguments.

    supp : float or sympy expression, optional
        The support for the RBF if it is compact. The RBF is set to zero where
        `r > supp`, regardless of what `expr` evaluates to. This can be a float
        or a sympy expression containing `eps`. This should be preferred over
        using a piecewise expression for compact RBFs due to difficulties sympy
        has with evaluating limits of piecewise expressions.

    limits : dict, optional
        Contains the values of the RBF or its derivatives at the center. For
        example, `{(0,1): 2*eps}` indicates that the derivative with respect to
        the second spatial dimension is `2*eps` at `x = c`. If this dictionary
        is provided and `tol` is not `None`, then it will be searched before
        computing the limit symbolically.

    cpd_order : int, optional
        If the RBF is known to be conditionally positive definite, then specify
        the order here. This is saved as an attribute and is otherwise unused
        by this class.

    Examples
    --------
    Instantiate an inverse quadratic RBF

    >>> from rbf.basis import *
    >>> iq_expr = 1/(1 + (EPS*R)**2)
    >>> iq = RBF(iq_expr)

    Evaluate an inverse quadratic at 10 points ranging from -5 to 5. Note that
    the evaluation points and centers are two dimensional arrays

    >>> x = np.linspace(-5.0, 5.0, 10)[:, None]
    >>> center = np.array([[0.0]])
    >>> values = iq(x, center)

    Instantiate a sinc RBF. This has a singularity at the RBF center and it
    must be handled separately by specifying a number for `tol`.

    >>> import sympy
    >>> sinc_expr = sympy.sin(R)/R
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
        return self._expr

    @property
    def tol(self):
        return self._tol

    @property
    def supp(self):
        return self._supp

    @property
    def limits(self):
        return self._limits

    @property
    def cpd_order(self):
        return self._cpd_order

    @property
    def eps_is_divisor(self):
        '''`True` if `eps` divides `r` in the RBF expression.'''
        if not hasattr(self, '_eps_is_divisor'):
            x = sympy.symbols('x')
            self._eps_is_divisor = self.expr.subs(R/EPS, x).free_symbols == {x}

        return self._eps_is_divisor

    @property
    def eps_is_factor(self):
        '''`True` if `eps` multiplies `r` in the RBF expression.'''
        if not hasattr(self, '_eps_is_factor'):
            x = sympy.symbols('x')
            self._eps_is_factor = self.expr.subs(EPS*R, x).free_symbols == {x}

        return self._eps_is_factor

    def __new__(cls, *args, **kwargs):
        # this keeps track of RBF and RBF subclass instances
        instance = object.__new__(cls)
        cls._INSTANCES += [weakref.ref(instance)]
        return instance

    def __init__(self, expr, tol=None, supp=None, limits=None, cpd_order=None):
        if not issubclass(type(expr), sympy.Expr):
            raise ValueError('`expr` must be a sympy expression.')

        # make sure that `expr` does not contain any symbols other than `r` and
        # `eps`
        other_symbols = expr.free_symbols.difference({R, EPS})
        if len(other_symbols) != 0:
            raise ValueError(
                '`expr` cannot contain any symbols other than `r` and `eps`.'
                )

        if not expr.has(R):
            raise ValueError('`expr` must contain the symbol `r`.')

        if not expr.has(EPS):
            # if `eps` is not in the expression then substitute `eps*r` for `r`
            expr = expr.subs(R, EPS*R)

        self._expr = expr

        if tol is not None:
            # make sure `tol` is a scalar or a sympy expression of `eps`
            tol = sympy.sympify(tol)
            other_symbols = tol.free_symbols.difference({EPS})
            if len(other_symbols) != 0:
                raise ValueError(
                    '`tol` cannot contain any symbols other than `eps`.'
                    )

        self._tol = tol

        if supp is not None:
            # make sure `supp` is a scalar or a sympy expression of `eps`
            supp = sympy.sympify(supp)
            other_symbols = supp.free_symbols.difference({EPS})
            if len(other_symbols) != 0:
                raise ValueError(
                    '`supp` cannot contain any symbols other than `eps`.'
                    )

        self._supp = supp

        if limits is None:
            limits = {}

        self._limits = limits
        self._cpd_order = cpd_order
        ## create the cache for numerical functions
        self._cache = {}

    def __call__(self, x, c, eps=1.0, diff=None, out=None, out_compat=False):
        '''
        Numerically evaluates the RBF or its derivatives.

        Parameters
        ----------
        x : (..., N, D) float array
            Evaluation points in D-dimensional space.

        c : (..., M, D) float array
            RBF centers in D-dimensional space.

        eps : float or float array, optional
            Shape parameter for each RBF.

        diff : (D,) int array, optional
            Derivative order for each dimension. For example, providing
            (2, 0, 1) would cause this function to return the RBF after
            differentiating it twice along the first dimension and once along
            the third dimension.

        out : (..., N, M) float array, optional
            Array in which the output is written.

        Returns
        -------
        (..., N, M) float array
            The RBFs with centers `c` evaluated at `x`.

        Notes
        -----
        The default method for converting the symbolic RBF to a numeric
        function limits the number of spatial dimensions `D` to 15. If there
        are more than 15 dimensions, the `lambdify` method is automatically
        used instead. However, functions produced by this method are unable
        to perform in-place operations. Hence, an exception will be raised if
        both `D` > 15 and `out` is specified, unless `out_compat` is also set
        to true by the caller to signal that they are aware that intermediate
        arrays still need to be allocated.

        The derivative order can be arbitrarily high, but some RBFs, such as
        Wendland and Matern, become numerically unstable when the derivative
        order exceeds 2.

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (..., None, None), 'x')
        ndim = x.shape[-1]

        c = np.asarray(c, dtype=float)
        assert_shape(c, (..., None, ndim), 'c')

        eps = np.asarray(eps, dtype=float)
        eps = np.broadcast_to(eps, c.shape[:-1])

        # if `diff` is not given then take no derivatives
        if diff is None:
            diff = (0,)*ndim

        else:
            # make sure diff is immutable
            diff = tuple(diff)
            assert_shape(diff, (ndim,), 'diff')

        # add numerical function to cache if not already
        if diff not in self._cache:
            self._add_diff_to_cache(diff)

        # reshape x from (..., n, d) to (d, ..., n, 1)
        x = x.transpose(-1, *range(x.ndim - 1))[..., None]
        # reshape c from (..., m, d) to (d, ..., 1, m)
        c = c.transpose(-1, *range(c.ndim - 1))[..., None, :]
        # reshape eps from (..., m) to (..., 1, m)
        eps = eps[..., None, :]
        # evaluate the cached function for the given `x`, `c`, and `eps`
        func = self._cache[diff]
        if out is None:
            out = func(*x, *c, eps)
        elif isinstance(func, np.ufunc):
            #dim <= 15 ufunc, supports in-place ops
            out = func(*x, *c, eps, out=out)
        elif out_compat:
            #caller is aware that intermediate arrays may be needed
            out[...] = func(*x, *c, eps)
        else:
            raise InplaceError(
                '''Array allocation is necessary in more than 15 dimensions.
                Hence, "true" in-place operations are not possible. Pass
                `out_compat=True` to emulate in-place operations. This
                preserves syntactic consistency with code in 15 dimensions
                or less, but incurs memory and performance overhead costs
                due to the creation of intermediate arrays.
                '''
                )
        
        return out

    def center_value(self, eps=1.0, diff=(0,)):
        '''
        Returns the value at the center of the RBF for the given `eps` and
        `diff`. This is a faster alternative to determining the center value
        with `__call__`.

        Parameters
        ----------
        eps : float, optional
            Shape parameter.

        diff : tuple, optional
            Derivative order for each dimension.

        Returns
        -------
        float

        '''
        diff = tuple(diff)
        if diff not in self._cache:
            self._add_diff_to_cache(diff)

        x = (0.0,)*len(diff)
        return self._cache[diff](*x, *x, eps)

    def __repr__(self):
        if self.supp is not None:
            out = '<%s: %s (support=%s)>' % (
                type(self).__name__, str(self.expr), str(self.supp)
                )
        else:
            out = '<%s: %s>' % (type(self).__name__, str(self.expr))

        return out

    def _add_diff_to_cache(self, diff, tempdir=None):
        '''
        Symbolically differentiates the RBF and then converts the expression to
        a function which can be evaluated numerically.
        '''
        logger.debug(
            'Creating a numerical function for %s with the derivative %s ...'
            % (self, str(diff))
            )

        dim = len(diff)
        c_sym = sympy.symbols('c:%s' % dim)
        x_sym = sympy.symbols('x:%s' % dim)
        r_sym = sympy.sqrt(sum((xi-ci)**2 for xi, ci in zip(x_sym, c_sym)))

        # substitute 'r' in the RBF expression with the cartesian spatial
        # variables and differentiate the RBF with respect to them
        expr = self.expr.subs(R, r_sym)
        for xi, order in zip(x_sym, diff):
            expr = expr.diff(xi, order)

        # if `tol` is given, form a separate expression for the RBF near its
        # center
        if self.tol is not None:
            if diff in self.limits:
                # use a user-specified limit if available
                lim = self.limits[diff]

            else:
                logger.debug(
                    'Symbolically evaluating the limit at the center ...'
                    )

                # evaluate the limit of the RBF at (x0=tol+c0, x1=c1, x2=c2,
                # ...) as tol goes to zero.
                lim = expr.subs(zip(x_sym[1:], c_sym[1:]))
                lim = lim.limit(x_sym[0], c_sym[0])
                logger.debug('Limit at the center: %s' % lim)

            # create a piecewise symbolic function which is `lim` when `r_sym
            # <= tol` and `expr` otherwise. Use `<=` so that the tolerance can
            # be exactly zero.
            expr = sympy.Piecewise((lim, r_sym <= self.tol), (expr, True))

        if self.supp is not None:
            # create a piecewise symbolic function which is `expr` when `r_sym
            # <= supp` and 0 otherwise.
            expr = sympy.Piecewise((expr, r_sym <= self.supp), (0, True))

        if dim <= 15:
            func = ufuncify(
                x_sym + c_sym + (EPS,), expr, backend='numpy', tempdir=tempdir
                )

        else:
            func = lambdify(x_sym + c_sym + (EPS,), expr, modules=['numpy'])

        self._cache[diff] = func
        logger.debug('The numeric function has been created and cached.')

    def clear_cache(self):
        '''
        Clears the cache of numeric functions. Makes a cache dictionary if it
        does not already exist.
        '''
        self._cache = {}

    def __getstate__(self):
        # This method is needed for RBF instances to be picklable. The cached
        # numerical functions are not picklable and so we need to remove them
        # from the state dictionary.

        # make a shallow copy of the instances __dict__ so that we do not mess
        # with it
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
        variable `r`. `r` is the Euclidean distance to the RBF center. The
        expression may optionally be a function of the shape parameter `eps`.
        If `eps` is not provided then `r` is substituted with `r*eps`. The
        symbolic variables `r` and `eps` can be created with
        `r, eps = sympy.symbols('r, eps')` or can be retrieved through the
        module-level attributes `R` and `EPS`.

    supp : float or sympy expression
        The support for the RBF. The RBF is set to zero where `r > supp`,
        regardless of what `expr` evaluates to. This can be a float or a sympy
        expression containing `eps`.

    tol : float or sympy expression, optional
        Distance from the RBF center within which the RBF expression or its
        derivatives are not numerically stable. The symbolically evaluated
        limit at the center is returned when evaluating points where
        `r <= tol`. This can be a float or a sympy expression containing `eps`.

        If the limit of the RBF at its center is known, then it can be manually
        specified with the `limits` arguments.

    limits : dict, optional
        Contains the values of the RBF or its derivatives at the center. For
        example, `{(0,1): 2*eps}` indicates that the derivative with respect to
        the second spatial dimension is `2*eps` at `x = c`. If this dictionary
        is provided and `tol` is not `None`, then it will be searched before
        computing the limit symbolically.

    cpd_order : int, optional
        If the RBF is known to be conditionally positive definite, then specify
        the order here. This is saved as an attribute and is otherwise unused
        by this class.

    '''
    def __init__(self, expr, supp, **kwargs):
        # make `supp` a required argument
        RBF.__init__(self, expr, supp=supp, **kwargs)

    def __call__(self, x, c, eps=1.0, diff=None):
        '''
        Numerically evaluates the RBF or its derivatives.

        Parameters
        ----------
        x : (N, D) float array
            Evaluation points in D-dimensional space.

        c : (M, D) float array
            RBF centers in D-dimensional space.

        eps : float, optional
            Shape parameter.

        diff : (D,) int array, optional
            Derivative order for each dimension. For example, providing
            (2, 0, 1) would cause this function to return the RBF after
            differentiating it twice along the first dimension and once along
            the third dimension.

        Returns
        -------
        out : (N, M) csc sparse matrix
            The RBFs with centers `c` evaluated at `x`.

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (None, None), 'x')
        ndim = x.shape[1]

        c = np.asarray(c, dtype=float)
        assert_shape(c, (None, ndim), 'c')

        if not np.isscalar(eps):
            raise NotImplementedError('`eps` must be a scalar.')

        if diff is None:
            diff = (0,)*ndim

        else:
            # make sure diff is immutable
            diff = tuple(diff)
            assert_shape(diff, (ndim,), 'diff')

        # add numerical function to cache if not already
        if diff not in self._cache:
            self._add_diff_to_cache(diff)

        # convert self.supp from a sympy expression to a float
        supp = float(self.supp.subs(EPS, eps))

        # find the nonzero entries based on distances between `x` and `c`
        xtree = cKDTree(x)
        ctree = cKDTree(c)
        # `idx` contains the indices of `x` that are within `supp` of each
        # point in `c`
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
            # `m` is the number of nodes in `x` close to `c[i]`
            m = len(idxi)
            data[n:n + m] = self._cache[diff](*x[idxi].T, *c[i], eps)
            rows[n:n + m] = idxi
            cols[n:n + m] = i
            n += m

        # convert to a csc_matrix
        out = csc_matrix((data, (rows, cols)), (len(x), len(c)))
        return out


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


def get_rbf(value):
    '''
    Returns the `RBF` corresponding to `value`. If `value` is a string, then
    this return the correspondingly named predefined `RBF`. If `value` is an
    RBF instance then this returns `value`.
    '''
    if issubclass(type(value), RBF):
        return value

    elif value in _PREDEFINED:
        return _PREDEFINED[value]

    else:
        raise ValueError(
            "Cannot interpret '%s' as an RBF. Use one of %s"
            % (value, set(_PREDEFINED.keys()))
            )


def clear_rbf_caches():
    '''
    Clear the caches of numerical functions for all the RBF instances
    '''
    for inst in RBF._INSTANCES:
        if inst() is not None:
            inst().clear_cache()


def add_precompiled_to_rbf_caches():
    '''
    Some commonly used numeric functions have been compiled at build-time (as
    universal functions) to save time at run-time. This loads those numeric
    functions into the corresponding RBF caches.
    '''
    metadata_file = Path(__file__).parent / '_rbf_ufuncs' / 'metadata.json'
    metadata = json.loads(metadata_file.read_text())
    for itm in metadata:
        inst = get_rbf(itm['rbf'])
        func = getattr(import_module(itm['module']), itm['function'])
        inst._cache[tuple(itm['diff'])] = func


## Instantiate some common RBFs
#####################################################################
# polyharmonic splines
phs8 = RBF(-(EPS*R)**8*sympy.log(EPS*R), tol=0.0, cpd_order=5)
phs7 = RBF( (EPS*R)**7,                  tol=0.0, cpd_order=4)
phs6 = RBF( (EPS*R)**6*sympy.log(EPS*R), tol=0.0, cpd_order=4)
phs5 = RBF(-(EPS*R)**5,                  tol=0.0, cpd_order=3)
phs4 = RBF(-(EPS*R)**4*sympy.log(EPS*R), tol=0.0, cpd_order=3)
phs3 = RBF( (EPS*R)**3,                  tol=0.0, cpd_order=2)
phs2 = RBF( (EPS*R)**2*sympy.log(EPS*R), tol=0.0, cpd_order=2)
phs1 = RBF(-(EPS*R),                     tol=0.0, cpd_order=1)

# inverse multiquadric
imq = RBF(1/sympy.sqrt(1 + (EPS*R)**2), cpd_order=0)

# inverse quadratic
iq = RBF(1/(1 + (EPS*R)**2), cpd_order=0)

# Gaussian
ga = RBF(sympy.exp(-(EPS*R)**2), cpd_order=0)

# multiquadric
mq = RBF(-sympy.sqrt(1 + (EPS*R)**2), cpd_order=1)

# exponential
exp = RBF(sympy.exp(-R/EPS), cpd_order=0)

# squared exponential
se = RBF(sympy.exp(-R**2/(2*EPS**2)), cpd_order=0)

# Matern
mat32 = RBF((1 + sympy.sqrt(3)*R/EPS) * sympy.exp(-sympy.sqrt(3)*R/EPS), tol=1e-8*EPS, cpd_order=0)
mat52 = RBF((1 + sympy.sqrt(5)*R/EPS + 5*R**2/(3*EPS**2)) * sympy.exp(-sympy.sqrt(5)*R/EPS), tol=1e-4*EPS, cpd_order=0)

# Wendland
wen10 = RBF((1 - R/EPS),                                  supp=EPS, tol=1e-8*EPS, cpd_order=0)
wen11 = RBF((1 - R/EPS)**3*(3*R/EPS + 1),                 supp=EPS, tol=1e-8*EPS, cpd_order=0)
wen12 = RBF((1 - R/EPS)**5*(8*R**2/EPS**2 + 5*R/EPS + 1), supp=EPS, tol=1e-8*EPS, cpd_order=0)

wen30 = RBF((1 - R/EPS)**2,                                   supp=EPS, tol=1e-8*EPS, cpd_order=0)
wen31 = RBF((1 - R/EPS)**4*(4*R/EPS + 1),                     supp=EPS, tol=1e-8*EPS, cpd_order=0)
wen32 = RBF((1 - R/EPS)**6*(35*R**2/EPS**2 + 18*R/EPS + 3)/3, supp=EPS, tol=1e-8*EPS, cpd_order=0)

# sparse Wendland
spwen10 = SparseRBF((1 - R/EPS),                                  supp=EPS, tol=1e-8*EPS, cpd_order=0)
spwen11 = SparseRBF((1 - R/EPS)**3*(3*R/EPS + 1),                 supp=EPS, tol=1e-8*EPS, cpd_order=0)
spwen12 = SparseRBF((1 - R/EPS)**5*(8*R**2/EPS**2 + 5*R/EPS + 1), supp=EPS, tol=1e-8*EPS, cpd_order=0)

spwen30 = SparseRBF((1 - R/EPS)**2,                                   supp=EPS, tol=1e-8*EPS, cpd_order=0)
spwen31 = SparseRBF((1 - R/EPS)**4*(4*R/EPS + 1),                     supp=EPS, tol=1e-8*EPS, cpd_order=0)
spwen32 = SparseRBF((1 - R/EPS)**6*(35*R**2/EPS**2 + 18*R/EPS + 3)/3, supp=EPS, tol=1e-8*EPS, cpd_order=0)

_PREDEFINED = {
    'phs8':phs8, 'phs7':phs7, 'phs6':phs6, 'phs5':phs5, 'phs4':phs4,
    'phs3':phs3, 'phs2':phs2, 'phs1':phs1, 'mq':mq, 'imq':imq, 'iq':iq,
    'ga':ga, 'exp':exp, 'se':se, 'mat32':mat32, 'mat52':mat52, 'wen10':wen10,
    'wen11':wen11, 'wen12':wen12, 'wen30':wen30, 'wen31':wen31, 'wen32':wen32,
    'spwen10':spwen10, 'spwen11':spwen11, 'spwen12':spwen12, 'spwen30':spwen30,
    'spwen31':spwen31, 'spwen32':spwen32
    }

add_precompiled_to_rbf_caches()
