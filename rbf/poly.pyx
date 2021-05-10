'''
This module is used for evaluating the monomial basis functions which are
commonly added to RBF interpolants
'''
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
from scipy.special import comb

from rbf.utils import assert_shape

from cython cimport boundscheck, wraparound


def mvmonos(x, degree, diff=None):
    '''
    Multivariate monomial basis functions.

    Parameters
    ----------
    x : (..., D) float array
        Positions where the monomials will be evaluated.

    degree : int or (M, D) int array
        If this is an int, it is the degree of polynomials spanned by the
        monomials. If this is an array, it is the power for each variable in
        each monomial.

    diff : (D,) int array, optional
        Derivative order for each variable.

    Returns
    -------
    (..., M) array
        Alternant matrix where each monomial is evaluated at `x`.

    Example
    -------

    >>> pos = np.array([[1.0], [2.0], [3.0]])
    >>> pows = np.array([[0], [1], [2]])
    >>> mvmonos(pos, pows)
    array([[ 1., 1., 1.],
           [ 1., 2., 4.],
           [ 1., 3., 9.]])

    >>> pos = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    >>> pows = np.array([[0, 0], [1, 0], [0, 1]])
    >>> mvmonos(pos, pows)
    array([[ 1., 1., 2.],
           [ 1., 2., 3.],
           [ 1., 3., 4.]])

    '''
    x = np.asarray(x, dtype=float)
    assert_shape(x, (..., None), 'x')
    ndim = x.shape[-1]

    if np.isscalar(degree):
        powers = monomial_powers(degree, ndim)
    else:
        powers = np.asarray(degree, dtype=int)
        assert_shape(powers, (None, ndim), 'powers')

    if diff is None:
        diff = np.zeros(ndim, dtype=int)
    else:
        diff = np.asarray(diff, dtype=int)
        assert_shape(diff, (ndim,), 'diff')

    x_flat = x.reshape((-1, ndim))
    out_flat = _mvmonos(x_flat, powers, diff)
    out = out_flat.reshape(x.shape[:-1] + (-1,))
    return out


@boundscheck(False)
@wraparound(False)
def _mvmonos(double[:, :] x, long[:, :] powers, long[:] diff):
    '''
    Cython evaluation of mvmonos.
    '''
    out_array = np.ones((x.shape[0], powers.shape[0]), dtype=float)
    cdef:
        long i, j, k, l
        # number of spatial dimensions
        long D = x.shape[1]
        # number of monomials
        long M = powers.shape[0]
        # number of positions where the monomials are evaluated
        long N = x.shape[0]
        long coeff, power
        # `out` is the memoryview of the numpy array `out_array`
        double[:, :] out = out_array

    for i in range(D):
        for j in range(M):
            # find the monomial coefficients after differentiation
            coeff = 1
            for k in range(diff[i]):
                coeff *= powers[j, i] - k

            # if the monomial coefficient is zero then make sure the power
            # is also zero to prevent a zero division error
            if coeff == 0:
                power = 0
            else:
                power = powers[j, i] - diff[i]

            for l in range(N):
                out[l, j] *= coeff*x[l, i]**power

    return out_array


@lru_cache()
def monomial_powers(degree, ndim):
    '''
    Returns an array containing the powers for the monomial basis functions
    spanning polynomials with the given degree and number of dimensions.
    Calling this with a degree of -1 will return an empty array (no monomial
    basis functions).

    Parameters
    ----------
    degree : int

    ndim : int

    Example
    -------
    This will return the powers of x and y for each monomial term in a two
    dimensional polynomial with degree 1.

    >>> monomial_powers(1, 2)
    >>> array([[0, 0],
               [1, 0],
               [0, 1]])
    '''
    nmonos = comb(degree + ndim, ndim, exact=True)
    out = np.zeros((nmonos, ndim), dtype=int)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out[count, var] += 1

            count += 1

    return out


def monomial_count(degree, ndim):
    '''
    Returns the number of monomial basis functions in a polynomial with the
    given degree and number of dimensions.

    Parameters
    ----------
    degree : int

    ndim : int

    '''
    return comb(degree + ndim, ndim, exact=True)
