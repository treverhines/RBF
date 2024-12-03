'''
This module provides functions for generating RBF-FD weights
'''
from __future__ import division
from functools import lru_cache
import logging

import numpy as np
import scipy.sparse as sp

from rbf.basis import phs3, get_rbf
from rbf.poly import monomial_count, monomial_powers, mvmonos
from rbf.utils import assert_shape, KDTree
from rbf.linalg import as_array

logger = logging.getLogger(__name__)


@lru_cache()
def _max_poly_order(size, dim):
    '''
    Returns the maximum polynomial order allowed for the given stencil size and
    number of dimensions
    '''
    order = -1
    while monomial_count(order + 1, dim) <= size:
        order += 1

    return order


def weights(x, s, diffs, coeffs=None, phi=phs3, order=None, eps=1.0,
            sum_terms=True):
    '''
    Returns the weights which map a function's values at `s` to an
    approximation of that function's derivative at `x`. The weights are
    computed using the RBF-FD method described in [1]

    Parameters
    ----------
    x : (..., D) float array
        Target points where the derivative is being approximated

    s : (..., M, D) float array
        Stencils for each target point

    diffs : (D,) int array or (K, D) int array
        Derivative orders for each spatial dimension. For example `[2, 0]`
        indicates that the weights should approximate the second derivative
        with respect to the first spatial dimension in two-dimensional space.
        `diffs` can also be a (K, D) array, where each (D,) sub-array is a term
        in a differential operator. For example, the two-dimensional Laplacian
        can be represented as `[[2, 0], [0, 2]]`.

    coeffs : (K, ...) float array, optional
        Coefficients for each term in the differential operator specified with
        `diffs`. The coefficients can vary between target points. Defaults to
        an array of ones.

    phi : rbf.basis.RBF instance or str, optional
        Type of RBF. See `rbf.basis` for the available options.

    order : int, optional
        Order of the added polynomial. This defaults to the highest derivative
        order. For example, if `diffs` is `[[2, 0], [0, 1]]`, then this is set
        to 2.

    eps : float or float array, optional
        Shape parameter for each RBF

    sum_terms : bool, optional
        If `False`, weights are returned for each term in `diffs` rather than
        their sum.

    Returns
    -------
    (..., M) float array or (K, ..., M) float array
        RBF-FD weights for each target point

    Examples
    --------
    Calculate the weights for a one-dimensional second order derivative.

    >>> x = np.array([1.0])
    >>> s = np.array([[0.0], [1.0], [2.0]])
    >>> diff = (2,)
    >>> weights(x, s, diff)
    array([ 1., -2., 1.])

    Calculate the weights for estimating an x derivative from three points in a
    two-dimensional plane

    >>> x = np.array([0.25, 0.25])
    >>> s = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 1.0]])
    >>> diff = (1, 0)
    >>> weights(x, s, diff)
    array([ -1., 1., 0.])

    References
    ----------
    [1] Fornberg, B. and N. Flyer. A Primer on Radial Basis Functions with
    Applications to the Geosciences. SIAM, 2015.

    '''
    x = np.asarray(x, dtype=float)
    assert_shape(x, (..., None), 'x')
    ndim = x.shape[-1]

    s = np.asarray(s, dtype=float)
    assert_shape(s, (..., None, ndim), 's')
    ssize = s.shape[-2]

    diffs = np.asarray(diffs, dtype=int)
    diffs = np.atleast_2d(diffs)
    assert_shape(diffs, (None, ndim), 'diffs')
    nterms = diffs.shape[0]

    if coeffs is None:
        coeffs = np.ones(nterms, dtype=float)
    else:
        coeffs = np.asarray(coeffs, dtype=float)
        assert_shape(coeffs, (nterms, ...), 'coeffs')

    bcast = np.broadcast_shapes(x.shape[:-1], s.shape[:-2], coeffs.shape[1:])
    x = np.broadcast_to(x, bcast + x.shape[-1:])
    s = np.broadcast_to(s, bcast + s.shape[-2:])

    phi = get_rbf(phi)
    # get the maximum polynomial order allowed for this stencil size
    max_order = _max_poly_order(ssize, ndim)
    if order is None:
        # If the polynomial order is not specified, make it equal to the
        # derivative order, provided that the stencil size is large enough.
        order = diffs.sum(axis=1).max()
        order = min(order, max_order)

    if order > max_order:
        raise ValueError('Polynomial order is too high for the stencil size')

    # center the stencil on `x` for improved numerical stability
    x = x[..., None, :]
    s = s - x
    x = np.zeros_like(x)
    # get the powers for the added monomials
    pwr = monomial_powers(order, ndim)
    nmonos = pwr.shape[0]
    # evaluate the RBF and monomials at each point in the stencil. This becomes
    # the left-hand-side
    A = phi(s, s, eps=eps)
    P = mvmonos(s, pwr)
    Pt = P.swapaxes(-2, -1)
    Z = np.zeros((*bcast, nmonos, nmonos), dtype=float)
    LHS = np.block([[A, P], [Pt, Z]])
    # Evaluate the RBF and monomials at the target points for each term in the
    # differential operator. This becomes the right-hand-side.
    a = np.empty((*bcast, ssize, nterms))
    p = np.empty((*bcast, nmonos, nterms))
    for i in range(nterms):
        # convert to an array because phi may be a sparse RBF
        a[..., i] = as_array(phi(x, s, eps=eps, diff=diffs[i]))[..., 0, :]
        p[..., i] = mvmonos(x, pwr, diff=diffs[i])[..., 0, :]

    coeffs = np.moveaxis(coeffs, 0, -1)[..., None, :]
    rhs = np.concatenate((a, p), axis=-2)
    rhs *= coeffs
    if sum_terms:
        rhs = rhs.sum(axis=-1, keepdims=True)

    w = np.linalg.solve(LHS, rhs)[..., :ssize, :]
    if sum_terms:
        w = w[..., 0]
    else:
        w = np.moveaxis(w, -1, 0)

    return w

def weight_matrix(x, p, n, diffs,
                  coeffs=None,
                  phi='phs3',
                  order=None,
                  eps=1.0,
                  sum_terms=True,
                  chunk_size=1000):
    '''
    Returns a weight matrix which maps a function's values at `p` to an
    approximation of that function's derivative at `x`. This is a convenience
    function which first creates stencils and then computes the RBF-FD weights
    for each stencil.

    Parameters
    ----------
    x : (N, D) float array
        Target points where the derivative is being approximated

    p : (M, D) array
        Source points. The derivatives will be approximated with a weighted sum
        of values at these point.

    n : int
        The stencil size. Each target point will have a stencil made of the `n`
        nearest neighbors from `p`

    diffs : (D,) int array or (K, D) int array
        Derivative orders for each spatial dimension. For example `[2, 0]`
        indicates that the weights should approximate the second derivative
        with respect to the first spatial dimension in two-dimensional space.
        `diffs` can also be a (K, D) array, where each (D,) sub-array is a term
        in a differential operator. For example the two-dimensional Laplacian
        can be represented as `[[2, 0], [0, 2]]`.

    coeffs : (K,) or (K, N) float array, optional
        Coefficients for each term in the differential operator specified with
        `diffs`. The coefficients can vary between target points. Defaults to
        an array of ones.

    phi : rbf.basis.RBF instance or str, optional
        Type of RBF. Select from those available in `rbf.basis` or create your
        own.

    order : int, optional
        Order of the added polynomial. This defaults to the highest derivative
        order. For example, if `diffs` is `[[2, 0], [0, 1]]`, then this is set
        to 2.

    eps : float, optional
        Shape parameter for each RBF

    sum_terms : bool, optional
        If `False`, a matrix will be returned for each term in `diffs` rather
        than their sum.

    chunk_size : int, optional
        Break the target points into chunks with this size to reduce the memory
        requirements

    Returns
    -------
    (N, M) coo sparse matrix or K-tuple of (N, M) coo sparse matrices

    Examples
    --------
    Create a second order differentiation matrix in one-dimensional space

    >>> x = np.arange(4.0)[:, None]
    >>> W = weight_matrix(x, x, 3, (2,))
    >>> W.toarray()
    array([[ 1., -2.,  1., 0.],
           [ 1., -2.,  1., 0.],
           [ 0.,  1., -2., 1.],
           [ 0.,  1., -2., 1.]])

    '''
    x = np.asarray(x, dtype=float)
    assert_shape(x, (None, None), 'x')
    nx, ndim = x.shape

    p = np.asarray(p, dtype=float)
    assert_shape(p, (None, ndim), 'p')

    diffs = np.asarray(diffs, dtype=int)
    diffs = np.atleast_2d(diffs)
    assert_shape(diffs, (None, ndim), 'diffs')
    nterms = diffs.shape[0]

    if coeffs is None:
        coeffs = np.ones(nterms, dtype=float)
    else:
        coeffs = np.asarray(coeffs, dtype=float)
        assert_shape(coeffs, (nterms, ...), 'coeffs')

    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]

    coeffs = np.broadcast_to(coeffs, (nterms, nx))

    _, stencils = KDTree(p).query(x, n)
    if chunk_size is None:
        data = weights(
            x, p[stencils], diffs,
            coeffs=coeffs,
            phi=phi,
            order=order,
            eps=eps,
            sum_terms=sum_terms
            )
    else:
        if sum_terms:
            data = np.empty((nx, n), dtype=float)
        else:
            data = np.empty((nterms, nx, n), dtype=float)

        for start in range(0, nx, chunk_size):
            stop = start + chunk_size
            data[..., start:stop, :] = weights(
                x[start:stop], p[stencils[start:stop]], diffs,
                coeffs=coeffs[:, start:stop],
                phi=phi,
                order=order,
                eps=eps,
                sum_terms=sum_terms
                )

    rows = np.repeat(range(nx), n)
    cols = stencils.ravel()
    if sum_terms:
        data = data.ravel()
        out = sp.coo_matrix((data, (rows, cols)), (nx, len(p)))
    else:
        data = data.reshape(nterms, -1)
        out = tuple(
            sp.coo_matrix((d, (rows, cols)), (nx, len(p))) for d in data
            )

    return out
