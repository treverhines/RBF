'''
Utility functions for scipy sparse matrices
'''
import logging

import numpy as np
import scipy.sparse as sp

from rbf.utils import assert_shape

from cython cimport boundscheck, wraparound
from libc.stdint cimport int32_t, int64_t

logger = logging.getLogger(__name__)


@boundscheck(False)
@wraparound(False)
def _coo_row_norms(double[:] data,
                   int32_t[:] row,
                   int64_t nrows,
                   int64_t order):
    '''
    Computes the row norms of a COO matrix
    '''
    cdef:
        int64_t i
        int64_t ndata = data.shape[0]
        double[:] arr = np.zeros((nrows,), dtype=float)

    for i in range(ndata):
        arr[row[i]] += abs(data[i])**order

    out = np.asarray(arr)
    out = out**(1.0/order)
    return out


def row_norms(A, order=2):
    '''
    Computes the norm of each row in `A`. This is more memory efficient than
    scipy.sparse.linalg.norm because it does not make a full copy of the matrix

    Parameters
    ----------
    A: (n, m) float sparse matrix
        CSC, CSR, BSR, or COO matrix for best efficiency

    order: int

    Returns
    -------
    (n,) float array

    '''
    if not sp.isspmatrix(A):
        raise ValueError('`A` must be a sparse matrix')

    # If A is csr, bsr, csc, or coo then the data will not be copied when
    # converting to coo. Otherwise, my request to not make a copy of the data
    # will be silently ignored.
    A = A.tocoo(copy=False)
    out = _coo_row_norms(A.data, A.row, A.shape[0], order)
    return out


def divide_rows(A, x, inplace=False):
    '''
    Divide the rows of the sparse matrix `A` by `x`. If `inplace` is `True` and
    `A` is CSC, CSR, BSR, or COO, then the operation is done in place.

    Parameters
    ----------
    A: (n, m) sparse matrix
        CSC, CSR, BSR, or COO matrix for best efficiency

    x: (n,) float array

    inplace: bool
        Whether to modify the data in `A` inplace. This is only possible if `A`
        is CSC, CSR, BSR, or COO.

    Returns
    -------
    (n, m) COO sparse matrix

    '''
    x = np.asarray(x, dtype=float)
    if not sp.isspmatrix(A):
        raise ValueError('`A` must be a sparse matrix')

    if inplace and not (sp.isspmatrix_csc(A) |
                        sp.isspmatrix_csr(A) |
                        sp.isspmatrix_bsr(A) |
                        sp.isspmatrix_coo(A)):
        logger.warning(
            'The data for the sparse matrix will not be modified in place. To '
            'do so, the sparse matrix should be CSC, CSR, BSR, or COO.')

    assert_shape(x, (A.shape[0],), 'x')
    out = A.tocoo(copy=not inplace)
    out.data /= x[out.row]
    return out


def add_rows(A, B, idx):
    '''
    This adds the sparse matrix `B` to rows `idx` of the sparse matrix `A`

    Parameters
    ----------
    A: (n1, m) sparse matrix
        CSC, CSR, BSR, or COO matrix for best efficiency

    B: (n2, m) sparse matrix
        CSC, CSR, BSR, or COO matrix for best efficiency

    idx: (n2,) int array
        rows of `A` that `B` will be added to

    Returns
    -------
    (n1, m) COO sparse matrix

    '''
    idx = np.asarray(idx, dtype=np.int32)
    if not sp.isspmatrix(A):
        raise ValueError('`A` must be a sparse matrix')

    if not sp.isspmatrix(B):
        raise ValueError('`B` must be a sparse matrix')

    assert_shape(B, (None, A.shape[1]), 'B')
    assert_shape(idx, (B.shape[0],), 'idx')

    A = A.tocoo(copy=False)
    B = B.tocoo(copy=False)

    data = np.hstack((A.data, B.data))
    row = np.hstack((A.row, idx[B.row]))
    col = np.hstack((A.col, B.col))
    out = sp.coo_matrix((data, (row, col)), shape=A.shape)
    return out


def expand_rows(A, rows, rout, copy=False):
    '''
    expand `A` along the rows.

    Parameters
    ----------
    A : (rin, cin) sparse matrix
        CSC, CSR, BSR, or COO matrix for best efficiency

    rows : (rin,) int array
        The row in the output matrix that each row in `A` will be assigned to

    rout : int
        The number of rows in the output matrix

    copy : bool,
        Whether the data in the output matrix should be a copy of the data in
        `A`

    Returns
    -------
    (rout, cin) COO sparse matrix

    '''
    if not sp.isspmatrix(A):
        raise ValueError('`A` must be a sparse matrix')

    A = A.tocoo(copy=copy)
    rin, cin = A.shape

    rows = np.asarray(rows, dtype=np.int32)
    assert_shape(rows, (rin,), 'rows')

    out = sp.coo_matrix((A.data, (rows[A.row], A.col)), shape=(rout, cin))
    return out


def expand_cols(A, cols, cout, copy=False):
    '''
    expand `A` along the columns.

    Parameters
    ----------
    A : (rin, cin) sparse matrix
        CSC, CSR, BSR, or COO matrix for best efficiency

    cols : (rin,) int array
        The column in the output matrix that each column in `A` will be
        assigned to

    cout : int
        The number of columns in the output matrix

    copy : bool,
        Whether the data in the output matrix should be a copy of the data in
        `A`

    Returns
    -------
    (rin, cout) COO sparse matrix

    '''
    if not sp.isspmatrix(A):
        raise ValueError('`A` must be a sparse matrix')

    A = A.tocoo(copy=copy)
    rin, cin = A.shape

    cols = np.asarray(cols, dtype=np.int32)
    assert_shape(cols, (cin,), 'cols')

    out = sp.coo_matrix((A.data, (A.row, cols[A.col])), shape=(rin, cout))
    return out
