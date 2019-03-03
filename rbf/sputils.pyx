'''
Utility functions for scipy sparse matrices
'''
import logging

import numpy as np
import scipy.sparse as sp

from rbf.utils import assert_shape

cimport numpy as np
from cython cimport boundscheck, wraparound

logger = logging.getLogger(__name__)

@boundscheck(False)
@wraparound(False)
cpdef np.ndarray _coo_row_norms(double[:] data, 
                                int[:] row, 
                                long nrows,
                                long order):
    '''
    Computes the row norms of a COO matrix
    '''
    cdef:
        long i
        long ndata = data.shape[0]
        double[:] arr = np.zeros((nrows,), dtype=float)

    for i in range(ndata):
        arr[row[i]] += abs(data[i])**order

    out = np.asarray(arr)
    out = out**(1.0/order)
    return out

    
def row_norms(A, order=2):
    '''
    Computes the norm of each row in `A`. This is more memory
    efficient than scipy.sparse.linalg.norm because it does not make a
    full copy of the matrix

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
    
    # If A is csr, bsr, csc, or coo then the data will not be copied
    # when converting to coo. Otherwise, my request to not make a copy
    # of the data will be silently ignored.
    A = A.tocoo(copy=False)
    out = _coo_row_norms(A.data, A.row, A.shape[0], order)
    return out    
    
    
def divide_rows(A, x, inplace=False):
    '''
    Divide the rows of the sparse matrix `A` by `x`. If `inplace` is
    `True` and `A` is CSC, CSR, BSR, or COO, then the operation is
    done in place.

    Parameters
    ----------
    A: (n, m) sparse matrix
        CSC, CSR, BSR, or COO matrix for best efficiency
    
    x: (n,) float array    

    inplace: bool
        Whether to modify the data in `A` inplace. This is only
        possible if `A` is CSC, CSR, BSR, or COO. 

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
            'The data for the sparse matrix will not being modified '
            'in place. To do so, the sparse matrix should be CSC, '
            'CSR, BSR, or COO.')

    assert_shape(x, (A.shape[0],), 'x')
    Acoo = A.tocoo(copy=not inplace)
    Acoo.data /= x[Acoo.row]
    return Acoo
    

def add_rows(A, B, idx):
    '''
    This adds the sparse matrix `B` to rows `idx` of the sparse matrix
    `A`

    Parameters
    ----------
    A: (n1, m) sparse matrix

    B: (n2, m) sparse matrix

    idx: (n2,) int array
        rows of `A` that `B` will be added to

    Returns
    -------
    (n1, m) CSC sparse matrix

    '''
    idx = np.asarray(idx, dtype=int)
    if not sp.isspmatrix(A):
        raise ValueError('`A` must be a sparse matrix')

    if not sp.isspmatrix(B):
        raise ValueError('`B` must be a sparse matrix')

    assert_shape(B, (None, A.shape[1]), 'B')
    assert_shape(idx, (B.shape[0],), 'idx')
    # coerce `A` to csc to ensure csc output
    A = A.tocsc(copy=False)
    # coerce `B` to coo to expand out its rows
    B = B.tocoo(copy=False)
    B = sp.csc_matrix((B.data, (idx[B.row], B.col)), shape=A.shape)
    # Now add the expanded `B` to `A`
    out = A + B
    return out
