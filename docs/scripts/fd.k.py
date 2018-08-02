''' 
In this example I explore ways to make iterative solvers more
effective with the RBF FD method

Notes:

- The row ordering for the matrix is important. Keeping the diagonal
structure seems to make the difference between GMRES converging or not
converging
'''
import numpy as np
from rbf.fd import weight_matrix
from rbf.basis import phs3
from rbf.geometry import contains
from rbf.nodes import menodes
import matplotlib.pyplot as plt
from scipy.sparse import vstack, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, gmres
from scipy.interpolate import LinearNDInterpolator

def callback(res,_itr=[0]):
    l2 = np.linalg.norm(res)
    print('residual norm on itr %s: %s' % (l2,_itr[0]))
    _itr[0] += 1

def expand_rows(A,idx,rout):
    '''
    Expand the rows of the (rin,c) matrix `A` into an (rout,c) matrix.

    Parameters
    ----------
    A : (rin,c) sparse matrix
    idx: (rin,) int array 
        mapping from the input rows to the output rows
    rout: int
        size of the output array    
    '''
    A = coo_matrix(A)
    idx = np.asarray(idx,dtype=int)
    shape = (rout,A.shape[1])
    out = csc_matrix((A.data,(idx[A.row],A.col)),shape=shape)
    return out

# Define the problem domain with line segments.
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],
                 [1.0,1.0],[1.0,2.0],[0.0,2.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])

N = 10000 # total number of nodes.

n = 20 # stencil size. Increase this will generally improve accuracy
       # at the expense of computation time.

basis = phs3 # radial basis function used to compute the weights. Odd
             # order polyharmonic splines (e.g., phs3) have always
             # performed well for me and they do not require the user
             # to tune a shape parameter. Use higher order
             # polyharmonic splines for higher order PDEs.

order = 2 # Order of the added polynomials. This should be at least as
          # large as the order of the PDE being solved (2 in this
          # case). Larger values may improve accuracy

# generate nodes
nodes,smpid = menodes(N,vert,smp) 
edge_idx, = (smpid>=0).nonzero() 
interior_idx, = (smpid==-1).nonzero() 
# create "left hand side" matrix
A_int = weight_matrix(nodes[interior_idx],nodes,diffs=[[2,0],[0,2]],
                      n=n,basis=basis,order=order)
A_int = expand_rows(A_int,interior_idx,N)

A_edg = weight_matrix(nodes[edge_idx],nodes,diffs=[0,0]) 
A_edg = expand_rows(A_edg,edge_idx,N)
A = A_int + A_edg

# create "right hand side" vector
d = np.zeros(N)
d[interior_idx] = -1.0
d[edge_idx] = 0.0

# find the solution at the nodes
u_soln,info = gmres(A,d,callback=callback) 
print('exited with info %s' % info)
# interpolate the solution on a grid
xg,yg = np.meshgrid(np.linspace(-0.05,2.05,500),np.linspace(-0.05,2.05,500))
points = np.array([xg.flatten(),yg.flatten()]).T                    
u_itp = LinearNDInterpolator(nodes,u_soln)(points)
# mask points outside of the domain
u_itp[~contains(points,vert,smp)] = np.nan 
ug = u_itp.reshape((500,500)) # fold back into a grid
# make a contour plot of the solution
fig,ax = plt.subplots()
p = ax.contourf(xg,yg,ug,np.linspace(-1e-10,0.16,9),cmap='viridis')
#ax.plot(nodes[:,0],nodes[:,1],'k.',markersize=2)
for s in smp:
  ax.plot(vert[s,0],vert[s,1],'k-',lw=2)

ax.set_aspect('equal')
fig.colorbar(p,ax=ax)
fig.tight_layout()
plt.savefig('../figures/fd.i.png')
plt.show()

