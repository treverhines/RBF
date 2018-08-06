''' 
This script demonstrates using the RBF-FD method to calculate static 
deformation of a three-dimensional elastic material subject to a 
uniform body force such as gravity. The domain has roller boundary 
condition on the bottom and sides. The top of domain has a free 
surface boundary condition. The top of the domain is generated 
randomly and is intended to simulate topography.
'''
import rbf
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, splu
from scipy.interpolate import griddata
from rbf.nodes import min_energy_nodes
from rbf.fd import weight_matrix, add_rows
from rbf.domain import topography
from rbf.fdbuild import (elastic3d_body_force,
                         elastic3d_surface_force,
                         elastic3d_displacement)
import logging
logging.basicConfig(level=logging.DEBUG)                         

## User defined parameters
#####################################################################

def taper_function(x,center,radius,order=10):
  ''' 
  This function is effectively 1.0 within a sphere with radius 
  *radius* centered at *center*.  The function quickly drops to 0.0 
  outside of the sphere. This is used to ensure that the user-defined 
  topography tapers to zero at the domain edges.
  '''
  r = np.sqrt((x[:,0]-center[0])**2 + (x[:,1]-center[1])**2)
  return 1.0/(1.0 + (r/radius)**order)

def topo_func(x):
  ''' 
  This function generates a random topography at *x*. It takes an 
  (N,2) array of surface positions and returns an (N,) array of 
  elevations.  For real-world applications this should be a function 
  that interpolates a DEM.
  '''
  np.random.seed(3)
  gp = rbf.gauss.gpse((0.0,0.01,0.25))
  gp += rbf.gauss.gpse((0.0,0.01,0.5))
  gp += rbf.gauss.gpse((0.0,0.01,1.0))
  out = gp.sample(x)
  out *= taper_function(x,[0.0,0.0],1.0)
  return out

def density_func(x):
  ''' 
  This function describes the desired node density at *x*. It takes an 
  (N,3) array of positions and returns an (N,) array of normalized 
  node densities. This function is normalized such that all values are 
  between 0.0 and 1.0.
  '''
  return np.ones(x.shape[0])
  #z = x[:,2]
  #out = np.zeros(x.shape[0])
  #out[z > -0.5] = 1.0
  #out[z <= -0.5] = 0.25
  #return out

# generates the domain according to topo_func 
vert,smp = topography(topo_func,[-1.3,1.3],[-1.3,1.3],1.0,n=60)
# number of nodes 
N = 30000
# size of RBF-FD stencils
n_pre = 15
n_full = 50
# Lame parameters
lamb = 1.0
mu = 1.0
# z component of body force 
body_force = 1.0

## Build and solve for topographic stress
#####################################################################

def find_orthogonals(n):
  ''' 
  Returns two arrays of normalized vector that are orthogonal to *n*.
  This is used to determine the directions along which zero traction 
  constraints will be imposed.
    
  Parameters
  ----------
  n : (N,2) float array
    
  Returns
  -------
  out1 : (N,2) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vectors in *n*.

  out2 : (N,2) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vectors in *n* and in *out1*
    
  '''
  out1 = np.empty_like(n,dtype=float)
  out1[:,0] = -np.sum(n,axis=1) + n[:,0]
  out1[:,1:] = n[:,[0]] 
  out1 /= np.linalg.norm(out1,axis=1)[:,None] # normalize length
  out2 = np.cross(n,out1) # find vectors that are orthogonal to *n* and *out1*
  out2 /= np.linalg.norm(out2,axis=1)[:,None] # normalize length
  return out1,out2


def build_system(nodes, indices, normals, n):
    print('building system with %s nodes' % nodes.shape[0])
    N = nodes.shape[0]

    free = indices['free']
    roller = indices['roller']
    interior_and_boundary = np.hstack((indices['interior'], 
                                       indices['roller'],
                                       indices['free']))
    interior_and_ghost = np.hstack((indices['interior'],
                                    indices['roller_ghosts'],
                                    indices['free_ghosts']))

    # allocate the left-hand-side matrix components
    G_xx = sp.csr_matrix((N,N))
    G_xy = sp.csr_matrix((N,N))
    G_xz = sp.csr_matrix((N,N))
    G_yx = sp.csr_matrix((N,N))
    G_yy = sp.csr_matrix((N,N))
    G_yz = sp.csr_matrix((N,N))
    G_zx = sp.csr_matrix((N,N))
    G_zy = sp.csr_matrix((N,N))
    G_zz = sp.csr_matrix((N,N))

    # build the "left hand side" matrices for body force constraints
    out = elastic3d_body_force(nodes[interior_and_boundary],nodes,
                               lamb=lamb,mu=mu,n=n)
    G_xx = add_rows(G_xx, out['xx'], interior_and_ghost)
    G_xy = add_rows(G_xy, out['xy'], interior_and_ghost)
    G_xz = add_rows(G_xz, out['xz'], interior_and_ghost)
    G_yx = add_rows(G_yx, out['yx'], interior_and_ghost)
    G_yy = add_rows(G_yy, out['yy'], interior_and_ghost)
    G_yz = add_rows(G_yz, out['yz'], interior_and_ghost)
    G_zx = add_rows(G_zx, out['zx'], interior_and_ghost)
    G_zy = add_rows(G_zy, out['zy'], interior_and_ghost)
    G_zz = add_rows(G_zz, out['zz'], interior_and_ghost)

    # build the "left hand side" matrices for free surface constraints
    out = elastic3d_surface_force(nodes[free],normals['free'],
                                  nodes,lamb=lamb,mu=mu,n=n)
    G_xx = add_rows(G_xx, out['xx'], free)
    G_xy = add_rows(G_xy, out['xy'], free)
    G_xz = add_rows(G_xz, out['xz'], free)
    G_yx = add_rows(G_yx, out['yx'], free)
    G_yy = add_rows(G_yy, out['yy'], free)
    G_yz = add_rows(G_yz, out['yz'], free)
    G_zx = add_rows(G_zx, out['zx'], free)
    G_zy = add_rows(G_zy, out['zy'], free)
    G_zz = add_rows(G_zz, out['zz'], free)

    # build the "left hand side" matrices for roller constraints
    # constrain displacements in the surface normal direction
    out = elastic3d_displacement(nodes[roller],nodes,
                                 lamb=lamb,mu=mu,n=1)
    normals_x = sp.diags(normals['roller'][:,0])
    normals_y = sp.diags(normals['roller'][:,1])
    normals_z = sp.diags(normals['roller'][:,2])

    G_xx = add_rows(G_xx, normals_x.dot(out['xx']), roller)
    G_xy = add_rows(G_xy, normals_y.dot(out['yy']), roller)
    G_xz = add_rows(G_xz, normals_z.dot(out['zz']), roller)

    out = elastic3d_surface_force(nodes[roller],normals['roller'],
                                  nodes,lamb=lamb,mu=mu,n=n)
    parallels_1,parallels_2 = find_orthogonals(normals['roller'])

    parallels_x = sp.diags(parallels_1[:,0])
    parallels_y = sp.diags(parallels_1[:,1])
    parallels_z = sp.diags(parallels_1[:,2])
    G_yx = add_rows(G_yx, parallels_x.dot(out['xx']) + parallels_y.dot(out['yx']) + parallels_z.dot(out['zx']), roller)
    G_yy = add_rows(G_yy, parallels_x.dot(out['xy']) + parallels_y.dot(out['yy']) + parallels_z.dot(out['zy']), roller)
    G_yz = add_rows(G_yz, parallels_x.dot(out['xz']) + parallels_y.dot(out['yz']) + parallels_z.dot(out['zz']), roller)

    parallels_x = sp.diags(parallels_2[:,0])
    parallels_y = sp.diags(parallels_2[:,1])
    parallels_z = sp.diags(parallels_2[:,2])
    G_zx = add_rows(G_zx, parallels_x.dot(out['xx']) + parallels_y.dot(out['yx']) + parallels_z.dot(out['zx']), roller)
    G_zy = add_rows(G_zy, parallels_x.dot(out['xy']) + parallels_y.dot(out['yy']) + parallels_z.dot(out['zy']), roller)
    G_zz = add_rows(G_zz, parallels_x.dot(out['xz']) + parallels_y.dot(out['yz']) + parallels_z.dot(out['zz']), roller)

    # stack it all together
    G_x = sp.hstack((G_xx, G_xy, G_xz))
    G_y = sp.hstack((G_yx, G_yy, G_yz))
    G_z = sp.hstack((G_zx, G_zy, G_zz))

    G = sp.vstack((G_x, G_y, G_z))
    G = G.tocsr()

    # create the right-hand-side vector
    d_x = np.zeros((N,))
    d_y = np.zeros((N,))
    d_z = np.zeros((N,))

    d_x[interior_and_ghost] = 0.0
    d_x[free] = 0.0
    d_x[roller] = 0.0

    d_y[interior_and_ghost] = 0.0
    d_y[free] = 0.0
    d_y[roller] = 0.0

    d_z[interior_and_ghost] = 1.0
    d_z[free] = 0.0
    d_z[roller] = 0.0

    d = np.hstack((d_x, d_y, d_z))

    print('done')
    return G,d    


# generate nodes. Note that this may take a while
boundary_groups = {'free': range(10, smp.shape[0]),
                   'roller': range(0, 10)}

nodes, indices, normals = min_energy_nodes(
                            N,vert,smp,
                            boundary_groups=boundary_groups,
                            boundary_groups_with_ghosts=['free','roller'],
                            rho=density_func,itr=50)

G_pre, _ = build_system(nodes, indices, normals, n_pre)
G_pre = G_pre.tocsc() # convert to csc for splu
G, d = build_system(nodes, indices, normals, n_full)

def callback(res,_itr=[0]):
  l2 = np.linalg.norm(res)
  print('gmres error on iteration %s: %s' % (_itr[0],l2))
  _itr[0] += 1
  
print('forming preconditioner')
lu = splu(G_pre)
M = LinearOperator(G_pre.shape,lu.solve)

print('solving with gmres')
u,info = gmres(G,d,M=M,callback=callback)
print('finished gmres with info %s' % info)

u = np.reshape(u,(3,-1))
u_x,u_y,u_z = u

# Calculate strain from displacements
D_x = weight_matrix(nodes,nodes,(1,0,0),n=n_full)
D_y = weight_matrix(nodes,nodes,(0,1,0),n=n_full)
D_z = weight_matrix(nodes,nodes,(0,0,1),n=n_full)
e_xx = D_x.dot(u_x)
e_yy = D_y.dot(u_y)
e_zz = D_z.dot(u_z)
e_xy = 0.5*(D_y.dot(u_x) + D_x.dot(u_y))
e_xz = 0.5*(D_z.dot(u_x) + D_x.dot(u_z))
e_yz = 0.5*(D_z.dot(u_y) + D_y.dot(u_z))
# Calculate stress from Hooks law
s_xx = (2*mu + lamb)*e_xx + lamb*e_yy + lamb*e_zz
s_yy = lamb*e_xx + (2*mu + lamb)*e_yy + lamb*e_zz
s_zz = lamb*e_xx + lamb*e_yy + (2*mu + lamb)*e_zz
s_xy = 2*mu*e_xy
s_xz = 2*mu*e_xz
s_yz = 2*mu*e_yz


interior_and_boundary = np.hstack((indices['interior'], 
                                   indices['roller'],
                                   indices['free']))

nodes = nodes[interior_and_boundary]
u_x,u_y,u_z = u_x[interior_and_boundary], u_y[interior_and_boundary], u_z[interior_and_boundary]
e_xx,e_yy,e_zz = e_xx[interior_and_boundary], e_yy[interior_and_boundary], e_zz[interior_and_boundary]
e_xy,e_xz,e_yz = e_xy[interior_and_boundary], e_xz[interior_and_boundary], e_yz[interior_and_boundary]
s_xx,s_yy,s_zz = s_xx[interior_and_boundary], s_yy[interior_and_boundary], s_zz[interior_and_boundary]
s_xy,s_xz,s_yz = s_xy[interior_and_boundary], s_xz[interior_and_boundary], s_yz[interior_and_boundary]


I2 = e_xx**2 + e_yy**2 + e_zz**2 + 2*e_xy**2 + 2*e_xz**2 + 2*e_yz**2
print('mean strain energy: %s' % np.mean(I2))

quit()
## Plot the results
#####################################################################
# remove ghost nodes

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_aspect('equal')

ax.plot_trisurf(vert[:,0],vert[:,1],vert[:,2],triangles=smp,
                edgecolor=(0.0,0.0,0.2,0.2),color=(0.2,0.2,0.2,0.1),lw=1.0,
                shade=False)
ax.quiver(nodes[:,0], nodes[:,1], nodes[:,2], u_x, u_y, u_z,
          length=0.25, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
quit()
