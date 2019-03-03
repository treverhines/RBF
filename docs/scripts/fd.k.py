''' 
WORK IN PROGRESS
'''
import numpy as np
import rbf
from rbf.fd import weight_matrix
from rbf.sputils import add_rows
from rbf.fdbuild import (elastic2d_body_force,
                         elastic2d_surface_force,
                         elastic2d_displacement)
from rbf.nodes import min_energy_nodes
from rbf.geometry import contains
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.integrate import ode
from scipy.interpolate import griddata

# define the problem domain
vert = np.array([[0.0, 0.0],
                 [2.0, 0.0],
                 [2.0, 1.0],
                 [1.0, 1.0],
                 [1.0, 2.0],
                 [0.0, 2.0]])
smp = np.array([[0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 0]])
times = np.linspace(0.0, 10.0, 20) # output times
n_nominal = 1000 # total number of nodes
lamb = 1.0
mu = 1.0
rho = 1.0
stencil_size = 30
order = 2
basis = rbf.basis.phs5
nodes, idx, normals = min_energy_nodes(
    n_nominal,
    vert,
    smp,
    boundary_groups={'all':range(len(smp))},
    boundary_groups_with_ghosts=['all'],
    include_vertices=False) 
n = nodes.shape[0]    

# create initial and boundary conditions
r = np.sqrt((nodes[idx['interior'], 0] - 0.5)**2 + 
            (nodes[idx['interior'], 1] - 0.5)**2)

u_init = np.zeros_like(nodes)
u_init[idx['interior'], 0] = 1.0/(1 + (r/0.2)**4)
u_init[idx['interior'], 1] = 1.0/(1 + (r/0.2)**4)

v_init = np.zeros_like(nodes)

z_init = np.hstack((u_init.T.flatten(), 
                    v_init.T.flatten()))

# construct a matrix that maps the displacements everywhere to the
# displacements at the interior and the boundary conditions
B_xx = sp.csc_matrix((n, n))
B_xy = sp.csc_matrix((n, n))
B_yx = sp.csc_matrix((n, n))
B_yy = sp.csc_matrix((n, n))

components = elastic2d_displacement(
    nodes[idx['interior']],
    nodes,
    lamb=lamb,
    mu=mu,
    n=stencil_size)
B_xx = add_rows(B_xx, components['xx'], idx['interior'])
B_yy = add_rows(B_yy, components['yy'], idx['interior'])

components = elastic2d_displacement(
    nodes[idx['boundary:all']],
    nodes,
    lamb=lamb,
    mu=mu,
    n=stencil_size)
B_xx = add_rows(B_xx, components['xx'], idx['boundary:all'])
B_yy = add_rows(B_yy, components['yy'], idx['boundary:all'])

components = elastic2d_surface_force(
    nodes[idx['boundary:all']],
    normals[idx['boundary:all']],
    nodes,
    lamb=lamb,
    mu=mu,
    n=stencil_size,
    order=order,
    basis=basis)
B_xx = add_rows(B_xx, components['xx'], idx['ghosts:all'])
B_xy = add_rows(B_xy, components['xy'], idx['ghosts:all'])
B_yx = add_rows(B_yx, components['yx'], idx['ghosts:all'])
B_yy = add_rows(B_yy, components['yy'], idx['ghosts:all'])

B = sp.vstack((sp.hstack((B_xx, B_xy)),
               sp.hstack((B_yx, B_yy)))).tocsc()
B += 1000.0*sp.eye(B.shape[0])
Binv = np.linalg.inv(B.A)


# construct a matrix that maps the displacements everywhere to the
# body force 
D_xx = sp.csc_matrix((n, n))
D_xy = sp.csc_matrix((n, n))
D_yx = sp.csc_matrix((n, n))
D_yy = sp.csc_matrix((n, n))

components = elastic2d_body_force(
    nodes[idx['interior']],
    nodes,
    lamb=lamb,
    mu=mu,
    n=stencil_size,
    order=order,
    basis=basis)
D_xx = add_rows(D_xx, components['xx'], idx['interior'])
D_xy = add_rows(D_xy, components['xy'], idx['interior'])
D_yx = add_rows(D_yx, components['yx'], idx['interior'])
D_yy = add_rows(D_yy, components['yy'], idx['interior'])

components = elastic2d_body_force(
    nodes[idx['boundary:all']],
    nodes,
    lamb=lamb,
    mu=mu,
    n=stencil_size,
    order=order,
    basis=basis)
D_xx = add_rows(D_xx, components['xx'], idx['boundary:all'])
D_xy = add_rows(D_xy, components['xy'], idx['boundary:all'])
D_yx = add_rows(D_xy, components['yx'], idx['boundary:all'])
D_yy = add_rows(D_yy, components['yy'], idx['boundary:all'])

# the ghost node components are left as zero

D = sp.vstack((sp.hstack((D_xx, D_xy)),
               sp.hstack((D_yx, D_yy)))).tocsc()

L = weight_matrix(nodes, nodes, diffs=[[6,0],[0,6]], 
                  n=stencil_size, 
                  basis=rbf.basis.phs7,
                  order=6)
L = sp.block_diag((L,L))
I = sp.eye(L.shape[0])
R = np.linalg.inv((I + 1e-14*L.T.dot(L)).A)


def f(t, z):
  ''' 
  Function used for time integration. This calculates the time 
  derivative of the current state vector. 
  '''
  u, v = z.reshape((2, -1))
  dudt = v
  h = Binv.dot(u)

  dvdt = rho*D.dot(h)

#  if t > 0.2:    
#    h = h.reshape((2,-1))
#    h2 = h2.reshape((2,-1))
#    plt.figure(1)
#    plt.quiver(nodes[:, 0], nodes[:, 1], 
#               h[0], h[1], scale=5.0, color='b')
#    plt.quiver(nodes[:, 0], nodes[:, 1], 
#               h2[0], h2[1], scale=5.0, color='r')
#    plt.figure(2)
#    plt.quiver(nodes[:, 0], nodes[:, 1], 
#               h2[0], h2[1], scale=5.0, color='r')
#    plt.show()
          
  dzdt = np.hstack((dudt, dvdt))
  return dzdt
              
# perform time integration with 'dopri5', which is Runge Kutta 
integrator = ode(f).set_integrator('dopri5',nsteps=1000)
integrator.set_initial_value(z_init, times[0])
soln = []

for t in times[1:]:
  # calculate state vector at time *t*
  z = integrator.integrate(t).reshape((2, 2, -1))
  #soln += [z[0]] # only save displacements
  plt.quiver(nodes[:, 0], nodes[:, 1], 
             z[0, 0], z[0, 1], scale=15.0)
  plt.show()             

quit()


# plot the results
fig,axs = plt.subplots(2,2,figsize=(7,7))
for i,t in enumerate(times[1:]):
  ax = axs.ravel()[i]
  xg,yg = np.mgrid[0.0:2.0:200j,0:2.0:200j]
  points = np.array([xg.ravel(),yg.ravel()]).T
  # interpolate the solution onto a grid
  u = np.empty(N)
  u[idx['interior']],u[idx['boundary:all']] = soln[i],u_bnd 
  ug = griddata(nodes,u,(xg,yg),method='linear')
  # mask the points outside of the domain
  ug.ravel()[~contains(points,vert,smp)] = np.nan 
  # plot the boudary
  for s in smp: ax.plot(vert[s,0],vert[s,1],'k-')
  ax.imshow(ug,extent=(0.0,2.0,0.0,2.0),origin='lower',vmin=-0.2,vmax=0.2,cmap='seismic')
  ax.set_aspect('equal')
  ax.text(0.6,0.85,'time : %s\nnodes : %s' % (t,N),transform=ax.transAxes,fontsize=10)
  ax.tick_params(labelsize=10)
  ax.set_xlim(-0.1,2.1)
  ax.set_ylim(-0.1,2.1)
    
plt.tight_layout()    
plt.savefig('../figures/fd.a.png')
plt.show()
