''' 
This modules contains functions for building frequently used RBF-FD 
weight matrices.
'''
from rbf.fd import weight_matrix
from scipy.sparse import csr_matrix

def elastic2d_body_force(x,p,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices used to calculate body 
  force in a two-dimensional homogeneous elastc medium.

  Parameters
  ----------
  x : (N,2) array
    Target points.
  
  p : (M,2) array
    Observation points.  
  
  lamb : float, optional
    First Lame parameter
  
  mu : float, optional
    Specond Lame parameter
    
  **kwargs :
    additional arguments passed to `weight_matrix`

  Returns
  -------
  out : (2,2) list of sparse matrices
    A collection of matrices [[D_xx,D_xy],[D_yx,D_yy]] which return 
    the body force [f_x,f_y] at `x` exerted by the material, when 
    dotted with the displacements [u_x,u_y] at `p`.

  '''
  # x component of force resulting from displacement in the x direction.
  coeffs_xx = [lamb+2*mu,mu]
  diffs_xx = [(2,0),(0,2)]
  # x component of force resulting from displacement in the y direction.
  coeffs_xy = [lamb+mu]
  diffs_xy = [(1,1)]
  # y component of force resulting from displacement in the x direction.
  coeffs_yx = [lamb+mu]
  diffs_yx = [(1,1)]
  # y component of force resulting from displacement in the y direction.
  coeffs_yy = [lamb+2*mu,mu]
  diffs_yy =  [(0,2),(2,0)]
  # make the differentiation matrices that enforce the PDE on the 
  # interior nodes.
  D_xx = weight_matrix(x,p,diffs_xx,coeffs=coeffs_xx,**kwargs)
  D_xy = weight_matrix(x,p,diffs_xy,coeffs=coeffs_xy,**kwargs)
  D_yx = weight_matrix(x,p,diffs_yx,coeffs=coeffs_yx,**kwargs)
  D_yy = weight_matrix(x,p,diffs_yy,coeffs=coeffs_yy,**kwargs)
  return D_xx,D_xy,D_yx,D_yy


def elastic2d_surface_force(x,nrm,p,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices that estimate surface
  traction forces at `x` resulting from displacements at `p`.

  Parameters
  ----------
  x : (N,2) array
    target points which reside on a surface.
  
  nrm : (N,2) array
    surface normal vectors at each point in `x`.

  p : (M,2) array
    observation points.  
  

  lamb : float
    first Lame parameter
  
  mu : float
    second Lame parameter
    
  **kwargs :
    additional arguments passed to `weight_matrix`

  Returns
  -------
  out : (2,2) list of sparse matrices
    A collection of matrices [[D_xx,D_xy],[D_yx,D_yy]] which return 
    the surface traction force [t_x,t_y] at `x` exerted by the 
    material, when dotted with the displacements [u_x,u_y] at `p`.

  '''
  # x component of traction force resulting from x displacement 
  coeffs_xx = [nrm[:,0]*(lamb+2*mu), nrm[:,1]*mu]
  diffs_xx =  [               (1,0),       (0,1)]
  # x component of traction force resulting from y displacement
  coeffs_xy = [nrm[:,0]*lamb, nrm[:,1]*mu]
  diffs_xy =  [        (0,1),       (1,0)]
  # y component of traction force resulting from x displacement
  coeffs_yx = [nrm[:,0]*mu, nrm[:,1]*lamb]
  diffs_yx =  [      (0,1),         (1,0)]
  # y component of force resulting from displacement in the y direction
  coeffs_yy = [nrm[:,0]*mu, nrm[:,1]*(lamb+2*mu)]
  diffs_yy =  [      (1,0),                (0,1)]
  # make the differentiation matrices that enforce the free surface boundary 
  # conditions.
  D_xx = weight_matrix(x,p,diffs_xx,coeffs=coeffs_xx,**kwargs)
  D_xy = weight_matrix(x,p,diffs_xy,coeffs=coeffs_xy,**kwargs)
  D_yx = weight_matrix(x,p,diffs_yx,coeffs=coeffs_yx,**kwargs)
  D_yy = weight_matrix(x,p,diffs_yy,coeffs=coeffs_yy,**kwargs)
  return D_xx, D_xy, D_yx, D_yy


def elastic2d_displacement(x,p,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices that estimates displacements 
  at `x` based on displacements at `p`. If `x` is in `p` then the 
  results will be an appropriately shaped identity matrix.

  Parameters
  ----------
  x : (N,2) array
    target points.
  
  p : (M,2) array
    observation points.  
  
  lamb : float
    first Lame parameter
  
  mu : float
    second Lame parameter
    
  **kwargs :
    additional arguments passed to `weight_matrix`

  Returns
  -------
  out : (2,2) list of sparse matrices
    A collection of matrices [[D_xx,D_xy],[D_yx,D_yy]] which return 
    the displacements [u_x,u_y] at `p` based on the displacements at 
    `x`.

  '''
  D_xx = weight_matrix(x,p,(0,0),**kwargs)
  D_yy = weight_matrix(x,p,(0,0),**kwargs)
  return D_xx, D_yy


def elastic3d_body_force(x,p,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices used to calculate body 
  force in a three-dimensional homogeneous elastic medium.

  Parameters
  ----------
  x : (N,3) array
    target points.
  
  p : (M,3) array
    observation points.  
  
  lamb : float
    first Lame parameter
  
  mu : float
    second Lame parameter
    
  **kwargs :
    additional arguments passed to `weight_matrix`

  Returns
  -------
  out : (3,3) list of sparse matrices
    A collection of matrices which return the body force at `x` 
    exerted by the material, when dotted with the displacements at 
    `p`.

  '''
  coeffs_xx = [lamb+2*mu,      mu,       mu]
  diffs_xx =  [  (2,0,0), (0,2,0),  (0,0,2)]
  coeffs_xy = [lamb+mu]
  diffs_xy =  [(1,1,0)]
  coeffs_xz = [lamb+mu]
  diffs_xz =  [(1,0,1)]
  coeffs_yx = [lamb+mu]
  diffs_yx =  [(1,1,0)]
  coeffs_yy = [     mu, lamb+2*mu,      mu]
  diffs_yy =  [(2,0,0),   (0,2,0), (0,0,2)]
  coeffs_yz = [lamb+mu]
  diffs_yz =  [(0,1,1)]
  coeffs_zx = [lamb+mu]
  diffs_zx =  [(1,0,1)]
  coeffs_zy = [lamb+mu]
  diffs_zy =  [(0,1,1)]
  coeffs_zz = [     mu,      mu, lamb+2*mu]
  diffs_zz =  [(2,0,0), (0,2,0),   (0,0,2)]
  D_xx = weight_matrix(x,p,diffs_xx,coeffs=coeffs_xx,**kwargs)
  D_xy = weight_matrix(x,p,diffs_xy,coeffs=coeffs_xy,**kwargs)
  D_xz = weight_matrix(x,p,diffs_xz,coeffs=coeffs_xz,**kwargs)
  D_yx = weight_matrix(x,p,diffs_yx,coeffs=coeffs_yx,**kwargs)
  D_yy = weight_matrix(x,p,diffs_yy,coeffs=coeffs_yy,**kwargs)
  D_yz = weight_matrix(x,p,diffs_yz,coeffs=coeffs_yz,**kwargs)
  D_zx = weight_matrix(x,p,diffs_zx,coeffs=coeffs_zx,**kwargs)
  D_zy = weight_matrix(x,p,diffs_zy,coeffs=coeffs_zy,**kwargs)
  D_zz = weight_matrix(x,p,diffs_zz,coeffs=coeffs_zz,**kwargs)
  return [[D_xx,D_xy,D_xz],
          [D_yx,D_yy,D_yz],
          [D_zx,D_zy,D_zz]]


def elastic3d_surface_force(x,nrm,p,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices that estimate surface
  traction forces at `x` resulting from displacements at `p`.

  Parameters
  ----------
  x : (N,3) array
    target points which reside on a surface.
  
  nrm : (N,3) array
    surface normal vectors at each point in `x`.

  p : (M,3) array
    observation points.  

  lamb : float
    first Lame parameter
  
  mu : float
    second Lame parameter
    
  **kwargs :
    additional arguments passed to `weight_matrix`

  Returns
  -------
  out : (3,3) list of sparse matrices
    A collection of matrices which return the surface traction force 
    at `x` exerted by the material, when dotted with the displacements 
    at `p`.

  '''
  coeffs_xx = [nrm[:,0]*(lamb+2*mu), nrm[:,1]*mu, nrm[:,2]*mu]
  diffs_xx =  [             (1,0,0),     (0,1,0),     (0,0,1)]
  coeffs_xy = [nrm[:,0]*lamb, nrm[:,1]*mu]
  diffs_xy =  [      (0,1,0),     (1,0,0)]
  coeffs_xz = [nrm[:,0]*lamb, nrm[:,2]*mu]
  diffs_xz =  [      (0,0,1),     (1,0,0)]
  coeffs_yx = [nrm[:,0]*mu, nrm[:,1]*lamb]
  diffs_yx =  [    (0,1,0),       (1,0,0)]
  coeffs_yy = [nrm[:,0]*mu, nrm[:,1]*(lamb+2*mu), nrm[:,2]*mu]
  diffs_yy =  [    (1,0,0),              (0,1,0),     (0,0,1)]
  coeffs_yz = [nrm[:,1]*lamb, nrm[:,2]*mu]
  diffs_yz =  [      (0,0,1),     (0,1,0)]
  coeffs_zx = [nrm[:,0]*mu, nrm[:,2]*lamb]
  diffs_zx =  [    (0,0,1),       (1,0,0)]
  coeffs_zy = [nrm[:,1]*mu, nrm[:,2]*lamb]
  diffs_zy =  [    (0,0,1),       (0,1,0)]
  coeffs_zz = [nrm[:,0]*mu, nrm[:,1]*mu, nrm[:,2]*(lamb+2*mu)]
  diffs_zz =  [    (1,0,0),     (0,1,0),              (0,0,1)]
  D_xx = weight_matrix(x,p,diffs_xx,coeffs=coeffs_xx,**kwargs)
  D_xy = weight_matrix(x,p,diffs_xy,coeffs=coeffs_xy,**kwargs)
  D_xz = weight_matrix(x,p,diffs_xz,coeffs=coeffs_xz,**kwargs)
  D_yx = weight_matrix(x,p,diffs_yx,coeffs=coeffs_yx,**kwargs)
  D_yy = weight_matrix(x,p,diffs_yy,coeffs=coeffs_yy,**kwargs)
  D_yz = weight_matrix(x,p,diffs_yz,coeffs=coeffs_yz,**kwargs)
  D_zx = weight_matrix(x,p,diffs_zx,coeffs=coeffs_zx,**kwargs)
  D_zy = weight_matrix(x,p,diffs_zy,coeffs=coeffs_zy,**kwargs)
  D_zz = weight_matrix(x,p,diffs_zz,coeffs=coeffs_zz,**kwargs)
  return [[D_xx,D_xy,D_xz],
          [D_yx,D_yy,D_yz],
          [D_zx,D_zy,D_zz]]


def elastic3d_displacement(x,p,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices that estimates displacements 
  at `x` based on displacements at `p`. If `x` is in `p` then the 
  results will be an appropriately shaped identity matrix.

  Parameters
  ----------
  x : (N,3) array
    target points.
  
  p : (M,3) array
    observation points.  
  
  lamb : float
    first Lame parameter
  
  mu : float
    second Lame parameter
    
  **kwargs :
    additional arguments passed to `weight_matrix`

  Returns
  -------
  out : (3,3) list of sparse matrices
    A collection of matrices which return the displacements at `p` 
    based on the displacements at `x`.

  '''
  D_xx = weight_matrix(x,p,(0,0,0),**kwargs)
  D_xy = csr_matrix((x.shape[0],p.shape[0]))
  D_xz = csr_matrix((x.shape[0],p.shape[0]))
  D_yx = csr_matrix((x.shape[0],p.shape[0]))
  D_yy = weight_matrix(x,p,(0,0,0),**kwargs)
  D_yz = csr_matrix((x.shape[0],p.shape[0]))
  D_zx = csr_matrix((x.shape[0],p.shape[0]))
  D_zy = csr_matrix((x.shape[0],p.shape[0]))
  D_zz = weight_matrix(x,p,(0,0,0),**kwargs)
  return [[D_xx,D_xy,D_xz],
          [D_yx,D_yy,D_yz],
          [D_zx,D_zy,D_zz]]


  

