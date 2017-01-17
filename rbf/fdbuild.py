''' 
This modules contains functions for building frequently used RBF-FD 
weight matrices.
'''
from rbf.fd import weight_matrix
from scipy.sparse import csr_sparse

def elastic2d_body_force(x,p,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices used to calculate body 
  force.

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
    additional arguments passed to *weight_matrix*

  Returns
  -------
  out : (2,2) list of sparse matrices
    A collection of matrices [[D_xx,D_xy],[D_yx,D_yy]] which return 
    the body force [f_x,f_y] at *x* exerted by the material, when 
    dotted with the displacements [u_x,u_y] at *p*.

  '''
  # x component of force resulting from displacement in the x direction.
  coeffs_xx = [lamb+2*mu,mu]
  diffs_xx = [(2,0),(0,2)]
  # x component of force resulting from displacement in the y direction.
  coeffs_xy = [lamb,mu]
  diffs_xy = [(1,1),(1,1)]
  # y component of force resulting from displacement in the x direction.
  coeffs_yx = [mu,lamb]
  diffs_yx = [(1,1),(1,1)]
  # y component of force resulting from displacement in the y direction.
  coeffs_yy = [lamb+2*mu, mu]
  diffs_yy =  [(0,2),(2,0)]
  # make the differentiation matrices that enforce the PDE on the 
  # interior nodes.
  D_xx = weight_matrix(x,p,diffs_xx,coeffs=coeffs_xx,**kwargs)
  D_xy = weight_matrix(x,p,diffs_xy,coeffs=coeffs_xy,**kwargs)
  D_yx = weight_matrix(x,p,diffs_yx,coeffs=coeffs_yx,**kwargs)
  D_yy = weight_matrix(x,p,diffs_yy,coeffs=coeffs_yy,**kwargs)
  return [[D_xx,D_xy],[D_yx,D_yy]]


def elastic2d_surface_force(x,p,normal,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices that estimate surface
  traction forces at *x* resulting from displacements at *p*.

  Parameters
  ----------
  x : (N,2) array
    target points which reside on a surface.
  
  p : (M,2) array
    observation points.  
  
  normal : (N,2) array
    surface normal vectors at each point in *x*.

  lamb : float
    first Lame parameter
  
  mu : float
    second Lame parameter
    
  **kwargs :
    additional arguments passed to *weight_matrix*

  Returns
  -------
  out : (2,2) list of sparse matrices
    A collection of matrices [[D_xx,D_xy],[D_yx,D_yy]] which return 
    the surface traction force [t_x,t_y] at *x* exerted by the 
    material, when dotted with the displacements [u_x,u_y] at *p*.

  '''  
  # x component of traction force resulting from x displacement 
  coeffs_xx = [normal[:,0]*(lamb+2*mu),normal[:,1]*mu]
  diffs_xx = [(1,0),(0,1)]
  # x component of traction force resulting from y displacement
  coeffs_xy = [normal[:,0]*lamb,normal[:,1]*mu]
  diffs_xy = [(0,1),(1,0)]
  # y component of traction force resulting from x displacement
  coeffs_yx = [normal[:,0]*mu,normal[:,1]*lamb]
  diffs_yx = [(0,1),(1,0)]
  # y component of force resulting from displacement in the y direction
  coeffs_yy = [normal[:,1]*(lamb+2*mu),normal[:,0]*mu]
  diffs_yy =  [(0,1),(1,0)]
  # make the differentiation matrices that enforce the free surface boundary 
  # conditions.
  D_xx = weight_matrix(x,p,diffs_xx,coeffs=coeffs_xx,**kwargs)
  D_xy = weight_matrix(x,p,diffs_xy,coeffs=coeffs_xy,**kwargs)
  D_yx = weight_matrix(x,p,diffs_yx,coeffs=coeffs_yx,**kwargs)
  D_yy = weight_matrix(x,p,diffs_yy,coeffs=coeffs_yy,**kwargs)
  # stack them together
  return [[D_xx,D_xy],[D_yx,D_yy]]


def elastic2d_displacement(x,p,lamb=1.0,mu=1.0,**kwargs):
  ''' 
  Returns a collection of weight matrices that estimates displacements 
  at *x* based on displacements at *p*. If *x* is in *p* then the 
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
    additional arguments passed to *weight_matrix*

  Returns
  -------
  out : (2,2) list of sparse matrices
    A collection of matrices [[D_xx,D_xy],[D_yx,D_yy]] which return 
    the displacements [u_x,u_y] at *p* based on the displacements at 
    *x*.

  '''
  D_xx = weight_matrix(x,p,(0,0),**kwargs)
  D_xy = csr_sparse((x.shape[0],p.shape[0]))
  D_yx = csr_sparse((x.shape[0],p.shape[0]))
  D_yy = weight_matrix(x,p,(0,0),**kwargs)
  # stack them together
  return [[D_xx,D_xy],[D_yx,D_yy]]


  

