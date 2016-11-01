import numpy as np
import matplotlib.pyplot as plt
import rbf.filter
import rbf.fd
import rbf.halton
from scipy.signal import periodogram
import sympy
import scipy.linalg
np.random.seed(3)

def spectral_diff_matrix(N,dt,diff):
  ''' 
  generates a periodic sinc differentation matrix. This is equivalent 
  
  Parameters
  ----------
    N : number of observations 
    dt : sample spacing
    diff : derivative order (max=2)

  '''
  scale = dt*N/(2*np.pi)
  dt = 2*np.pi/N
  t,h = sympy.symbols('t,h')
  sinc = sympy.sin(sympy.pi*t/h)/((2*sympy.pi/h)*sympy.tan(t/2))
  if diff == 0:
    sinc_diff = sinc
  else:
    sinc_diff = sinc.diff(*(t,)*diff)

  func = sympy.lambdify((t,h),sinc_diff,'numpy')
  times = dt*np.arange(N)
  val = func(times,dt)
  if diff == 0:
    val[0] = 1.0
  elif diff == 1:
    val[0] = 0.0
  elif diff == 2:
    val[0] = -(np.pi**2/(3*dt**2)) - 1.0/6.0

  D = scipy.linalg.circulant(val)/scale**diff
  return D



N_obs = 500
N_post = 500
#x_obs = rbf.halton.halton(N_obs,2)
x_obs = np.linspace(0.0,1.0,N_obs)[:,None]
dt = x_obs[1] - x_obs[0]
#x_obs = np.sort(x_obs,axis=0)
x_post = x_obs# np.random.uniform(0.0,1.0,N_obs)[:,None])
#x_obs = np.linspace(0.0,1.0,N_obs)[:,None]
#x_post = rbf.halton.halton(N_post,2)
#x_post = np.linspace(0.0,1.0,N_post)[:,None]
#x_obs = np.linspace(0.0,1.0,N_obs)[:,None]**2
#x_obs = np.random.normal(0.5,0.25,N_obs)[:,None]**2
#x_obs = np.sort(x_obs,axis=0)
#x_post = np.linspace(0.0,1.0,N_post)[:,None]**2
#x_post = np.random.uniform(0.0,1.5,N_post)[:,None]
#x_post = np.sort(x_post,axis=0)
#x_post = np.linspace(0.0,1.4,N_post)[:,None]
sigma_obs = 1.0*np.ones(N_obs)
sigma_obs[100:200] = 10000.0

u_obs = np.random.normal(0.0,sigma_obs)

bnd_vert = np.array([[100.4,0.2],
                     [100.8,0.4]])
bnd_smp = np.array([[0,1]])                     

cutoff = 10.0
sigma_bar = np.sqrt(N_post/np.sum(1.0/sigma_obs**2))
penalty = np.sqrt((2*np.pi*cutoff)**4*sigma_bar**2)

K = rbf.fd.weight_matrix(x_obs,x_post,[[0]],size=20).toarray()
C_obs_inv = np.diag(1.0/sigma_obs**2)
Dn = rbf.fd.weight_matrix(x_post,x_post,[[2]],size=20).toarray()
#Dn = spectral_diff_matrix(N_obs,dt,2)
Gg = np.linalg.inv(K.T.dot(C_obs_inv).dot(K) + 1.0/penalty**2*Dn.T.dot(Dn)).dot(K.T).dot(C_obs_inv)
u_post = Gg.dot(u_obs)

val,vec = np.linalg.eig(Gg)
idx = np.argsort(val)[::-1]
val = val[idx]
vec = vec[:,idx]

idx, = np.nonzero(val > 0.5)
val = val[idx]
vec = vec[:,idx]
print(val.shape)
print(vec.shape)

plt.figure(1)
plt.plot(val)
plt.figure(2)
plt.plot(x_obs,val[None,:]*vec,color='k',lw=1)
plt.plot(x_obs,val[-1]*vec[:,-1],color='r',lw=2)
plt.figure(3)
plt.plot(x_obs,u_obs,'ko')
plt.plot(x_post,u_post,'b-')
plt.show()
#print(np.trace(C_obs_inv))
#print(np.trace(Gg))
plt.figure(2)
plt.scatter(x_post[:,0],x_post[:,1],s=100,c=u_post,edgecolor='none',cmap='seismic')
plt.colorbar()
plt.show()


quit()
#u_post = Gg.dot(u_obs)
#plt.imshow(Gg)
#plt.colorbar()
#plt.show()
sigma_post = np.sqrt(np.diag(np.linalg.inv(K.T.dot(C_obs_inv).dot(K) + 1.0/penalty**2*Dn.T.dot(Dn))))
print(u_post.shape)
#u_post2,sigma_post = rbf.filter.filter(x_obs,u_obs,cutoff=cutoff,sigma=sigma_obs)

plt.figure(1)
plt.plot(x_obs,u_obs,'k.')
plt.plot(x_post,u_post,'b-o')
plt.fill_between(x_post[:,0],u_post-sigma_post,u_post+sigma_post,color='b',alpha=0.2)

plt.figure(2)  
freq1,pow1 = periodogram(u_post,N_post/2.0,scaling='density')
#freq2,pow2 = periodogram(u_obs,N_obs,scaling='density')

def expected(f,n):
  return 1.0/(1.0 + (f/cutoff)**(2*n))

plt.loglog(freq1,expected(freq1,2)**2,'k--')
plt.loglog(freq1[1:],pow1[1:])
#plt.loglog(freq2[1:],pow2[1:])
plt.show()
