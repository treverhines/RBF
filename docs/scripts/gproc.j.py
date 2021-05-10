'''
Use Gaussian process regression to perform 2-D interpolation of
scattered data and then differentiate the interpolant.
'''
import matplotlib.pyplot as plt
import numpy as np 
from rbf.basis import spwen32
from rbf.gproc import gpiso, gppoly

# get a verbose log of what is going on
import logging
logging.basicConfig(level=logging.INFO)

np.random.seed(1)

def true_function(x):
  '''
  An arbitrary function that we are trying to recover through
  interpolation
  '''
  return np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])

def true_x_derivative(x):
  '''
  The x derivative of the true function, which we also want to recover
  '''
  return 2*np.pi*np.cos(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
  
# generate `nobs` observation locations 
nobs = 100
xobs = np.random.uniform(0.0,1.0,(nobs,2))
# evaluate the true function at the obsevation points
uobs = true_function(xobs)
# generate random uncertainties for each observation
sobs = np.random.uniform(0.1,0.2,(nobs,))
# add noise to the observations
uobs += np.random.normal(0.0,sobs)

# generate a grid of interpolation points
x1itp,x2itp = np.meshgrid(np.linspace(0.0,1.0,100),
                          np.linspace(0.0,1.0,100))
x1itp = x1itp.flatten()
x2itp = x2itp.flatten()
xitp = np.array([x1itp,x2itp]).T

## Perform Gaussian process regression

# this is the prior covariance function, a wenland function. It is
# compact and the subsequent matrix operations will use sparse
# matrices. We can perform Gaussian process regression on large
# datasets with this choice of prior, provided that the lengthscale of
# the prior is much less than the size of the domain (which is not
# true for this demo)
basis = spwen32 

# define hyperparameters for the prior. Tune these parameters to get a
# satisfactory interpolant. These can also be chosen with maximum
# likelihood methods.
prior_mean = 0.0 
prior_var = 1.0 
prior_lengthscale = 0.8 # this controls the sparsity 

# create the prior Gaussian process 
prior_gp = gpiso(basis, var=prior_var, eps=prior_lengthscale)
# add a first order polynomial to the prior to make it suitable for
# data with linear trends
prior_gp += gppoly(1) 

# condition the prior on the observations, creating a new Gaussian
# process for the posterior.
posterior_gp = prior_gp.condition(xobs, uobs, dcov=np.diag(sobs**2))

# differentiate the posterior with respect to x
derivative_gp = posterior_gp.differentiate((1, 0))

# evaluate the posterior and posterior derivative at the interpolation
# points. calling the GaussianProcess instances will return their mean
# and standard deviation at the provided points.
post_mean, post_std = posterior_gp(xitp)
diff_mean, diff_std = derivative_gp(xitp)


## Plotting

# plot the true function
utrue = true_function(xitp)
plt.figure(1)
plt.title('True function')
p = plt.tripcolor(xitp[:,0],xitp[:,1],utrue,cmap='viridis')
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar(p)
plt.tight_layout()

# plot the interpolant
plt.figure(2)
plt.title('RBF interpolant (dots are observations)')
p = plt.tripcolor(xitp[:,0],xitp[:,1],post_mean,cmap='viridis')
# plot the observations
plt.scatter(xobs[:,0],xobs[:,1],c=uobs,s=40,edgecolor='k',
            vmin=p.get_clim()[0],vmax=p.get_clim()[1])
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar(p)
plt.tight_layout()


# plot the interpolant uncertainty
plt.figure(3)
plt.title('RBF interpolant uncertainty (dots are observed uncertainty)')
p = plt.tripcolor(xitp[:,0],xitp[:,1],post_std,cmap='viridis')
# plot the observations
plt.scatter(xobs[:,0],xobs[:,1],c=sobs,s=40,edgecolor='k',
            vmin=p.get_clim()[0],vmax=p.get_clim()[1])
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar(p)
plt.tight_layout()

# print the weighted RMS, this should be close to 1 if the predicted
# uncertainties are accurate
wrms = np.sqrt( np.mean( ( (post_mean - utrue) / post_std )**2 ) )
print('weighted RMSE for interpolant : %.4f' % wrms)


# plot the true function derivative
difftrue = true_x_derivative(xitp)
plt.figure(4)
plt.title('True function x derivative')
p = plt.tripcolor(xitp[:,0],xitp[:,1],difftrue,cmap='viridis')
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar(p)
plt.tight_layout()

# plot the interpolant derivative
plt.figure(5)
plt.title('RBF interpolant x derivative')
p = plt.tripcolor(xitp[:,0],xitp[:,1],diff_mean,cmap='viridis')
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar(p)
plt.tight_layout()


# plot the interpolant derivative uncertainty
plt.figure(6)
plt.title('RBF interpolant x derivative uncertainty')
p = plt.tripcolor(xitp[:,0],xitp[:,1],diff_std,cmap='viridis')
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar(p)
plt.tight_layout()

# print the weighted RMS, this should be close to 1 if the predicted
# uncertainties are accurate
wrms = np.sqrt( np.mean( ( (diff_mean - difftrue) / diff_std )**2 ) )
print('weighted RMSE for interpolant derivative : %.4f' % wrms)


plt.show()









