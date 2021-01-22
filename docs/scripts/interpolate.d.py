'''
Demonstrates optimizing the shape parameter for the RBFInterpolant when using
an RBF other than a polyharmonic spline
'''
import numpy as np
import matplotlib.pyplot as plt

from rbf.interpolate import RBFInterpolant

np.random.seed(0)

def frankes_test_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    term1 = 0.75 * np.exp(-(9*x1-2)**2/4 - (9*x2-2)**2/4)
    term2 = 0.75 * np.exp(-(9*x1+1)**2/49 - (9*x2+1)/10)
    term3 = 0.5 * np.exp(-(9*x1-7)**2/4 - (9*x2-3)**2/4)
    term4 = -0.2 * np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    y = term1 + term2 + term3 + term4
    return y

phi = 'ga' # use a gaussian RBF
degree = 0 # degree of the added polynomial
y = np.random.random((100, 2)) # observation points
d = frankes_test_function(y) # observed values at y
x = np.mgrid[0:1:200j, 0:1:200j].reshape(2, -1).T # interpolation points
k = 5 # number of subgroups used for k-fold cross validation

def cross_validation(epsilon):
    groups = [range(i, len(y), k) for i in range(k)]
    error = 0.0
    for i in range(k):
        train = np.hstack([groups[j] for j in range(k) if j != i])
        test = groups[i]
        interp = RBFInterpolant(
            y[train],
            d[train],
            phi=phi,
            order=degree,
            eps=epsilon
            )
        error += ((interp(y[test]) - d[test])**2).sum()

    mse = error / len(y)
    return mse

# range of epsilon values to test
test_epsilons = 10**np.linspace(-0.5, 2.5, 1000)
mses = [cross_validation(eps) for eps in test_epsilons]

best_mse = np.min(mses)
best_epsilon = test_epsilons[np.argmin(mses)]
print('best epsilon: %.2e (MSE=%.2e)' % (best_epsilon, best_mse))

interp = RBFInterpolant(y, d, phi=phi, order=degree, eps=best_epsilon)

fig, ax = plt.subplots()
ax.loglog(test_epsilons, mses, 'k-')
ax.grid(ls=':', color='k')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('cross validation MSE')
ax.plot(best_epsilon, best_mse, 'ko')
ax.text(
    best_epsilon,
    best_mse,
    '(%.2e, %.2e)' % (best_epsilon, best_mse),
    va='top'
    )
fig.tight_layout()

fig, axs = plt.subplots(2, 1, figsize=(6, 8))
p = axs[0].tripcolor(x[:, 0], x[:, 1], interp(x))
axs[0].scatter(y[:, 0], y[:, 1], c='k', s=5)
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)
axs[0].set_title(
    'RBF interpolant ($\phi$=%s, degree=%s, $\epsilon$=%.2f)'
    % (phi, degree, best_epsilon)
    )
axs[0].set_xlabel('$x_0$')
axs[0].set_ylabel('$x_1$')
axs[0].grid(ls=':', color='k')
axs[0].set_aspect('equal')
fig.colorbar(p, ax=axs[0])

error = np.abs(interp(x) - frankes_test_function(x))
p = axs[1].tripcolor(x[:, 0], x[:, 1], error)
axs[1].scatter(y[:, 0], y[:, 1], c='k', s=5)
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].set_title('|error|')
axs[1].set_xlabel('$x_0$')
axs[1].set_ylabel('$x_1$')
axs[1].grid(ls=':', color='k')
axs[1].set_aspect('equal')
fig.colorbar(p, ax=axs[1])

fig.tight_layout()
plt.show()
