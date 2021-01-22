'''
Demonstrates optimizing the smoothing parameter for the RBFInterpolator when
the data contains noise
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

# use a 3rd order polyharmonic spline
phi = 'phs3'
# degree of the added polynomial
degree = 1
# shape parameter
epsilon = 1.0
# observation points
y = np.random.random((100, 2))
# observed values at y with noise
d = frankes_test_function(y) + np.random.normal(0.0, 0.1, len(y))
# interpolation points
x = np.mgrid[0:1:200j, 0:1:200j].reshape(2, -1).T
# number of subgroups used for k-fold cross validation
k = 5

def cross_validation(smoothing):
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
            eps=epsilon,
            sigma=smoothing
            )
        error += ((interp(y[test]) - d[test])**2).sum()

    mse = error / len(y)
    return mse

# range of epsilon values to test
test_smoothings = 10**np.linspace(-4.0, 3.0, 1000)
mses = [cross_validation(s) for s in test_smoothings]

best_mse = np.min(mses)
best_smoothing = test_smoothings[np.argmin(mses)]
print('best smoothing parameter: %.2e (MSE=%.2e)' % (best_smoothing, best_mse))

interp = RBFInterpolant(
    y, d,
    phi=phi,
    order=degree,
    eps=epsilon,
    sigma=best_smoothing
    )

fig, ax = plt.subplots()
ax.loglog(test_smoothings, mses, 'k-')
ax.grid(ls=':', color='k')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('cross validation MSE')
ax.plot(best_smoothing, best_mse, 'ko')
ax.text(
    best_smoothing,
    best_mse,
    '(%.2e, %.2e)' % (best_smoothing, best_mse),
    va='top'
    )
fig.tight_layout()

fig, axs = plt.subplots(2, 1, figsize=(6, 8))
p = axs[0].tripcolor(x[:, 0], x[:, 1], interp(x))
axs[0].scatter(y[:, 0], y[:, 1], c='k', s=5)
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)
axs[0].set_title(
    'RBF interpolant ($\phi$=%s, degree=%s, $\lambda$=%.2e)'
    % (phi, degree, best_smoothing)
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
