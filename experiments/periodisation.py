import numpy as np
import matplotlib.pyplot as plt

from yagregrf.utility.covariances import matern_ptw

# Domain settings
nDom = 2001
dom = np.linspace(-2.2, 2.2, nDom, endpoint=True)

# Parameters for the covariance functions
ell = 0.4
nu = 2.0


def cov_fcn(x):
    return matern_ptw(x, ell, nu)


# Create figure with narrower aspect ratio for half-page width
fig, ax = plt.subplots(figsize=(4, 2))

# Calculate pristine covariance
pristineCov = np.array([cov_fcn(x) for x in dom])

# Plot pristine covariance
ax.plot(dom, pristineCov, color='tab:red', label=r'$\varphi$', linewidth=1.5)

# Naive periodisation
nDom4 = nDom // 4
nPrd = 10
alpha = 1.0
naivePrdCov = np.copy(pristineCov)

for n in range(1, nPrd):
    naivePrdCov += np.array([cov_fcn(x + alpha * n) for x in dom])
    naivePrdCov += np.array([cov_fcn(x - alpha * n) for x in dom])

# Plot naive periodisation with cropped domain
ax.plot(2. * dom[nDom4:-nDom4],
        naivePrdCov[nDom4:-nDom4],
        color='tab:green',
        label=r'$\tilde{\varphi}_{1}^{(\pi)}$',
        linewidth=1.5)

# Periodisation of CRF
alpha = 2.0
crfPrdCov = np.copy(pristineCov)

for n in range(1, nPrd):
    crfPrdCov += np.array([cov_fcn(x + alpha * n) for x in dom])
    crfPrdCov += np.array([cov_fcn(x - alpha * n) for x in dom])

ax.plot(
    dom,
    crfPrdCov,
    color='tab:blue',
    label=r'$\varphi_{2}^{(\pi)}$',
    linewidth=1.5)

# Circulant embedding periodisation


def crop_cov(x):
    return cov_fcn(x) if np.abs(x) <= 1. else 0.


cePrdCov = np.array([crop_cov(x) for x in dom])

for n in range(1, 2):
    cePrdCov += np.array([crop_cov(x + alpha * n) for x in dom])
    cePrdCov += np.array([crop_cov(x - alpha * n) for x in dom])

ax.plot(
    dom,
    cePrdCov,
    color='k',
    label=r'$\varphi_{1, CE}^{(\pi)}$',
    linewidth=1.5,
    linestyle='--')

# Improve appearance for publication quality
ax.set_xlabel(r'distance $\delta$', fontsize=12)
ax.set_ylabel(r'Covariance', fontsize=12)

# Set legend to an unobtrusive position outside the plot
ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

# Add horizontal line at y=0 and set limits
ax.axhline(0, color='gray', linewidth=1.0, linestyle='-')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-0.2, np.max(pristineCov) + 0.5)

# add target domain boundaries
ax.axvline(-1., 0., 1.5, color='gray', linewidth=1.0, linestyle='-')
ax.axvline(1., 0., 1.5, color='gray', linewidth=1.0, linestyle='-')

# Set specific y-axis ticks
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-0.2, 0, 0.5, 1, 1.5])

# Set tick parameters for a balanced layout
ax.tick_params(axis='both', labelsize=10)

# Remove vertical lines at ends of the x-axis
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.grid(which='both', color='gray', linestyle='--', linewidth=0.3, alpha=0.5)

# Save figure in a format compatible with LaTeX without borders
plt.tight_layout(pad=0)
plt.savefig(
    "covariance_periodisation_narrow.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0)
plt.show()
