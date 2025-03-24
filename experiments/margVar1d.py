import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import yagregrf.utility.covariances as covs

from numpy.random import standard_normal
from scipy.fft import rfft, dct, dst

from yagregrf.utility.series import sin_series, cos_series

# domain
domExt = 1.
nDof = 400

dom = np.linspace(0., domExt, nDof + 2, endpoint=True)

print("\n- total grid points: " + str(nDof) + "\n")


# covariance
corrLength = 0.15
smoothness = 2.0


def cov_fcn(x):
    return covs.matern_covariance_ptw(x, corrLength, smoothness)


def cov_ftrans(s):
    return covs.matern_fourier_ptw(s, corrLength, smoothness, 1)


# coefficient
print("Computing Coefficients")

grid = np.linspace(0., domExt, nDof + 2, endpoint=False)

ftEval = np.array([cov_ftrans(0.5 * m) for m in range(nDof + 1)])

coeff = np.sqrt(ftEval)

# sampling
nSamp = int(8e5)

dirSample = []
neuSample = []

for n in range(nSamp):

    if (n % 10000 == 0):
        print(str(n) + " realisations computed.")

    dirSample += [sin_series(standard_normal(nDof + 1) * coeff)]
    neuSample += [cos_series(standard_normal(nDof + 1) * coeff)]


print("Analysing statistics")

# sample marginal variance
dirCov = np.cov(dirSample, rowvar=False)
neuCov = np.cov(neuSample, rowvar=False)

dirMv = np.var(dirSample, axis=0)
neuMv = np.var(neuSample, axis=0)


# Improve plot aesthetics for scientific publication
fig, ax = plt.subplots(1, 1, figsize=(4, 4))  # Adjust size for A4 paper

mpl.rcParams['figure.dpi'] = 300

# Line width and font size
lw = 3.
fontsize = 11

# Plotting with improved styling
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.grid(which='both', linestyle='--', linewidth=0.6)

# Plot the sample variances
ax.plot(dom, neuMv, color='tab:blue', label='Neumann', linewidth=lw, zorder=2)
ax.plot(dom, dirMv, color='tab:red', label='Dirichlet', linewidth=lw, zorder=2)

# Baseline at y=0 and y=1
ax.plot(dom, np.zeros_like(dom), color='k', linewidth=0.5 * lw, zorder=7)
ax.plot(
    dom,
    np.ones_like(dom),
    linestyle='--',
    linewidth=lw,
    color='dimgray',
    zorder=1)

# Labels with appropriate font sizes
ax.set_xlabel(r'location $x$', fontsize=fontsize, labelpad=10)
ax.set_ylabel(r'sample variance $\sigma^2(x)$', fontsize=fontsize, labelpad=10)

# Legend configuration
ax.legend(loc='best', prop={'size': fontsize}, framealpha=1.)

# Aspect ratio and layout adjustments
ax.set_aspect(aspect=0.35)
plt.tight_layout()

# Show or save plot
plt.show()

# Optionally, save for high-quality export to LaTeX
fig.savefig('marginal_variances_1d_dir_neu.pdf',
            bbox_inches='tight', format='pdf')
