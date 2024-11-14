import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from numpy.random import standard_normal
from scipy.fft import ifft, ifft2, fftshift, ifftshift, dct, dst, fft2

import covariance_functions as covs
import utility


# domain
domExt = 1.
nDof = 400

print("\n- degrees of freedom: " + str(nDof) + "\n")


# covariance
corrLength = 0.15
smoothness = 2.0


def cov_fcn(x):
    return covs.matern_covariance_ptw(x, corrLength, smoothness)


def cov_ftrans(s):
    return covs.matern_fourier_ptw(s, corrLength, smoothness, 1)


# coefficient
print("Computing Coefficients")

bcFTEval = np.array([cov_ftrans(0.5 * m) for m in range(nDof + 1)])
prdFTEval = np.array([cov_ftrans(m) for m in range(nDof)])


bcCoeff = np.sqrt(bcFTEval)
prdCoeff = np.sqrt(prdFTEval)

# sampling
nSamp = int(5e5)

dirSample = []
neuSample = []
prdSample = []

print("Start sampling")

for n in range(nSamp):

    if (n % 10000 == 0):
        print(str(n) + " realisations computed.")

    dirStoch = standard_normal(nDof + 1)
    neuStoch = standard_normal(nDof + 1)
    prdStoch = standard_normal(nDof) + 1.j * standard_normal(nDof)

    dstEval = utility.sin_series(bcCoeff * dirStoch)
    dctEval = utility.cos_series(bcCoeff * neuStoch)

    dirSample += [dstEval]
    neuSample += [dctEval]

    prdEval = ifft(prdCoeff * prdStoch, norm="forward")

    prdSample += [np.real(prdEval)]
    prdSample += [np.imag(prdEval)]

prdCov = np.cov(prdSample, rowvar=False)
dirCov = np.cov(dirSample, rowvar=False)
neuCov = np.cov(neuSample, rowvar=False)

dirMv = np.var(dirSample, axis=0)
neuMv = np.var(neuSample, axis=0)

grid = np.linspace(0., domExt, nDof + 2, endpoint=True)
priCov = utility.evaluate_isotropic_covariance_1d(cov_fcn, grid)

# Create a figure and GridSpec layout
fig1 = plt.figure(figsize=(4, 4))
fig2 = plt.figure(figsize=(8, 4))

# Set up GridSpec for a 2x2 plot on the left and 1x1 plot on the right
gs = GridSpec(2, 2, width_ratios=[1, 1], hspace=-0.3, wspace=-0.12)

# Left 2x2 grid for covariance matrices
ax00 = fig1.add_subplot(gs[0, 0])
ax01 = fig1.add_subplot(gs[0, 1])
ax10 = fig1.add_subplot(gs[1, 0])
ax11 = fig1.add_subplot(gs[1, 1])

ax2 = fig2.add_subplot()  # Span both rows for the right plot

# Set higher resolution for export
mpl.rcParams['figure.dpi'] = 600

# Colormap and color range settings for all subplots
vmin, vmax = 0., 2.0
cmap = plt.cm.turbo

# Line width and font size
lw = 3.
fontsize = 12

ax2.tick_params(axis='both', which='major', labelsize=fontsize)
ax2.grid(which='both', linestyle='--', linewidth=0.6)

# Plot the sample variances
ax2.plot(
    grid,
    neuMv,
    color='tab:blue',
    label='Neumann',
    linewidth=lw,
    zorder=2)
ax2.plot(
    grid,
    dirMv,
    color='tab:red',
    label='Dirichlet',
    linewidth=lw,
    zorder=2)

# Baseline at y=0 and y=1
ax2.plot(grid, np.zeros_like(grid), color='k', linewidth=0.5 * lw, zorder=7)
ax2.plot(
    grid,
    np.ones_like(grid),
    linestyle='--',
    linewidth=lw,
    color='dimgray',
    zorder=1)

# Labels with appropriate font sizes
ax2.set_xlabel(r'location $x$', fontsize=fontsize, labelpad=10)
ax2.set_ylabel(
    r'sample marginal variance $\sigma^2(x)$',
    fontsize=fontsize,
    labelpad=10)

# Legend configuration
ax2.legend(loc='best', prop={'size': fontsize}, framealpha=1.)

# Aspect ratio and layout adjustments
ax2.set_aspect(aspect=0.38)

pos2 = ax2.get_position()  # Get the original position of ax2
new_pos2 = [pos2.x0 + 0.1, pos2.y0, pos2.width,
            pos2.height]  # Shift ax2 slightly to the left
ax2.set_position(new_pos2)  # Apply the new position

# Plotting with imshow
priCovIm = ax00.imshow(priCov, vmin=vmin, vmax=vmax, cmap=cmap)
neuCovIm = ax01.imshow(neuCov, vmin=vmin, vmax=vmax, cmap=cmap)
prdCovIm = ax10.imshow(prdCov, vmin=vmin, vmax=vmax, cmap=cmap)
dirCovIm = ax11.imshow(dirCov, vmin=vmin, vmax=vmax, cmap=cmap)

cbarticks = [0, 0.5, 1, 1.5, 2]  # Desired ticks for colorbars

fig1.colorbar(
    priCovIm,
    ax=ax00,
    ticks=cbarticks,
    fraction=0.4,
    pad=0.05,
    shrink=0.55).ax.tick_params(
        labelsize=10)
fig1.colorbar(
    neuCovIm,
    ax=ax01,
    ticks=cbarticks,
    fraction=0.4,
    pad=0.05,
    shrink=0.55).ax.tick_params(
        labelsize=10)
fig1.colorbar(
    prdCovIm,
    ax=ax10,
    ticks=cbarticks,
    fraction=0.4,
    pad=0.05,
    shrink=0.55).ax.tick_params(
        labelsize=10)
fig1.colorbar(
    dirCovIm,
    ax=ax11,
    ticks=cbarticks,
    fraction=0.4,
    pad=0.05,
    shrink=0.55).ax.tick_params(
        labelsize=10)

for ax in [ax00, ax01, ax10, ax11]:
    ax.tick_params(which='both', size=0, labelsize=0, color='white')
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_aspect('equal')

fig1.savefig('covariance1d.pdf', bbox_inches='tight', format='pdf', dpi=600)
fig2.savefig('margVar1d.pdf', bbox_inches='tight', format='pdf', dpi=600)

plt.show()
