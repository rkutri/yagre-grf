import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import yagregrf.utility.covariances as covs

from numpy.random import standard_normal
from scipy.fft import dct, dctn, dst, dstn

from yagregrf.utility.evaluation import norm
from yagregrf.utility.series import sin_series, cos_series


# domain
dim = 2
domExt = 1.
nDof = 150


print("\n- grid points per dimension: " + str(nDof))
print("- total grid points: " + str(nDof**dim) + "\n")


# covariance
corrLength = 0.2
smoothness = 1.5


def cov_fcn(x):
    return covs.matern_covariance_ptw(x, corrLength, smoothness)


def cov_ftrans(s):
    return covs.matern_fourier_ptw(s, corrLength, smoothness, 2)


# coefficient
print("Computing Coefficients")
# offset of 2 for boundary values
fourierEval = np.zeros((nDof + 1, nDof + 1))

for i in range(nDof):
    for j in range(nDof):
        fourierEval[i, j] = cov_ftrans(0.5 * norm([i, j]))

coeff = np.sqrt(fourierEval)

# sampling
nSamp = int(1e4)

ddSample = []
dnSample = []
ndSample = []
nnSample = []

for n in range(nSamp):

    # Dirichlet - Dirichlet
    ddEval = np.zeros((nDof + 2, nDof + 2))
    ddCoeff = standard_normal((nDof + 1, nDof + 1)) * coeff

    for row in range(nDof + 1):
        ddEval[row, :] = sin_series(ddCoeff[row, :])
    for col in range(nDof + 2):
        ddEval[:, col] = sin_series(ddEval[:-1, col])

    ddSample += [ddEval]

    # Dirichlet - Neumann
    dnEval = np.zeros((nDof + 2, nDof + 2))
    dnCoeff = standard_normal((nDof + 1, nDof + 1)) * coeff

    for row in range(nDof + 1):
        dnEval[row, :] = sin_series(dnCoeff[row, :])
    for col in range(nDof + 2):
        dnEval[:, col] = cos_series(dnEval[:-1, col])

    dnSample += [dnEval]

    # Neumann - Dirichlet
    ndEval = np.zeros((nDof + 2, nDof + 2))
    ndCoeff = standard_normal((nDof + 1, nDof + 1)) * coeff

    for row in range(nDof + 1):
        ndEval[row, :] = cos_series(ndCoeff[row, :])
    for col in range(nDof + 2):
        ndEval[:, col] = sin_series(ndEval[:-1, col])

    ndSample += [ndEval]

    # Neumann - Neumann
    nnEval = np.zeros((nDof + 2, nDof + 2))
    nnCoeff = standard_normal((nDof + 1, nDof + 1)) * coeff

    for row in range(nDof + 1):
        nnEval[row, :] = cos_series(nnCoeff[row, :])
    for col in range(nDof + 2):
        nnEval[:, col] = cos_series(nnEval[:-1, col])

    nnSample += [nnEval]

    if (n % 500 == 0):
        print(str(n) + " realisations computed.")

sample = [0.5 * (nnSample[i] + ndSample[i] + dnSample[i] + ddSample[i])
          for i in range(nSamp)]

print("Analysing statistics")

nnMargVar = np.var(np.array(nnSample), axis=0)
ndMargVar = np.var(np.array(ndSample), axis=0)
dnMargVar = np.var(np.array(dnSample), axis=0)
ddMargVar = np.var(np.array(ddSample), axis=0)

margVar = np.var(np.array(sample), axis=0)


# save large figure
fig, ax = plt.subplots(figsize=(7, 7))

# plot entire marginal variance using turbo colormap
cax = ax.imshow(margVar, vmin=0., vmax=4., cmap=plt.cm.turbo)

nGridHalf = (nDof + 2) // 2

nnMargVar = nnMargVar[:nGridHalf, nGridHalf:]
ndMargVar = ndMargVar[:nGridHalf, nGridHalf:]
dnMargVar = dnMargVar[:nGridHalf, nGridHalf:]
ddMargVar = ddMargVar[:nGridHalf, nGridHalf:]


useRectangle = True

if useRectangle:

    rect = Rectangle((nGridHalf - 1, 0), nGridHalf, nGridHalf,
                     linewidth=5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

else:
    # gray-out the upper right hand corner
    cornerMask = np.zeros_like(margVar, dtype=bool)

    cornerMask[:nGridHalf, nGridHalf:] = True
    cornerMargVar = np.ma.masked_where(~cornerMask, margVar - 0.5)

    ax.imshow(cornerMargVar, cmap='Greys')

cbar = fig.colorbar(cax, ax=ax, shrink=0.75, aspect=15, location='left')
cbar.ax.tick_params(labelsize=40, size=20, width=5)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()

plt.savefig('margVar_combined.png', dpi=600, format='png')
plt.close()

# save smaller figures

plt.figure(figsize=(3, 3))
plt.imshow(nnMargVar, vmin=0., vmax=4., cmap=plt.cm.turbo)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('margVar_nn.png', dpi=600, format='png')
plt.close()


plt.figure(figsize=(3, 3))
plt.imshow(ndMargVar, vmin=0., vmax=4., cmap=plt.cm.turbo)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('margVar_nd.png', dpi=600, format='png')
plt.close()

plt.figure(figsize=(3, 3))
plt.imshow(dnMargVar, vmin=0., vmax=4., cmap=plt.cm.turbo)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('margVar_dn.png', dpi=600, format='png')
plt.close()

plt.figure(figsize=(3, 3))
plt.imshow(ddMargVar, vmin=0., vmax=4., cmap=plt.cm.turbo)
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('margVar_dd.png', dpi=600, format='png')
plt.close()
