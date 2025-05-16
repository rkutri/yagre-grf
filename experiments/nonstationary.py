import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cs

import experiments.scriptUtility as util

from yagregrf.sampling.spde import SPDEEngine2DDeep
from yagregrf.sampling.dnaSPDE import DNASPDEEngine2DDeep
from yagregrf.sampling.dnaFourier import DNAFourierEngine2D

from yagregrf.utility.covariances import matern_fourier_ptw
from yagregrf.utility.accumulation import CovarianceAccumulator, MarginalVarianceAccumulator

# Parameters
DIM = 2

varCoeff = 5.
varField = 1.

ellCoeff = 0.2
effectiveEll = 0.1

nuCoeff = 8.
nuField = 1.

effectiveKappa = np.sqrt(2.*nuField) / effectiveEll

alphaSPDE = 1.
alphaDNA = 1.

dofPerDim = 128
nSamp = 10000

def cov_ftrans_callable(s):
    return varCoeff * matern_fourier_ptw(s, ellCoeff, nuCoeff, DIM)

# Set up model problem
correlationRF = DNAFourierEngine2D(cov_ftrans_callable, dofPerDim)
correlationCoeff = effectiveKappa**2 * np.exp(correlationRF.generate_realisation())

spdeRF = SPDEEngine2DDeep(varField, correlationCoeff, nuField, dofPerDim, alphaSPDE, [False, False])
dnaRF = DNASPDEEngine2DDeep(varField, correlationCoeff, nuField, dofPerDim, alphaDNA)


anisotropy = np.array([[2.8, 0.4],
                       [0.4, 0.7]])
spdeRF.anisotropy = anisotropy
dnaRF.anisotropy = anisotropy

spdeMV = MarginalVarianceAccumulator(spdeRF.nDof)
dnaMV = MarginalVarianceAccumulator(dnaRF.nDof)

spdeCov = CovarianceAccumulator(dofPerDim)
dnaCov = CovarianceAccumulator(dofPerDim)

def print_progress(n, nSamp, nUpdates=9):
    assert nSamp > nUpdates
    if n % (nSamp // (nUpdates + 1)) == 0:
        if n == 0:
            print("Start sampling")
        else:
            print(f"{n} realisations computed")

for n in range(nSamp):
    print_progress(n, nSamp)

    spdeRealisation = spdeRF.generate_realisation()
    dnaRealisation = dnaRF.generate_realisation()

    spdeMV.update(spdeRealisation.flatten())
    dnaMV.update(dnaRealisation.flatten())

    spdeSlice = util.extract_diagonal_slice(spdeRealisation)
    dnaSlice = util.extract_diagonal_slice(dnaRealisation)

    spdeCov.update(spdeSlice)
    dnaCov.update(dnaSlice)

# Pick a realisation to plot (the last one)
spdeRealPlot = spdeRealisation.reshape((dofPerDim, dofPerDim))
dnaRealPlot = dnaRealisation.reshape((dofPerDim, dofPerDim))

# Marginal variances
spdeVar = spdeMV.marginalVariance.reshape((dofPerDim, dofPerDim))
dnaVar = dnaMV.marginalVariance.reshape((dofPerDim, dofPerDim))

vminMV = spdeVar.min()
vmaxMV = spdeVar.max()

spdeCovariance = np.maximum(spdeCov.covariance, 0.) 
dnaCovariance = np.maximum(dnaCov.covariance, 0.)


vminCov = spdeCovariance.min()
vmaxCov = spdeCovariance.max()

#norm = cs.LogNorm(vmin, vmax)
normMV = cs.Normalize(vminMV, vmaxMV)
normCov = cs.LogNorm(vminCov, vmaxCov)

# Correlation coefficient field
correlationCoeff2D = correlationCoeff.reshape((dofPerDim, dofPerDim))

# Set up figure
fig, axs = plt.subplots(2, 4, figsize=(10, 20))

def show_image(ax, data, title, cmap,  norm=None):
    im = ax.imshow(data, origin='lower', cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


show_image(axs[0][0], correlationCoeff2D, "Coefficient", 'coolwarm')
show_image(axs[1][0], np.log(correlationCoeff2D), "Logarithm of Coefficient", 'coolwarm')
show_image(axs[0][1], np.flipud(dnaRealPlot), "DNA Realisation", 'coolwarm')
show_image(axs[1][1], np.flipud(spdeRealPlot), "SPDE Realisation", 'coolwarm')
show_image(axs[0][2], dnaVar, "DNA Marginal Variance", 'Spectral_r', normMV)
show_image(axs[1][2], spdeVar, "SPDE Marginal Variance", 'Spectral_r', normMV)
show_image(axs[0][3], np.flipud(spdeCovariance), "SPDE Covariance Matrix", 'Spectral_r', normCov)
show_image(axs[1][3], np.flipud(dnaCovariance), "DNA Covariance Matrix", 'Spectral_r', normCov)

plt.tight_layout()
plt.show()

