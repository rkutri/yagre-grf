import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from yagregrf.sampling.dnaFourier import DNAFourierEngine2D
from yagregrf.sampling.circulantEmbedding import ApproximateCirculantEmbeddingEngine2D
from yagregrf.utility.covariances import matern_ptw, matern_fourier_ptw

DIM = 2
ell = 0.25
nu = 1.0
n = 500
var = 1.
padding = 0

np.random.seed(59)

def cov_fcn(x): return var * matern_ptw(x, ell, nu)
def cov_fourier(x): return var * matern_fourier_ptw(x, ell, nu, dim=DIM)

print(f"setting up DNA with {n} grid points per dimension")
dnaRF = DNAFourierEngine2D(cov_fourier, n)
print("generating realisation")
dnaRealisation = dnaRF.generate_realisation()

print(f"setting up approximate CE with {n} grid points per dimension")
aceRF = ApproximateCirculantEmbeddingEngine2D(cov_fcn, n, padding=padding)
print("generating realisation")
aceRealisation = aceRF.generate_realisation()[0]

# Plotting
fig, (axCE, axDNA) = plt.subplots(1, 2, figsize=(8, 3))
plt.subplots_adjust(wspace=-0.04)  # Reduce space between images

colorMap = 'seismic'
maxAbsValue = np.max(np.abs([dnaRealisation, aceRealisation]))

axCE.imshow(
    aceRealisation,
    cmap=colorMap,
    vmin=-maxAbsValue,
    vmax=maxAbsValue)

dnaImage = axDNA.imshow(
    dnaRealisation,
    cmap=colorMap,
    vmin=-maxAbsValue,
    vmax=maxAbsValue)

axCE.axis('off')
axDNA.axis('off')

axCE.set_title("Approximate CE")
axDNA.set_title("DNA using DST/DCT")

# Add colorbar with more padding (distance) from the image
divider = make_axes_locatable(axDNA)
cax = divider.append_axes("right", size="10%", pad=0.4)
cbar = plt.colorbar(dnaImage, cax=cax)
cbar.ax.tick_params(labelsize=10)

# Save to PDF at 300 dpi
pdfFileName = 'realisation_comparison.pdf'
dpiValue = 300
fig.savefig(pdfFileName, dpi=dpiValue, format='pdf', bbox_inches='tight')

print(f"Saved figure as '{pdfFileName}' with {dpiValue} dpi.")

