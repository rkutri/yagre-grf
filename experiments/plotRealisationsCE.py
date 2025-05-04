import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

from matplotlib.colors import LinearSegmentedColormap

from yagregrf.sampling.dnaFourier import DNAFourierEngine2D
from yagregrf.sampling.circulantEmbedding import ApproximateCirculantEmbeddingEngine2D

from yagregrf.utility.covariances import matern_ptw, matern_fourier_ptw, gaussian_ptw, gaussian_fourier_ptw


DIM = 2
ell = 0.25
nu = 1.0
n = 500
var = 1.

padding = 0

np.random.seed(59)

#cov_fcn = lambda x: gaussian_ptw(x, ell)
#cov_fourier = lambda x: gaussian_fourier_ptw(x, ell, dim=DIM)


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

# Create masks for the left and right halves
maskLeft = np.zeros_like(aceRealisation, dtype=bool)
maskLeft[:, :aceRealisation.shape[1] // 2] = True
maskRight = ~maskLeft

# Create the combined image
combinedImage = np.zeros_like(aceRealisation)
combinedImage[maskLeft] = aceRealisation[maskLeft]
combinedImage[maskRight] = dnaRealisation[maskRight]

# Plotting
fig, ax = plt.subplots(figsize=(3, 2.2))

# Create a softened version of seismic


def desaturate_seismic(scale):

    base = plt.get_cmap('seismic', 256)
    colors = base(np.linspace(0, 1, 256))

    desaturated = []

    for r, g, b, a in colors:

        h, s, v = cs.rgb_to_hsv(r, g, b)
        s *= scale

        r2, g2, b2 = cs.hsv_to_rgb(h, s, v)
        desaturated.append((r2, g2, b2, a))

    return LinearSegmentedColormap.from_list(
        'desaturated_seismic', desaturated)


#colorMap = desaturate_seismic(scale=0.85)
colorMap = 'seismic'

# Combine images into one for computing range
maxAbsValue = np.max(np.abs([dnaRealisation, aceRealisation]))

# Plotting with symmetric color limits around 0
imageHandle = ax.imshow(
    combinedImage,
    cmap=colorMap,
    vmin=-maxAbsValue,
    vmax=maxAbsValue)

ax.axis('off')

# Draw vertical line in the center (x = width // 2)
height, width = combinedImage.shape
xMid = width // 2
lineWidth = 3  # thickness of the line
lineExtension = 0.05 * height  # extend a bit beyond the image vertically

ax.axvline(
    x=xMid - 0.5,
    color='black',
    linewidth=lineWidth,
    ymin=-lineExtension / height,
    ymax=1 + lineExtension / height,
    clip_on=False
)

fontSize = 6

# Add separate titles manually in axes coordinates
ax.text(0.24, 1.02, "Approximate CE", transform=ax.transAxes,
        ha='center', va='bottom', fontsize=fontSize)
ax.text(0.765, 1.02, "DNA using DST/DCT", transform=ax.transAxes,
        ha='center', va='bottom', fontsize=fontSize)

cbar = plt.colorbar(imageHandle, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=6)

plt.tight_layout()

# Save to PDF at 300 dpi
pdfFileName = 'realisation_comparison.pdf'
dpiValue = 300
fig.savefig(pdfFileName, dpi=dpiValue, format='pdf', bbox_inches='tight')

print(f"Saved figure as '{pdfFileName}' with {dpiValue} dpi.")
