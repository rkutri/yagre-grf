import matplotlib.pyplot as plt

from numpy import linspace
from dnaSampling import DNAGaussianRandomField1d, DNAGaussianRandomField2d
from covariances import matern_fourier_ptw

corrLength = 0.1
smoothness = 1.5

nGrid = 100

nSamp = int(1e3)

for DIM in [1, 2]:

    def cov_ftrans(s):
        return matern_fourier_ptw(s, corrLength, smoothness, DIM)

    DNARF = DNAGaussianRandomField1d if DIM == 1 else DNAGaussianRandomField2d
    dnaRF = DNARF(cov_ftrans, nGrid)

    samples = dnaRF.generate(nSamp)

    fig, axes = plt.subplots(2, 2)

    fig.suptitle(f"DNA GRF in {DIM}d")

    if DIM == 1:

        nPlot = 50
        assert nPlot + 3 <= nSamp

        grid = linspace(0., 1., nGrid, endpoint=True)

        axes[0, 0].plot(grid, samples[0])
        axes[0, 1].plot(grid, samples[1])
        axes[1, 0].plot(grid, samples[2])

        for i in range(3, 3 + nPlot):
            axes[1, 1].plot(grid, samples[i], alpha=0.5, color='blue')

    else:
        for i, ax in enumerate(axes.flat):

            im = ax.imshow(samples[i], cmap='viridis', interpolation='nearest')
            ax.axis('off')
            fig.colorbar(im, ax=ax)

    plt.show()
