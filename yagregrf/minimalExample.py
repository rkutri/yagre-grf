import matplotlib.pyplot as plt

from numpy import linspace

from sampling.randomField import RandomField
from sampling.dnaFourier import DNAFourierEngine1d, DNAFourierEngine2d
from utility.covariances import matern_fourier_ptw

corrLength = 0.1
smoothness = 1.5

nGrid = [1000, 150]

nSamp = [int(1e3), int(5e1)]

dimIdx = 0

for DIM in [1, 2]:

    def cov_ftrans_callable(s):
        return matern_fourier_ptw(s, corrLength, smoothness, DIM)

    if (DIM == 1):
        dnaEngine = DNAFourierEngine1d(cov_ftrans_callable, nGrid[dimIdx])
    elif (DIM == 2):
        dnaEngine = DNAFourierEngine2d(cov_ftrans_callable, nGrid[dimIdx])
    else:
        raise NotImplementedError(
            "Currently only dimensions 1 and 2 are implemented")

    dnaRF = RandomField(dnaEngine)
    samples = dnaRF.generate(nSamp[dimIdx])

    fig, axes = plt.subplots(2, 2)

    fig.suptitle(f"{nSamp[dimIdx]} realisations of the DNA GRF in {DIM}d")

    if DIM == 1:

        nPlot = 50
        assert nPlot + 3 <= nSamp[dimIdx]

        grid = linspace(0., 1., nGrid[dimIdx], endpoint=True)

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

    dimIdx += 1
