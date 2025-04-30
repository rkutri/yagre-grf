import matplotlib.pyplot as plt

from numpy import linspace

from sampling.dnaFourier import DNAFourierEngine1D, DNAFourierEngine2D
from sampling.spde import SPDEEngine2D
from sampling.dnaSPDE import DNASPDEEngine2D
from sampling.circulantEmbedding import CirculantEmbedding2DEngine, ApproximateCirculantEmbedding2DEngine
from utility.covariances import matern_ptw, matern_fourier_ptw

corrLength = 0.2
smoothness = 1
var = 1.

nGrid = [100, 75]

nSamp = [int(1e2), int(5e0)]


engines2D = {
     "dna_fourier": DNAFourierEngine2D,
     "dna_spde": DNASPDEEngine2D,
     "spde": SPDEEngine2D,
     "ce": CirculantEmbedding2DEngine,
     "aCE": ApproximateCirculantEmbedding2DEngine
}


dimIdx = 0

for DIM in [1, 2]:

    def cov_ftrans_callable(s):
        return matern_fourier_ptw(s, corrLength, smoothness, DIM)

    def cov_callable(x):
        return matern_ptw(x, corrLength, smoothness)

    if (DIM == 1):
        dimIdx +=1
        continue

        fig, axes = plt.subplots(2, 2)
        print("Sampling 1D fields using DNA in Fourier basis")

        rf = DNAFourierEngine1D(cov_ftrans_callable, nGrid[dimIdx])

        nPlot = 50
        assert nPlot + 3 <= nSamp[dimIdx]

        samples = [rf.generate_realisation() for _ in range(nSamp[dimIdx])]

        grid = linspace(0., 1., nGrid[dimIdx], endpoint=True)

        axes[0, 0].plot(grid, samples[0])
        axes[0, 1].plot(grid, samples[1])
        axes[1, 0].plot(grid, samples[2])

        for i in range(3, 3 + nPlot):
            axes[1, 1].plot(grid, samples[i], alpha=0.5, color='blue')

        plt.show()

    if DIM == 2:

        for method, engine in engines2D.items():

            print(f"Sampling in 2D using {method}")

            if method == "dna_fourier":
                rf = engine(cov_ftrans_callable, nGrid[dimIdx], scaling=2.)
            elif method == "dna_spde":
                rf = engine(var, corrLength, smoothness, nGrid[dimIdx], 1.)
            elif method == "spde":
                rf = engine(var, corrLength, smoothness, nGrid[dimIdx], 1.2)
            elif method == "ce":
                rf = engine(cov_callable, nGrid[dimIdx])
            elif method == "aCE":
                rf = engine(cov_callable, nGrid[dimIdx])
     
            realisations = []
            for n in range(nSamp[dimIdx]):

                if method not in ["ce", "aCE"]:
                    realisations.append(rf.generate_realisation())
                else:
                    realisation, _ = rf.generate_realisation()
                    realisations.append(realisation)

            fig, axes = plt.subplots(2, 2)

            fig.suptitle(f"{nSamp[dimIdx]} realisations using {method} in {DIM}D")

            for i, ax in enumerate(axes.flat):

                im = ax.imshow(realisations[i], cmap='viridis', interpolation='nearest')
                ax.axis('off')
                fig.colorbar(im, ax=ax)

            plt.show()

    dimIdx += 1
