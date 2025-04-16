import os
import csv
import sys
import time

import numpy as np
import experiments.scriptUtility as util

from numpy.linalg import norm

from yagregrf.sampling.dnaFourier import DNAFourierEngine2D
from yagregrf.sampling.circulantEmbedding import (
    CirculantEmbedding2DEngine,
    ApproximateCirculantEmbedding2DEngine
)
from yagregrf.utility.covariances import (
    cauchy_ptw, cauchy_fourier_ptw,
    gaussian_ptw, gaussian_fourier_ptw,
    matern_ptw, matern_fourier_ptw
)
from yagregrf.utility.accumulation import CovarianceAccumulator
from yagregrf.utility.evaluation import evaluate_isotropic_covariance_1d
from filename import create_data_string


def max_error(matdiff): return np.max(np.abs(matdiff))


if len(sys.argv) < 2:
    print("Usage: python3 script.py <filenameID> (must be two digits)")
    sys.exit(1)

filenameID = sys.argv[1]

if not (filenameID.isdigit() and len(filenameID) == 2):
    print("Error: filenameID must be exactly two digits (e.g., '02', '15').")
    sys.exit(1)

print(f"Filename ID set to: '{filenameID}'")

dofPerDim = [16, 32, 64, 128, 256]

models = [
    "cauchy",
    "gaussian",
    "matern_smooth",
    "matern_nonsmooth",
    "exponential"
]

DIM = 2
nSamp = 100
nAvg = 100

# CE produces two realisations per FFT, so we only need half the sample size
# in that case
assert nSamp % 2 == 0

dataBaseDir = 'data'

covParams = {
    "cauchy": {"ell": 0.1},
    "gaussian": {"ell": 0.1},
    "matern_smooth": {"ell": 0.2, "nu": 8.},
    "matern_nonsmooth": {"ell": 0.05, "nu": 1.},
    "exponential": {"ell": 0.1}
}

covFcns = {
    "cauchy": 
        lambda x: cauchy_ptw(x, covParams["cauchy"]["ell"]),
    "gaussian":
        lambda x: gaussian_ptw(x, covParams["gaussian"]["ell"]),
    "matern_smooth":
        lambda x: matern_ptw(x, covParams["matern_smooth"]["ell"],
                                covParams["matern_smooth"]["nu"]),
    "matern_nonsmooth":
        lambda x: matern_ptw(x, covParams["matern_nonsmooth"]["ell"],
                                covParams["matern_nonsmooth"]["nu"]),
    "exponential":
        lambda x: matern_ptw(x, covParams["exponential"]["ell"], 0.5)
}

pwSpecs = {
    "cauchy":
        lambda x: cauchy_fourier_ptw(x, covParams["cauchy"]["ell"], DIM),
    "gaussian":
        lambda x: gaussian_fourier_ptw(x, covParams["gaussian"]["ell"], DIM),
    "matern_smooth":
        lambda x: matern_fourier_ptw(x, covParams["matern_smooth"]["ell"],
                                        covParams["matern_smooth"]["nu"], DIM),
    "matern_nonsmooth":
        lambda x: matern_fourier_ptw(x, covParams["matern_nonsmooth"]["ell"],
                                        covParams["matern_nonsmooth"]["nu"], DIM),
    "exponential":
        lambda x: matern_fourier_ptw(x, covParams["exponential"]["ell"], 0.5, DIM)
}


def print_sampling_progress(n, nSamp, nUpdates=9):
    assert nSamp > nUpdates
    if n % (nSamp // (nUpdates + 1)) == 0:
        if n == 0:
            print("Start sampling")
        else:
            print(f"{n} iterations done")


meshWidths = []

costData = {
    "dna": {modelCov: [] for modelCov in models},
    "ce": {modelCov: [] for modelCov in models},
    "aCE": {modelCov: [] for modelCov in models}
}

errorData = {
    "maxError": {
        "dna": {modelCov: [] for modelCov in models},
        "aCE": {modelCov: [] for modelCov in models}
    },
    "froError": {
        "dna": {modelCov: [] for modelCov in models},
        "aCE": {modelCov: [] for modelCov in models}
    }
}

for nGrid in dofPerDim:

    print(f"\n\n\nRunning experiments with {nGrid} dofs per dimension")
    print("--------------------------------------------------")

    for modelCov in models:

        print(f"\n\n- Benchmarking {modelCov} covariance")

        print("\n\n- Running DNA Sampling")

        dnaRF = DNAFourierEngine2D(pwSpecs[modelCov], nGrid)
        dnaCov = CovarianceAccumulator(nGrid)

        avgCost = 0.

        for n in range(nSamp):
            print_sampling_progress(n, nSamp)
            if n < nAvg:
                startTime = time.perf_counter()
                realisation = dnaRF.generate_realisation()
                endTime = time.perf_counter()
                avgCost += (endTime - startTime - avgCost) / (n + 1)
            else:
                realisation = dnaRF.generate_realisation()

            diagSlice = util.extract_diagonal_slice(realisation)
            dnaCov.update(diagSlice)

        print(f"Average time per realisation: {avgCost}")
        costData["dna"][modelCov].append(avgCost)

        print("\n\n- Running vanilla Circulant Embedding")

        embeddingPossible = True

        try:
            ceRF = CirculantEmbedding2DEngine(covFcns[modelCov], nGrid)

        except RuntimeError as e:

            print(f"Error in CE setup: {e}")

            avgCost = np.inf
            embeddingPossible = False

        if embeddingPossible:

            ceCov = CovarianceAccumulator(nGrid)

            avgCost = 0.

            for n in range(nSamp // 2):

                print_sampling_progress(n, nSamp)

                if n < nAvg:

                    startTime = time.perf_counter()
                    rls1, rls2 = ceRF.generate_realisation()
                    endTime = time.perf_counter()

                    # Half the cost, as we produce two realisations
                    avgCost += 0.5 * (endTime - startTime - avgCost) / (n + 1)

                else:
                    rls1, rls2 = ceRF.generate_realisation()

                diagSlice = util.extract_diagonal_slice(rls1)
                ceCov.update(diagSlice)

                diagSlice = util.extract_diagonal_slice(rls2)
                ceCov.update(diagSlice)

            print(f"Average time per realisation: {avgCost}")

        costData["ce"][modelCov].append(avgCost)

        print("\n\n- Running Approximate Circulant Embedding")

        aCERF = ApproximateCirculantEmbedding2DEngine(covFcns[modelCov], nGrid)
        aCECov = CovarianceAccumulator(nGrid)

        avgCost = 0.

        for n in range(nSamp // 2):

            print_sampling_progress(n, nSamp)

            if n < nAvg:

                startTime = time.perf_counter()

                rls1, rls2 = aCERF.generate_realisation()

                endTime = time.perf_counter()

                # half the cost as we generate two realisations
                avgCost += 0.5 * (endTime - startTime - avgCost) / (n + 1)

            else:
                rls1, rls2 = aCERF.generate_realisation()

            diagSlice = util.extract_diagonal_slice(rls1)
            aCECov.update(diagSlice)

            diagSlice = util.extract_diagonal_slice(rls2)
            aCECov.update(diagSlice)

        print(f"Average time per realisation: {avgCost}")
        costData["aCE"][modelCov].append(avgCost)

        diagonalGrid = np.sqrt(DIM) * np.linspace(0., 1., nGrid)

        trueCov = evaluate_isotropic_covariance_1d(
            covFcns[modelCov], diagonalGrid)
        trueCovFrob = norm(trueCov, ord='fro')

        dnaError = trueCov - dnaCov.covariance
        aCEError = trueCov - aCECov.covariance

        maxErrorDNA = max_error(dnaError)
        froErrorDNA = norm(dnaError, ord='fro') / trueCovFrob

        maxErrorACE = max_error(aCEError)
        froErrorACE = norm(aCEError, ord='fro') / trueCovFrob

        errorData["maxError"]["dna"][modelCov].append(maxErrorDNA)
        errorData["froError"]["dna"][modelCov].append(froErrorDNA)

        errorData["maxError"]["aCE"][modelCov].append(maxErrorACE)
        errorData["froError"]["aCE"][modelCov].append(froErrorACE)

experimentConfig = [
    ("cost", "cost"),
    ("error", "err")
]

errorTypes = ["maxError", "froError"]

# TODO: write the parameters to file

# TODO: write data to one file for cost-mw comparison (DNA vs. CE) and one
#       file for cost-error comparison (DNA vs aCE)

