import os
import csv
import sys
import time

import numpy as np
import experiments.scriptUtility as util

from numpy.linalg import norm

from yagregrf.sampling.dnaFourier import DNAFourierEngine2D
from yagregrf.sampling.circulantEmbedding import CirculantEmbedding2DEngine, ApproximateCirculantEmbedding2DEngine
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
ell1 = 0.05
ell2 = 0.1
ell3 = 0.2
nu1 = 2
nu2 = 8

nSamp = 100
nAvg = 100

# CE produces two realisations per FFT, so we only need half the sample size
# in that case
assert nSamp % 2 == 0

dataBaseDir = 'data'

covFcns = {
    "cauchy": lambda x: cauchy_ptw(x, ell2),
    "gaussian": lambda x: gaussian_ptw(x, ell3),
    "matern_smooth": lambda x: matern_ptw(x, ell3, nu2),
    "matern_nonsmooth": lambda x: matern_ptw(x, ell2, nu1),
    "exponential": lambda x: matern_ptw(x, ell2, 0.5)
}

pwSpecs = {
    "cauchy": lambda x: cauchy_fourier_ptw(x, ell2, DIM),
    "gaussian": lambda x: gaussian_fourier_ptw(x, ell3, DIM),
    "matern_smooth": lambda x: matern_fourier_ptw(x, ell3, nu2, DIM),
    "matern_nonsmooth": lambda x: matern_fourier_ptw(x, ell2, nu1, DIM),
    "exponential": lambda x: matern_fourier_ptw(x, ell1, 0.5, DIM)
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

# FIXME

for dataVariable, prefix in experimentConfig:

    for error in errorTypes:
        for modelCov in models:

            ell = ell2  # replace with actual logic if ell varies per model
            nu = nu1    # replace with actual logic if nu varies per model

            dataString = create_data_string(DIM, 1., ell, nu, nSamp, prefix)
            paramDir = dataString
            outDir = os.path.join(dataBaseDir, dataVariable, error, paramDir)
            os.makedirs(outDir, exist_ok=True)

            filename = os.path.join(outDir, dataString + f"_{filenameID}.csv")

            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)

                if prefix == "cost":
                    writer.writerow(["mesh_widths"] +
                                    [1. / n for n in dofPerDim])
                    writer.writerow(["dna_" + modelCov] +
                                    costData["dna"][modelCov])
                    writer.writerow(["ce_" + modelCov] +
                                    costData["ce"][modelCov])

                if prefix == "err":
                    costRowTitle = "dna_" + modelCov + "_cost"
                    errorRowTitle = "dna_" + modelCov + "_" + error
                    writer.writerow([costRowTitle] + costData["dna"][modelCov])
                    writer.writerow(
                        [errorRowTitle] +
                        errorData[error]["dna"][modelCov])

                    costRowTitle = "aCE_" + modelCov + "_cost"
                    errorRowTitle = "aCE_" + modelCov + "_" + error
                    writer.writerow([costRowTitle] + costData["aCE"][modelCov])
                    writer.writerow(
                        [errorRowTitle] +
                        errorData[error]["aCE"][modelCov])
