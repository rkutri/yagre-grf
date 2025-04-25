import os
import csv
import sys
import time
import tracemalloc

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


def max_error(matdiff):

    if np.any(np.isnan(matdiff)):
        raise RuntimeError("NaN in Matrix difference")

    return np.max(np.abs(matdiff))


if len(sys.argv) < 2:
    print("Usage: python3 script.py <filenameID> (must be two digits)")
    sys.exit(1)

filenameID = sys.argv[1]

if not (filenameID.isdigit() and len(filenameID) == 2):
    print("Error: filenameID must be exactly two digits (e.g., '02', '15').")
    sys.exit(1)

print(f"Filename ID set to: '{filenameID}'")

dofPerDim = [8, 16, 32, 64, 128, 256, 512]

models = [
    "gaussian",
    "matern",
    "exponential"
]

DIM = 2
nSamp = int(1e1)
nAvg = 10000

# dataBaseDir = 'data'
dataBaseDir = os.path.join("experiments", "publicationData")

# CE produces two realisations per FFT, so we only need half the sample size
# in that case
assert nSamp % 2 == 0

covParams = {
    "gaussian": {"ell": 0.1},
    "matern": {"ell": 0.1, "nu": 6.},
    "exponential": {"ell": 0.1}
}

covFcns = {
    "gaussian":
        lambda x: gaussian_ptw(x, covParams["gaussian"]["ell"]),
    "matern":
        lambda x: matern_ptw(x, covParams["matern"]["ell"],
                             covParams["matern"]["nu"]),
    "exponential":
        lambda x: matern_ptw(x, covParams["exponential"]["ell"], 0.5)
}

pwSpecs = {
    "gaussian":
        lambda x: gaussian_fourier_ptw(x, covParams["gaussian"]["ell"], DIM),
    "matern":
        lambda x: matern_fourier_ptw(x, covParams["matern"]["ell"],
                                     covParams["matern"]["nu"], DIM),
    "exponential":
        lambda x: matern_fourier_ptw(
            x, covParams["exponential"]["ell"], 0.5, DIM)
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


memoryData = {
    "dna": {modelCov: [] for modelCov in models},
    "ce": {modelCov: [] for modelCov in models},
    "aCE": {modelCov: [] for modelCov in models}
}


errorData = {
    "maxError": {
        "dna": {modelCov: [] for modelCov in models},
        "ce": {modelCov: [] for modelCov in models},
        "aCE": {modelCov: [] for modelCov in models}
    },
    "froError": {
        "dna": {modelCov: [] for modelCov in models},
        "ce": {modelCov: [] for modelCov in models},
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

        avgMem = 0.
        avgCost = 0.

        for n in range(nSamp):

            print_sampling_progress(n, nSamp)

            if n < nAvg:

                tracemalloc.start()
                startTime = time.perf_counter()

                realisation = dnaRF.generate_realisation()

                endTime = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()

                tracemalloc.stop()

                # convert to MB
                peak /= 1e6

                avgMem += (peak - avgMem) / (n + 1)
                avgCost += (endTime - startTime - avgCost) / (n + 1)

            else:
                realisation = dnaRF.generate_realisation()

            diagSlice = util.extract_diagonal_slice(realisation)
            dnaCov.update(diagSlice)

        print(f"Average peak memory: {avgMem}")
        print(f"Average time per realisation: {avgCost}")

        memoryData["dna"][modelCov].append(avgMem)
        costData["dna"][modelCov].append(avgCost)

        print("\n\n- Running vanilla Circulant Embedding")

        embeddingPossible = True

        ceCov = CovarianceAccumulator(nGrid)

        try:
            maxPadding = 1024
            ceRF = CirculantEmbedding2DEngine(
                covFcns[modelCov], nGrid, maxPadding=1024)

        except RuntimeError as e:

            print(f"Error in CE setup: {e}")

            avgMem = np.inf
            avgCost = np.inf
            embeddingPossible = False

        if embeddingPossible:

            avgMem = 0.
            avgCost = 0.

            for n in range(nSamp // 2):

                print_sampling_progress(n, nSamp)

                if n < nAvg:

                    tracemalloc.start()
                    startTime = time.perf_counter()

                    rls1, rls2 = ceRF.generate_realisation()

                    endTime = time.perf_counter()
                    _, peak = tracemalloc.get_traced_memory()

                    # convert to MB
                    peak /= 1e6

                    tracemalloc.stop()

                    # full memory cost of the algorithm
                    avgMem += (peak - avgMem) / (n + 1)

                    # Half the cost, as we produce two realisations
                    avgCost += 0.5 * (endTime - startTime - avgCost) / (n + 1)

                else:
                    rls1, rls2 = ceRF.generate_realisation()

                diagSlice = util.extract_diagonal_slice(rls1)
                ceCov.update(diagSlice)

                diagSlice = util.extract_diagonal_slice(rls2)
                ceCov.update(diagSlice)

        print(f"Average peak memory: {avgMem}")
        print(f"Average time per realisation: {avgCost}")

        memoryData["ce"][modelCov].append(avgMem)
        costData["ce"][modelCov].append(avgCost)

        print("\n\n- Running Approximate Circulant Embedding")

        aCERF = ApproximateCirculantEmbedding2DEngine(covFcns[modelCov], nGrid)
        aCECov = CovarianceAccumulator(nGrid)

        avgMem = 0.
        avgCost = 0.

        for n in range(nSamp // 2):

            print_sampling_progress(n, nSamp)

            if n < nAvg:

                tracemalloc.start()
                startTime = time.perf_counter()

                rls1, rls2 = aCERF.generate_realisation()

                endTime = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()

                tracemalloc.stop()

                # convert to MB
                peak /= 1e6

                # full memory cost of the algorithm
                avgMem += (peak - avgMem) / (n + 1)

                # half the cost as we generate two realisations
                avgCost += 0.5 * (endTime - startTime - avgCost) / (n + 1)

            else:
                rls1, rls2 = aCERF.generate_realisation()

            diagSlice = util.extract_diagonal_slice(rls1)
            aCECov.update(diagSlice)

            diagSlice = util.extract_diagonal_slice(rls2)
            aCECov.update(diagSlice)

        print(f"Average peak memory: {avgMem}")
        print(f"Average time per realisation: {avgCost}")

        memoryData["aCE"][modelCov].append(avgMem)
        costData["aCE"][modelCov].append(avgCost)

        diagonalGrid = np.sqrt(DIM) * np.linspace(0., 1., nGrid)

        trueCov = evaluate_isotropic_covariance_1d(
            covFcns[modelCov], diagonalGrid)
        trueCovFrob = norm(trueCov, ord='fro')

        dnaError = trueCov - dnaCov.covariance
        ceError = trueCov - ceCov.covariance
        aCEError = trueCov - aCECov.covariance

        maxErrorDNA = max_error(dnaError)
        froErrorDNA = norm(dnaError, ord='fro') / trueCovFrob

        maxErrorCE = max_error(ceError)
        froErrorCE = norm(ceError, ord='fro') / trueCovFrob

        maxErrorACE = max_error(aCEError)
        froErrorACE = norm(aCEError, ord='fro') / trueCovFrob

        errorData["maxError"]["dna"][modelCov].append(maxErrorDNA)
        errorData["froError"]["dna"][modelCov].append(froErrorDNA)

        errorData["maxError"]["ce"][modelCov].append(maxErrorCE)
        errorData["froError"]["ce"][modelCov].append(froErrorCE)

        errorData["maxError"]["aCE"][modelCov].append(maxErrorACE)
        errorData["froError"]["aCE"][modelCov].append(froErrorACE)


errorTypes = ["maxError", "froError"]

subDir = os.path.join(dataBaseDir, "circulantEmbedding")

# cost comparison data
outDir = os.path.join(subDir, "cost")
os.makedirs(outDir, exist_ok=True)

filename = os.path.join(outDir, "run_" +
                        f"{int(nSamp // 1000)}k_{filenameID}.csv")

with open(filename, mode='w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["meshWidths"] + [1. / n for n in dofPerDim])

    for method in ["dna", "ce", "aCE"]:
        for modelCov in models:
            writer.writerow([method + "_" + modelCov] +
                            costData[method][modelCov])

# error comparison data
for eType in errorTypes:

    outDir = os.path.join(subDir, "error", eType)
    os.makedirs(outDir, exist_ok=True)

    filename = os.path.join(outDir, "run_" + f"{filenameID}.csv")

    with open(filename, mode='w', newline='') as file:

        writer = csv.writer(file)

        for method in ["dna", "aCE"]:
            for modelCov in models:

                methodTitle = method + "_" + modelCov
                writer.writerow([methodTitle + "_cost"] +
                                costData[method][modelCov])
                writer.writerow([methodTitle + "_" + eType] +
                                errorData[eType][method][modelCov])

# memory comparison data
outDir = os.path.join(subDir, "memory")
os.makedirs(outDir, exist_ok=True)

filename = os.path.join(outDir, "run_" + f"{filenameID}.csv")

with open(filename, mode='w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["meshWidths"] + [1. / n for n in dofPerDim])

    for method in ["dna", "ce", "aCE"]:
        for modelCov in models:
            writer.writerow([method + "_" + modelCov] +
                            memoryData[method][modelCov])
