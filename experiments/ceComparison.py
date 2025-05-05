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
    CirculantEmbeddingEngine2D,
    ApproximateCirculantEmbeddingEngine2D
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

dataBaseDir = os.path.join("experiments", "publicationData")

dofPerDim = [8, 16, 32, 64, 128, 256, 512]

models = [
    "gaussian",
    "matern",
    "exponential"
]

DIM = 2
nSamp = int(1e4)
nAvg = 5000
maxPadding = 1024


# CE produces two realisations per FFT, so we only need half the sample size
# in that case
assert nSamp % 2 == 0

covParams = {
    "gaussian": {"ell": 0.1},
    "matern": {"ell": 0.1, "nu": 5.},
    "exponential": {"ell": 0.1}
}

variance = 0.1

covFcns = {
    "gaussian":
        lambda x: gaussian_ptw(
            x,
            covParams["gaussian"]["ell"],
            margVar=variance),
    "matern":
        lambda x: matern_ptw(x, covParams["matern"]["ell"],
                             covParams["matern"]["nu"],
                             margVar=variance),
    "exponential":
        lambda x: matern_ptw(
            x,
            covParams["exponential"]["ell"],
            nu=0.5,
            margVar=variance)
}

pwSpecs = {
    "gaussian":
        lambda x: gaussian_fourier_ptw(
            x, covParams["gaussian"]["ell"], dim=DIM, margVar=variance),
    "matern":
        lambda x: matern_fourier_ptw(x, covParams["matern"]["ell"],
                                     covParams["matern"]["nu"],
                                     dim=DIM, margVar=variance),
    "exponential":
        lambda x: matern_fourier_ptw(
            x, covParams["exponential"]["ell"], 0.5, dim=DIM, margVar=variance)
}


def print_sampling_progress(n, nSamp, nUpdates=9):
    assert nSamp > nUpdates
    if n % (nSamp // (nUpdates + 1)) == 0:
        if n == 0:
            print("Start sampling")
        else:
            print(f"{n} iterations done")


problemSize = []
costData = {"dna": [], "aCE": [], "ce": {modelCov: [] for modelCov in models}}
memoryData = {"dna": [], "aCE": [], "ce": {modelCov: []
                                           for modelCov in models}}

# covariance used to benchmark runtime and memory for dna and aCE
benchmarkCov = "exponential"

embeddingPossibleFlag = [True for modelCov in models]


print(f"Benchmarking {benchmarkCov} covariance")

for nGrid in dofPerDim:

    print(f"\n\n\nRunning experiments with {nGrid} dofs per dimension")
    print("--------------------------------------------------")

    problemSize.append(nGrid**DIM)

    print("\n\n- Running DNA Sampling")

    dnaRF = DNAFourierEngine2D(pwSpecs[benchmarkCov], nGrid)
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

    memoryData["dna"].append(avgMem)
    costData["dna"].append(avgCost)

    print("\n\n- Running vanilla Circulant Embedding")

    for i, modelCov in enumerate(models):

        print(f"\n- Benchmarking {modelCov} covariance")

        if embeddingPossibleFlag[i]:

            try:
                ceRF = CirculantEmbeddingEngine2D(
                    covFcns[modelCov], nGrid, maxPadding=maxPadding)

            except RuntimeError as e:

                print(f"Error in CE setup: {e}")
                embeddingPossibleFlag[i] = False

                costData["ce"][modelCov].append(np.inf)
                memoryData["ce"][modelCov].append(np.inf)

            if embeddingPossibleFlag[i]:

                ceCov = CovarianceAccumulator(nGrid)

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

                        tracemalloc.stop()

                        # convert to MB
                        peak /= 1e6

                        # full memory cost of the algorithm
                        avgMem += (peak - avgMem) / (n + 1)

                        # half the cost as we generate two realisations
                        avgCost += 0.5 * \
                            (endTime - startTime - avgCost) / (n + 1)

                    else:
                        rls1, rls2 = aCERF.generate_realisation()

                    diagSlice = util.extract_diagonal_slice(rls1)
                    ceCov.update(diagSlice)

                    diagSlice = util.extract_diagonal_slice(rls2)
                    ceCov.update(diagSlice)

                print(f"Average peak memory: {avgMem}")
                print(f"Average time per realisation: {avgCost}")

                memoryData["ce"][modelCov].append(avgMem)
                costData["ce"][modelCov].append(avgCost)

        else:

            print(f"Skipping CE for {modelCov} covariance, as previous " +
                  "embedding was not possible")

            costData["ce"][modelCov].append(np.inf)
            memoryData["ce"][modelCov].append(np.inf)

    print("\n\n- Running Approximate Circulant Embedding")

    aCERF = ApproximateCirculantEmbeddingEngine2D(covFcns[benchmarkCov], nGrid)
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
            rls1, rls2 = ceRF.generate_realisation()

        diagSlice = util.extract_diagonal_slice(rls1)
        aCECov.update(diagSlice)

        diagSlice = util.extract_diagonal_slice(rls2)
        aCECov.update(diagSlice)

    print(f"Average peak memory: {avgMem}")
    print(f"Average time per realisation: {avgCost}")

    memoryData["aCE"].append(avgMem)
    costData["aCE"].append(avgCost)

subDir = os.path.join(dataBaseDir, "circulantEmbedding")

# cost comparison data
outDir = os.path.join(subDir, "cost")
os.makedirs(outDir, exist_ok=True)

filename = os.path.join(outDir, "run_" +
                        f"{int(nSamp // 1000)}k_{filenameID}.csv")

with open(filename, mode='w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["problemSize"] + problemSize)

    for method in ["dna", "aCE"]:
        writer.writerow([method] + costData[method])

    for modelCov in models:
        writer.writerow(["ce_" + modelCov] + costData["ce"][modelCov])


# memory comparison data
outDir = os.path.join(subDir, "memory")
os.makedirs(outDir, exist_ok=True)

filename = os.path.join(outDir, "run_" +
                        f"{int(nSamp // 1000)}k_{filenameID}.csv")

with open(filename, mode='w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(["problemSize"] + problemSize)

    for method in ["dna", "aCE"]:
        writer.writerow([method] + memoryData[method])

    for modelCov in models:
        writer.writerow(["ce_" + modelCov] + memoryData["ce"][modelCov])
