import os
import csv
import sys
import time
import tracemalloc

import numpy as np
import experiments.scriptUtility as util

from numpy.linalg import norm

from yagregrf.sampling.spde import SPDEEngine2D
from yagregrf.sampling.dnaSPDE import DNASPDEEngine2D
from yagregrf.sampling.dnaFourier import DNAFourierEngine2D
from yagregrf.utility.covariances import matern_ptw, matern_fourier_ptw
from yagregrf.utility.accumulation import CovarianceAccumulator, MarginalVarianceAccumulator
from yagregrf.utility.evaluation import evaluate_isotropic_covariance_1d
from filename import create_data_string


def max_error(matdiff): return np.max(np.abs(matdiff))


# Check if an argument was provided
if len(sys.argv) < 2:
    print("Usage: python3 script.py <filenameID> (must be two digits)")
    sys.exit(1)

filenameID = sys.argv[1]

# Ensure it's exactly two digits
if not (filenameID.isdigit() and len(filenameID) == 2):
    print("Error: filenameID must be exactly two digits (e.g., '02', '15').")
    sys.exit(1)

print(f"Filename ID set to: '{filenameID}'")

# Parameters
DIM = 2
var = 0.1
ell = 0.05
nu = 1.
nSamp = int(5e4)

kappa = np.sqrt(2. * nu) / ell
beta = 0.5 * (1. + nu)

dofPerDim = [16, 32, 64, 128, 256]

deltas = np.array([0., 2., 4.]) * ell
# 2 delta in each direction
oversampling = [1. + 2. * delta for delta in deltas]
print(f"oversampling deltas for this run: {deltas}")

# used in estimation of average time per sample and avg memory current
nAvg = 1000

# fixed number of dofs used for marginal variance plot
nFixedMV = dofPerDim[-1]

# fixed oversampling size used when decreasing mesh width. Corresponds
# to 2 times correlation length heuristic.
osHeuristic = 1. + 4. * ell  # 2 ell in each direction/dimension
osFixedSPDE = min(oversampling, key=lambda x: abs(x - osHeuristic))

dataBaseDir = os.path.join('experiments', 'publicationData')


def print_sampling_progress(n, nSamp, nUpdates=9):

    assert nSamp > nUpdates

    if n % (nSamp // (nUpdates + 1)) == 0:

        if n == 0:
            print("Start sampling")
        else:
            print(f"{n} realisations computed")


def cov_fcn(r):
    return var * matern_ptw(r, ell, nu)


def cov_ftrans_callable(s):
    return var * matern_fourier_ptw(s, ell, nu, DIM)


osData = {
    "maxError":
    {
        "DNA_fourier": [],
        "DNA_spde": [],
        "SPDE": [[] for _ in oversampling]
    },
    "froError":
    {
        "DNA_fourier": [],
        "DNA_spde": [],
        "SPDE": [[] for _ in oversampling]
    }
}

memData = {
    "DNA_fourier": {"memory": [], "maxError": [], "froError": []},
    "DNA_spde": {"memory": [], "maxError": [], "froError": []},
    "SPDE_osFix": {"memory": [], "maxError": [], "froError": []}
}

cData = {
    "DNA_fourier": {"cost": [], "maxError": [], "froError": []},
    "DNA_spde": {"cost": [], "maxError": [], "froError": []},
    "SPDE_osFix": {"cost": [], "maxError": [], "froError": []}
}

mvData = {
    "pos": [],
    "DNA_fourier": [],
    "DNA_spde": [],
    "SPDE": [[] for _ in oversampling]
}

for nDof in dofPerDim:

    print(f"\n\n\nRunning experiments with {nDof} dofs per dimension")
    print("--------------------------------------------------")

    print(f"\n\n- Running DNA Sampling in Fourier basis")

    dnaFourierRF = DNAFourierEngine2D(cov_ftrans_callable, nDof)
    dnaFourierCov = CovarianceAccumulator(nDof)
    dnaFourierMV = MarginalVarianceAccumulator(nDof)

    avgMem = 0.
    avgCost = 0.

    for n in range(nSamp):

        print_sampling_progress(n, nSamp)

        if n < nAvg:

            tracemalloc.start()
            startTime = time.perf_counter()

            realisation = dnaFourierRF.generate_realisation()

            endTime = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()

            # convert to MB
            peak /= 1e6

            avgMem += (peak - avgMem) / (n + 1)
            avgCost += (endTime - startTime - avgCost) / (n + 1)

            tracemalloc.stop()

        else:
            realisation = dnaFourierRF.generate_realisation()

        diagSlice = util.extract_diagonal_slice(realisation)
        dnaFourierCov.update(diagSlice)
        dnaFourierMV.update(diagSlice)

    print(f"\naverage peak memory usage: {avgMem} MB")
    print(f"Average time per realisation: {avgCost}")

    memData["DNA_fourier"]["memory"].append(avgMem)
    cData["DNA_fourier"]["cost"].append(avgCost)

    print(f"\n\n- Running DNA Sampling using SPDE approach")

    dnaSPDERF = DNASPDEEngine2D(var, ell, nu, nDof, 1.)
    dnaSPDECov = CovarianceAccumulator(nDof)
    dnaSPDEMV = MarginalVarianceAccumulator(nDof)

    avgMem = 0.
    avgCost = 0.

    for n in range(nSamp):

        print_sampling_progress(n, nSamp)

        if n < nAvg:

            tracemalloc.start()
            startTime = time.perf_counter()

            realisation = dnaSPDERF.generate_realisation()

            endTime = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()

            # convert to MB
            peak /= 1e6

            avgMem += (peak - avgMem) / (n + 1)
            avgCost += (endTime - startTime - avgCost) / (n + 1)

            tracemalloc.stop()

        else:
            realisation = dnaSPDERF.generate_realisation()

        diagSlice = util.extract_diagonal_slice(realisation)
        dnaSPDECov.update(diagSlice)
        dnaSPDEMV.update(diagSlice)

    print(f"\naverage peak memory usage: {avgMem} MB")
    print(f"Average time per realisation: {avgCost}")

    memData["DNA_spde"]["memory"].append(avgMem)
    cData["DNA_spde"]["cost"].append(avgCost)

    print(f"\n\n- Sampling using SPDE approach with oversampling")

    spdeCovList = [CovarianceAccumulator(nDof) for _ in oversampling]
    spdeMVList = [MarginalVarianceAccumulator(nDof) for _ in oversampling]

    for alphaIdx, alpha in enumerate(oversampling):

        print(f"\noversampling with alpha = {alpha}")

        nOsDof = int(np.ceil(alpha * nDof))
        if nOsDof % 2 == 1:
            nOsDof += 1
        osWidth = (nOsDof - nDof) // 2

        print(f"oversampling width: {osWidth} degrees of freedom\n")

        spdeRF = SPDEEngine2D(var, ell, nu, nOsDof, alpha,
                              useDirBC=[False, False])

        spdeCov = spdeCovList[alphaIdx]
        spdeMV = spdeMVList[alphaIdx]

        avgMem = 0.
        avgCost = 0.

        for n in range(nSamp):

            print_sampling_progress(n, nSamp)

            if n < nAvg:

                tracemalloc.start()
                startTime = time.perf_counter()

                realisation = spdeRF.generate_realisation()

                endTime = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()

                # convert to MB
                peak /= 1e6

                avgMem += (peak - avgMem) / (n + 1)
                avgCost += (endTime - startTime - avgCost) / (n + 1)

                tracemalloc.stop()

            else:
                realisation = spdeRF.generate_realisation()

            diagSlice = util.extract_diagonal_slice(realisation)

            if osWidth > 0:
                diagSlice = diagSlice[osWidth:-osWidth]

            spdeCov.update(diagSlice)

            if nDof == nFixedMV:
                spdeMV.update(diagSlice)

        print(f"\naverage peak memory usage: {avgMem} MB")
        print(f"Average time per realisation: {avgCost}")

        if np.abs(osFixedSPDE - alpha) < 1e-8:

            memData["SPDE_osFix"]["memory"].append(avgMem)
            cData["SPDE_osFix"]["cost"].append(avgCost)

    diagonalGrid = util.extract_diagonal_from_mesh(dnaSPDERF.mesh)

    trueCov = evaluate_isotropic_covariance_1d(cov_fcn, diagonalGrid)
    trueCovFrob = norm(trueCov, ord='fro')

    if nDof == nFixedMV:
        mvData["pos"] = diagonalGrid

    dnaFError = trueCov - dnaFourierCov.covariance
    dnaSError = trueCov - dnaSPDECov.covariance

    maxErrorDNAFourier = max_error(dnaFError)
    froErrorDNAFourier = norm(dnaFError, ord='fro') / trueCovFrob

    maxErrorDNASPDE = max_error(dnaSError)
    froErrorDNASPDE = norm(dnaSError, ord='fro') / trueCovFrob

    osData["maxError"]["DNA_fourier"].append(maxErrorDNAFourier)
    osData["maxError"]["DNA_spde"].append(maxErrorDNASPDE)
    osData["froError"]["DNA_fourier"].append(froErrorDNAFourier)
    osData["froError"]["DNA_spde"].append(froErrorDNASPDE)

    memData["DNA_fourier"]["maxError"].append(maxErrorDNAFourier)
    memData["DNA_fourier"]["froError"].append(froErrorDNAFourier)
    memData["DNA_spde"]["maxError"].append(maxErrorDNASPDE)
    memData["DNA_spde"]["froError"].append(froErrorDNASPDE)

    cData["DNA_fourier"]["maxError"].append(maxErrorDNAFourier)
    cData["DNA_fourier"]["froError"].append(froErrorDNAFourier)
    cData["DNA_spde"]["maxError"].append(maxErrorDNASPDE)
    cData["DNA_spde"]["froError"].append(froErrorDNASPDE)

    if nDof == nFixedMV:

        mvData["DNA_fourier"] = dnaFourierMV.marginalVariance
        mvData["DNA_spde"] = dnaSPDEMV.marginalVariance

    for i, spdeCov in enumerate(spdeCovList):

        spdeError = trueCov - spdeCov.covariance

        maxErrorSPDE = max_error(spdeError)
        froErrorSPDE = norm(spdeError, ord='fro') / trueCovFrob

        osData["maxError"]["SPDE"][i].append(maxErrorSPDE)
        osData["froError"]["SPDE"][i].append(froErrorSPDE)

        if np.abs(osFixedSPDE - oversampling[i]) < 1e-8:

            memData["SPDE_osFix"]["maxError"].append(maxErrorSPDE)
            memData["SPDE_osFix"]["froError"].append(froErrorSPDE)

            cData["SPDE_osFix"]["maxError"].append(maxErrorSPDE)
            cData["SPDE_osFix"]["froError"].append(froErrorSPDE)

        if nDof == nFixedMV:
            mvData["SPDE"][i] = spdeMVList[i].marginalVariance

experimentConfig = [
    ("oversampling", "os"),
    ("memory", "mem"),
    ("cost", "cost"),
    ("marginalVariance", "mv")
]

errorTypes = ["maxError", "froError"]


for dataVariable, prefix in experimentConfig:

    for i, error in enumerate(errorTypes):

        # Skip mv for all but the first error type
        if prefix == "mv" and i > 0:
            continue

        dataString = create_data_string(DIM, var, ell, nu, nSamp, prefix)

        if prefix == "mv":
            outDir = os.path.join(dataBaseDir, dataVariable, dataString)
        else:
            outDir = os.path.join(dataBaseDir, dataVariable, error, dataString)

        os.makedirs(outDir, exist_ok=True)

        filename = os.path.join(outDir, dataString + f"_{filenameID}.csv")

        with open(filename, mode='w', newline='') as file:

            writer = csv.writer(file)

            if prefix == "os":

                writer.writerow(["mesh_widths"] + [1. / n for n in dofPerDim])
                writer.writerow(["DNA_fourier"] + osData[error]["DNA_fourier"])
                writer.writerow(["DNA_spde"] + osData[error]["DNA_spde"])

                for i, alpha in enumerate(oversampling):
                    writer.writerow(
                        [f"SPDE_alpha{int(100*alpha)}"] +
                        osData[error]["SPDE"][i]
                    )

            elif prefix == "mem":

                writer.writerow(
                    ["DNA_fourier_memory"] +
                    memData["DNA_fourier"]["memory"])
                writer.writerow(["DNA_fourier_" + error] +
                                memData["DNA_fourier"][error])
                writer.writerow(
                    ["DNA_spde_memory"] +
                    memData["DNA_spde"]["memory"])
                writer.writerow(["DNA_spde_" + error] +
                                memData["DNA_spde"][error])
                writer.writerow(
                    ["SPDE_osFix_memory"] +
                    memData["SPDE_osFix"]["memory"])
                writer.writerow(["SPDE_osFix_" + error] +
                                memData["SPDE_osFix"][error])

            elif prefix == "cost":

                writer.writerow(
                    ["DNA_fourier_cost"] +
                    cData["DNA_fourier"]["cost"])
                writer.writerow(["DNA_fourier_" + error] +
                                cData["DNA_fourier"][error])
                writer.writerow(["DNA_spde_cost"] + cData["DNA_spde"]["cost"])
                writer.writerow(["DNA_spde_" + error] +
                                cData["DNA_spde"][error])
                writer.writerow(
                    ["SPDE_osFix_cost"] +
                    cData["SPDE_osFix"]["cost"])
                writer.writerow(["SPDE_osFix_" + error] +
                                cData["SPDE_osFix"][error])

            elif prefix == "mv":

                writer.writerow(["position"] + mvData["pos"].tolist())
                writer.writerow(
                    ["DNA_fourier"] +
                    mvData["DNA_fourier"].tolist())
                writer.writerow(["DNA_spde"] + mvData["DNA_spde"].tolist())

                for i, alpha in enumerate(oversampling):
                    writer.writerow(
                        [f"SPDE_alpha{int(100*alpha)}"] +
                        mvData["SPDE"][i].tolist())
