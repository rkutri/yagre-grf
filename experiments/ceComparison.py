import os
import csv
import sys
import time

import numpy as np
import experiments.scriptUtility as util

from numpy.linalg import norm

from yagregrf.sampling.spde import SPDEEngine2d
from yagregrf.sampling.dnaSPDE import DNASPDEEngine2d
from yagregrf.sampling.dnaFourier import DNAFourierEngine2d
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

dofPerDim = [16, 32, 64, 128, 256]

# covariance functions used for comparsion
models = [
    "cauchy",
    "gaussian",
    "matern_smooth",
    "matern_nonsmooth",
    "exponential"]

DIM = 2

ell1 = 0.05
ell2 = 0.1
ell3 = 0.2

nu1 = 2
nu2 = 8

covFcns = {
    "cauchy": lambda x: cauchy_ptw(x, ell2),
    "gaussian": lambda x: gaussian_ptw(x, ell2),
    "matern_smooth" lambda x: matern_ptw(x, ell3, nu2),
    "matern_nonsmooth" lambda x: matern_ptw(x, ell1, nu1),
    "exponential" lambda x: matern_ptw(x, ell2)
}

pwSpecs = {
    "cauchy": lambda x cauchy_fourier_2d_ptw(x, ell2),
    "gaussian": lambda x: gaussian_fourier_2d_ptw(x, ell2),
    "matern_smooth" lambda x: matern_fourier_ptw(x, ell3, nu2, DIM),
    "matern_nonsmooth" lambda x: matern_fourier_ptw(x, ell1, nu1, DIM),
    "exponential" lambda x: exponential_fourier_2d_ptw(x, ell2)
}

# used in estimation of average time per realisation
nAvg = 10000


def print_sampling_progress(n, nSamp, nUpdates=9):

    assert nSamp > nUpdates

    if n % (nSamp // (nUpdates + 1)) == 0:

        if n == 0:
            print("Start sampling")
        else:
            print(f"{n} realisations computed")


meshWidths = []

costData = {
    "dna": {modelCov: [] for modelCov in models},
    "ce": {modelCov: [] for modelCov in models},
    "approxCE": {modelCov: [] for modelCov in models}
}

errorData = {
    "dna": {modelCov: [] for modelCov in models},
    "approxCE": {modelCov: [] for modelCov in models}
}


for nDof in dofPerDim:

    print(f"\n\n\nRunning experiments with {nDof} dofs per dimension")
    print("--------------------------------------------------")

    for modelCov in models:

        print(f"\n\n- Benchmarking {modelCov} covariance")

        print("\n\n- Running DNA Sampling")

        dnaFourierRF = DNAFourierEngine2d(pwSpecs[modelCov], nDof)
        dnaFourierCov = CovarianceAccumulator(nDof)

        avgCost = 0.

        for n in range(nSamp):

            print_sampling_progress(n, nSamp)

            if n < nAvg:

                startTime = time.perf_counter()

                realisation = dnaFourierRF.generate_realisation()

                endTime = time.perf_counter()
                avgCost += (endTime - startTime - avgCost) / (n + 1)

            else:
                realisation = dnaFourierRF.generate_realisation()

            diagSlice = util.extract_diagonal_slice(realisation)
            dnaFourierCov.update(diagSlice)

        print(f"Average time per realisation: {avgCost}")

        costData["dna"][modelCov].append(avgCost)

        print("\n\n- Running vanilla Circulant Embedding")

        ceRF = CirculantEmbedding2dEngine(...)
        ceCov = CovarianceAccumulator(nDof)

        avgCost = 0.

        for n in range(nSamp):

            print_sampling_progress(n, nSamp)

            if n < nAvg:

                startTime = time.perf_counter()

                realisation = ceRF.generate_realisation()

                endTime = time.perf_counter()
                avgCost += (endTime - startTime - avgCost) / (n + 1)

            else:
                realisation = dnaSPDERF.generate_realisation()

            diagSlice = util.extract_diagonal_slice(realisation)
            ceCov.update(diagSlice)

        print(f"Average time per realisation: {avgCost}")

        costData["ce"][modelCov].append(avgCost)

        print("\n\n- Running Approximate Circulant Embedding")

        aCERF = CirculantEmbedding2dEngine(...)
        aCECov = CovarianaCEAccumulator(nDof)

        avgCost = 0.

        for n in range(nSamp):

            print_sampling_progress(n, nSamp)

            if n < nAvg:

                startTime = time.perf_counter()

                realisation = aCERF.generate_realisation()

                endTime = time.perf_counter()
                avgCost += (endTime - startTime - avgCost) / (n + 1)

            else:
                realisation = dnaSPDERF.generate_realisation()

            diagSlice = util.extract_diagonal_slice(realisation)
            aCECov.update(diagSlice)

        print(f"Average time per realisation: {avgCost}")

        costData["approxCE"][modelCov].append(avgCost)

        # TODO

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

        dataString = create_data_string(DIM, var, ell, nu, nSamp, prefix)

        for error in errorTypes:

            paramDir = dataString

            if prefix == "mv":
                outDir = os.path.join("data", dataVariable, paramDir)
            else:
                outDir = os.path.join("data", dataVariable, error, paramDir)

            os.makedirs(outDir, exist_ok=True)

            filename = os.path.join(outDir, dataString + f"_{filenameID}.csv")

            with open(filename, mode='w', newline='') as file:

                writer = csv.writer(file)

                if prefix == "os":

                    writer.writerow(["mesh_widths"] +
                                    [1. / n for n in dofPerDim])
                    writer.writerow(
                        ["DNA_fourier"] +
                        osData[error]["DNA_fourier"])
                    writer.writerow(["DNA_spde"] + osData[error]["DNA_spde"])

                    for i, alpha in enumerate(oversampling):
                        writer.writerow(
                            [f"SPDE_alpha{int(100*alpha)}"] +
                            osData[error]["SPDE"][i])

                if prefix == "mem":

                    writer.writerow(
                        ["DNA_fourier_memory"] +
                        memData["DNA_fourier"]["memory"])
                    writer.writerow(
                        ["DNA_fourier_" + error] +
                        memData["DNA_fourier"][error])

                    writer.writerow(
                        ["DNA_spde_memory"] +
                        memData["DNA_spde"]["memory"])
                    writer.writerow(["DNA_spde_" + error] +
                                    memData["DNA_spde"][error])

                    writer.writerow(
                        ["SPDE_osFix_memory"] +
                        memData["SPDE_osFix"]["memory"])
                    writer.writerow(
                        ["SPDE_osFix_" + error] +
                        memData["SPDE_osFix"][error])

                if prefix == "cost":

                    writer.writerow(
                        ["DNA_fourier_cost"] +
                        cData["DNA_fourier"]["cost"])
                    writer.writerow(
                        ["DNA_fourier_" + error] +
                        cData["DNA_fourier"][error])

                    writer.writerow(
                        ["DNA_spde_cost"] +
                        cData["DNA_spde"]["cost"])
                    writer.writerow(["DNA_spde_" + error] +
                                    cData["DNA_spde"][error])

                    writer.writerow(
                        ["SPDE_osFix_cost"] +
                        cData["SPDE_osFix"]["cost"])
                    writer.writerow(
                        ["SPDE_osFix_" + error] +
                        cData["SPDE_osFix"][error])

                if prefix == "mv" and error == errorTypes[0]:

                    writer.writerow(["position"] + mvData["pos"].tolist())

                    writer.writerow(
                        ["DNA_fourier"] +
                        mvData["DNA_fourier"].tolist())
                    writer.writerow(["DNA_spde"] + mvData["DNA_spde"].tolist())

                    for i, alpha in enumerate(oversampling):
                        writer.writerow(
                            [f"SPDE_alpha{int(100*alpha)}"] +
                            mvData["SPDE"][i].tolist())
