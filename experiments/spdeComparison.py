import os
import csv
import sys
import time
import tracemalloc

import numpy as np
import experiments.scriptUtility as util

from yagregrf.sampling.randomField import RandomField
from yagregrf.sampling.spde import SPDEEngine2d
from yagregrf.sampling.dnaSPDE import DNASPDEEngine2d
from yagregrf.sampling.dnaFourier import DNAFourierEngine2d
from yagregrf.utility.covariances import matern_ptw, matern_fourier_ptw
from yagregrf.utility.accumulation import CovarianceAccumulator
from yagregrf.utility.evaluation import evaluate_isotropic_covariance_1d
from filename import create_data_string

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

kappa = np.sqrt(2 * nu) / ell
beta = 0.5 * (1. + nu)

dofPerDim = [8, 16, 32, 64, 128, 256]
oversampling = [1., 1.25, 1.5, 2.]

# used in estimation of average time per sample and avg memory current
nAvg = 1000

# fixed number of dofs used when only increasing oversampling size
nFixedSPDE = dofPerDim[-1]

# fixed oversampling size used when decreasing mesh width. Corresponds 
# to 2 times correlation length heuristic. Use oversampling value which
# is closest to the heuristic
osHeuristic = 1. + 2. * DIM * ell # 2 ell in each direction
osFixedSPDE = min(oversampling, key=lambda x: abs(x - osHeuristic))


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
    "DNA_fourier": [],
    "DNA_spde": [],
    "SPDE": [[] for _ in oversampling]
}
memData = {
    "DNA_fourier": {"memory": [], "error": []},
    "DNA_spde": {"memory": [], "error": []},
#    "SPDE_nFix": {"memory": [], "error": []},
    "SPDE_osFix": {"memory": [], "error": []}
    }
cData = {
    "DNA_fourier": {"cost": [], "error": []},
    "DNA_spde": {"cost": [], "error": []},
#    "SPDE_nFix": {"cost": [], "error": []},
    "SPDE_osFix": {"cost": [], "error": []}
    }

for nDof in dofPerDim:

    sampleSize = 1

    print(f"\n\n\nRunning experiments with {nDof} dofs per dimension")
    print("--------------------------------------------------")

    print(f"\n\n- Running DNA Sampling in Fourier basis")

    dnaFourierRF = RandomField(DNAFourierEngine2d(cov_ftrans_callable, nDof))
    dnaFourierCov = CovarianceAccumulator(nDof)

    avgMem = 0.
    avgCost = 0.

    for n in range(nSamp):

        print_sampling_progress(n, nSamp)

        if n < nAvg:

            tracemalloc.start()
            startTime = time.perf_counter()

            realisation = dnaFourierRF.generate(sampleSize)[0]

            endTime = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()

            # convert to MB
            peak /= 1e6

            avgMem += (peak - avgMem) / (n + 1)
            avgCost += (endTime - startTime - avgCost) / (n + 1)

            tracemalloc.stop()

        else:
            realisation = dnaFourierRF.generate(sampleSize)[0]

        diagSlice = util.extract_diagonal_slice(realisation)
        dnaFourierCov.update(diagSlice)

    print(f"\naverage peak memory usage: {avgMem} MB")
    print(f"Average time per realisation: {avgCost}")

    memData["DNA_fourier"]["memory"].append(avgMem)
    cData["DNA_fourier"]["cost"].append(avgCost)

    print(f"\n\n- Running DNA Sampling using SPDE approach")

    dnaSPDERF = RandomField(DNASPDEEngine2d(var, ell, nu, nDof, 1.))
    dnaSPDECov = CovarianceAccumulator(nDof)

    avgMem = 0.
    avgCost = 0.

    for n in range(nSamp):

        print_sampling_progress(n, nSamp)

        if n < nAvg:

            tracemalloc.start()
            startTime = time.perf_counter()

            realisation = dnaSPDERF.generate(sampleSize)[0]

            endTime = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()

            # convert to MB
            peak /= 1e6

            avgMem += (peak - avgMem) / (n + 1)
            avgCost += (endTime - startTime - avgCost) / (n + 1)

            tracemalloc.stop()

        else:
            realisation = dnaSPDERF.generate(sampleSize)[0]

        diagSlice = util.extract_diagonal_slice(realisation)
        dnaSPDECov.update(diagSlice)

    print(f"\naverage peak memory usage: {avgMem} MB")
    print(f"Average time per realisation: {avgCost}")

    memData["DNA_spde"]["memory"].append(avgMem)
    cData["DNA_spde"]["cost"].append(avgCost)

    print(f"\n\n- Sampling using SPDE approach with oversampling")

    spdeCovList = [CovarianceAccumulator(nDof) for _ in oversampling]

    for alphaIdx, alpha in enumerate(oversampling):

        print(f"\noversampling with alpha = {alpha}")

        nOsDof = int(np.ceil(alpha * nDof))
        if nOsDof % 2 == 1:
            nOsDof += 1
        osWidth = (nOsDof - nDof) // 2

        print(f"oversampling width: {osWidth} degrees of freedom\n")

        spdeRF = RandomField(
            SPDEEngine2d(var, ell, nu, nOsDof, alpha, useDirBC=[False, False])
        )

        spdeCov = spdeCovList[alphaIdx]

        avgMem = 0.
        avgCost = 0.

        for n in range(nSamp):

            print_sampling_progress(n, nSamp)

            if np.abs(osFixedSPDE - alpha) < 1e-8 and n < nAvg:

                tracemalloc.start()
                startTime = time.perf_counter()

                realisation = spdeRF.generate(sampleSize)[0]

                endTime = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()

                # convert to MB
                peak /= 1e6

                avgMem += (peak - avgMem) / (n + 1)
                avgCost += (endTime - startTime - avgCost) / (n + 1)

                tracemalloc.stop()

            else:
                realisation = spdeRF.generate(sampleSize)[0]

            diagSlice = util.extract_diagonal_slice(realisation)

            if osWidth > 0:
                diagSlice = diagSlice[osWidth:-osWidth]

            spdeCov.update(diagSlice)

#         if nDof == nFixedSPDE:
# 
#             print(f"\naverage peak memory usage: {avgMem} MB")
#             print(f"Average time per realisation: {avgCost}")
# 
#             memData["SPDE_nFix"]["memory"].append(avgMem)
#             cData["SPDE_nFix"]["cost"].append(avgCost)
# 
        if np.abs(osFixedSPDE - alpha) < 1e-8:

            print(f"\naverage peak memory usage: {avgMem} MB")
            print(f"Average time per realisation: {avgCost}")

            memData["SPDE_osFix"]["memory"].append(avgMem)
            cData["SPDE_osFix"]["cost"].append(avgCost)

    diagonalGrid = util.extract_diagonal_from_mesh(dnaSPDERF.engine.mesh)
    trueCov = evaluate_isotropic_covariance_1d(cov_fcn, diagonalGrid)

    # l2ErrorDNAFourier = l2_error(dnaFourierCov.covariance  
    # l2ErrorDNASPDE = 

    maxErrorDNAFourier = np.max(np.abs(trueCov - dnaFourierCov.covariance))
    maxErrorDNASPDE = np.max(np.abs(trueCov - dnaSPDECov.covariance))

    osData["DNA_fourier"].append(maxErrorDNAFourier)
    osData["DNA_spde"].append(maxErrorDNASPDE)

    memData["DNA_fourier"]["error"].append(maxErrorDNAFourier)
    memData["DNA_spde"]["error"].append(maxErrorDNASPDE)

    cData["DNA_fourier"]["error"].append(maxErrorDNAFourier)
    cData["DNA_spde"]["error"].append(maxErrorDNASPDE)

    for i, spdeCov in enumerate(spdeCovList):

        maxErrorSPDE = np.max(np.abs(trueCov - spdeCov.covariance))

        osData["SPDE"][i].append(np.max(np.abs(trueCov - spdeCov.covariance)))

#         if nDof == nFixedSPDE:

#             memData["SPDE_nFix"]["error"].append(maxErrorSPDE)
#             cData["SPDE_nFix"]["error"].append(maxErrorSPDE)

        if np.abs(osFixedSPDE - oversampling[i]) < 1e-8:

            memData["SPDE_osFix"]["error"].append(maxErrorSPDE)
            cData["SPDE_osFix"]["error"].append(maxErrorSPDE)


experimentConfig = [
    ("oversampling", "os"),
    ("memory", "mem"),
    ("cost", "cost")
]

for dataDir, prefix in experimentConfig:

    dataString = create_data_string(DIM, var, ell, nu, nSamp, prefix)

    paramDir = dataString
    outDir = os.path.join("data", dataDir, paramDir)
    os.makedirs(outDir, exist_ok=True)

    filename = os.path.join(outDir, dataString + f"_{filenameID}.csv")

    with open(filename, mode='w', newline='') as file:

        writer = csv.writer(file)

        if prefix == "os":

            writer.writerow(["mesh_widths"] + [1. / n for n in dofPerDim])
            writer.writerow(["DNA_fourier"] + osData["DNA_fourier"])
            writer.writerow(["DNA_spde"] + osData["DNA_spde"])

            for i, alpha in enumerate(oversampling):
                writer.writerow(
                    [f"SPDE_alpha{int(100*alpha)}"] +
                    osData["SPDE"][i])

        if prefix == "mem":

            writer.writerow(
                ["DNA_fourier_memory"] +
                memData["DNA_fourier"]["memory"])
            writer.writerow(
                ["DNA_fourier_error"] +
                memData["DNA_fourier"]["error"])

            writer.writerow(
                ["DNA_spde_memory"] +
                memData["DNA_spde"]["memory"])
            writer.writerow(["DNA_spde_error"] + memData["DNA_spde"]["error"])

#             writer.writerow(["SPDE_nFix_memory"] + memData["SPDE_nFix"]["memory"])
#             writer.writerow(["SPDE_nFix_error"] + memData["SPDE_nFix"]["error"])

            writer.writerow(["SPDE_osFix_memory"] + memData["SPDE_osFix"]["memory"])
            writer.writerow(["SPDE_osFix_error"] + memData["SPDE_osFix"]["error"])

        if prefix == "cost":

            writer.writerow(
                ["DNA_fourier_cost"] +
                cData["DNA_fourier"]["cost"])
            writer.writerow(
                ["DNA_fourier_error"] +
                cData["DNA_fourier"]["error"])

            writer.writerow(["DNA_spde_cost"] + cData["DNA_spde"]["cost"])
            writer.writerow(["DNA_spde_error"] + cData["DNA_spde"]["error"])

#             writer.writerow(["SPDE_nFix_cost"] + cData["SPDE_nFix"]["cost"])
#             writer.writerow(["SPDE_nFix_error"] + cData["SPDE_nFix"]["error"])

            writer.writerow(["SPDE_osFix_cost"] + cData["SPDE_osFix"]["cost"])
            writer.writerow(["SPDE_osFix_error"] + cData["SPDE_osFix"]["error"])
