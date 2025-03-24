import os
import csv
import numpy as np

from spde import SPDESampler
from covariance_functions import matern_covariance_ptw
from utility import extract_diagonal_slice, evaluate_isotropic_covariance_1d
from spdeUtility import extract_cropped_diagonal_from_mesh


filenameID = "40"


ell = 0.25
nu = 1.

dofPerDim = [8, 16, 32, 64, 128]
oversampling = [1., 1.25, 1.5, 1.75]

kappa = np.sqrt(2. * nu) / ell
beta = 0.5 * (nu + 1.)

nSamp = int(5e4)

dataDir = 'data'
dataSubDir = 'spde'
outDir = os.path.join(dataDir, dataSubDir)

os.makedirs(outDir, exist_ok=True)

filename = os.path.join(
    outDir,
    f"spde_data_ell{int(100*ell)}_nu{int(nu)}_{nSamp // 1000}k_{filenameID}.csv")

maxErrorCRF = []
maxErrorSPDE = []

alphaIdx = 0

for alpha in oversampling:

    print(f"oversampling with alpha = {alpha}")

    dofFactor = 1. + 2. * (alpha - 1.)

    osDofPerDim = [int(np.rint(dofFactor * N)) for N in dofPerDim]

    maxErrorCRF.append([])
    maxErrorSPDE.append([])

    dofIdx = 0

    for nDof in osDofPerDim:

        print(f"\nsetting up solvers with {nDof} dofs per dimension")

        osWidth = (nDof - dofPerDim[dofIdx]) // 2
        print(f"oversampling width: {osWidth} degrees of freedom\n")

        spdeSampler_nn = SPDESampler(kappa, beta, nDof, alpha, [False, False])
        spdeSampler_nd = SPDESampler(kappa, beta, nDof, alpha, [False, True])
        spdeSampler_dn = SPDESampler(kappa, beta, nDof, alpha, [True, False])
        spdeSampler_dd = SPDESampler(kappa, beta, nDof, alpha, [True, True])

        spdeSliceSamples = []
        crfSliceSamples = []

        for n in range(nSamp):

            if n % (nSamp // 5) == 0:

                if (n == 0):
                    print("Start sampling")
                else:
                    print(f"{n} realisations computed")

            fullSPDESol_nn = spdeSampler_nn.generate_realisation()
            fullSPDESol_nd = spdeSampler_nd.generate_realisation()
            fullSPDESol_dn = spdeSampler_dn.generate_realisation()
            fullSPDESol_dd = spdeSampler_dd.generate_realisation()

            fullCRFSol = 0.5 * \
                (fullSPDESol_nn + fullSPDESol_nd + fullSPDESol_dn + fullSPDESol_dd)

            # extract the diagonal and crop the oversampling part
            if osWidth == 0:
                spdeDiagCropped = extract_diagonal_slice(fullSPDESol_nn)
                crfDiagCropped = extract_diagonal_slice(fullCRFSol)
            else:
                spdeDiagCropped = extract_diagonal_slice(fullSPDESol_nn)[
                    osWidth:-osWidth]
                crfDiagCropped = extract_diagonal_slice(fullCRFSol)[
                    osWidth:-osWidth]

            spdeSliceSamples.append(spdeDiagCropped)
            crfSliceSamples.append(crfDiagCropped)

        spdeCov = np.cov(spdeSliceSamples, rowvar=False)
        crfCov = np.cov(crfSliceSamples, rowvar=False)

        def cov_fcn(r): return matern_covariance_ptw(r, ell, nu)
        diagonalGrid = extract_cropped_diagonal_from_mesh(
            spdeSampler_nn.mesh, osWidth)
        trueCov = evaluate_isotropic_covariance_1d(cov_fcn, diagonalGrid)

        spdeError = np.max(np.abs(trueCov - spdeCov))
        crfError = np.max(np.abs(trueCov - crfCov))

        print(f"spde error: {spdeError}")
        print(f"crf error: {crfError}\n\n")

        maxErrorSPDE[alphaIdx].append(spdeError)
        maxErrorCRF[alphaIdx].append(crfError)

        dofIdx += 1

    alphaIdx += 1

# Writing to a CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    hdrRow = ['Mesh width']

    for alpha in oversampling:
        hdrRow.append(f'α = {alpha}, SPDE')
        hdrRow.append(f'α = {alpha}, CRF')

    writer.writerow(hdrRow)

    # Write the data row by row
    for i in range(len(dofPerDim)):
        writer.writerow([1. / dofPerDim[i],
                         maxErrorSPDE[0][i], maxErrorCRF[0][i],
                         maxErrorSPDE[1][i], maxErrorCRF[1][i],
                         maxErrorSPDE[2][i], maxErrorCRF[2][i],
                         maxErrorSPDE[3][i], maxErrorCRF[3][i],
                         ])
