import os
import csv

import numpy as np

from glob import glob

from filename import create_data_string


# Parameters
DIM = 2
ell = 0.05
nu = 1.0
nSampBatch = int(1e5)

filenamePrefix = create_data_string(DIM, ell, nu, nSampBatch, "OS")

baseDir = 'data'
subDir = 'oversampling'

inDir = os.path.join(baseDir, subDir)
inDir = os.path.join(inDir, filenamePrefix)

filePattern = os.path.join(inDir, filenamePrefix + "_*.csv")

csvFiles = glob(filePattern)

nBatch = len(csvFiles)

outFilename = create_data_string(
    DIM,
    ell,
    nu,
    nSampBatch,
    "OS_ACCUMULATED") + f"_{nBatch}batches.csv"

if not csvFiles:
    raise RuntimeError("No files found matching the pattern:\n" + filePattern)

methods = []
errorData = {}

dofValues = None

for filename in csvFiles:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        if dofValues is None:
            # Extract DoF values from the first row
            dofValues = [float(value) for value in rows[0][1:]]
            # Convert DoF values to mesh widths
            meshWidths = [1.0 / nDof for nDof in dofValues]

        for row in rows[1:]:  # Skip header row
            method = row[0]
            errors = [float(value) for value in row[1:]]

            if method not in errorData:
                errorData[method] = []
            errorData[method].append(errors)

            if method not in methods:
                methods.append(method)

numFiles = len(csvFiles)
averagedData = {
    method: np.mean(
        errorData[method],
        axis=0).tolist() for method in methods}

ciFactor = 1.96
errorBars = {
    method: (ciFactor * np.std(errorData[method], axis=0)).tolist() for method in methods}

# Write to output CSV
with open(os.path.join(baseDir, outFilename), mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["Method"] + [str(h) for h in meshWidths])

    for method in methods:
        writer.writerow([method] + averagedData[method])

        writer.writerow([f"{method}_std"] + errorBars[method])
