import os
import csv

import numpy as np

from glob import glob

from filename import create_data_string


# Parameters
DIM = 2
ell = 0.2
nu = 1.
nSampBatch = int(1e1)


baseDir = 'data'
dataConfig = [("oversampling", "os"),
              ("memory", "mem"),
              ("cost", "cost")]

for subDir, prefix in dataConfig:

    filenamePrefix = create_data_string(DIM, ell, nu, nSampBatch, prefix)

    inDir = os.path.join(baseDir, subDir)
    inDir = os.path.join(inDir, filenamePrefix)

    filePattern = os.path.join(inDir, filenamePrefix + "_*.csv")

    csvFiles = glob(filePattern)

    nBatch = len(csvFiles)

    outFilename = create_data_string(DIM, ell, nu, nSampBatch,
            prefix + "_ACCUMULATED") + f"_{nBatch}batches.csv"

    if not csvFiles:
        raise RuntimeError("No files found matching the pattern:\n" + filePattern)

    methods = []
    errorData = {}

    indepVar = None

    for filename in csvFiles:
        with open(filename, 'r') as file:

            reader = csv.reader(file)
            rows = list(reader)

            # first row contains the independent variable
            if indepVar is None:
                method = rows[0][0]
                indepVar = [float(value) for value in rows[0][1:]]

            for row in rows[1:]:  
                method = row[0]
                errors = [float(value) for value in row[1:]]

                if method not in errorData:
                    errorData[method] = []

                errorData[method].append(errors)

                if method not in methods:
                    methods.append(method)

    numFiles = len(csvFiles)
    averagedData = {method: np.mean(errorData[method], axis=0).tolist()
                        for method in methods}

    # confidence interval ~ 95%
    ciFactor = 1.96
    errorBars = {method: (ciFactor * np.std(errorData[method],
                          axis=0)).tolist() for method in methods}

    # Write to output CSV
    with open(os.path.join(baseDir, outFilename), mode='w', newline='') as file:

        writer = csv.writer(file)

        writer.writerow(["Method"] + [str(x) for x in indepVar])

        for method in methods:
            writer.writerow([method] + averagedData[method])

            writer.writerow([f"{method}_std"] + errorBars[method])
