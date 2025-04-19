import os
import csv
import numpy as np
from glob import glob
from filename import create_data_string

# Parameters
DIM = 2
var = 0.1
ell = 0.05
nu = 1.
nSampBatch = int(5e4)

errorTypes = ["maxError", "froError"]

#baseDir = 'data'
baseDir = os.path.join('experiments', 'publicationData')
dataConfig = [("oversampling", "os"),
              ("memory", "mem"),
              ("cost", "cost"),
              ("marginalVariance", "mv")]

for subDir, prefix in dataConfig:

    if prefix == "mv":
        relevantErrors = [None]  # only process once
    else:
        relevantErrors = errorTypes

    for error in relevantErrors:

        filenamePrefix = create_data_string(
            DIM, var, ell, nu, nSampBatch, prefix)

        if prefix == "mv":
            inDir = os.path.join(baseDir, subDir, filenamePrefix)
        else:
            inDir = os.path.join(baseDir, subDir, error, filenamePrefix)

        filePattern = os.path.join(inDir, filenamePrefix + "_*.csv")
        csvFiles = glob(filePattern)
        nBatch = len(csvFiles)

        print(f"{subDir} has {nBatch} files")

        if not csvFiles:
            raise RuntimeError(
                "No files found matching the pattern:\n" + filePattern)

        if prefix == "mv":
            outFilename = create_data_string(
                DIM, var, ell, nu, nSampBatch,
                prefix + "_ACCUMULATED") + f"_{nBatch}batches.csv"
        else:
            outFilename = create_data_string(
                DIM, var, ell, nu, nSampBatch,
                prefix + "_ACCUMULATED") + f"_{nBatch}batches_" + error + ".csv"

        methods = []
        variableData = {}
        margVarSamples = {}
        errorSamples = {}
        errorBars = {}

        for filename in csvFiles:
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)

                if prefix == "os":
                    # first row contains mesh widths
                    meshWidths = [float(h) for h in rows[0][1:]]
                    variableData["meshWidths"] = meshWidths

                    for row in rows[1:]:
                        method = row[0]
                        if method not in methods:
                            methods.append(method)
                        if method not in errorSamples:
                            errorSamples[method] = []
                        errorSamples[method].append(
                            [float(x) for x in row[1:]])

                elif prefix == "mv":
                    pos = [float(x) for x in rows[0][1:]]
                    variableData["position"] = pos

                    for row in rows[1:]:
                        method = row[0]
                        if method not in methods:
                            methods.append(method)
                        if method not in margVarSamples:
                            margVarSamples[method] = []
                        margVarSamples[method].append(
                            [float(x) for x in row[1:]])

                else:
                    for row in rows:
                        label = row[0].rsplit('_', 1)
                        method = label[0]
                        quantity = label[1]

                        if method not in methods:
                            methods.append(method)

                        if quantity == error:
                            if method not in errorSamples:
                                errorSamples[method] = []
                            errorSamples[method].append(
                                [float(x) for x in row[1:]])
                        else:
                            variableData[method] = [float(x) for x in row[1:]]

        if prefix == "mv":
            averagedMargVar = {
                method: np.mean(margVarSamples[method], axis=0).tolist()
                for method in methods
            }
        else:
            averagedErrors = {
                method: np.mean(errorSamples[method], axis=0).tolist()
                for method in methods
            }

            ciFactor = 1.96  # 95% confidence interval
            errorBars = {
                method: (
                    ciFactor *
                    np.std(
                        errorSamples[method],
                        axis=0)).tolist()
                for method in methods
            }

        # Write to output CSV
        with open(os.path.join(baseDir, outFilename), mode='w', newline='') as file:
            writer = csv.writer(file)

            if prefix == "os":
                writer.writerow(["meshWidths"] + variableData["meshWidths"])
                for method in methods:
                    writer.writerow([method] + averagedErrors[method])
                    writer.writerow([method + "_bars"] + errorBars[method])

            elif prefix == "mv":
                writer.writerow(["position"] + variableData["position"])
                for method in methods:
                    writer.writerow([method] + averagedMargVar[method])

            else:
                for method in methods:
                    writer.writerow([method + "_" + subDir] +
                                    variableData[method])
                    writer.writerow([method + "_" + error] +
                                    averagedErrors[method])
                    writer.writerow([method + "_bars"] + errorBars[method])
