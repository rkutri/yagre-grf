import os
import csv

import numpy as np

from glob import glob

from filename import create_data_string


# Parameters
DIM = 2
var = 0.1
ell = 0.2
nu = 3.
nSampBatch = int(5e1)

errorTypes = ["maxError", "froError"]

for error in errorTypes:

    baseDir = 'data'
    dataConfig = [("oversampling", "os"),
                  ("memory", "mem"),
                  ("cost", "cost"),
                  ("marginalVariance", "mv")]

    for subDir, prefix in dataConfig:

        filenamePrefix = create_data_string(
            DIM, var, ell, nu, nSampBatch, prefix)

        inDir = os.path.join(baseDir, subDir, error)
        inDir = os.path.join(inDir, filenamePrefix)

        filePattern = os.path.join(inDir, filenamePrefix + "_*.csv")

        csvFiles = glob(filePattern)

        nBatch = len(csvFiles)

        outFilename = create_data_string(DIM, var, ell, nu, nSampBatch,
                                         prefix + "_ACCUMULATED") + f"_{nBatch}batches_" + error + ".csv"

        if not csvFiles:
            raise RuntimeError(
                "No files found matching the pattern:\n" +
                filePattern)

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

                    # first row contains the mesh widhts
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
                            [float(var) for var in row[1:]])

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

                            errors = [float(value) for value in row[1:]]
                            errorSamples[method].append(errors)

                        else:
                            variableData[method] = [float(x) for x in row[1:]]

        if prefix == "mv":
            averagedMargVar = {
                method: np.mean(
                    margVarSamples[method],
                    axis=0).tolist() for method in methods}
        else:
            averagedErrors = {method: np.mean(errorSamples[method], axis=0).tolist()
                              for method in methods}

            # confidence interval ~ 95%
            ciFactor = 1.96
            errorBars = {method: (ciFactor * np.std(errorSamples[method],
                                                    axis=0)).tolist() for method in methods}

        # Write to output CSV
        with open(os.path.join(baseDir, outFilename), mode='w', newline='') as file:

            writer = csv.writer(file)

            if prefix == "os":

                writer.writerow(["meshWidths"] + variableData["meshWidths"])

                for method in methods:

                    writer.writerow([method] + averagedErrors[method])

                    barRow = method + "_bars"
                    writer.writerow([barRow] + errorBars[method])

            elif prefix == "mv":

                writer.writerow(["position"] + variableData["position"])

                for method in methods:
                    writer.writerow([method] + averagedMargVar[method])

            else:

                for method in methods:

                    variableRow = method + "_" + subDir
                    writer.writerow([variableRow] + variableData[method])

                    errorRow = method + "_" + error
                    writer.writerow([errorRow] + averagedErrors[method])

                    barRow = errorRow.replace(error, "bars")
                    writer.writerow([barRow] + errorBars[method])
