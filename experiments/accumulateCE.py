import os
import csv
import numpy as np
from glob import glob
from filename import create_data_string

nSampBatch = int(5e3)
errorType = "maxError"

baseDir = os.path.join('experiments', 'testData', 'circulantEmbedding')
dataConfig = ["memory", "cost", "error"]

for subDir in dataConfig:
    # Determine input directory
    if subDir == "error":
        inDir = os.path.join(baseDir, subDir, errorType)
    else:
        inDir = os.path.join(baseDir, subDir)

    filePattern = os.path.join(inDir, f"run_{int(nSampBatch // 1000)}k_*.csv")
    csvFiles = glob(filePattern)
    nBatch = len(csvFiles)

    print(f"averaging {subDir} data over {nBatch} files")

    if not csvFiles:
        raise RuntimeError(
            f"No files found matching the pattern:\n{filePattern}")

    outFilename = os.path.join(
        baseDir, f"{subDir}_averaged_{nBatch}batches.csv")

    methods = ["dna", "ce", "aCE"]

    # Initialize containers
    if subDir == "error":
        xData = {"dna": {}, "aCE": {}}
        yData = {"dna": {}, "aCE": {}}

    else:
        xData = {"meshWidths": []}
        yData = {method: {} for method in methods}

    # Load data
    for filename in csvFiles:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

            if subDir == "error":
                for row in rows:
                    method, modelCov, variable = row[0].split('_')

                    if variable == "cost":
                        if method in xData:
                            xData[method].setdefault(modelCov, []).append(
                                [float(x) for x in row[1:]])
                        else:
                            raise RuntimeError(
                                f"invalid method for cost: {method}")
                    elif variable == "maxError":
                        if method in yData:
                            yData[method].setdefault(modelCov, []).append(
                                [float(x) for x in row[1:]])
                        else:
                            raise RuntimeError(
                                f"invalid method for maxError: {method}")
                    else:
                        raise RuntimeError(f"unknown variable: {variable}")

            else:
                xData["meshWidths"] = [float(x) for x in rows[0][1:]]
                for row in rows[1:]:
                    method, modelCov = row[0].split('_')
                    if method not in yData:
                        raise RuntimeError(
                            f"invalid method {method} for {subDir} data")
                    yData[method].setdefault(modelCov, []).append(
                        [float(x) for x in row[1:]])

    # Averaging
    if subDir == "error":

        xAveraged = {
            method: {
                modelCov: np.mean(xData[method][modelCov], axis=0).tolist()
                for modelCov in xData[method]
            } for method in xData
        }
    else:
        xAveraged = xData["meshWidths"]

    yAveraged = {
        method: {
            modelCov: np.mean(yData[method][modelCov], axis=0).tolist()
            for modelCov in yData[method]
        } for method in yData
    }

    # Confidence interval (only needed for error)
    CIFactor = 1.96
    if subDir == "error":
        errorBars = {
            method: {
                modelCov: (
                    CIFactor *
                    np.std(
                        yData[method][modelCov],
                        axis=0)).tolist()
                for modelCov in yData[method]
            } for method in yData
        }

    # Write output
    with open(outFilename, 'w', newline='') as f:
        writer = csv.writer(f)

        if subDir == "error":
            for method in yAveraged:
                for modelCov in yAveraged[method]:
                    writer.writerow(
                        [f"{method}_{modelCov}_cost"] +
                        xAveraged[method][modelCov])
                    writer.writerow(
                        [f"{method}_{modelCov}_error"] +
                        yAveraged[method][modelCov])
                    writer.writerow(
                        [f"{method}_{modelCov}_errorBar"] +
                        errorBars[method][modelCov])
        else:
            writer.writerow(["meshWidth"] + xAveraged)
            for method in yAveraged:
                for modelCov in yAveraged[method]:
                    writer.writerow(
                        [f"{method}_{modelCov}"] +
                        yAveraged[method][modelCov])
