import os
import csv
import numpy as np
from glob import glob
from filename import create_data_string

nSampBatch = int(1e4)
errorType = "maxError"

baseDir = os.path.join('experiments', 'publicationData', 'circulantEmbedding')
dataConfig = ["memory", "cost"]

for subDir in dataConfig:

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

    xData = {"problemSize": []}
    yData = {}

    # Load data
    for filename in csvFiles:
        with open(filename, 'r') as file:

            reader = csv.reader(file)
            rows = list(reader)

            xData["problemSize"] = [float(x) for x in rows[0][1:]]

            for row in rows[1:]:

                method = row[0]

                if method not in yData:
                    yData[method] = [[float(x) for x in row[1:]]]

                else:
                    yData[method].append([float(x) for x in row[1:]])

    # Averaging
    xAveraged = xData["problemSize"]
    yAveraged = {
        method: np.mean(yData[method], axis=0).tolist() for method in yData
    }

    # Write output
    with open(outFilename, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["problemSize"] + xAveraged)
        for method in yAveraged:
            writer.writerow([f"{method}"] + yAveraged[method])
