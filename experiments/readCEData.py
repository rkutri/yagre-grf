import os
import csv
import re
from collections import defaultdict
from glob import glob


def read_averaged_data(baseDir, nBatch):

    dataConfig = ["memory", "cost", "error"]

    averagedData = {}

    for subDir in dataConfig:

        inFilename = f"{subDir}_averaged_{nBatch}batches.csv"
        inFile = os.path.join(baseDir, inFilename)

        print(f"Reading {inFile}")

        with open(inFile, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if subDir == "error":
            xData = defaultdict(dict)
            yData = defaultdict(dict)
            errorBars = defaultdict(dict)

            for row in rows:
                key, *values = row
                method, modelCov, variable = key.split("_")
                data = [float(x) for x in values]

                if variable == "cost":
                    xData[method][modelCov] = data
                elif variable == "error":
                    yData[method][modelCov] = data
                elif variable == "errorBar":
                    errorBars[method][modelCov] = data
                else:
                    raise RuntimeError(f"Unknown variable: {variable}")

            averagedData[subDir] = {
                "xData": dict(xData),
                "yData": dict(yData),
                "errorBars": dict(errorBars)
            }

        else:
            meshWidths = [float(x) for x in rows[0][1:]]
            yData = defaultdict(dict)

            for row in rows[1:]:
                key, *values = row
                method, modelCov = key.split("_")
                data = [float(x) for x in values]
                yData[method][modelCov] = data

            averagedData[subDir] = {
                "meshWidths": meshWidths,
                "yData": dict(yData)
            }

    return averagedData
