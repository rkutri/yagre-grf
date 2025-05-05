import os
import csv
import re
from collections import defaultdict
from glob import glob


def read_averaged_data(baseDir, nBatch):

    dataConfig = ["cost", "memory"]

    averagedData = {}

    for q in dataConfig:

        inFilename = f"{q}_averaged_{nBatch}batches.csv"
        inFile = os.path.join(baseDir, inFilename)

        print(f"Reading {inFile}")

        with open(inFile, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            problemSize = [float(x) for x in rows[0][1:]]
            yData = defaultdict(dict)

            for row in rows[1:]:
                key, *values = row
                method = key
                data = [float(x) for x in values]
                yData[method] = data

            averagedData[q] = {
                "problemSize": problemSize,
                "yData": dict(yData)
            }

    return averagedData
