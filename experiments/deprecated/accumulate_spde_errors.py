import os
import csv
import numpy as np
from glob import glob

# Parameters
ell = 0.25
nu = 1.0
nSamp = int(5e4)

outputFilename = f"ACCUMULATED_SPDE_ERRORS_ell{int(100*ell)}_nu{int(nu)}_{nSamp // 1000}k_with_error.csv"

baseInDir = 'data'
subInDir = 'spde'
inDir = os.path.join(baseInDir, subInDir)

filenamePrefix = f"spde_data_ell{int(100 * ell)}_nu{int(nu)}_{nSamp // 1000}k_"
filePattern = os.path.join(inDir, filenamePrefix + "*.csv")

csvFiles = glob(filePattern)

if not csvFiles:
    raise RuntimeError("No files found matching the pattern.")

meshWidths = []
dataAccumulator = []

# Initialize lists to accumulate all batch data for each row (for std dev)
allBatchData = []

for filename in csvFiles:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        if not meshWidths:
            header = rows[0]
            meshWidths = [float(row[0]) for row in rows[1:]]
            dataAccumulator = [[float(value)
                                for value in row[1:]] for row in rows[1:]]
            allBatchData = [[[] for _ in row[1:]] for row in rows[1:]]
        else:
            for i, row in enumerate(rows[1:]):
                values = [float(value) for value in row[1:]]
                dataAccumulator[i] = [
                    sum(x) for x in zip(
                        dataAccumulator[i], values)]
                for j, value in enumerate(values):
                    allBatchData[i][j].append(value)

numFiles = len(csvFiles)
averagedData = [[val / numFiles for val in row] for row in dataAccumulator]

# Calculate standard deviation for error bars
errorBars = [[np.std(allBatchData[i][j]) for j in range(len(row))]
             for i, row in enumerate(dataAccumulator)]

# Write to output CSV
with open(outputFilename, mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow([f'Averaged over {numFiles} files with nSamp = {nSamp}'])
    writer.writerow(header + [f"{col}_std" for col in header[1:]])

    for i, (row, error) in enumerate(zip(averagedData, errorBars)):
        writer.writerow([meshWidths[i]] + row + error)

print(f"Averaged results with error bars saved to {outputFilename}")
