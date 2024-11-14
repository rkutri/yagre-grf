import os
import csv


def generateFilenameIDs(nIds):
    return [f"{i:02d}" for i in range(1, nIds + 1)]


nDof = 2500
nSamp = 100000
alpha = 2

nFilenameIDs = 4

filenameIDs = generateFilenameIDs(nFilenameIDs)
print(f"average data from IDs: {filenameIDs}")

dataDir = 'data'
dataSubDir = 'cov'

inDir = os.path.join(dataDir, dataSubDir)

csvFiles = []

for filenameID in filenameIDs:

    filename = os.path.join(
        inDir,
        f"cov_error_data_n{nDof}_{nSamp // 1000}k_alpha{int(alpha)}_{filenameID}.csv")
    csvFiles.append(filename)

# Initialize storage for data
data = {}
nFiles = len(csvFiles)


def parseCsv(file):

    print(f"Processing file: {file}")

    with open(file, mode='r') as f:

        reader = csv.reader(f)
        headers = next(reader)

        if 'header' not in data:
            data['header'] = headers

        for row in reader:

            key = row[0]  # nu/kernel value
            values = list(map(float, row[1:]))

            if key not in data:
                data[key] = values

            else:
                data[key] = [x + y for x, y in zip(data[key], values)]


for file in csvFiles:
    parseCsv(file)

print("Calculating averages")
for key in data:
    if key != 'header':

        data[key] = [x / nFiles for x in data[key]]

outputFile = f"ACCUMULATED_COVARIANCE_ERRORS_n{nDof}_{nSamp // 1000}k_alpha{int(alpha)}.csv"

print(f"Writing output to {outputFile}")

with open(outputFile, mode='w', newline='') as f:

    writer = csv.writer(f)
    writer.writerow(data['header'])  # Write header

    for key, values in data.items():

        if key != 'header':
            writer.writerow([key] + values)

print("Processing complete!")
