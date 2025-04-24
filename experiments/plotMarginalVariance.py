import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext
from experiments.filename import create_data_string

# =============================================================================
# Parameters for Filename Generation
# =============================================================================

DIM = 2
var = 0.1
ell = 0.15
nu = 1.
nSampBatch = int(5e4)
nBatch = 5

# baseDir = 'data'
baseDir = os.path.join('experiments', 'publicationData')
fileStr = create_data_string(DIM, var, ell, nu, nSampBatch,
                             "mv_ACCUMULATED") \
    + f"_{nBatch}batches.csv"
fileName = os.path.join(baseDir, fileStr)

# =============================================================================
# Plot Appearance Settings (variables)
# =============================================================================

lineWidth = 1.5
markerSize = 6
fontSizeLabel = 10
fontSizeTicks = 8
tickLabelSize = 6
fontSizeLegend = 6
legendMarkerSize = 6

figWidth = 5.0  # inches
figHeight = 5.0  # inches

# =============================================================================
# Read Data from CSV
# =============================================================================

methods = []

pos = []
margVar = {}

with open(fileName, mode='r') as file:

    reader = csv.reader(file)
    rows = list(reader)

    print(rows[0][0])

    assert rows[0][0] == "position"
    pos = [float(x) for x in rows[0][1:]]

    for row in rows[1:]:

        method = row[0]

        print(method)

        if method not in methods:
            methods.append(method)

        if method not in margVar:
            margVar[method] = [float(x) for x in row[1:]]


# =============================================================================
# Define Plot Styles
# =============================================================================

dnaMethods = ["DNA_fourier", "DNA_spde"]
spdeMethods = [m for m in methods if "SPDE_alpha" in m]

dnaColors = {'DNA_fourier': 'tab:green', 'DNA_spde': 'tab:red'}
spdeColors = ['tab:orange', 'tab:blue', 'tab:purple']

linestyles = ['-', '--', '-.', ':']

markers = ['o', 's', 'D', '^', 'p']


# =============================================================================
fig, ax = plt.subplots(figsize=(figWidth, figHeight))

# =============================================================================
# Plot Data
# =============================================================================

# Plot SPDE methods
for i, method in enumerate(spdeMethods):

    linestyle = linestyles[0]
    color = spdeColors[i]
    linewidth = 1.5

    ax.plot(
        pos,
        margVar[method],
        linewidth=linewidth,
        linestyle=linestyle,
        label=method.replace("_", " "),
        color=color,
        alpha=0.9
    )

for i, method in enumerate(dnaMethods):

    linestyle = linestyles[1]
    color = dnaColors[method]
    linewidth = 2.5

    ax.plot(
        pos,
        margVar[method],
        linewidth=linewidth,
        linestyle=linestyle,
        label=method.replace("_", " "),
        color=color,
        alpha=0.9
    )

# =============================================================================
# Customize Axes
# =============================================================================
ax.set_xlabel('position', fontsize=fontSizeLabel)
ax.set_ylabel('Marginal Variance Estimate', fontsize=fontSizeLabel)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Set tick label sizes for both major and minor ticks on both axes
ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)

# =============================================================================
# Legend Setup
# =============================================================================
# Compute marker scale for legend relative to markerSize using
# legendMarkerSize variable
markerScaleValue = legendMarkerSize / markerSize

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    fontsize=fontSizeLegend,
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    frameon=True,
    framealpha=0.9,
    markerscale=markerScaleValue
)
for text in legend.get_texts():
    text.set_alpha(1)
for line in legend.get_lines():
    line.set_alpha(1)

# =============================================================================
# Final Layout Adjustments and Save Figure
# =============================================================================
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve space for the legend
ax.set_aspect('auto', adjustable='box')

plt.savefig(
    f"./spde_comparison_margVar.pdf",
    format='pdf',
    dpi=300,
    bbox_inches='tight')
plt.show()
