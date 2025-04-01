import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext
from experiments.filename import create_data_string

# =============================================================================
# Parameters for Filename Generation
# =============================================================================
ell = 0.2
nu = 1.0
nSampBatch = int(1e5)
nBatch = 4

baseDir = 'data'
fileStr = create_data_string(
    ell,
    nu,
    nSampBatch,
    "ACCUMULATED_ERRORS_OVERSAMPLING") + f"_{nBatch}batches.csv"
fileName = os.path.join(baseDir, fileStr)

# =============================================================================
# Plot Appearance Settings (variables)
# =============================================================================
lineWidth = 0.8         # Thinner lines (for error bars and triangle)
markerSize = 6          # Marker size for plotted data
fontSizeLabel = 12      # Axis labels font size
fontSizeTicks = 10      # Font size for manually set tick labels
tickLabelSize = 8       # Desired tick label font size
fontSizeLegend = 8      # Legend text font size
legendMarkerSize = 4    # Desired marker size in the legend

# Figure size (for 3 figures per row in an 11pt DIN A4 LaTeX publication)
figWidth = 5.0  # inches
figHeight = 5.0  # inches

# =============================================================================
# Read Data from CSV
# =============================================================================
meshWidths = []
methods = []
errors = {}
errorsStd = {}

with open(fileName, mode='r') as file:
    reader = csv.reader(file)
    rows = list(reader)

    # First row: mesh widths (skip first column header)
    meshWidths = [float(val) for val in rows[0][1:]]

    # Subsequent rows: alternate between error values and (optional) std errors
    i = 1
    while i < len(rows):
        method = rows[i][0]
        methods.append(method)
        errors[method] = [float(val) for val in rows[i][1:]]
        if i + 1 < len(rows) and "std" in rows[i + 1][0].lower():
            errorsStd[method] = [float(val) for val in rows[i + 1][1:]]
            i += 2
        else:
            errorsStd[method] = None
            i += 1

# =============================================================================
# Define Plot Styles
# =============================================================================
# Separate methods into SPDE methods and DNA methods:
spdeMethods = [m for m in methods if "spde_alpha" in m]
dnaMethods = ["DNA_fourier", "DNA_spde"]

linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 2, 1, 2))]
spdeColor = 'tab:blue'
dnaColors = {'DNA_fourier': 'tab:red', 'DNA_spde': 'tab:green'}

# =============================================================================
# Create Figure and Axes
# =============================================================================
fig, ax = plt.subplots(figsize=(figWidth, figHeight))

# =============================================================================
# Plot Data
# =============================================================================
# Plot SPDE methods (same color, different linestyles)
for i, method in enumerate(spdeMethods):
    linestyle = linestyles[i % len(linestyles)]
    ax.errorbar(
        meshWidths,
        errors[method],
        yerr=errorsStd[method],
        fmt='o',
        linestyle=linestyle,
        label=method.replace("_", " "),
        markersize=markerSize,
        markeredgewidth=1.5,
        elinewidth=lineWidth,
        capsize=5,
        color=spdeColor,
        alpha=0.9
    )

# Plot DNA methods (unique colors)
for method in dnaMethods:
    if method in errors:
        ax.errorbar(
            meshWidths,
            errors[method],
            yerr=errorsStd[method],
            fmt='s',
            linestyle='-',
            label=method.replace("_", " "),
            markersize=markerSize,
            markeredgewidth=1.5,
            elinewidth=lineWidth,
            capsize=5,
            color=dnaColors[method],
            alpha=0.9
        )

# =============================================================================
# Draw Convergence Rate Triangle
# =============================================================================
rate = 0.5
base = 0.025
height = rate * base
xTriangle = [0.05, 0.05 + base]
yTriangle = [0.02, 0.02 + height]

ax.plot([xTriangle[0], xTriangle[1]], [yTriangle[0],
                                       yTriangle[0]], color='black', lw=lineWidth)
ax.plot([xTriangle[1], xTriangle[1]], [yTriangle[0],
                                       yTriangle[1]], color='black', lw=lineWidth)
ax.plot([xTriangle[0], xTriangle[1]], [yTriangle[0],
                                       yTriangle[1]], color='black', lw=lineWidth)

ax.text(
    xTriangle[0] +
    0.5 *
    base,
    yTriangle[0] -
    0.006,
    '1',
    fontsize=fontSizeTicks,
    ha='center')
ax.text(
    xTriangle[1] +
    0.003,
    yTriangle[1] -
    0.6 *
    height,
    '1/2',
    fontsize=fontSizeTicks,
    va='center')

# =============================================================================
# Customize Axes
# =============================================================================
ax.set_xlabel('Mesh width $h$', fontsize=fontSizeLabel)
ax.set_ylabel('Error estimate', fontsize=fontSizeLabel)
ax.set_xscale('log')
ax.set_yscale('log')
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
    './spde_oversampling.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight')
plt.show()
