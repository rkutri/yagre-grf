import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext
from experiments.filename import create_data_string
from collections import OrderedDict

# =============================================================================
# Parameters for Filename Generation
# =============================================================================

DIM = 2
var = 0.1
ellData = [0.05, 0.1, 0.2]
nu = 1.0
nSampBatch = int(5e4)
nBatch = 5

variables = [("oversampling", "os"), ("memory", "mem"), ("cost", "cost")]

errorTypes = ["maxError", "froError"]
errorType = errorTypes[0]

legendLabels = [
    r'DNA - Fourier',
    r'DNA - Lagrange',
    r'SPDE - vanilla',
    r'SPDE - heuristic'
]

yLabel = r'Monte-Carlo estimate of maximal covariance error'
xLabels = {
    "oversampling": r'mesh width h',
    "memory": r'peak memory (MB)',
    "cost": r'runtime (s)'
}

# Create figure with 3x3 grid of subplots (3 rows, 3 columns)
fig, axs = plt.subplots(len(ellData), 3, figsize=(9, 7))

for i, ell in enumerate(ellData):

    for j, (variable, prefix) in enumerate(variables):

        print(f"generating plot for {variable}, ell={ell}")

        #    baseDir = 'data'
        baseDir = os.path.join('experiments', 'publicationData')
        fileStr = create_data_string(DIM, var, ell, nu, nSampBatch,
                                     prefix + "_ACCUMULATED") \
            + f"_{nBatch}batches_" + errorType + ".csv"
        fileName = os.path.join(baseDir, fileStr)

        # =============================================================================
        # Plot Appearance Settings (variables)
        # =============================================================================

        lineWidth = 1.4
        markerSize = 4
        fontSizeXLabel = 10
        fontSizeYLabel = 12
        fontSizeTicks = 10
        tickLabelSize = 6
        fontSizeLegend = 11
        legendMarkerSize = 6
        lineAlpha = 0.9

        # =============================================================================
        # Read Data from CSV
        # =============================================================================

        meshWidths = []
        methods = []
        xData = {}
        errors = {}
        errorBars = {}

        with open(fileName, mode='r') as file:

            reader = csv.reader(file)
            rows = list(reader)

            for ii, row in enumerate(rows):

                if variable == "oversampling":

                    isErrorBar = False

                    if ii == 0:
                        meshWidths = [float(x) for x in row[1:]]

                    label = row[0]

                    if label.count('_') > 1:
                        isErrorBar = True
                        method = label.rsplit('_', 1)[0]

                    else:
                        isErrorBar = False
                        method = label

                else:

                    label = row[0].rsplit('_', 1)

                    method = label[0]
                    variableName = label[1]

                    isErrorBar = (variableName == "bars")

                print(f"\nreading data for {method}")

                if method not in methods:
                    methods.append(method)

                if isErrorBar:
                    if method not in errorBars:
                        errorBars[method] = [float(x) for x in row[1:]]
                else:

                    if variable == "oversampling":
                        if method not in errors:
                            errors[method] = [float(x) for x in row[1:]]
                    else:
                        if variableName == errorType:
                            if method not in errors:
                                errors[method] = [float(x) for x in row[1:]]

                        elif variableName == variable:
                            if method not in xData:
                                xData[method] = [float(x) for x in row[1:]]

                        else:
                            raise RuntimeError(
                                f"unknown variable name: {variableName}")

        # =============================================================================
        # Define Plot Styles
        # =============================================================================

        colors = {
            'SPDE': ['tab:purple', 'tab:blue', 'tab:orange'],
            'DNA_fourier': 'tab:green',
            'DNA_spde': 'tab:red'
        }

        linestyles = ['-', '--', '-.', ':']

        markers = ['o', 's', 'D', '^', 'p']

        # =============================================================================
        # Plot Data
        # =============================================================================

        ax = axs[i, j]  # Select subplot axis

        # Plot SPDE methods
        for iii, method in enumerate(methods):

            if method == "meshWidths":
                continue

            linestyle = linestyles[0]
            marker = markers[iii -
                             1] if variable == "oversampling" else markers[iii]

            if method == "SPDE_alpha100":
                color = colors['SPDE'][0]

            elif "SPDE" in method:
                color = colors['SPDE'][1]
            else:
                color = colors[method]

            if variable == "oversampling":
                xArray = meshWidths
            else:
                xArray = xData[method]

            yArray = errors[method]

            ax.errorbar(
                xArray,
                yArray,
                yerr=errorBars[method],
                fmt=marker,
                linestyle=linestyle,
                label=method.replace("_", " "),
                markersize=markerSize,
                markeredgewidth=1.5,
                elinewidth=lineWidth,
                capsize=5,
                color=color,
                alpha=lineAlpha
            )

        # =============================================================================
        # Draw Rate Indication Triangle
        # =============================================================================

        if variable == "memory":

            baseScale = 1.5
            rate = -1.

            x0 = 0.03
            y0 = 0.004

            labelOffset = 0.01

        elif variable == "cost":

            baseScale = 1.4
            rate = -1.

            x0 = 0.005
            y0 = 0.007

            labelOffset = 0.001

        elif variable == "oversampling":

            baseScale = 2.0
            rate = -2.

            x0 = 0.02
            y0 = 0.01

            labelOffset = 0.02

        else:
            raise RuntimeError("")

        x1 = x0 * baseScale
        y1 = y0 * np.power(baseScale, rate)

        ax.plot([x0, x1], [y0, y1], color='black', lw=lineWidth)
        ax.plot([x0, x1], [y1, y1], color='black', lw=lineWidth)
        ax.plot([x0, x0], [y0, y1], color='black', lw=lineWidth)

        ax.text(x0 - labelOffset, 1.1 * y1, f"{rate}", color='k',
                horizontalalignment='center', verticalalignment='bottom', fontsize=0.5 * fontSizeTicks)

        # =============================================================================
        # Customize Axes
        # =============================================================================

        errorLabel = "maximal error" if errorType == "maxError" else "relative Frobenius error"

        if i == 1 and j == 0:
            ax.set_ylabel(yLabel, fontsize=fontSizeYLabel)

        if i == 2:
            ax.set_xlabel(xLabels[variable], fontsize=fontSizeXLabel)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

        # Set tick label sizes for both major and minor ticks on both axes
        ax.tick_params(axis='both', which='both', labelsize=tickLabelSize)

# =============================================================================
# Legend Setup
# =============================================================================

handles, _ = axs[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    legendLabels,
    fontsize=fontSizeLegend,
    loc='center right',
    bbox_to_anchor=(1.0, 0.5),
    frameon=True,
    framealpha=0.9,
    markerscale=legendMarkerSize / markerSize
)

# =============================================================================
# Final Layout Adjustments and Save Figure
# =============================================================================

fig.subplots_adjust(
    left=0.07,
    right=0.77,
    top=0.97,
    bottom=0.1,
    wspace=0.2,
    hspace=0.18)
fig.savefig(
    './spde_comparison.pdf',
    format='pdf',
    dpi=300)
# plt.show()
