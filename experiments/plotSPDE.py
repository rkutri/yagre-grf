import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext, NullFormatter
from matplotlib.font_manager import FontProperties
from experiments.filename import create_data_string
from collections import OrderedDict

# =============================================================================
# Parameters for Filename Generation
# =============================================================================

DIM = 2
var = 0.1
ellData = [0.05, 0.1, 0.2]
nu = 1.0
nSampBatch = int(2e4)
nBatch = 25

variables = [("oversampling", "os"), ("memory", "mem"), ("cost", "cost")]

errorType = "maxError"


yLabel = r'Monte-Carlo estimate of maximal covariance error'
xLabels = {
    "oversampling": r'problem size (dofs)',
    "memory": r'peak memory (MB)',
    "cost": r'runtime (s)'
}

# Create figure with 3x3 grid of subplots (3 rows, 3 columns)
fig, axs = plt.subplots(len(ellData), 3, figsize=(9, 6))

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

        lineWidth = 1.5
        markerSize = 3
        fontSizeXLabel = 13
        fontSizeYLabel = 14
        fontSizeTicks = 9
        tickLabelSize = 10
        fontSizeLegend = 14
        legendMarkerSize = 4
        lineAlpha = 0.9

        # =============================================================================
        # Read Data from CSV
        # =============================================================================

        problemSize = []
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
                        problemSize = [float(x) for x in row[1:]]

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

                    isErrorBar = (variableName == "bar")

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

            if method == "problemSize":
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
                xArray = problemSize
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
                capsize=3,
                capthick=0.5 * lineWidth,
                color=color,
                alpha=lineAlpha
            )

        # =============================================================================
        # Draw Rate Indication Triangle
        # =============================================================================

        if variable == "oversampling":

            if i == 0:
                x0 = 2000
                y0 = 0.02
                labelOffset = 2750

            elif i == 1:
                x0 = 1750
                y0 = 0.01
                labelOffset = 2250

            elif i == 2:
                x0 = 1250
                y0 = 0.005
                labelOffset = 1750

            baseScale = 2.0
            rate = -0.5

            x1 = x0 * baseScale
            y1 = y0 / np.power(baseScale, rate)

            ax.plot([x0, x1], [y1, y0], color='black', lw=lineWidth)
            ax.plot([x0, x1], [y1, y1], color='black', lw=lineWidth)
            ax.plot([x1, x1], [y0, y1], color='black', lw=lineWidth)

            ax.text(x1 + labelOffset, 0.95 * y0, f"1/2", color='k',
                    horizontalalignment='center', verticalalignment='bottom', fontsize=fontSizeTicks)

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

        # Hide minor tick labels
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

# =============================================================================
# Legend Setup
# =============================================================================

dnaLabels = [r'DST/DCT', r'Q1 FEM']
spdeLabels = [r'no oversampling', r'oversampling $2\ell$']

handles, _ = axs[0, 0].get_legend_handles_labels()

dnaHandles = handles[:2]
spdeHandles = handles[2:]

dnaLegend = plt.legend(
    handles=dnaHandles, labels=dnaLabels, title=r'DNA',
    title_fontproperties=FontProperties(weight='bold', size=1.15*fontSizeLegend),
    fontsize=fontSizeLegend, bbox_to_anchor=(2.125, 2.7), frameon=True,
    framealpha=0.9, markerscale=legendMarkerSize / markerSize)

plt.gca().add_artist(dnaLegend)

plt.legend(
    handles=spdeHandles, labels=spdeLabels, title=r'SPDE',
    title_fontproperties=FontProperties(weight='bold', size=1.15*fontSizeLegend),
    fontsize=fontSizeLegend, bbox_to_anchor=(2.6, 2.), frameon=True,
    framealpha=0.9, markerscale=legendMarkerSize / markerSize)
# =============================================================================
# Final Layout Adjustments and Save Figure
# =============================================================================

fig.subplots_adjust(
    left=0.08,
    right=0.72,
    top=0.98,
    bottom=0.08,
    wspace=0.29,
    hspace=0.18)
fig.savefig(
    './spde_comparison.pdf',
    format='pdf',
    dpi=300)
# plt.show()
