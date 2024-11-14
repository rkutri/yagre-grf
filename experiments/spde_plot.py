import numpy as np
import matplotlib.pyplot as plt
import csv

nAlpha = 4

oversampling = [1., 1.25, 1.5, 1.75]

# Reading data from CSV file
mesh_widths = []
errors_spde = [[] for _ in range(nAlpha)]
errors_crf = [[] for _ in range(nAlpha)]
errors_spde_std = [[] for _ in range(nAlpha)]  # For error bars
errors_crf_std = [[] for _ in range(nAlpha)]  # For error bars

filename = "ACCUMULATED_SPDE_ERRORS_ell25_nu1_50k_with_error.csv"
with open(filename, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip info row
    next(reader)  # Skip header row
    for row in reader:
        mesh_widths.append(float(row[0]))
        for i in range(nAlpha):
            errors_spde[i].append(float(row[1 + 2 * i]))
            errors_crf[i].append(float(row[2 + 2 * i]))
            errors_spde_std[i].append(float(row[9 + 2 * i]))
            errors_crf_std[i].append(float(row[10 + 2 * i]))

# Set up plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each oversampling data for SPDE and CRF
linestyles = ['o--', '^--', 's--', 'D--']
for i in range(nAlpha):
    ax.errorbar(
        mesh_widths,
        errors_spde[i],
        yerr=errors_spde_std[i],
        fmt=linestyles[i],
        label=f'α = {oversampling[i]} - SPDE',
        markersize=9,
        markeredgewidth=1.5,
        elinewidth=2,
        capsize=8,
        color='tab:blue')

    ax.errorbar(
        mesh_widths,
        errors_crf[i],
        yerr=errors_crf_std[i],
        fmt=linestyles[i],
        label=f'α = {oversampling[i]} - CRF',
        markersize=9,
        markeredgewidth=1.5,
        elinewidth=2,
        capsize=8,
        color='tab:red')

# Customize axes
ax.set_xlabel('mesh width h', fontsize=14)
ax.set_ylabel('estimate of maximal covariance error', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

handles, labels = ax.get_legend_handles_labels()
handles = [h[0] for h in handles]

# Add legend
ax.legend(
    handles,
    labels,
    fontsize=12,
    loc='upper left',
    bbox_to_anchor=(1, 1), frameon=False)

# Show plot
plt.tight_layout()
plt.savefig('./spde_comparison.pdf')
plt.show()
