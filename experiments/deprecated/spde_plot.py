import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import linregress

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
fig, ax = plt.subplots(figsize=(10, 5))

# Plot each oversampling data for SPDE and DNA
linestyles = ['o--', '^--', 's--', 'D--']
for i in range(nAlpha):
    ax.errorbar(
        mesh_widths,
        errors_spde[i],
        yerr=errors_spde_std[i],
        fmt=linestyles[i],
        label=f'α = {oversampling[i]} - SPDE',
        markersize=7,
        markeredgewidth=1.5,
        elinewidth=1.5,
        capsize=6,
        color='tab:blue',
        alpha=0.9)

    ax.errorbar(
        mesh_widths,
        errors_crf[i],
        yerr=errors_crf_std[i],
        fmt=linestyles[i],
        label=f'α = {oversampling[i]} - DNA',
        markersize=7,
        markeredgewidth=1.5,
        elinewidth=1.5,
        capsize=6,
        color='tab:red',
        alpha=0.9)

rate1 = 1.
rate2 = 0.5

base1 = 0.02 
base2 = 0.03

height1 = rate1 * base1
height2 = rate2 * base2

x_triangle1 = [0.083, 0.083 + base1]
y_triangle1 = [0.025, 0.025 + height1]

x_triangle2 = [0.075, 0.075 + base2]
y_triangle2 = [0.035, 0.035 + height2]

# # Draw triangle
# ax.plot(
#     [x_triangle1[0], x_triangle1[1]], [y_triangle1[0], y_triangle1[0]],
#     color='black', lw=2.5)  # Base
# ax.plot(
#     [x_triangle1[1], x_triangle1[1]], [y_triangle1[0], y_triangle1[1]],
#     color='black', lw=2.5)  # Height
# ax.plot(
#     [x_triangle1[0], x_triangle1[1]], [y_triangle1[0], y_triangle1[1]],
#     color='black', lw=2.5)  # Hypotenuse
# 
# # Annotate catheti lengths
# ax.text(x_triangle1[0] + 0.5 * base1, y_triangle1[0] - 0.002,f'{1}', fontsize=12, ha='center')
# ax.text(x_triangle1[1] + 0.002, y_triangle1[1] - 0.6 * height1,
#         f'{1}', fontsize=12, va='center')


# Draw triangle
ax.plot(
    [x_triangle2[0], x_triangle2[1]], [y_triangle2[0], y_triangle2[0]],
    color='black', lw=2.5)  # Base
ax.plot(
    [x_triangle2[1], x_triangle2[1]], [y_triangle2[0], y_triangle2[1]],
    color='black', lw=2.5)  # Height
ax.plot(
    [x_triangle2[0], x_triangle2[1]], [y_triangle2[0], y_triangle2[1]],
    color='black', lw=2.5)  # Hypotenuse

# Annotate catheti lengths
ax.text(x_triangle2[0] + 0.5 * base2, y_triangle2[0] - 0.009,f'{1}', fontsize=12, ha='center')
ax.text(x_triangle2[1] + 0.003, y_triangle2[1] - 0.6 * height2,
        '1/2', fontsize=12, va='center')

# Customize axes
ax.set_xlabel('mesh width h', fontsize=16)
ax.set_ylabel('estimate of maximal covariance error', fontsize=16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    fontsize=14,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    frameon=False)

# Set alpha for legend text and markers
for text in legend.get_texts():
    text.set_alpha(1)  # Text fully opaque

for line in legend.get_lines():
    line.set_alpha(1)  # Marker lines fully opaque

# Show plot
plt.tight_layout()
plt.savefig('./spde_comparison.pdf', format='pdf', dpi=800)
plt.show()

