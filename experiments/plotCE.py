import matplotlib.pyplot as plt
import numpy as np
import os
from experiments.readCEData import read_averaged_data

methods = ["dna", "ce", "aCE"]
modelCovs = ["cauchy", "gaussian", "matern", "exponential"]

colors = {
    "dna": "tab:blue",
    "ce": "tab:orange",
    "aCE": "tab:green",
}

linestyles = {
    "Exp": "-",
    "Matern": "--"
}

fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.3), constrained_layout=True)

baseDir = os.path.join("data", "circulantEmbedding")
errorType = "maxError"
nBatch = 2

averagedData = read_averaged_data(baseDir, nBatch)

# === 1. Cost vs Mesh Width ===
ax = axes[0]
meshWidths = averagedData["cost"]["meshWidths"]

for method in averagedData["cost"]["yData"]:
    for modelCov in averagedData["cost"]["yData"][method]:
        cost = averagedData["cost"]["yData"][method][modelCov]
        ax.plot(meshWidths, cost,
                label=f"{method}, {modelCov}",
                color=colors.get(method, "black"),
                linestyle=linestyles.get(modelCov, "-"))

ax.set_xlabel("Mesh width")
ax.set_ylabel("Cost [s]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Cost vs. Mesh Width")
ax.grid(True, which="both", ls=":")
ax.legend(frameon=False, fontsize=7)

# === 2. Memory vs Mesh Width ===
ax = axes[1]
meshWidths = averagedData["memory"]["meshWidths"]

for method in averagedData["memory"]["yData"]:
    for modelCov in averagedData["memory"]["yData"][method]:
        memory = averagedData["memory"]["yData"][method][modelCov]
        ax.plot(meshWidths, memory,
                label=f"{method}, {modelCov}",
                color=colors.get(method, "black"),
                linestyle=linestyles.get(modelCov, "-"))

ax.set_xlabel("Mesh width")
ax.set_ylabel("Memory [MB]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Memory vs. Mesh Width")
ax.grid(True, which="both", ls=":")
ax.legend(frameon=False, fontsize=7)

# === 3. Error vs Cost (dna + aCE only) ===
ax = axes[2]

for method in ["dna", "aCE"]:
    for modelCov in averagedData["error"]["yData"][method]:
        cost = averagedData["error"]["xData"][method][modelCov]
        error = averagedData["error"]["yData"][method][modelCov]
        ax.plot(cost, error,
                label=f"{method}, {modelCov}",
                color=colors[method],
                linestyle=linestyles.get(modelCov, "-"))

ax.set_xlabel("Cost [s]")
ax.set_ylabel("Max error")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Error vs. Cost")
ax.grid(True, which="both", ls=":")
ax.legend(frameon=False, fontsize=7)

plt.show()

# Save to PDF
os.makedirs("figures", exist_ok=True)
output_file = os.path.join("figures", "summary_comparison.pdf")
plt.savefig(output_file, bbox_inches="tight", format='pdf')
print(f"Saved plot to {output_file}")

