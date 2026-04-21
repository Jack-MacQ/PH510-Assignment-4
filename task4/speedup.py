"""
PH510 Assignment 4 - Task 4
Parallel scaling analysis and speedup plots.

This script reads measured wall-clock timings for the parallel parameter
scan and the final production calculation, then constructs standard
strong-scaling diagnostics:

    1. Speedup as a function of MPI rank count
    2. Parallel efficiency as a function of MPI rank count

The measured speedup curves are compared directly with ideal linear
scaling. All figures and summary data are written to the directory

    Speedup-Results/

for convenient inclusion in the report.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
Pylint Score: 10/10
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("Speedup-Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Measured wall-clock timings
# ---------------------------------------------------------------------------

PROCS = np.array([1, 2, 4, 8, 16])

SCAN_TIMES = np.array([208.858800, 125.764732, 64.047925, 33.381782, 18.421478])
FINAL_TIMES = np.array([6.826406, 4.299692, 2.187627, 1.177863, 0.656053])

# ---------------------------------------------------------------------------
# Derived scaling metrics
# ---------------------------------------------------------------------------

scan_speedup = SCAN_TIMES[0] / SCAN_TIMES
final_speedup = FINAL_TIMES[0] / FINAL_TIMES

scan_efficiency = 100.0 * scan_speedup / PROCS
final_efficiency = 100.0 * final_speedup / PROCS

summary_path = OUTPUT_DIR / "task4_scaling_summary.txt"
with open(summary_path, "w", encoding="utf-8") as handle:
    handle.write("PH510 Assignment 4 - Task 4 scaling summary\n")
    handle.write("=" * 48 + "\n\n")

    handle.write(f"MPI ranks:\n{PROCS}\n\n")
    handle.write(f"Scan times (s):\n{SCAN_TIMES}\n\n")
    handle.write(f"Final times (s):\n{FINAL_TIMES}\n\n")

    handle.write(f"Scan speedup:\n{np.round(scan_speedup, 3)}\n\n")
    handle.write(f"Final speedup:\n{np.round(final_speedup, 3)}\n\n")

    handle.write(f"Scan efficiency (%):\n{np.round(scan_efficiency, 1)}\n\n")
    handle.write(f"Final efficiency (%):\n{np.round(final_efficiency, 1)}\n")

print("Scan speedups:", np.round(scan_speedup, 3))
print("Final speedups:", np.round(final_speedup, 3))
print()
print("Scan efficiencies (%):", np.round(scan_efficiency, 1))
print("Final efficiencies (%):", np.round(final_efficiency, 1))
print()
print(f"Saved summary data to {summary_path}")

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
    }
)

BLUE = "#3266ad"
GREEN = "#1d9e75"
GRAY = "#888780"

# ---------------------------------------------------------------------------
# Figure 1: speedup
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(PROCS, PROCS, linestyle="--", linewidth=1.2, color=GRAY,
        label="Ideal linear scaling")

ax.plot(PROCS, scan_speedup, marker="o", linewidth=1.8, markersize=6,
        color=BLUE, label="Parameter scan")

ax.plot(PROCS, final_speedup, marker="D", linewidth=1.8, markersize=6,
        color=GREEN, label="Final production run")

ax.set_xlabel("Number of MPI ranks")
ax.set_ylabel("Speedup")
ax.set_xlim(0.5, 17)
ax.set_ylim(0, 17)
ax.xaxis.set_major_locator(mticker.FixedLocator(PROCS.tolist()))
ax.yaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(loc="upper left")
ax.set_title("Parallel speedup for Task 4 VMC")

fig.tight_layout()

speedup_png = OUTPUT_DIR / "task4_speedup.png"
fig.savefig(speedup_png, dpi=300, bbox_inches="tight")
print(f"Saved {speedup_png}")

# ---------------------------------------------------------------------------
# Figure 2: efficiency
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(6, 3.8))

ax2.axhline(100, linestyle="--", linewidth=1.2, color=GRAY,
            label="Ideal efficiency")

ax2.plot(PROCS, scan_efficiency, marker="o", linewidth=1.8, markersize=6,
         color=BLUE, label="Parameter scan")

ax2.plot(PROCS, final_efficiency, marker="D", linewidth=1.8, markersize=6,
         color=GREEN, label="Final production run")

for p, se, fe in zip(PROCS, scan_efficiency, final_efficiency):
    ax2.annotate(f"{se:.0f}%", (p, se), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=8, color=BLUE)
    ax2.annotate(f"{fe:.0f}%", (p, fe), textcoords="offset points",
                 xytext=(0, -13), ha="center", fontsize=8, color=GREEN)

ax2.set_xlabel("Number of MPI ranks")
ax2.set_ylabel("Parallel efficiency (%)")
ax2.set_xlim(0.5, 17)
ax2.set_ylim(0, 115)
ax2.xaxis.set_major_locator(mticker.FixedLocator(PROCS.tolist()))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(25))
ax2.legend(loc="upper right")
ax2.set_title("Parallel efficiency for Task 4 VMC")

fig2.tight_layout()

efficiency_png = OUTPUT_DIR / "task4_efficiency.png"
fig2.savefig(efficiency_png, dpi=300, bbox_inches="tight")
print(f"Saved {efficiency_png}")

plt.show()
