#!/usr/bin/env python3
"""Grafica speedup vs número de procesos.
Uso: python3 plot_speedup.py results_cpp.csv [results_go.csv ...]
Genera speedup.png
"""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.5))
max_k = 1
for path in sys.argv[1:]:
    ks, sp = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ks.append(int(row["procesos"]))
            sp.append(float(row["speedup"]))
    max_k = max(max_k, max(ks))
    ax.plot(ks, sp, marker="o", label=path.replace("results_", "").replace(".csv", ""))

ax.plot([1, max_k], [1, max_k], "--", color="gray", label="ideal")
ax.set_xlabel("Número de procesos")
ax.set_ylabel("Speedup (T_secuencial / T_paralelo)")
ax.set_title("Speedup vs número de procesos")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("speedup.png", dpi=150)
print("Guardado speedup.png")