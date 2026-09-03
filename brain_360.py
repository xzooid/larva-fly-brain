"""
brain_360.py -- the connectome in 3D, slow 360-degree view.
Spectral embedding (Laplacian eigenmaps) lays the graph out in 3D from
connectivity alone; ~20k strongest synapses drawn as lines; neurons colored
by partition (sensory red / intrinsic blue / descending green).
Output: brain_360.gif + brain_360_still.png  (pure matplotlib, no GL)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

rows = []
with open("data/fly_larva/nodes.csv") as f:
    for r in csv.reader(f):
        if not r or r[0].startswith("#"):
            continue
        rows.append(r)
cell = np.array([r[4] for r in rows])
n = len(rows)

W = np.zeros((n, n), dtype=np.float32)
with open("data/fly_larva/edges.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        W[int(p[1]), int(p[0])] += int(p[2])
A = W + W.T
np.fill_diagonal(A, 0.0)
print(f"{n} neurons, {np.count_nonzero(W)} directed connections")

deg = A.sum(axis=1)
L = np.diag(deg) - A
print("computing spectral embedding (eigen-decomposition)...")
evals, evecs = np.linalg.eigh(L)
coords = evecs[:, 1:4]
for k in range(3):
    c = coords[:, k]
    coords[:, k] = 2 * (c - c.min()) / (c.max() - c.min() + 1e-9) - 1
print("3D layout ready")

sens = np.where(cell == "sensory")[0]
dn = np.where((cell == "DN-VNC") | (cell == "DN-SEZ"))[0]
intr = np.array([i for i in range(n) if i not in set(sens) and i not in set(dn)])
part = np.zeros(n, dtype=int)
part[sens], part[intr], part[dn] = 0, 1, 2
PCOL = ["#d62728", "#4a90d9", "#2ca02c"]

Wt = np.triu(W, 1)
k = 20000
idx = np.argpartition(Wt.ravel(), -k)[-k:]
ii, jj = np.unravel_index(idx, Wt.shape)
segs = np.stack([coords[ii], coords[jj]], axis=1)
print(f"drawing {len(segs)} synapse lines")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Line3DCollection(segs, colors="0.55", linewidths=0.35, alpha=0.22))
ax.scatter(coords[sens, 0], coords[sens, 1], coords[sens, 2],
           s=13, c=PCOL[0], alpha=0.55, depthshade=False)
ax.scatter(coords[intr, 0], coords[intr, 1], coords[intr, 2],
           s=13, c=PCOL[1], alpha=0.55, depthshade=False)
ax.scatter(coords[dn, 0], coords[dn, 1], coords[dn, 2],
           s=15, c=PCOL[2], alpha=0.8, depthshade=False)

ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
ax.set_axis_off()
ax.view_init(elev=22, azim=0)
ax.set_title("Larval Drosophila brain connectome (2,956 neurons, spectral 3D layout)")

from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          label=l, markersize=8)
                   for c, l in zip(PCOL, ["sensory (afferent)", "intrinsic", "descending (efferent)"])],
          loc="upper left", fontsize=9)

from PIL import Image
frames = []
N_FRAMES = 240
for f in range(N_FRAMES):
    ax.view_init(elev=22, azim=360.0 * f / N_FRAMES)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(buf.copy())
    if f == 0:
        Image.fromarray(buf).save("brain_360_still.png")

imgs = [Image.fromarray(fr) for fr in frames]
imgs[0].save("brain_360.gif", save_all=True, append_images=imgs[1:],
             duration=90, loop=0)
print(f"saved brain_360.gif ({N_FRAMES} frames) + brain_360_still.png")
