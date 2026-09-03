"""
brain_viz.py -- watch the brain think: the connectome as a living graph.
================================================================================
Uses the neuron embedding coordinates (_pos column of nodes.csv -- a 2D
projection of the connectome's own anatomy/connectivity layout) to draw all
2,956 neurons as a graph. During a 100 ms poke we color each neuron by how
recently it spiked, so you SEE the impulse ripple through the wiring.

Top ~3,000 strongest synapses are drawn as the wiring skeleton.
Right panel: population firing rates (sensory / intrinsic / descending).

Output: brain_activity.gif (+ still brain_activity_peak.png)
"""
import ast
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ============================ 1. DATA ============================
rows = []
with open("data/fly_larva/nodes.csv") as f:
    for r in csv.reader(f):
        if not r or r[0].startswith("#"):
            continue
        rows.append(r)

cell = np.array([r[4] for r in rows])
pos = np.zeros((len(rows), 2))
for i, r in enumerate(rows):
    s = r[7].strip('"')
    try:
        pos[i] = ast.literal_eval(s[s.find("["): s.find("]") + 1])
    except Exception:
        pos[i] = [0.0, 0.0]
n = len(rows)
print(f"loaded {n} neurons, pos range x[{pos[:,0].min():.2f},{pos[:,0].max():.2f}] "
      f"y[{pos[:,1].min():.2f},{pos[:,1].max():.2f}]")

# ============================ 2. CONNECTOME ============================
W_raw = np.zeros((n, n), dtype=np.float32)
with open("data/fly_larva/edges.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        W_raw[int(p[1]), int(p[0])] += int(p[2])
np.fill_diagonal(W_raw, 0.0)
rowsum = W_raw.sum(axis=1); rowsum[rowsum == 0] = 1.0
Wn = (W_raw / rowsum[:, None]) * 8.0

sens = np.where(cell == "sensory")[0]
dn = np.where((cell == "DN-VNC") | (cell == "DN-SEZ"))[0]
intr = np.array([i for i in range(n) if i not in set(sens) and i not in set(dn)])
part = np.zeros(n, dtype=np.int32)
part[sens], part[intr], part[dn] = 0, 1, 2
PCOL = ["#d62728", "#4a90d9", "#2ca02c"]

# top edges for the skeleton
Wt = np.triu(W_raw, 1)
idx = np.argpartition(Wt.ravel(), -3000)[-3000:]
ii, jj = np.unravel_index(idx, Wt.shape)
print(f"edge skeleton: {len(ii)} strongest synapses")

# ============================ 3. SIMULATE (brain only) ============================
rng = np.random.default_rng(42)
pools = np.array_split(rng.permutation(sens), 6)
POKE_POOLS = [1, 3, 5]

DT_B = 0.025                      # ms
TAU, TAU_SYN, THETA = 10.0, 2.0, 1.0
REF = int(2.0 / DT_B)
I_STIM = 2.0
T0, T1 = 900.0, 1600.0            # ms window
POKE_A, POKE_B = 1000.0, 1400.0   # ms
FRAME_MS = 10.0                   # ms of sim per animation frame

V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
refrac = np.zeros(n, dtype=np.int32)
spikes = []
n_sub = int((T1 - T0) / DT_B)

for k in range(n_sub):
    t = T0 + k * DT_B
    poking = POKE_A <= t < POKE_B
    if poking:
        I_ext[pools[1]] = I_STIM
        I_ext[pools[3]] = I_STIM
        I_ext[pools[5]] = I_STIM
    else:
        I_ext[pools[1]] = 0.0
        I_ext[pools[3]] = 0.0
        I_ext[pools[5]] = 0.0

    s *= np.exp(-DT_B / TAU_SYN)
    V += DT_B * ((-V + s + I_ext) / TAU)
    V[refrac > 0] = 0.0; refrac[refrac > 0] -= 1
    fired = (V >= THETA) & (refrac <= 0)
    if fired.any():
        for j in np.where(fired)[0]:
            spikes.append((t, int(j)))
        s += Wn[:, fired].sum(axis=1)
        V[fired] = 0.0; refrac[fired] = REF

spk = np.array(spikes).reshape(-1, 2) if spikes else np.empty((0, 2))
print(f"spikes in window: {len(spk)}")

# ============================ 4. ANIMATE ============================
frames_t = np.arange(T0, T1, FRAME_MS)
edges = LineCollection([(pos[i], pos[j]) for i, j in zip(ii, jj)],
                       colors="0.75", linewidths=0.3, alpha=0.35, zorder=1)
deg = np.log10(1 + np.count_nonzero(W_raw, axis=1))
sizes = 3 + 22 * (deg - deg.min()) / max(1e-6, (deg.max() - deg.min()))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 11), height_ratios=[3, 1])
ax1.add_collection(edges)
base = ax1.scatter(pos[:, 0], pos[:, 1], c=[PCOL[p] for p in part],
                   s=sizes, alpha=0.45, linewidths=0, zorder=2)
act = ax1.scatter([], [], c="white", s=0, cmap="hot", vmin=0, vmax=6,
                  edgecolors="black", linewidths=0.3, zorder=3)
ax1.set_xlim(pos[:, 0].min() - 0.05, pos[:, 0].max() + 0.05)
ax1.set_ylim(pos[:, 1].min() - 0.05, pos[:, 1].max() + 0.05)
ax1.set_aspect("equal")
ax1.set_title("larval brain connectome -- 2,956 neurons, live activity")
from matplotlib.lines import Line2D
ax1.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=l)
                    for c, l in zip(PCOL, ["sensory", "intrinsic", "descending"])],
           loc="upper right", fontsize=9)

bins = np.arange(T0, T1 + 30, 25.0)
for pi in range(3):
    sel = part == pi
    tt = spk[np.isin(spk[:, 1], np.where(sel)[0]), 0]
    h, _ = np.histogram(tt, bins=bins)
    ax2.plot(bins[:-1], h / 0.025 / max(1, int(sel.sum())), color=PCOL[pi], lw=1.5)
ax2.axvspan(POKE_A, POKE_B, color="gray", alpha=0.25)
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("mean rate (Hz)")
ax2.grid(alpha=0.3)
vline = ax2.axvline(T0, color="k", lw=1)
ax2.set_xlim(T0, T1)

frames = []
for t in frames_t:
    w = spk[(spk[:, 0] >= t - FRAME_MS) & (spk[:, 0] < t)]
    if len(w):
        jj_, cc = np.unique(w[:, 1].astype(int), return_counts=True)
    else:
        jj_, cc = np.array([], dtype=int), np.array([])
    act.set_offsets(np.c_[pos[jj_, 0], pos[jj_, 1]] if len(jj_) else np.empty((0, 2)))
    act.set_sizes(30 + 70 * cc if len(cc) else np.array([]))
    act.set_array(cc if len(cc) else np.array([]))
    vline.set_xdata([t])
    ax1.set_title(f"poke RIGHT -- t = {t:.0f} ms")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(buf.copy())
    if abs(t - (POKE_A + POKE_B) / 2) < 1:
        peak = buf.copy()

from PIL import Image
Image.fromarray(peak).save("brain_activity_peak.png")
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save("brain_activity.gif", save_all=True, append_images=imgs[1:],
             duration=80, loop=0)
print(f"saved brain_activity.gif ({len(frames)} frames) + brain_activity_peak.png")
