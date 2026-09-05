"""
brain_viz_lesions.py -- the lesion experiment, in the BRAIN, not the body.
================================================================================
Three brains, same poke (right-side pools, 100 ms), side by side:
   1. baseline (intact)     2. KC lesion (mushroom body)   3. LN lesion (inhibitory)
Each panel: all 2,956 neurons laid out by their real embedding coordinates
(_pos from nodes.csv), flashing as they spike; strongest ~3,000 synapses as
the wiring skeleton. Bottom: population firing rates (sensory/intrinsic/
descending) per condition.

Why this is the RIGHT view: the body videos look similar because the reflex
leg kick is similar; the DIFFERENCE lives in the brain (LN lesion -> extra
descending drive). Here you SEE it.

Output: brain_lesions.gif (3-panel composite) + brain_lesions_peak.png
Run:    python brain_viz_lesions.py        (pure matplotlib, no GL needed)
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

W_raw = np.zeros((n, n), dtype=np.float32)
with open("data/fly_larva/edges.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        W_raw[int(p[1]), int(p[0])] += int(p[2])
np.fill_diagonal(W_raw, 0.0)

sens = np.where(cell == "sensory")[0]
dn = np.where((cell == "DN-VNC") | (cell == "DN-SEZ"))[0]
intr = np.array([i for i in range(n) if i not in set(sens) and i not in set(dn)])
part = np.zeros(n, dtype=np.int32)
part[sens], part[intr], part[dn] = 0, 1, 2
PCOL = ["#d62728", "#4a90d9", "#2ca02c"]
PNAMES = ["sensory", "intrinsic", "descending"]

LESIONS = {"KC": np.where(cell == "KC")[0], "LN": np.where(cell == "LN")[0]}
pools = np.array_split(np.random.default_rng(42).permutation(sens), 6)
POKE_POOLS = [1, 3, 5]

# edge skeleton
Wt = np.triu(W_raw, 1)
idx = np.argpartition(Wt.ravel(), -3000)[-3000:]
ii, jj = np.unravel_index(idx, Wt.shape)

# ============================ 2. SIMULATE 3 BRAINS ============================
DT_B = 0.025                      # ms
TAU, TAU_SYN, THETA = 10.0, 2.0, 1.0
REF = int(2.0 / DT_B)
I_STIM = 2.0
T0, T1 = 900.0, 1600.0
POKE_A, POKE_B = 1000.0, 1400.0
FRAME_MS = 10.0

def simulate(lesion=None):
    W = W_raw.copy()
    if lesion is not None:
        idx = LESIONS[lesion]
        W[:, idx] = 0.0
        W[idx, :] = 0.0
    rowsum = W.sum(axis=1); rowsum[rowsum == 0] = 1.0
    Wn = (W / rowsum[:, None]) * 8.0

    V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
    refrac = np.zeros(n, dtype=np.int32)
    spikes = []
    for k in range(int((T1 - T0) / DT_B)):
        t = T0 + k * DT_B
        poking = POKE_A <= t < POKE_B
        for pi in POKE_POOLS:
            I_ext[pools[pi]] = I_STIM if poking else 0.0
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
    return spk

conds = [("baseline", None), ("KC lesion", "KC"), ("LN lesion", "LN")]
spikes_by = {}
for name, lesion in conds:
    spk = simulate(lesion)
    spikes_by[name] = spk
    desc = spk[np.isin(spk[:, 1], dn)]
    d_dur = int(((desc[:, 0] >= POKE_A) & (desc[:, 0] < POKE_B)).sum())
    print(f"{name:12s}: {len(spk):6d} spikes | descending during poke: {d_dur}")

# ============================ 3. ANIMATE ============================
frames_t = np.arange(T0, T1, FRAME_MS)
fig, axes = plt.subplots(len(conds), 2, figsize=(15, 10),
                         gridspec_kw={"width_ratios": [3, 1]})
fig.subplots_adjust(hspace=0.35)

# shared edge skeleton per panel
edges = LineCollection([(pos[i], pos[j]) for i, j in zip(ii, jj)],
                       colors="0.75", linewidths=0.3, alpha=0.3, zorder=1)
deg = np.log10(1 + np.count_nonzero(W_raw, axis=1))
sizes = 3 + 22 * (deg - deg.min()) / max(1e-6, (deg.max() - deg.min()))

scatters = []
ax_rate = axes[0][1]
bins = np.arange(T0, T1 + 30, 25.0)
vline = None
for ai, (name, lesion) in enumerate(conds):
    ax = axes[ai][0]
    ax.add_collection(LineCollection([(pos[i], pos[j]) for i, j in zip(ii, jj)],
                                     colors="0.75", linewidths=0.3, alpha=0.3, zorder=1))
    ax.scatter(pos[:, 0], pos[:, 1], c=[PCOL[p] for p in part],
               s=sizes, alpha=0.4, linewidths=0, zorder=2)
    act = ax.scatter([], [], s=0, c="white", cmap="hot", vmin=0, vmax=6,
                     edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_xlim(pos[:, 0].min() - 0.05, pos[:, 0].max() + 0.05)
    ax.set_ylim(pos[:, 1].min() - 0.05, pos[:, 1].max() + 0.05)
    ax.set_aspect("equal")
    ax.set_title(f"{name}  (desc@poke: {int(((spikes_by[name][np.isin(spikes_by[name][:,1], dn), 0] >= POKE_A) & (spikes_by[name][np.isin(spikes_by[name][:,1], dn), 0] < POKE_B)).sum())})")
    scatters.append(act)

    ar = axes[ai][1]
    for pi in range(3):
        sel = part == pi
        tt = spikes_by[name][np.isin(spikes_by[name][:, 1], np.where(sel)[0]), 0]
        h, _ = np.histogram(tt, bins=bins)
        ar.plot(bins[:-1], h / 0.025 / max(1, int(sel.sum())), color=PCOL[pi], lw=1.2)
    ar.axvspan(POKE_A, POKE_B, color="gray", alpha=0.25)
    ar.set_xlim(T0, T1)
    ar.set_ylabel(f"{name}\nrate (Hz)", fontsize=8)
    ar.grid(alpha=0.3)
    if ai == 0:
        ar.legend(PNAMES, fontsize=7, loc="upper left")

vline = axes[0][1].axvline(T0, color="k", lw=1)

from PIL import Image
frames_out = []
for ti, t in enumerate(frames_t):
    for ai, (name, lesion) in enumerate(conds):
        spk = spikes_by[name]
        w = spk[(spk[:, 0] >= t - FRAME_MS) & (spk[:, 0] < t)]
        if len(w):
            jj_, cc = np.unique(w[:, 1].astype(int), return_counts=True)
        else:
            jj_, cc = np.array([], dtype=int), np.array([])
        act = scatters[ai]
        act.set_offsets(np.c_[pos[jj_, 0], pos[jj_, 1]] if len(jj_) else np.empty((0, 2)))
        act.set_sizes(30 + 70 * cc if len(cc) else np.array([]))
        act.set_array(cc if len(cc) else np.array([]))
        axes[ai][0].set_title(f"{name}  t = {t:.0f} ms")
    vline.set_xdata([t])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames_out.append(buf.copy())
    if abs(t - (POKE_A + POKE_B) / 2) < 1:
        peak = buf.copy()

Image.fromarray(peak).save("brain_lesions_peak.png")
imgs = [Image.fromarray(f) for f in frames_out]
imgs[0].save("brain_lesions.gif", save_all=True, append_images=imgs[1:],
             duration=80, loop=0)
print(f"saved brain_lesions.gif ({len(frames_out)} frames) + brain_lesions_peak.png")
