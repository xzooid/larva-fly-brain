"""
lif_full_brain.py -- the whole larval brain in a box.
The SAME loop as toy_lif.py, now with the real 2,956-neuron connectome.

THE EXPERIMENT: "Poke the senses" -- drive the 434 sensory (afferent) neurons
with a current pulse for 100 ms. Watch the impulse travel through real wiring:
     sensory -> interneurons -> descending (the brain's output to the body)

DECISIONS (all yours to justify later):
  * row-normalize W (synaptic scaling), then multiply by global GAIN (swept)
  * zero the diagonal (no self-connections)
  * refractory period 2 ms (real neurons can't refire instantly)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0

# ---------------- parameters ----------------
dt = 0.5; T = 500.0; tau = 10.0; tau_syn = 2.0; theta = 1.0
refrac_ms = 2.0
STIM_ON, STIM_OFF = 50.0, 150.0
I_STIM = 2.0

# ---------------- load + build W ----------------
DATA = "data/fly_larva"
nodes = []
with open(f"{DATA}/nodes.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        nodes.append({"idx": int(p[0]), "cell_type": p[4]})
n = len(nodes)

W = np.zeros((n, n), dtype=np.float32)
with open(f"{DATA}/edges.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        W[int(p[1]), int(p[0])] += int(p[2])     # W[post, pre]
np.fill_diagonal(W, 0.0)

rowsum = W.sum(axis=1)
rowsum[rowsum == 0] = 1.0
Wn = (W / rowsum[:, None]) * GAIN
print(f"brain: {n} neurons, {np.count_nonzero(W):,} connections, gain = {GAIN}")

# ---------------- partition ----------------
SENSORY = {"sensory"}
DESCENDING = {"DN-VNC", "DN-SEZ", "pre-DN-VNC", "pre-DN-SEZ"}
sens = np.array([i for i, nd in enumerate(nodes) if nd["cell_type"] in SENSORY])
desc = np.array([i for i, nd in enumerate(nodes) if nd["cell_type"] in DESCENDING])
intr = np.array([i for i in range(n) if i not in set(sens) and i not in set(desc)])
part = np.full(n, -1, dtype=np.int32)
part[sens], part[intr], part[desc] = 0, 1, 2
pnames = ["sensory", "intrinsic", "descending"]

# ---------------- state ----------------
V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
refrac = np.zeros(n, dtype=np.int32)
spikes = []
n_steps = int(T / dt)
STIM_ON_S, STIM_OFF_S = int(STIM_ON / dt), int(STIM_OFF / dt)

# ---------------- the loop (same skeleton as toy_lif.py) ----------------
for ti in range(n_steps):
    t = ti * dt
    if ti == STIM_ON_S:
        I_ext[sens] = I_STIM
    if ti == STIM_OFF_S:
        I_ext[sens] = 0.0

    s *= np.exp(-dt / tau_syn)
    V += dt * ((-V + s + I_ext) / tau)
    V[refrac > 0] = 0.0
    refrac[refrac > 0] -= 1

    fired = (V >= theta) & (refrac <= 0)
    if fired.any():
        for j in np.where(fired)[0]:
            spikes.append((t, int(j)))
        s += Wn[:, fired].sum(axis=1)
        V[fired] = 0.0
        refrac[fired] = int(refrac_ms / dt)

# ---------------- report ----------------
spk = np.array(spikes)
print(f"\ntotal spikes in {T:.0f} ms: {len(spk)}")
for pi in range(3):
    sel = part == pi
    pre  = int(((spk[:, 0] < STIM_ON)  & np.isin(spk[:, 1], np.where(sel)[0])).sum())
    dur  = int(((spk[:, 0] >= STIM_ON) & (spk[:, 0] < STIM_OFF) & np.isin(spk[:, 1], np.where(sel)[0])).sum())
    post = int(((spk[:, 0] >= STIM_OFF) & np.isin(spk[:, 1], np.where(sel)[0])).sum())
    print(f"  {pnames[pi]:11s} spikes  pre {pre:5d} | during-stimulus {dur:5d} ({dur/0.1:8.0f} Hz) | post {post:5d}")

dspk = spk[np.isin(spk[:, 1], np.where(part == 2)[0])]
if len(dspk):
    first = dspk[dspk[:, 0] >= STIM_ON]
    if len(first):
        print(f"\nfirst descending (motor-output) spike: t = {first[0,0]:.1f} ms"
              f" -> sensorimotor latency {first[0,0]-STIM_ON:.1f} ms after stimulus onset")
    else:
        print("\nno descending spikes during/after stimulus -- cascade died out!")
else:
    print("\nno descending spikes at all")

# ---------------- plot ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
colors = ["#d62728", "#1f77b4", "#2ca02c"]
for pi in range(3):
    sel = part == pi
    t = spk[np.isin(spk[:, 1], np.where(sel)[0]), 0]
    ax1.plot(t, np.full(len(t), pi), "|", color=colors[pi], ms=1.5, alpha=0.5)
ax1.axvspan(STIM_ON, STIM_OFF, color="gray", alpha=0.25, label="stimulus")
ax1.set_yticks([0, 1, 2]); ax1.set_yticklabels(pnames); ax1.set_ylim(-0.5, 2.5)
ax1.set_title(f"Whole larval brain responds to a 100 ms sensory poke (gain={GAIN})")
ax1.legend(loc="upper right", fontsize=9); ax1.grid(alpha=0.3)

bins = np.arange(0, T, 10.0)
for pi in range(3):
    sel = part == pi
    t = spk[np.isin(spk[:, 1], np.where(sel)[0]), 0]
    hist, _ = np.histogram(t, bins=bins)
    ax2.plot(bins[:-1], hist / 0.01 / max(1, int(sel.sum())), color=colors[pi], label=pnames[pi])
ax2.axvspan(STIM_ON, STIM_OFF, color="gray", alpha=0.25)
ax2.set_ylabel("mean firing rate (Hz)"); ax2.set_xlabel("time (ms)")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("lif_full_brain.png", dpi=130)
print("\nsaved lif_full_brain.png")

# ---------------- exercises ----------------
# 1. GAIN=4  -> cascade dies (few descending spikes).   2. GAIN=20 -> seizure.
# 3. Comment the two I_ext lines -> a quiet brain. Proves response is stimulus-driven.
# 4. (lesion) Zero the Kenyon cells' rows and watch descending output drop.
