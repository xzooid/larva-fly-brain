import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
EI   = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # I/E balance ratio
DELAY_MS = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0  # synaptic delay

# ---------------- parameters (same as lif_full_brain.py) ----------------
dt = 0.5; T = 500.0; tau = 10.0; tau_syn = 2.0; theta = 1.0
refrac_ms = 2.0
STIM_ON, STIM_OFF = 50.0, 150.0
I_STIM = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

# ---------------- load + build W (signed!) ----------------
DATA = "data/fly_larva"
nodes = []
with open(f"{DATA}/nodes.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        nodes.append({"idx": int(p[0]), "cell_type": p[4]})
n = len(nodes)

INHIBITORY_CLASSES = {"MBON", "LN"}          # Dale's law table (see docstring)
sign = np.ones(n, dtype=np.float32)
for i, nd in enumerate(nodes):
    if nd["cell_type"] in INHIBITORY_CLASSES:
        sign[i] = -1.0
# HYPOTHESIS TEST: a fraction of SENSORY neurons are inhibitory (as observed in
# some insect sensory systems -- e.g., GABAergic mechanosensory neurons).
# If the sensory->sensory recurrent loop is the seizure bottleneck, adding
# inhibition INTO that loop should stabilize high gain. argv[4] = fraction.
# NOTE: must run BEFORE W *= sign below (sign must be final before signing).
FRAC_INH_SENS = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
if FRAC_INH_SENS > 0:
    rng = np.random.default_rng(0)                       # seeded: reproducible
    sens_idx = np.array([i for i, nd in enumerate(nodes) if nd["cell_type"] in {"sensory"}])
    k = max(1, int(round(FRAC_INH_SENS * len(sens_idx))))
    extra = rng.choice(sens_idx, size=k, replace=False)
    sign[extra] = -1.0
    print(f"hypothesis: {k} sensory neurons treated as inhibitory ({FRAC_INH_SENS:.0%})")

n_inh = int((sign < 0).sum())
print(f"Dale's law: {n_inh} inhibitory neurons "
      f"({', '.join(sorted(INHIBITORY_CLASSES))}), {n - n_inh} excitatory (default)")

W = np.zeros((n, n), dtype=np.float32)
with open(f"{DATA}/edges.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        W[int(p[1]), int(p[0])] += int(p[2])     # W[post, pre]
np.fill_diagonal(W, 0.0)

W *= sign[None, :]                              # Dale's law: sign of the SOURCE

# Balanced-network normalization (van Vreeswijk & Sompolinsky 1996):
# scale excitatory and inhibitory input SEPARATELY per neuron, so that
# each neuron gets total E input = GAIN and total I input = GAIN * EI.
# This guarantees inhibition can veto excitation -- not just be a whisper.
W_pos = np.clip(W, 0, None)
W_neg = np.clip(W, None, 0)
rowE = W_pos.sum(axis=1)
rowI = -W_neg.sum(axis=1)
rowE[rowE == 0] = 1.0
rowI[rowI == 0] = 1.0
Wn = GAIN * (W_pos / rowE[:, None] - EI * W_neg / rowI[:, None])
print(f"brain: {n} neurons, gain = {GAIN}, I/E ratio = {EI}")

# ---------------- partition ----------------
SENSORY = {"sensory"}
DESCENDING = {"DN-VNC", "DN-SEZ", "pre-DN-VNC", "pre-DN-SEZ"}
sens = np.array([i for i, nd in enumerate(nodes) if nd["cell_type"] in SENSORY])
desc = np.array([i for i, nd in enumerate(nodes) if nd["cell_type"] in DESCENDING])
intr = np.array([i for i in range(n) if i not in set(sens) and i not in set(desc)])
part = np.full(n, -1, dtype=np.int32)
part[sens], part[intr], part[desc] = 0, 1, 2
pnames = ["sensory", "intrinsic", "descending"]

# ---------------- state + loop ----------------
V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
refrac = np.zeros(n, dtype=np.int32)
spikes = []
n_steps = int(T / dt)
STIM_ON_S, STIM_OFF_S = int(STIM_ON / dt), int(STIM_OFF / dt)

# synaptic delay buffer: a spike fired now is delivered DELAY_MS later
delay_steps = int(round(DELAY_MS / dt))
spike_buf = np.zeros((delay_steps + 1, n), dtype=bool)

for ti in range(n_steps):
    t = ti * dt
    if ti == STIM_ON_S:
        I_ext[sens] = I_STIM
    if ti == STIM_OFF_S:
        I_ext[sens] = 0.0

    if delay_steps > 0:
        delivered = spike_buf[0]                   # spikes from DELAY_MS ago
        spike_buf[:-1] = spike_buf[1:]             # shift the buffer
        spike_buf[-1] = False
    else:
        delivered = None

    s *= np.exp(-dt / tau_syn)
    V += dt * ((-V + s + I_ext) / tau)
    V[refrac > 0] = 0.0
    refrac[refrac > 0] -= 1

    fired = (V >= theta) & (refrac <= 0)
    if fired.any():
        for j in np.where(fired)[0]:
            spikes.append((t, int(j)))
        if delay_steps > 0:
            spike_buf[-1] |= fired                # queue for later delivery
        else:
            s += Wn[:, fired].sum(axis=1)
        V[fired] = 0.0
        refrac[fired] = int(refrac_ms / dt)

    if delivered is not None and delivered.any():
        s += Wn[:, delivered].sum(axis=1)

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
ax1.set_title(f"Dale's law brain (gain={GAIN}): stimulus -> motor output")
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
plt.savefig("lif_dales.png", dpi=130)
print("\nsaved lif_dales.png")
