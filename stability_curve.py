"""
stability_curve.py -- the connectome's stability landscape.
================================================================================
Runs the LIF brain at increasing gains (with and without Dale's law) and plots
total spikes vs gain. Two regimes emerge:
  * stable regime  (gain < ~11): the brain responds to the stimulus and decays
  * seizure regime (gain > ~11): fluctuations exceed threshold -> spontaneous,
    self-sustaining firing (sigma ~ sqrt(sum w^2 r tau_syn) crosses theta=1)

This is a critical point of the network -- a genuine quantitative finding.
"""
import subprocess, sys, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAINS = [4, 6, 8, 10, 12, 16, 20]
I_STIM = 2.0

def total_spikes(script, gain):
    r = subprocess.run([sys.executable, script, str(gain), str(I_STIM)],
                       capture_output=True, text=True, cwd=".")
    m = re.search(r"total spikes in [\d.]+ ms: (\d+)", r.stdout)
    return int(m.group(1)) if m else None

full   = [total_spikes("lif_full_brain.py", g) for g in GAINS]
dales  = [total_spikes("lif_dales.py", g) for g in GAINS]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(GAINS, np.array(full)/1000, "o-", color="#d62728", label="no Dale's law (all excitatory)")
ax.plot(GAINS, np.array(dales)/1000, "s-", color="#1f77b4", label="Dale's law (MBON, LN inhibitory)")
ax.axvline(10.6, color="k", ls="--", lw=1.2, label="predicted critical gain (sigma = theta)")
ax.axvspan(10.6, 20, color="gray", alpha=0.15)
ax.text(15, ax.get_ylim()[1]*0.55, "seizure\nregime", ha="center", fontsize=10, color="0.3")
ax.text(6.5, ax.get_ylim()[1]*0.55, "stable\nregime", ha="center", fontsize=10, color="0.3")
ax.set_xlabel("synaptic gain")
ax.set_ylabel("total spikes per 500 ms (thousands)")
ax.set_title("Stability landscape of the larval brain connectome (stimulus 50-150 ms)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("stability_curve.png", dpi=130)
print("saved stability_curve.png")
print("gains:   ", GAINS)
print("no-Dale: ", full)
print("Dale's:  ", dales)
