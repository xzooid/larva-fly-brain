"""
toy_lif.py -- the smallest possible "brain": 2 neurons talking to each other.
================================================================================
WHY THIS FILE EXISTS
   The larva connectome is a 3,016 x 3,016 matrix W. This toy is the SAME object,
   just 2 x 2. Every concept here scales up unchanged to the real connectome:
   the W matrix, the membrane equation, the spike-and-reset, the plot.

THE MODEL (Leaky Integrate-and-Fire, LIF)
   Each neuron has a membrane potential V that:
     - integrates incoming current I:      tau * dV/dt = -(V - V_rest) + I
     - "leaks" back toward rest on a timescale tau (membrane time constant)
     - FIRES (spikes) when V crosses threshold theta, then resets to 0.

   Synapses are simple exponential currents: when neuron j spikes, it adds
   W[i, j] to the synaptic current of neuron i, which then decays with tau_syn.
   (Real synapses work like this -- a bump of current that decays in ~2 ms.)

   W CONVENTION (we use this everywhere, including the real connectome):
     rows  = POSTsynaptic neuron (the receiver)
     cols  = PREsynaptic neuron  (the sender)
     so input to neuron i = sum over j of W[i, j] * (spikes of j recently)

THE EXPERIMENT
   Neuron A gets a constant external drive -> it fires tonically (~90 Hz).
   Neuron B gets NO drive; it only hears A through a strong synapse W[B, A] = 4.5,
   and weakly whispers back to A through W[A, B] = 1.5 (feedback).

   QUESTIONS TO ANSWER BY RUNNING:
   1. Does B fire? How many A-spikes did it take to push B to threshold?
   2. Does B fire on EVERY A-spike? If not, why not? (state at the time matters!)
   3. What does the feedback do to A? (look for tiny bumps on A's trace)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe: writes PNG, needs no display
import matplotlib.pyplot as plt

# ---------------- parameters (normalized units; real neurons: tau ~ 10 ms) ----
dt      = 0.1      # simulation step, ms
T       = 300.0    # total time, ms
tau     = 10.0     # membrane time constant, ms
tau_syn = 2.0      # synaptic current decay, ms
theta   = 1.0      # spike threshold
V_rest  = 0.0

n = 2
# W[i, j] = strength of synapse FROM j TO i.  0 = no synapse.
W = np.array([[0.0, 1.5],     # row 0 = neuron A: receives 1.5 from B (feedback)
              [7.0, 0.0]])    # row 1 = neuron B: receives 7.0 from A (the big driver)
# Why 7.0? Each A-spike pushes B's voltage to a peak of about
#   W * (tau_syn/(tau - tau_syn)) * (e^-t/tau - e^-t/tau_syn) ~= W * 0.134
# so W = 7.0 -> peak ~0.94: just below threshold. B fires only when a bit
# of previous input is still lingering -- i.e. WHEN THE STATE ALLOWS IT.
names = ["A (driver)", "B (follower)"]

# ---------------- state ----------------
V = np.zeros(n)                # membrane potentials
s = np.zeros(n)                # synaptic input currents
I_ext = np.zeros(n)
I_ext[0] = 1.5                 # constant drive ONLY to A

n_steps = int(T / dt)
spikes = []                    # list of (time_ms, neuron_index)
V_hist = np.zeros((n_steps, n))

# ---------------- the simulation loop ----------------
for t_i in range(n_steps):
    t = t_i * dt

    # 1) synaptic currents decay
    s *= np.exp(-dt / tau_syn)

    # 2) integrate membranes (Euler step of the LIF equation)
    V += dt * ((-V + s + I_ext) / tau)

    # 3) spike detection + reset + broadcast
    fired = V >= theta
    if fired.any():
        for j in np.where(fired)[0]:
            spikes.append((t, j))
            s += W[:, j]       # presynaptic neuron j spikes: add its outgoing weights
        V[fired] = 0.0

    V_hist[t_i] = V

# ---------------- report ----------------
spk = np.array(spikes)
print(f"simulated {T:.0f} ms of brain time (dt = {dt} ms, {n_steps} steps)")
for j in range(n):
    times = spk[spk[:, 1] == j, 0]
    rate = 1000.0 * len(times) / T
    print(f"  {names[j]:16s} fired {len(times):3d} times  ->  {rate:6.1f} Hz")

a_times = spk[spk[:, 1] == 0, 0]
b_times = spk[spk[:, 1] == 1, 0]
if len(b_times):
    first_a = a_times[a_times < b_times[0]]
    print(f"  B's first spike at t = {b_times[0]:.1f} ms, after {len(first_a)} A-spike(s) -> temporal summation!")
if len(a_times) and len(b_times):
    near = sum(any(abs(b - a) < 5.0 for b in b_times) for a in a_times)
    print(f"  {near}/{len(a_times)} A-spikes were followed by a B-spike within 5 ms")

# ---------------- plot ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})

ts = np.arange(n_steps) * dt
colors = ["#d62728", "#1f77b4"]
for j in range(n):
    ax1.plot(ts, V_hist[:, j], color=colors[j], lw=1.2, label=names[j])
ax1.axhline(theta, color="k", ls="--", lw=0.8, label="threshold")
ax1.set_ylabel("membrane potential V")
ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("Two neurons, one connectome: A drives B, B whispers back")
ax1.grid(alpha=0.3)

ax2.eventplot([a_times, b_times], colors=colors, lineoffsets=[1, 0], linelengths=0.6)
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["B", "A"])
ax2.set_xlabel("time (ms)"); ax2.set_ylabel("spikes")
ax2.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("lif_2neurons.png", dpi=130)
print("saved lif_2neurons.png")

# ---------------- exercises (try them!) ----------------
# 1. Set W[1, 0] = 0.0 -> B goes silent. Proves A was driving B.
# 2. Set W[0, 1] = 0.0 -> A's rate unchanged. Proves B is only a whisper.
# 3. Raise I_ext[0] to 3.0 -> both rates rise. The brain speeds up with more drive.
