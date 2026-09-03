"""
embodied_fly.py -- the larval brain, embodied in the MuJoCo fly.
================================================================================
THE LOOP (afferent -> brain -> efferent, every physics step):

  body (claw touch sensors)  --I_ext-->  sensory neurons (434)
                                          |  (connectome, gain 8: stable)
                                          v
                                  interneurons (1,598)
                                          |
                                          v
                          descending neurons: DN-VNC (legs), DN-SEZ (head)
                                          |
                 firing rate (low-pass) --> joint targets (position servos)
                                          v
                                       the fly moves

  * 6 claw-touch sensors -> 6 fixed sensory pools (seeded, reproducible)
  * DN-VNC neurons split into 6 groups (one per leg) -> that leg's 8 joints
  * DN-SEZ neurons -> head + abdomen joints
  * joint target = stance baseline + k * (group rate - rest rate), clamped

UNITS (the bug we fixed): MuJoCo's timestep is in SECONDS (0.0001 s); the LIF
equation uses milliseconds (tau = 10 ms). ALWAYS convert: dt_brain_ms =
dt_phys_s * 1000 / SUB.

THE DEMO: a 100 ms "poke" (current injection into the right-T2 sensory pool)
-> spikes propagate through the real connectome -> legs twitch.

Run:
    python embodied_fly.py                       # plots only
    MUJOCO_GL=glfw xvfb-run -a python embodied_fly.py --video   # + GIF
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO = ("--video" in sys.argv) or os.environ.get("MUJOCO_VIDEO") == "1"

# ============================ 1. THE BRAIN ============================
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
        W[int(p[1]), int(p[0])] += int(p[2])       # W[post, pre]
np.fill_diagonal(W, 0.0)
rowsum = W.sum(axis=1); rowsum[rowsum == 0] = 1.0
Wn = (W / rowsum[:, None]) * 8.0                   # gain 8 = stable operating point

cell = np.array([nd["cell_type"] for nd in nodes])
sens = np.where(cell == "sensory")[0]              # afferents
dn_vnc = np.where(cell == "DN-VNC")[0]             # leg-command neurons
dn_sez = np.where(cell == "DN-SEZ")[0]             # head/feeding neurons
desc_all = np.concatenate([dn_vnc, dn_sez])
intr = np.array([i for i in range(n) if i not in set(sens) and i not in set(desc_all)])
part = np.zeros(n, dtype=np.int32)
part[sens] = 0; part[intr] = 1; part[desc_all] = 2

# sensory pools: 6 fixed groups (one per claw), reproducible
POOL_NAMES = ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]
rng = np.random.default_rng(42)
pools = np.array_split(rng.permutation(sens), 6)

# descending groups: 6 fixed groups (one per leg), reproducible
rng2 = np.random.default_rng(7)
leg_groups = np.array_split(rng2.permutation(dn_vnc), 6)

# ============================ 2. THE BODY ============================
import mujoco
CANDIDATES = ["flybody_assets",
              "flybody/flybody/fruitfly/assets",
              "../flybody/flybody/fruitfly/assets",
              "~/flybrain/flybody/flybody/fruitfly/assets"]
ASSETS = next(p for p in CANDIDATES if os.path.exists(os.path.expanduser(p) + "/floor.xml"))
m = mujoco.MjModel.from_xml_path(f"{ASSETS}/floor.xml")
d = mujoco.MjData(m)
DT_PHYS = float(m.opt.timestep)                    # SECONDS (0.0001)
print(f"physics timestep: {DT_PHYS*1000:.2f} ms")

LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2"]
LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
act = {}
for leg in LEGS:
    act[leg] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_{leg}")
                for j in LEG_JOINTS]
    assert all(a >= 0 for a in act[leg]), f"missing actuator for {leg}"
SEZ_ACTS = ["head_abduct", "head_twist", "head", "rostrum",
            "haustellum_abduct", "haustellum", "abdomen_abduct", "abdomen"]
sez_act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in SEZ_ACTS]
assert all(a >= 0 for a in sez_act)

# ============================ 3. PARAMETERS ============================
SUB = 4
DT_B = DT_PHYS * 1000.0 / SUB                     # brain dt in MILLISECONDS!
TAU, TAU_SYN, THETA = 10.0, 2.0, 1.0              # ms, ms, threshold
REF = int(2.0 / DT_B)                             # refractory ~2 ms
TAU_RATE = 50.0                                   # ms, rate low-pass
GAIN_JOINT = 0.006                                # rad per Hz above rest
CLAMP = 1.0                                       # max joint delta, rad
I_STIM = 2.0                                      # poke current
TOUCH_GAIN = 0.05                                 # claw force -> sensory current

POKE_POOLS = [1, 3, 5]                            # right side: T1R, T2R, T3R
POKE_ON, POKE_OFF = 1.0, 1.4                      # s (after settle)
T_TOTAL = 4.0                                     # s
RENDER_EVERY = 160                                # physics steps per video frame

# ============================ 4. STATE ============================
V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
refrac = np.zeros(n, dtype=np.int32)
spikes = []
r_leg = np.zeros(6); r_sez = 0.0
ts = []; legpos = {leg: [[] for _ in LEG_JOINTS] for leg in LEGS}; rates = []

def joint_pos(aid):
    # read by NAME: actuator and joint share a name. actuator_trnid is WRONG
    # for tendon actuators (trnid[0] is a tendon id that collides with a joint id)
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(d.qpos[m.jnt_qposadr[jid]])

def brain_substep(t):
    global s, V
    s *= np.exp(-DT_B / TAU_SYN)
    V += DT_B * ((-V + s + I_ext) / TAU)
    V[refrac > 0] = 0.0; refrac[refrac > 0] -= 1
    fired = (V >= THETA) & (refrac <= 0)
    if fired.any():
        for j in np.where(fired)[0]:
            spikes.append((t, int(j)))
        s += Wn[:, fired].sum(axis=1)
        V[fired] = 0.0; refrac[fired] = REF
        return np.where(fired)[0]
    return np.empty(0, dtype=np.int64)

# settle (brain runs, no input, body relaxes) then lock the stance baseline
print("settling 1 s...")
for _ in range(int(1.0 / DT_PHYS)):
    for _ in range(SUB):
        brain_substep(0.0)
    mujoco.mj_step(m, d)
base_joint = {leg: np.array([joint_pos(a) for a in act[leg]]) for leg in LEGS}
base_sez = np.array([joint_pos(a) for a in sez_act])
print("settled. stance locked.")

# renderer for video (created once; skipped gracefully if GL unavailable)
renderer = None
frames = []
if VIDEO:
    try:
        renderer = mujoco.Renderer(m, 480, 640)
        print("renderer ready -> capturing video during the run")
    except Exception as e:
        print("video disabled:", e)

# ============================ 5. MAIN LOOP ============================
n_steps = int(T_TOTAL / DT_PHYS)
poke_spikes = 0
leg_cnt = np.zeros(6, dtype=np.int64)
sez_cnt = 0
CALIB_END = 0.5                                    # s: measure resting rates first
calib_samples = []                                 # (r_leg, r_sez) during calibration
r_rest = np.zeros(6)
calibrated = False
for step in range(n_steps):
    t = step * DT_PHYS
    poking = POKE_ON <= t < POKE_OFF
    calibrating = t < CALIB_END

    # lock in resting rates the moment calibration ends
    if not calibrating and not calibrated:
        r_rest = np.mean([c[0] for c in calib_samples], axis=0)
        calibrated = True
        print(f"resting rates calibrated: {np.round(r_rest, 1)} Hz")

    I_ext[:] = 0.0                                   # reset afferent input each step
    for pi in POKE_POOLS:
        I_ext[pools[pi]] = I_STIM if poking else 0.0

    # afferent: claw touch sensors (sensordata[27:33]) -> their pools
    touch = np.maximum(0.0, d.sensordata[27:33])
    for pi in range(6):
        I_ext[pools[pi]] += TOUCH_GAIN * touch[pi]

    leg_cnt[:] = 0; sez_cnt = 0
    for _ in range(SUB):
        fired_idx = brain_substep(t)
        if len(fired_idx):
            leg_cnt += np.array([np.isin(fired_idx, g).sum() for g in leg_groups])
            sez_cnt += int(np.isin(fired_idx, dn_sez).sum())
            if poking:
                poke_spikes += len(fired_idx)

    # rate estimates: exponential low-pass, r += (nsp - r*dt) / TAU_RATE
    r_leg += (leg_cnt - r_leg * DT_PHYS) / (TAU_RATE / 1000.0)
    r_sez += (sez_cnt - r_sez * DT_PHYS) / (TAU_RATE / 1000.0)

    if calibrating:
        calib_samples.append((r_leg.copy(), r_sez))
    else:
        # efferent: rates -> joint targets (position servos)
        for li, leg in enumerate(LEGS):
            delta = np.clip(GAIN_JOINT * (r_leg[li] - r_rest[li]), -CLAMP, CLAMP)
            for k, a in enumerate(act[leg]):
                d.ctrl[a] = base_joint[leg][k] + delta
        delta_s = np.clip(GAIN_JOINT * r_sez, -CLAMP, CLAMP)
        for k, a in enumerate(sez_act):
            d.ctrl[a] = base_sez[k] + delta_s

    mujoco.mj_step(m, d)

    # video: capture a frame every RENDER_EVERY steps, DURING the run
    if renderer is not None and step % RENDER_EVERY == 0:
        renderer.update_scene(d, camera="track2")
        frames.append(renderer.render().copy())
        if len(frames) % 50 == 0:
            print(f"  video frame {len(frames)} (t = {t:.2f} s)")

    if step % 100 == 0:
        ts.append(t)
        for leg in LEGS:
            for k, a in enumerate(act[leg]):
                legpos[leg][k].append(joint_pos(a))
        rates.append(r_leg.copy())



# ============================ 6. REPORT ============================
spk = np.array(spikes)
print(f"\nbrain spikes total: {len(spk)} | during poke: {poke_spikes}")
desc = spk[np.isin(spk[:, 1], desc_all)]
print(f"descending spikes  before poke: {(desc[:,0] < POKE_ON).sum()} | "
      f"during poke: {((desc[:,0] >= POKE_ON) & (desc[:,0] < POKE_OFF)).sum()} | "
      f"after: {(desc[:,0] >= POKE_OFF).sum()}")
print("\nleg movement (peak deviation from stance, per leg):")
best_leg, best_dev = None, 0.0
for li, leg in enumerate(LEGS):
    arrs = [np.abs(np.array(legpos[leg][k]) - base_joint[leg][k]) for k in range(8)]
    dev = max(a.max() for a in arrs)
    if dev > best_dev:
        best_leg, best_dev = leg, dev
    print(f"  {leg:10s} peak deviation {dev:.3f} rad")
print(f"-> most active leg: {best_leg} ({best_dev:.3f} rad)")

# ============================ 7. PLOTS ============================
rates_arr = np.array(rates)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
colors = ["#d62728", "#1f77b4", "#2ca02c"]
for pi in range(3):
    sel = part == pi
    tt = spk[np.isin(spk[:, 1], np.where(sel)[0]), 0]
    ax1.plot(tt, np.full(len(tt), pi), "|", color=colors[pi], ms=1.2, alpha=0.5)
ax1.axvspan(POKE_ON, POKE_OFF, color="gray", alpha=0.3)
ax1.set_yticks([0, 1, 2]); ax1.set_yticklabels(["sensory", "intrinsic", "descending"])
ax1.set_title(f"Embodied larva brain: {int((POKE_OFF-POKE_ON)*1000)} ms poke -> spikes -> legs twitch")
ax1.grid(alpha=0.3)

ax2.axvspan(POKE_ON, POKE_OFF, color="gray", alpha=0.3)
bl = best_leg if best_leg else "T2_right"
for k, name in enumerate(LEG_JOINTS[:2] + [LEG_JOINTS[5]]):
    c = ["#d62728", "#1f77b4", "#2ca02c"][k]
    ax2.plot(ts, np.array(legpos[bl][[0, 1, 5][k]]) - base_joint[bl][[0, 1, 5][k]],
             color=c, lw=1.4, label=f"{name} {bl} (delta)")
ax2.set_ylabel("joint angle delta (rad)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

for li in range(6):
    ax3.plot(ts, rates_arr[:, li], lw=1.2, label=f"leg {POOL_NAMES[li]}")
ax3.axvspan(POKE_ON, POKE_OFF, color="gray", alpha=0.3)
ax3.set_xlabel("time (s)"); ax3.set_ylabel("DN-VNC group rate (Hz)")
ax3.legend(fontsize=8, ncol=3); ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("embodied_result.png", dpi=130)
print("saved embodied_result.png")

# ============================ 8. SAVE VIDEO ============================
if VIDEO and frames:
    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames]
    dur = int(RENDER_EVERY * DT_PHYS * 1000)          # ms per frame
    pil_frames[0].save("embodied_fly.gif", save_all=True,
                       append_images=pil_frames[1:], duration=dur, loop=0)
    print(f"saved embodied_fly.gif ({len(frames)} frames, {dur} ms/frame)")
