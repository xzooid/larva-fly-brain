"""
experiments.py -- the perturbation battery: does the wiring decide the behavior?
================================================================================
Runs the embodied fly under several conditions and compares them:

  1. control            (no poke)
  2. poke RIGHT         (sensory pools 1,3,5)
  3. poke LEFT          (sensory pools 0,2,4)
  4. poke BOTH          (all 6 pools)
  5. poke RIGHT + KC lesion   (remove the 144 Kenyon cells -- mushroom body)
  6. poke RIGHT + LN lesion   (remove the 110 LN inhibitory interneurons)

Per condition we measure:
  * descending spikes during the poke (the motor command volume)
  * peak joint deviation per leg (the behavior)
  * the most active leg, and a left/right asymmetry index

LESIONS: a lesioned neuron is removed entirely -- all its inputs AND outputs
are zeroed, then row normalization is redone (synaptic scaling of survivors).

Interpretation notes:
  * The sensory pools are seeded but arbitrary partitions of the 434 sensory
    neurons (the mirror data lacks per-leg identity), so "left vs right" tests
    the METHOD; the lesion conditions test real network structure.
  * Kenyon cells are the famous mushroom-body learning neurons; LN are
    inhibitory local interneurons (Dale's-law candidates from our earlier work).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

# ============================ 1. BRAIN DATA ============================
DATA = "data/fly_larva"
nodes = []
with open(f"{DATA}/nodes.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        nodes.append(p[4])
n = len(nodes)
cell = np.array(nodes)
sens = np.where(cell == "sensory")[0]
dn_vnc = np.where(cell == "DN-VNC")[0]
dn_sez = np.where(cell == "DN-SEZ")[0]
desc_all = np.concatenate([dn_vnc, dn_sez])

W_raw = np.zeros((n, n), dtype=np.float32)
with open(f"{DATA}/edges.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        W_raw[int(p[1]), int(p[0])] += int(p[2])       # W[post, pre]
np.fill_diagonal(W_raw, 0.0)

LESIONS = {
    "KC": np.where(cell == "KC")[0],       # Kenyon cells (mushroom body)
    "LN": np.where(cell == "LN")[0],       # local interneurons (inhibitory)
}
print(f"brain: {n} neurons | KC: {len(LESIONS['KC'])} | LN: {len(LESIONS['LN'])}")

# sensory pools (seeded, as in embodied_fly.py) and leg groups
POOL_NAMES = ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]
pools = np.array_split(np.random.default_rng(42).permutation(sens), 6)
leg_groups = np.array_split(np.random.default_rng(7).permutation(dn_vnc), 6)

# ============================ 2. BODY ============================
CANDIDATES = ["flybody_assets",
              "flybody/flybody/fruitfly/assets",
              "../flybody/flybody/fruitfly/assets",
              "~/flybrain/flybody/flybody/fruitfly/assets"]
ASSETS = next(p for p in CANDIDATES if os.path.exists(os.path.expanduser(p) + "/floor.xml"))
m = mujoco.MjModel.from_xml_path(f"{ASSETS}/floor.xml")
DT_PHYS = float(m.opt.timestep)

LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2"]
LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
act = {}
for leg in LEGS:
    act[leg] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_{leg}")
                for j in LEG_JOINTS]
    assert all(a >= 0 for a in act[leg])
SEZ_ACTS = ["head_abduct", "head_twist", "head", "rostrum",
            "haustellum_abduct", "haustellum", "abdomen_abduct", "abdomen"]
sez_act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in SEZ_ACTS]
assert all(a >= 0 for a in sez_act)

# ============================ 3. PARAMETERS (validated config) ============================
SUB = 4
DT_B = DT_PHYS * 1000.0 / SUB
TAU, TAU_SYN, THETA = 10.0, 2.0, 1.0
REF = int(2.0 / DT_B)
TAU_RATE = 50.0
GAIN_JOINT = 0.006
CLAMP = 1.0
I_STIM = 2.0
TOUCH_GAIN = 0.05
GAIN = 8.0
POKE_ON, POKE_OFF = 1.0, 1.4
T_TOTAL = 4.0

def joint_pos(d, aid):
    # read by NAME: actuator and joint share a name. actuator_trnid is WRONG
    # for tendon actuators (trnid[0] is a tendon id that collides with a joint id)
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(d.qpos[m.jnt_qposadr[jid]])

# ============================ 4. ONE CONDITION ============================
def run_condition(name, poke_pools, lesion=None):
    W = W_raw.copy()
    if lesion is not None:
        idx = LESIONS[lesion]
        W[:, idx] = 0.0                        # no inputs to lesioned neurons
        W[idx, :] = 0.0                        # no outputs from lesioned neurons
    rowsum = W.sum(axis=1); rowsum[rowsum == 0] = 1.0
    Wn = (W / rowsum[:, None]) * GAIN

    d = mujoco.MjData(m)                       # fresh physics state
    V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
    refrac = np.zeros(n, dtype=np.int32)
    spikes = []
    r_leg = np.zeros(6)
    r_sez = 0.0
    sez_cnt = 0
    leg_cnt = np.zeros(6, dtype=np.int64)

    def substep(t):
        nonlocal s, V
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

    # settle 1 s (brain idle, body drops onto floor)
    for _ in range(int(1.0 / DT_PHYS)):
        for _ in range(SUB):
            substep(0.0)
        mujoco.mj_step(m, d)
    base_joint = {leg: np.array([joint_pos(d, a) for a in act[leg]]) for leg in LEGS}
    base_sez = np.array([joint_pos(d, a) for a in sez_act])

    # main loop
    leg_dev = np.zeros(6)
    desc_during = 0
    for step in range(int(T_TOTAL / DT_PHYS)):
        t = step * DT_PHYS
        poking = POKE_ON <= t < POKE_OFF
        I_ext[:] = 0.0
        for pi in poke_pools:
            I_ext[pools[pi]] = I_STIM if poking else 0.0
        touch = np.maximum(0.0, d.sensordata[27:33])
        for pi in range(6):
            I_ext[pools[pi]] += TOUCH_GAIN * touch[pi]

        leg_cnt[:] = 0; sez_cnt = 0
        for _ in range(SUB):
            fi = substep(t)
            if len(fi):
                leg_cnt += np.array([np.isin(fi, g).sum() for g in leg_groups])
                sez_cnt += int(np.isin(fi, dn_sez).sum())
                if poking:
                    desc_during += int(np.isin(fi, desc_all).sum())
        r_leg += (leg_cnt - r_leg * DT_PHYS) / (TAU_RATE / 1000.0)
        r_sez += (sez_cnt - r_sez * DT_PHYS) / (TAU_RATE / 1000.0)

        for li, leg in enumerate(LEGS):
            delta = np.clip(GAIN_JOINT * r_leg[li], -CLAMP, CLAMP)
            for k, a in enumerate(act[leg]):
                d.ctrl[a] = base_joint[leg][k] + delta
        delta_s = np.clip(GAIN_JOINT * r_sez, -CLAMP, CLAMP)
        for k, a in enumerate(sez_act):
            d.ctrl[a] = base_sez[k] + delta_s
        mujoco.mj_step(m, d)

        for li, leg in enumerate(LEGS):
            for k, a in enumerate(act[leg]):
                dev = abs(joint_pos(d, a) - base_joint[leg][k])
                if dev > leg_dev[li]:
                    leg_dev[li] = dev

    spk = np.array(spikes).reshape(-1, 2) if spikes else np.empty((0, 2))
    desc = spk[np.isin(spk[:, 1], desc_all)]
    desc_before = int((desc[:, 0] < POKE_ON).sum())
    desc_after = int((desc[:, 0] >= POKE_OFF).sum())
    best = int(np.argmax(leg_dev))
    asym = (leg_dev[[1, 3, 5]].sum() - leg_dev[[0, 2, 4]].sum()) / \
           max(1e-6, leg_dev.sum())
    return {"name": name, "lesion": lesion,
            "desc_before": desc_before, "desc_during": desc_during, "desc_after": desc_after,
            "leg_dev": leg_dev, "best_leg": LEGS[best], "best_dev": leg_dev[best],
            "asym": asym}

# ============================ 5. RUN THE BATTERY ============================
CONDITIONS = [
    ("control (no poke)",       [],            None),
    ("poke RIGHT",              [1, 3, 5],     None),
    ("poke LEFT",               [0, 2, 4],     None),
    ("poke BOTH",               [0, 1, 2, 3, 4, 5], None),
    ("poke RIGHT + KC lesion",  [1, 3, 5],     "KC"),
    ("poke RIGHT + LN lesion",  [1, 3, 5],     "LN"),
]

results = []
for name, pools_, lesion in CONDITIONS:
    print(f"\n=== {name} ===")
    res = run_condition(name, pools_, lesion)
    results.append(res)
    print(f"  descending spikes  before {res['desc_before']} | during {res['desc_during']} | after {res['desc_after']}")
    print(f"  leg deviations: " + " ".join(f"{l}={v:.3f}" for l, v in zip(LEGS, res["leg_dev"])))
    print(f"  most active: {res['best_leg']} ({res['best_dev']:.3f} rad) | L/R asymmetry {res['asym']:+.2f}")

# ============================ 6. TABLE + FIGURE ============================
print("\n" + "=" * 78)
print(f"{'condition':26s} {'desc@poke':>9s} {'best leg':>10s} {'dev':>6s} {'asym':>6s}")
print("-" * 78)
for res in results:
    print(f"{res['name']:26s} {res['desc_during']:9d} {res['best_leg']:>10s} "
          f"{res['best_dev']:6.3f} {res['asym']:+6.2f}")

devs = np.array([r["leg_dev"] for r in results])          # conditions x legs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
im = ax1.imshow(devs, cmap="viridis", aspect="auto")
ax1.set_xticks(range(6)); ax1.set_xticklabels(LEGS, rotation=45, ha="right")
ax1.set_yticks(range(len(results)))
ax1.set_yticklabels([r["name"] for r in results])
ax1.set_title("Leg response by condition (peak deviation, rad)")
for i in range(len(results)):
    for j in range(6):
        ax1.text(j, i, f"{devs[i, j]:.2f}", ha="center", va="center",
                 color="w" if devs[i, j] < devs.max() / 2 else "k", fontsize=8)
fig.colorbar(im, ax=ax1, shrink=0.8)

ax2.barh(range(len(results)), [r["desc_during"] for r in results], color="#1f77b4")
ax2.set_yticks(range(len(results)))
ax2.set_yticklabels([r["name"] for r in results])
ax2.set_xlabel("descending spikes during poke")
ax2.set_title("Motor-command volume by condition")
ax2.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("experiment_battery.png", dpi=130)
print("\nsaved experiment_battery.png")
