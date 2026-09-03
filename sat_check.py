"""
sat_check.py -- is T1_left's identical 0.769 rad a saturation artifact?
Measures, per condition:
  * r_peak    : peak low-passed rate per leg group (the BRAIN-side signal)
  * grp_spk   : raw spike counts per leg group during the poke
  * delta_max : peak |delta| commanded to each leg's joints
  * for the max-deviation joint of each leg: base, qpos, joint range, at-limit?
  * ctrl clamp: whether d.ctrl was clamped by the actuator's ctrlrange
Reads joint positions by JOINT NAME (fixes the tarsus2 tendon-transmission
mapping problem found in diagnose_saturation.py).
"""
import os
import numpy as np
import mujoco

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
        W_raw[int(p[1]), int(p[0])] += int(p[2])
np.fill_diagonal(W_raw, 0.0)

LESIONS = {"KC": np.where(cell == "KC")[0], "LN": np.where(cell == "LN")[0]}
pools = np.array_split(np.random.default_rng(42).permutation(sens), 6)
leg_groups = np.array_split(np.random.default_rng(7).permutation(dn_vnc), 6)

CANDIDATES = ["flybody_assets", "flybody/flybody/fruitfly/assets",
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

def joint_q(d, leg, j):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{leg}")
    return float(d.qpos[m.jnt_qposadr[jid]])

def sez_joint_q(d, a):
    # SEZ actuators share their name with the joint they drive
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, a)
    return float(d.qpos[m.jnt_qposadr[jid]])

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

def run(name, poke_pools, lesion=None):
    W = W_raw.copy()
    if lesion is not None:
        idx = LESIONS[lesion]
        W[:, idx] = 0.0
        W[idx, :] = 0.0
    rowsum = W.sum(axis=1); rowsum[rowsum == 0] = 1.0
    Wn = (W / rowsum[:, None]) * GAIN

    d = mujoco.MjData(m)
    V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
    refrac = np.zeros(n, dtype=np.int32)
    spikes = []
    r_leg = np.zeros(6); r_sez = 0.0
    leg_cnt = np.zeros(6, dtype=np.int64); sez_cnt = 0
    grp_spk = np.zeros(6, dtype=np.int64)
    desc_during = 0
    r_peak = np.zeros(6)
    delta_max = np.zeros(6)
    dev_val = np.zeros(6); dev_joint = [None] * 6
    dev_q = np.zeros(6); dev_lim = np.zeros((6, 2)); dev_atlimit = np.zeros(6, dtype=bool)

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

    for _ in range(int(1.0 / DT_PHYS)):
        for _ in range(SUB):
            substep(0.0)
        mujoco.mj_step(m, d)
    base_joint = {leg: np.array([joint_q(d, leg, j) for j in LEG_JOINTS]) for leg in LEGS}
    base_sez = np.array([sez_joint_q(d, name) for name in SEZ_ACTS])

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
                    for li, g in enumerate(leg_groups):
                        grp_spk[li] += int(np.isin(fi, g).sum())
        r_leg += (leg_cnt - r_leg * DT_PHYS) / (TAU_RATE / 1000.0)
        r_sez += (sez_cnt - r_sez * DT_PHYS) / (TAU_RATE / 1000.0)
        r_peak = np.maximum(r_peak, r_leg)

        for li, leg in enumerate(LEGS):
            delta = float(np.clip(GAIN_JOINT * r_leg[li], -CLAMP, CLAMP))
            delta_max[li] = max(delta_max[li], abs(delta))
            for k, a in enumerate(act[leg]):
                d.ctrl[a] = base_joint[leg][k] + delta
        ds = float(np.clip(GAIN_JOINT * r_sez, -CLAMP, CLAMP))
        for k, a in enumerate(sez_act):
            d.ctrl[a] = base_sez[k] + ds
        mujoco.mj_step(m, d)

        for li, leg in enumerate(LEGS):
            for k, j in enumerate(LEG_JOINTS):
                q = joint_q(d, leg, j)
                dev = abs(q - base_joint[leg][k])
                if dev > dev_val[li]:
                    dev_val[li] = dev; dev_joint[li] = j
                    dev_q[li] = q
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{j}_{leg}")
                    lim = m.jnt_range[jid]
                    dev_lim[li] = lim
                    dev_atlimit[li] = bool(m.jnt_limited[jid]) and (q <= lim[0] + 0.005 or q >= lim[1] - 0.005)

    print(f"\n=== {name} ===")
    print(f"  descending during poke: {desc_during}")
    print(f"  group spikes during poke (per leg group): {grp_spk}")
    print(f"  peak rate r_leg (Hz):                    {np.round(r_peak, 1)}")
    print(f"  peak |delta| commanded (rad):            {np.round(delta_max, 3)}")
    for li, leg in enumerate(LEGS):
        atl = "AT-LIMIT" if dev_atlimit[li] else "in-range "
        print(f"  {leg:9s} max dev {dev_val[li]:.3f} at {dev_joint[li]:12s} "
              f"qpos {dev_q[li]:+.3f} range [{dev_lim[li][0]:+.2f}, {dev_lim[li][1]:+.2f}] {atl}")

for name, pools_, lesion in [
    ("poke RIGHT",             [1, 3, 5], None),
    ("poke LEFT",              [0, 2, 4], None),
    ("poke RIGHT + LN lesion", [1, 3, 5], "LN"),
]:
    run(name, pools_, lesion)
