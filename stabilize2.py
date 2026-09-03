"""
stabilize2.py -- focused retry: hover trim + symmetric gyro controller.
Usage:
    python stabilize2.py trim [amp]                 # no controller, find gentle flight
    python stabilize2.py stab  [amp] [kp] [kd] [poke]
Prints: max z, max tilt, tilt at end, z at end.
Controller: amplitude modulation SYMMETRIC about base (total thrust constant),
clamped to +/-0.5, plus wing-roll mean offset for pitch, gyro damping.
"""
import os
import sys
import numpy as np
import mujoco

MODE = sys.argv[1] if len(sys.argv) > 1 else "trim"
BASE_AMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
KP = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
KD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
POKE = len(sys.argv) > 5 and sys.argv[5] == "poke"
AIR = len(sys.argv) > 6 and sys.argv[6] == "air"

# ---------------- brain ----------------
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
pools = np.array_split(np.random.default_rng(42).permutation(sens), 6)
leg_groups = np.array_split(np.random.default_rng(7).permutation(dn_vnc), 6)

# ---------------- body ----------------
CANDIDATES = ["flybody_assets", "flybody/flybody/fruitfly/assets",
              "../flybody/flybody/fruitfly/assets",
              "~/flybrain/flybody/flybody/fruitfly/assets"]
ASSETS = next(p for p in CANDIDATES if os.path.exists(os.path.expanduser(p) + "/floor.xml"))
m = mujoco.MjModel.from_xml_path(f"{ASSETS}/floor.xml")
m.opt.timestep = 5e-5
WING_FLUID = [1.0, 0.5, 1.5, 1.7, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "fluid" in nm:
        m.geom_fluid[g] = WING_FLUID
for a in range(m.nu):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
    if nm.startswith("wing_"):
        m.actuator_gainprm[a, 0] = 18.0
for j in range(m.njnt):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
    if nm.startswith("wing_"):
        m.dof_damping[m.jnt_dofadr[j]] = 0.00777

DT = float(m.opt.timestep)
LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2"]
act = {leg: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_{leg}")
             for j in LEG_JOINTS] for leg in LEGS}
SEZ_ACTS = ["head_abduct", "head_twist", "head", "rostrum",
            "haustellum_abduct", "haustellum", "abdomen_abduct", "abdomen"]
sez_act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in SEZ_ACTS]
wing_a = {s: {dof: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"wing_{dof}_{s}")
              for dof in ("yaw", "roll", "pitch")} for s in ("left", "right")}

def joint_pos(d, aid):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(d.qpos[m.jnt_qposadr[jid]])

def rot_vec_quat(v, q):
    w, x, y, z = q
    t = 2.0 * np.cross((x, y, z), v)
    return v + w * t + np.cross((x, y, z), t)

SUB = 4
DT_B = DT * 1000.0 / SUB
TAU, TAU_SYN, THETA = 10.0, 2.0, 1.0
REF = int(2.0 / DT_B)
TAU_RATE = 50.0
GAIN_JOINT = 0.006
CLAMP = 1.0
I_STIM = 2.0
TOUCH_GAIN = 0.05
FLAP_F = 200.0
POKE_ON, POKE_OFF = 1.5, 1.9
T = 3.5 if POKE else 3.0

d = mujoco.MjData(m)
if AIR:                                          # flight-arena release
    d.qpos[0] = 0.0; d.qpos[1] = 0.0; d.qpos[2] = 0.4
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
refrac = np.zeros(n, dtype=np.int32)
spikes = []
r_leg = np.zeros(6); r_sez = 0.0
leg_cnt = np.zeros(6, dtype=np.int64); sez_cnt = 0
desc_during = 0
ts, zs, tilts = [], [], []

def substep(t):
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

# build Wn
rowsum = W_raw.sum(axis=1); rowsum[rowsum == 0] = 1.0
Wn = (W_raw / rowsum[:, None]) * 8.0

# settle (skip ground contact in air mode)
if not AIR:
    for _ in range(int(0.3 / DT)):
        for _ in range(SUB):
            substep(0.0)
        mujoco.mj_step(m, d)
base_joint = {leg: np.array([joint_pos(d, a) for a in act[leg]]) for leg in LEGS}
base_sez = np.array([joint_pos(d, a) for a in sez_act])

for step in range(int(T / DT)):
    t = step * DT
    poking = POKE and POKE_ON <= t < POKE_OFF
    I_ext[:] = 0.0
    if poking:
        for pi in [1, 3, 5]:
            I_ext[pools[pi]] = I_STIM
    touch = np.maximum(0.0, d.sensordata[27:33])
    for pi in range(6):
        I_ext[pools[pi]] += TOUCH_GAIN * touch[pi]

    # ---- controller ----
    if MODE == "stab":
        q = d.qpos[3:7]
        up = rot_vec_quat(np.array([0.0, 0.0, 1.0]), q)
        e_world = np.cross(up, np.array([0.0, 0.0, 1.0]))
        e_body = rot_vec_quat(e_world, np.array([q[0], -q[1], -q[2], -q[3]]))
        gyr = d.sensordata[3:6]
        c = float(np.clip(KP * e_body[0] - KD * gyr[0], -0.5, 0.5))
        pc = float(np.clip(KP * e_body[1] - KD * gyr[1], -0.5, 0.5))
    else:
        c, pc = 0.0, 0.0

    ramp = min(1.0, t / (0.5 if AIR else 1.0))   # soft start
    amp_t = (0.0 if AIR else 0.3) + (BASE_AMP - (0.0 if AIR else 0.3)) * ramp
    w = 2 * np.pi * FLAP_F * t
    for side, sgn in (("left", -1), ("right", 1)):
        amp = amp_t * (1 + sgn * c)             # symmetric: total = 2*amp_t
        d.ctrl[wing_a[side]["pitch"]] = amp * np.sin(w)
        d.ctrl[wing_a[side]["roll"]] = pc + 0.6 * BASE_AMP * np.sin(w + np.pi / 2)
        d.ctrl[wing_a[side]["yaw"]] = 0.3 * BASE_AMP * np.sin(w + np.pi)

    leg_cnt[:] = 0; sez_cnt = 0
    for _ in range(SUB):
        fi = substep(t)
        if len(fi):
            leg_cnt += np.array([np.isin(fi, g).sum() for g in leg_groups])
            sez_cnt += int(np.isin(fi, dn_sez).sum())
            if poking:
                desc_during += int(np.isin(fi, desc_all).sum())
    r_leg += (leg_cnt - r_leg * DT) / (TAU_RATE / 1000.0)
    r_sez += (sez_cnt - r_sez * DT) / (TAU_RATE / 1000.0)
    for li, leg in enumerate(LEGS):
        delta = float(np.clip(GAIN_JOINT * r_leg[li], -CLAMP, CLAMP))
        for k, a in enumerate(act[leg]):
            d.ctrl[a] = base_joint[leg][k] + delta
    ds = float(np.clip(GAIN_JOINT * r_sez, -CLAMP, CLAMP))
    for k, a in enumerate(sez_act):
        d.ctrl[a] = base_sez[k] + ds

    mujoco.mj_step(m, d)
    if not np.isfinite(d.qpos).all():
        print("NaN at t =", round(t, 2)); break

    if step % int(0.01 / DT) == 0:
        z = float(d.qpos[2])
        q = d.qpos[3:7]
        up = rot_vec_quat(np.array([0.0, 0.0, 1.0]), q)
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        ts.append(t); zs.append(z); tilts.append(tilt)

zs = np.array(zs); tilts = np.array(tilts)
np.savez(f"trace_{MODE}_amp{BASE_AMP}_kp{KP}_poke{int(POKE)}.npz",
         ts=ts, zs=zs, tilts=tilts, desc_during=desc_during)
print(f"{MODE} amp={BASE_AMP} kp={KP} kd={KD} poke={POKE}: "
      f"max z {zs.max():+.3f} | max tilt {tilts.max():5.1f} deg | "
      f"tilt@end {tilts[-3:].mean():5.1f} | z@end {zs[-3:].mean():+.3f} | "
      f"desc@poke {desc_during}")
