"""
flight_test.py -- will it fly? balance? fall? die?
================================================================================
Three experiments on the (aerodynamics-enabled) fly:

  A) HOVER ATTEMPT  : flap wings at 218 Hz (flybody's flight params), measure
                      height + body tilt. Does it leave the ground?
  B) POKE IN FLIGHT : same flapping, but poke the brain's sensory pools
                      mid-air -> descending drive to the legs. Does it keep
                      balancing, or tumble and fall?
  C) SEIZURE ("die"): on the ground, gain 20 + poke all sensory pools -> the
                      brain seizes (we know this regime). Does the fly convulse
                      and collapse? Physics still finite?

AERODYNAMICS: the base floor.xml has zero fluid coefficients, so we enable the
flybody flight config: fluidcoef=[1.0, 0.5, 1.5, 1.7, 1.0] on the wing geoms,
wing actuator gainprm=18, wing joint damping ~0.0078, timestep 1e-4 s (100 samples per 218 Hz wingbeat)
(flight needs a finer dt; flybody's own flight envs use 5e-5).

Output: flight_results.png (height/tilt curves) + metrics + optional GIFs.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO = ("--video" in sys.argv) or os.environ.get("MUJOCO_VIDEO") == "1"

# ============================ BRAIN ============================
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

# ============================ BODY (patched for flight) ============================
import mujoco
CANDIDATES = ["flybody_assets", "flybody/flybody/fruitfly/assets",
              "../flybody/flybody/fruitfly/assets",
              "~/flybrain/flybody/flybody/fruitfly/assets"]
ASSETS = next(p for p in CANDIDATES if os.path.exists(os.path.expanduser(p) + "/floor.xml"))
m = mujoco.MjModel.from_xml_path(f"{ASSETS}/floor.xml")

# --- enable flight ---
m.opt.timestep = 1e-4
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

DT_PHYS = float(m.opt.timestep)
print(f"flight config: dt = {DT_PHYS*1e6:.0f} us, wing fluid + gain 18")

LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2"]
LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
act = {leg: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_{leg}")
             for j in LEG_JOINTS] for leg in LEGS}
SEZ_ACTS = ["head_abduct", "head_twist", "head", "rostrum",
            "haustellum_abduct", "haustellum", "abdomen_abduct", "abdomen"]
sez_act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in SEZ_ACTS]
wing_pitch = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"wing_pitch_{s}")
              for s in ("left", "right")]

def joint_pos(d, aid):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(d.qpos[m.jnt_qposadr[jid]])

def body_state(d):
    z = float(d.qpos[2])
    q = d.qpos[3:7]
    # body "up" axis = rotate (0,0,1) by quaternion (w,x,y,z)
    ux = 2 * (q[1] * q[3] + q[0] * q[2])
    uy = 2 * (q[2] * q[3] - q[0] * q[1])
    uz = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    tilt = float(np.degrees(np.arccos(np.clip(uz, -1, 1))))
    return z, tilt, (ux, uy, uz)

SUB = 4
DT_B = DT_PHYS * 1000.0 / SUB
TAU, TAU_SYN, THETA = 10.0, 2.0, 1.0
REF = int(2.0 / DT_B)
TAU_RATE = 50.0
GAIN_JOINT = 0.006
CLAMP = 1.0
I_STIM = 2.0
TOUCH_GAIN = 0.05
FLAP_F = 200.0
FLAP_A = 1.0

# ============================ RUN ============================
def run(name, gain, poke_pools, flap, poke_on, poke_off, T, video_name=None):
    W = W_raw.copy()
    rowsum = W.sum(axis=1); rowsum[rowsum == 0] = 1.0
    Wn = (W / rowsum[:, None]) * gain

    d = mujoco.MjData(m)
    V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
    refrac = np.zeros(n, dtype=np.int32)
    spikes = []
    r_leg = np.zeros(6); r_sez = 0.0
    leg_cnt = np.zeros(6, dtype=np.int64); sez_cnt = 0
    leg_dev = np.zeros(6)
    ts, zs, tilts = [], [], []
    nan_hit = False

    renderer = None
    frames = []
    if VIDEO and video_name:
        try:
            renderer = mujoco.Renderer(m, 480, 640)
        except Exception as e:
            print(f"  [{name}] video disabled: {e}")

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

    # settle grounded, brain idle
    for _ in range(int(0.3 / DT_PHYS)):
        for _ in range(SUB):
            substep(0.0)
        mujoco.mj_step(m, d)
    base_joint = {leg: np.array([joint_pos(d, a) for a in act[leg]]) for leg in LEGS}
    base_sez = np.array([joint_pos(d, a) for a in sez_act])

    for step in range(int(T / DT_PHYS)):
        t = step * DT_PHYS
        poking = poke_on <= t < poke_off
        I_ext[:] = 0.0
        for pi in poke_pools:
            I_ext[pools[pi]] = I_STIM if poking else 0.0
        touch = np.maximum(0.0, d.sensordata[27:33])
        for pi in range(6):
            I_ext[pools[pi]] += TOUCH_GAIN * touch[pi]

        # flapping wings: 3-DOF with insect-like phase offsets
        # (pitch ~0, roll +90 deg, yaw +180 deg -- from the wing sweep:
        #  pitch-only barely lifts; 3-DOF launches)
        if flap:
            w = FLAP_A * np.sin(2 * np.pi * FLAP_F * t)
            for side in ("left", "right"):
                d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"wing_pitch_{side}")] = w
                d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"wing_roll_{side}")]  = 0.6 * FLAP_A * np.sin(2 * np.pi * FLAP_F * t + np.pi / 2)
                d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"wing_yaw_{side}")]   = 0.3 * FLAP_A * np.sin(2 * np.pi * FLAP_F * t + np.pi)

        leg_cnt[:] = 0; sez_cnt = 0
        for _ in range(SUB):
            fi = substep(t)
            if len(fi):
                leg_cnt += np.array([np.isin(fi, g).sum() for g in leg_groups])
                sez_cnt += int(np.isin(fi, dn_sez).sum())
        r_leg += (leg_cnt - r_leg * DT_PHYS) / (TAU_RATE / 1000.0)
        r_sez += (sez_cnt - r_sez * DT_PHYS) / (TAU_RATE / 1000.0)

        for li, leg in enumerate(LEGS):
            delta = float(np.clip(GAIN_JOINT * r_leg[li], -CLAMP, CLAMP))
            for k, a in enumerate(act[leg]):
                d.ctrl[a] = base_joint[leg][k] + delta
        ds = float(np.clip(GAIN_JOINT * r_sez, -CLAMP, CLAMP))
        for k, a in enumerate(sez_act):
            d.ctrl[a] = base_sez[k] + ds

        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            nan_hit = True
            break

        if renderer is not None and step % int(0.02 / DT_PHYS) == 0:
            renderer.update_scene(d, camera="side")
            frames.append(renderer.render().copy())

        if step % int(0.01 / DT_PHYS) == 0:
            z, tilt, _ = body_state(d)
            ts.append(t); zs.append(z); tilts.append(tilt)
            for li, leg in enumerate(LEGS):
                for k, a in enumerate(act[leg]):
                    dev = abs(joint_pos(d, a) - base_joint[leg][k])
                    if dev > leg_dev[li]:
                        leg_dev[li] = dev

    spk = np.array(spikes).reshape(-1, 2) if spikes else np.empty((0, 2))
    desc = spk[np.isin(spk[:, 1], desc_all)]
    desc_during = int(((desc[:, 0] >= poke_on) & (desc[:, 0] < poke_off)).sum())

    if renderer is not None and frames:
        from PIL import Image
        g = [Image.fromarray(f) for f in frames]
        g[0].save(video_name, save_all=True, append_images=g[1:],
                  duration=int(0.02 / DT_PHYS * DT_PHYS * 1000), loop=0)
        print(f"  [{name}] saved {video_name} ({len(frames)} frames)")

    return {"name": name, "ts": np.array(ts), "zs": np.array(zs),
            "tilts": np.array(tilts), "desc_during": desc_during,
            "leg_dev": leg_dev, "nan": nan_hit}

# ============================ A/B/C ============================
print("\n=== A: hover attempt (flap 218 Hz, no poke) ===")
A = run("hover", 8.0, [], True, 99.0, 99.0, 2.0, "flight_hover.gif")
print(f"  max height z = {A['zs'].max():.4f} m  (standing z ~ -0.005)  "
      f"airborne: {bool(A['zs'].max() > 0.01)}  | tilt max {A['tilts'].max():.1f} deg | NaN: {A['nan']}")

print("\n=== B: poke in flight (poke right side mid-air) ===")
B = run("poke-flight", 8.0, [1, 3, 5], True, 1.0, 1.4, 2.5, "flight_poke.gif")
print(f"  max height z = {B['zs'].max():.4f} m  | desc spikes during poke: {B['desc_during']}")
z_after = B['zs'][B['ts'] > 2.3]
print(f"  height at end: {z_after.mean():.4f} m  | tilt max {B['tilts'].max():.1f} deg | NaN: {B['nan']}")

print("\n=== C: seizure on the ground (gain 20, poke all) ===")
C = run("seizure", 20.0, [0, 1, 2, 3, 4, 5], False, 1.0, 1.4, 2.0, "seizure.gif")
print(f"  desc spikes during poke: {C['desc_during']}  | NaN: {C['nan']}")
print(f"  leg deviations: {np.round(C['leg_dev'], 3)}")
print(f"  tilt max: {C['tilts'].max():.1f} deg  | min z: {C['zs'].min():.4f} m")

# ============================ PLOT ============================
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

ax = axes[0]
ax.plot(A["ts"], A["zs"], color="#1f77b4", lw=1.5, label="hover attempt (flap only)")
ax.plot(B["ts"], B["zs"], color="#d62728", lw=1.5, label="poke in flight")
ax.axvspan(1.5, 1.9, color="gray", alpha=0.25, label="poke")
ax.axhline(0.0, color="k", ls="--", lw=0.8)
ax.set_ylabel("body height z (m)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title("A/B: will it fly? (wings enabled: fluidcoef, gain 18, 218 Hz flap)")

ax = axes[1]
ax.plot(A["ts"], A["tilts"], color="#1f77b4", lw=1.5, label="hover attempt")
ax.plot(B["ts"], B["tilts"], color="#d62728", lw=1.5, label="poke in flight")
ax.axvspan(1.5, 1.9, color="gray", alpha=0.25)
ax.set_ylabel("body tilt from upright (deg)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(C["ts"], C["zs"], color="#2ca02c", lw=1.5, label="height z")
ax.plot(C["ts"], C["tilts"], color="#ff7f0e", lw=1.5, label="tilt (deg)")
ax.axvspan(1.0, 1.4, color="gray", alpha=0.25, label="poke (gain 20)")
ax.set_xlabel("time (s)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title("C: seizure regime (gain 20, all pools) -- convulse? collapse?")

plt.tight_layout()
plt.savefig("flight_results.png", dpi=130)
print("\nsaved flight_results.png")

