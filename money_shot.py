"""
money_shot.py -- THE video: walk, take off, get poked mid-air, keep flying.
================================================================================
One continuous take, four phases (all in one 10-second run):

  t 0.0-3.6 s   WALK    : open-loop tripod gait (scripted leg pattern).
                          The brain is attached but quiet (no poke, touch
                          coupling disabled during the walk so the gait is
                          clean -- documented approximation).
  t 3.6-4.4 s   TAKE OFF: legs fold, wings ramp up (sub-stall amp), a hop
                          clears the ground before the downstroke hits.
  t 4.4-7.5 s   FLIGHT  : stable hover (passive aerodynamic stability).
                          Brain fully coupled: touch -> sensory, descending ->
                          legs (as in stabilize_video.py).
  t 7.5-7.9 s   POKE    : 400 ms current into right-side sensory pools.
                          Watch the legs kick mid-air and the fly recover.

Camera: fixed-orientation chase cam (never orbits). 333 frames @ 30 ms.

Env flags for testing (no video):
  MONEY_TEST=1 python money_shot.py      # prints metrics, no rendering
  TAKE_AMP=0.30 HOP=1 python money_shot.py   # tune takeoff

Run with video: MUJOCO_GL=glfw xvfb-run -a python money_shot.py --video
Output: money_shot.gif + money_shot_metrics.txt
"""
import os
import sys
import numpy as np
import mujoco

VIDEO = ("--video" in sys.argv) or os.environ.get("MUJOCO_VIDEO") == "1"
TEST = os.environ.get("MONEY_TEST") == "1"

# takeoff tuning knobs (env-overridable)
TAKE_AMP = float(os.environ.get("TAKE_AMP", "0.30"))
HOP = os.environ.get("HOP", "1") == "1"

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
rowsum = W_raw.sum(axis=1); rowsum[rowsum == 0] = 1.0
Wn = (W_raw / rowsum[:, None]) * 8.0
pools = np.array_split(np.random.default_rng(42).permutation(sens), 6)
leg_groups = np.array_split(np.random.default_rng(7).permutation(dn_vnc), 6)

# ============================ BODY ============================
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

# ============================ PARAMS ============================
WALK_F = 2.6                       # Hz gait
WALK_AMP = 0.25                    # leg swing amplitude (rad) -- tuned for stability
TAKEOFF_T0 = 3.6
FLAP_CRUISE = 0.25
POKE_ON, POKE_OFF = 7.5, 7.9
T_END = 10.0
I_STIM = 2.0
SUB = 4
DT_B = DT * 1000.0 / SUB
TAU, TAU_SYN, THETA = 10.0, 2.0, 1.0
REF = int(2.0 / DT_B)
TAU_RATE = 50.0
GAIN_JOINT = 0.006
CLAMP = 1.0
TOUCH_GAIN = 0.05
FRAME_EVERY = int(0.03 / DT)

# tripod groups for the gait
TRIPOD_A = [0, 3, 4]               # T1L, T2R, T3L
TRIPOD_B = [1, 2, 5]               # T1R, T2L, T3R

# ============================ STATE ============================
d = mujoco.MjData(m)
V = np.zeros(n); s = np.zeros(n); I_ext = np.zeros(n)
refrac = np.zeros(n, dtype=np.int32)
spikes = []
r_leg = np.zeros(6); r_sez = 0.0
leg_cnt = np.zeros(6, dtype=np.int64); sez_cnt = 0
desc_during = 0
ts, zs, tilts, xs = [], [], [], []

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

# settle grounded 0.5 s (brain idle)
for _ in range(int(0.5 / DT)):
    for _ in range(SUB):
        substep(0.0)
    mujoco.mj_step(m, d)
base_joint = {leg: np.array([joint_pos(d, a) for a in act[leg]]) for leg in LEGS}
base_sez = np.array([joint_pos(d, a) for a in sez_act])

renderer = None
frames = []
if VIDEO:
    try:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance = 0.5
        cam.azimuth = 90.0
        cam.elevation = 20.0
        print("renderer ready (chase cam)")
    except Exception as e:
        print("video disabled:", e)

def drive_legs(t, phase, base):
    """set leg ctrls for the current phase"""
    if phase == "walk":
        for li, leg in enumerate(LEGS):
            grpA = li in TRIPOD_A
            phi = 0.0 if grpA else np.pi
            swing = np.sin(2 * np.pi * WALK_F * t + phi)
            # coxa_abduct swings the leg fore/aft
            d.ctrl[act[leg][0]] = base[leg][0] + 0.22 * swing
            # femur + tibia lift during the back half of the swing
            lift = 0.5 * (1.0 + swing)
            d.ctrl[act[leg][4]] = base[leg][4] + 0.15 * lift      # femur
            d.ctrl[act[leg][5]] = base[leg][5] + 0.25 * lift      # tibia
    elif phase == "fold":        # tuck legs for takeoff
        for li, leg in enumerate(LEGS):
            for k in range(8):
                d.ctrl[act[leg][k]] = base[leg][k]
            d.ctrl[act[leg][5]] = base[leg][5] + 0.5              # tibia tuck
            d.ctrl[act[leg][4]] = base[leg][4] + 0.2              # femur tuck
    elif phase == "hop":         # explosive leg extension -> airborne
        for li, leg in enumerate(LEGS):
            d.ctrl[act[leg][4]] = base[leg][4] - 0.6
            d.ctrl[act[leg][5]] = base[leg][5] - 0.9
    else:                        # flight: brain-driven (same as stabilize)
        for li, leg in enumerate(LEGS):
            delta = float(np.clip(GAIN_JOINT * r_leg[li], -CLAMP, CLAMP))
            for k, a in enumerate(act[leg]):
                d.ctrl[a] = base[leg][k] + delta

def drive_sez(t, phase, base):
    if phase == "flight":
        ds = float(np.clip(GAIN_JOINT * r_sez, -CLAMP, CLAMP))
        for k, a in enumerate(sez_act):
            d.ctrl[a] = base[k] + ds
    else:
        for k, a in enumerate(sez_act):
            d.ctrl[a] = base[k]

# ============================ MAIN LOOP ============================
n_steps = int(T_END / DT)
hop_done = False
wing_amp = 0.0
wing_phase_running = False
for step in range(n_steps):
    t = step * DT

    # ---- phase selection ----
    if t < TAKEOFF_T0:
        phase = "walk"
    elif t < TAKEOFF_T0 + 0.4:
        phase = "fold"                       # 3.6-4.0: tuck legs
    elif t < TAKEOFF_T0 + 0.55:
        phase = "hop" if HOP else "fold"     # 4.0-4.15: jump
        if HOP and not hop_done:
            hop_done = True
    else:
        phase = "flight"

    # wings: ramp up during fold/hop, cruise after
    if t >= TAKEOFF_T0:
        wing_amp = min(TAKE_AMP, wing_amp + (TAKE_AMP / 0.5) * DT)
    w = 2 * np.pi * 200.0 * t

    # ---- brain input ----
    poking = POKE_ON <= t < POKE_OFF
    I_ext[:] = 0.0
    if poking:
        for pi in [1, 3, 5]:
            I_ext[pools[pi]] = I_STIM
    if phase == "flight":
        touch = np.maximum(0.0, d.sensordata[27:33])
        for pi in range(6):
            I_ext[pools[pi]] += TOUCH_GAIN * touch[pi]

    # ---- actuators ----
    drive_legs(t, phase, base_joint)
    drive_sez(t, phase, base_sez)
    if phase in ("fold", "hop", "flight"):
        amp = TAKE_AMP if t < TAKEOFF_T0 + 1.0 else FLAP_CRUISE
        for side in ("left", "right"):
            d.ctrl[wing_a[side]["pitch"]] = amp * np.sin(w)
            d.ctrl[wing_a[side]["roll"]] = 0.6 * amp * np.sin(w + np.pi / 2)
            d.ctrl[wing_a[side]["yaw"]] = 0.3 * amp * np.sin(w + np.pi)
    else:
        for side in ("left", "right"):
            for dof in ("pitch", "roll", "yaw"):
                d.ctrl[wing_a[side][dof]] = 0.0

    # ---- brain step ----
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

    mujoco.mj_step(m, d)
    if not np.isfinite(d.qpos).all():
        print("NaN at t =", round(t, 2)); break

    # ---- capture ----
    if renderer is not None and step % FRAME_EVERY == 0:
        cam.lookat[:] = d.qpos[0:3]
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())
    if step % int(0.01 / DT) == 0:
        z = float(d.qpos[2])
        q = d.qpos[3:7]
        up = rot_vec_quat(np.array([0.0, 0.0, 1.0]), q)
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        ts.append(t); zs.append(z); tilts.append(tilt); xs.append(float(d.qpos[0]))

# ============================ REPORT ============================
zs = np.array(zs); tilts = np.array(tilts); xs = np.array(xs)
walk_end = int(np.searchsorted(ts, TAKEOFF_T0))
fly_mask = ts >= TAKEOFF_T0 + 0.6
dx_walk = xs[walk_end] - xs[0]
print(f"walk: dx = {dx_walk:.3f} m over {TAKEOFF_T0:.1f} s")
print(f"takeoff: max z = {zs.max():.3f} m | max tilt after 4.2s = "
      f"{tilts[ts >= 4.2].max():.1f} deg")
print(f"flight: z@7.5s = {zs[ts >= 7.5][0] if (ts>=7.5).any() else float('nan'):.3f} m, "
      f"tilt@7.5s = {tilts[ts >= 7.5][0] if (ts>=7.5).any() else float('nan'):.1f} deg")
print(f"poke: desc spikes during poke = {desc_during}")
print(f"end: tilt = {tilts[-3:].mean():.1f} deg, z = {zs[-3:].mean():+.3f} m, NaN = {not np.isfinite(zs).all()}")
with open("money_shot_metrics.txt", "w") as f:
    f.write(f"walk_dx={dx_walk:.3f}\ntakeoff_max_z={zs.max():.3f}\n"
            f"desc_during_poke={desc_during}\nend_tilt={tilts[-3:].mean():.1f}\n")

if renderer is not None and frames:
    from PIL import Image
    gif = [Image.fromarray(fr) for fr in frames]
    gif[0].save("money_shot.gif", save_all=True, append_images=gif[1:],
                duration=int(FRAME_EVERY * DT * 1000), loop=0)
    print(f"saved money_shot.gif ({len(frames)} frames)")
