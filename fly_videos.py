"""
fly_videos.py -- the lesion experiment, as videos.
Runs poke RIGHT under three conditions (with CORRECTED name-based joint reads),
captures video of each, and builds a side-by-side comparison GIF:
  1. baseline (intact brain)   -> poke_right.gif
  2. KC lesion (mushroom body) -> kc_lesion.gif
  3. LN lesion (inhibitory)    -> ln_lesion.gif
  + composite_lesions.gif (3 panels) + lesion_comparison.png + metrics table
Run:  MUJOCO_GL=glfw xvfb-run -a python fly_videos.py --video
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

LESIONS = {"KC": np.where(cell == "KC")[0], "LN": np.where(cell == "LN")[0]}
pools = np.array_split(np.random.default_rng(42).permutation(sens), 6)
leg_groups = np.array_split(np.random.default_rng(7).permutation(dn_vnc), 6)

# ============================ BODY ============================
import mujoco
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
SEZ_ACTS = ["head_abduct", "head_twist", "head", "rostrum",
            "haustellum_abduct", "haustellum", "abdomen_abduct", "abdomen"]
sez_act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in SEZ_ACTS]

def joint_pos(d, aid):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(d.qpos[m.jnt_qposadr[jid]])

# ============================ PARAMS (validated) ============================
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
RENDER_EVERY = 160

# ============================ RUN ONE CONDITION ============================
def run(name, lesion=None, video_name=None):
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
    leg_dev = np.zeros(6)
    desc_during = 0

    renderer = None
    frames = []
    if VIDEO and video_name:
        try:
            renderer = mujoco.Renderer(m, 480, 640)
            print(f"  [{name}] renderer ready")
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

    for _ in range(int(1.0 / DT_PHYS)):
        for _ in range(SUB):
            substep(0.0)
        mujoco.mj_step(m, d)
    base_joint = {leg: np.array([joint_pos(d, a) for a in act[leg]]) for leg in LEGS}
    base_sez = np.array([joint_pos(d, a) for a in sez_act])

    for step in range(int(T_TOTAL / DT_PHYS)):
        t = step * DT_PHYS
        poking = POKE_ON <= t < POKE_OFF
        I_ext[:] = 0.0
        for pi in [1, 3, 5]:
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
            delta = float(np.clip(GAIN_JOINT * r_leg[li], -CLAMP, CLAMP))
            for k, a in enumerate(act[leg]):
                d.ctrl[a] = base_joint[leg][k] + delta
        ds = float(np.clip(GAIN_JOINT * r_sez, -CLAMP, CLAMP))
        for k, a in enumerate(sez_act):
            d.ctrl[a] = base_sez[k] + ds
        mujoco.mj_step(m, d)

        if renderer is not None and step % RENDER_EVERY == 0:
            renderer.update_scene(d, camera="track2")
            frames.append(renderer.render().copy())

        for li, leg in enumerate(LEGS):
            for k, a in enumerate(act[leg]):
                dev = abs(joint_pos(d, a) - base_joint[leg][k])
                if dev > leg_dev[li]:
                    leg_dev[li] = dev

    spk = np.array(spikes).reshape(-1, 2) if spikes else np.empty((0, 2))
    desc = spk[np.isin(spk[:, 1], desc_all)]
    desc_before = int((desc[:, 0] < POKE_ON).sum())
    desc_after = int((desc[:, 0] >= POKE_OFF).sum())

    if renderer is not None and frames:
        from PIL import Image
        gif = [Image.fromarray(f) for f in frames]
        gif[0].save(video_name, save_all=True, append_images=gif[1:],
                    duration=int(RENDER_EVERY * DT_PHYS * 1000), loop=0)
        print(f"  [{name}] saved {video_name} ({len(frames)} frames)")

    return {"name": name, "desc_before": desc_before, "desc_during": desc_during,
            "desc_after": desc_after, "leg_dev": leg_dev, "frames": frames}

# ============================ RUN ALL ============================
results = []
for name, lesion, vn in [
    ("baseline (intact)", None, "poke_right.gif"),
    ("KC lesion", "KC", "kc_lesion.gif"),
    ("LN lesion", "LN", "ln_lesion.gif"),
]:
    print(f"=== {name} ===")
    results.append(run(name, lesion, vn))

# ============================ REPORT ============================
print("\n" + "=" * 72)
print(f"{'condition':20s} {'desc@poke':>9s} {'best leg':>10s} {'dev':>6s}")
print("-" * 72)
for r in results:
    best = int(np.argmax(r["leg_dev"]))
    print(f"{r['name']:20s} {r['desc_during']:9d} {LEGS[best]:>10s} {r['leg_dev'][best]:6.3f}")

devs = np.array([r["leg_dev"] for r in results])
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(6)
w = 0.27
for i, r in enumerate(results):
    ax.bar(x + (i - 1) * w, r["leg_dev"], w, label=r["name"])
ax.set_xticks(x); ax.set_xticklabels(LEGS, rotation=45, ha="right")
ax.set_ylabel("peak joint deviation (rad)")
ax.set_title("Lesion experiment: poke RIGHT, leg responses")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("lesion_comparison.png", dpi=130)
print("saved lesion_comparison.png")

# ============================ COMPOSITE GIF ============================
if VIDEO and all(r["frames"] for r in results):
    from PIL import Image
    nf = min(len(r["frames"]) for r in results)
    comp = []
    for i in range(nf):
        row = np.hstack([np.array(r["frames"][i]) for r in results])
        comp.append(Image.fromarray(row))
    comp[0].save("composite_lesions.gif", save_all=True, append_images=comp[1:],
                 duration=int(RENDER_EVERY * DT_PHYS * 1000), loop=0)
    print("saved composite_lesions.gif (baseline | KC lesion | LN lesion)")
