"""
thrust_calib.py -- "wind tunnel": measure wing thrust vs amplitude.
================================================================================
Release the fly from rest in mid-air (z = 0.5, upright, no ground), flap at a
given amplitude, and fit the first ~30 ms of z(t) to z0 + v0 t + 0.5 a0 t^2.
Then  net_aero_force = m * (a0 + g)  -->  the thrust produced by that amplitude.
The HOVER amplitude is where thrust == weight (a0 == 0).

Output: thrust_calib.png (thrust vs amplitude, weight line) + hover amp.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

CANDIDATES = ["flybody_assets", "flybody/flybody/fruitfly/assets",
              "../flybody/flybody/fruitfly/assets",
              "~/flybrain/flybody/flybody/fruitfly/assets"]
ASSETS = next(p for p in CANDIDATES if os.path.exists(os.path.expanduser(p) + "/floor.xml"))
m = mujoco.MjModel.from_xml_path(f"{ASSETS}/floor.xml")
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

DT = float(m.opt.timestep)
MASS = float(m.body_mass.sum())
G = 9.81
WEIGHT = MASS * G
print(f"fly mass = {MASS:.2e} kg | weight = {WEIGHT:.2e} N")

wing_a = {s: {dof: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"wing_{dof}_{s}")
              for dof in ("yaw", "roll", "pitch")} for s in ("left", "right")}

def thrust_at(amp, f=200.0):
    d = mujoco.MjData(m)
    # release from rest, upright, mid-air
    d.qpos[0] = 0.0; d.qpos[1] = 0.0; d.qpos[2] = 0.5
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    ts, zs = [], []
    for step in range(300):                      # 30 ms
        t = step * DT
        w = 2 * np.pi * f * t
        for side in ("left", "right"):
            d.ctrl[wing_a[side]["pitch"]] = amp * np.sin(w)
            d.ctrl[wing_a[side]["roll"]] = 0.6 * amp * np.sin(w + np.pi / 2)
            d.ctrl[wing_a[side]["yaw"]] = 0.3 * amp * np.sin(w + np.pi)
        mujoco.mj_step(m, d)
        ts.append(t); zs.append(float(d.qpos[2]))
    # fit z(t) ~ z0 + v0 t + 0.5 a0 t^2
    A = np.vstack([np.ones_like(ts), ts, 0.5 * np.array(ts) ** 2]).T
    coef, *_ = np.linalg.lstsq(A, np.array(zs), rcond=None)
    a0 = coef[2]
    thrust = MASS * (a0 + G)
    return thrust, a0

amps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
thrusts = []
print(f"{'amp':>5s} {'thrust (N)':>12s} {'a0 (m/s2)':>10s}")
for amp in amps:
    th, a0 = thrust_at(amp)
    thrusts.append(th)
    print(f"{amp:5.2f} {th:12.2e} {a0:10.2f}")

thrusts = np.array(thrusts)
amps_a = np.array(amps)
# hover amp: where thrust curve crosses weight
hover = float(np.interp(WEIGHT, thrusts, amps_a)) if thrusts.min() < WEIGHT < thrusts.max() else None
print(f"\nweight = {WEIGHT:.2e} N -> hover amplitude = {hover:.2f} rad" if hover else
      f"\nweight {WEIGHT:.2e} N outside measured thrust range {thrusts.min():.2e}-{thrusts.max():.2e}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(amps_a, thrusts * 1e3, "o-", color="#1f77b4", label="measured thrust")
ax.axhline(WEIGHT * 1e3, color="r", ls="--", lw=1.2, label=f"weight ({WEIGHT*1e3:.2f} mN)")
if hover:
    ax.axvline(hover, color="g", ls=":", lw=1.2, label=f"hover amp = {hover:.2f} rad")
ax.set_xlabel("wing amplitude (rad)")
ax.set_ylabel("thrust (mN)")
ax.set_title("Wing thrust vs amplitude (200 Hz, 3-DOF) -- the 'wind tunnel'")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("thrust_calib.png", dpi=130)
print("saved thrust_calib.png")
