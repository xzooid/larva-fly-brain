"""
wing_sweep.py -- can ANY naive flapping lift the fly? (no brain, just wings)
Sweeps wing parameters and measures max height over 1.0 s of flapping:
  * amplitude (0.8 / 1.5 / 2.5 rad)
  * 3-DOF wings (yaw/roll/pitch with insect-like phase offsets) vs pitch-only
  * wingbeat frequency (150 / 218 / 300 Hz)
Prints max height z; airborne = z > 0.01 (standing z ~ -0.005).
"""
import numpy as np
import mujoco
import os, sys

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
T = 1.0
wing_acts = {s: {dof: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"wing_{dof}_{s}")
                 for dof in ("yaw", "roll", "pitch")}
             for s in ("left", "right")}

# insect-like phase offsets: pitch ~0, roll ~+90deg, yaw ~+180deg (approx)
def run(label, f, amp, three_dof):
    d = mujoco.MjData(m)
    # settle 0.3 s
    for _ in range(int(0.3 / DT)):
        mujoco.mj_step(m, d)
    z0 = float(d.qpos[2])
    zmax = z0
    for step in range(int(T / DT)):
        t = step * DT
        for s in ("left", "right"):
            d.ctrl[wing_acts[s]["pitch"]] = amp * np.sin(2 * np.pi * f * t)
            if three_dof:
                d.ctrl[wing_acts[s]["roll"]]  = amp * 0.6 * np.sin(2 * np.pi * f * t + np.pi / 2)
                d.ctrl[wing_acts[s]["yaw"]]   = amp * 0.3 * np.sin(2 * np.pi * f * t + np.pi)
        mujoco.mj_step(m, d)
        z = float(d.qpos[2])
        if z > zmax:
            zmax = z
        if not np.isfinite(d.qpos).all():
            return f"{label}: NaN at t={t:.2f}"
    dz = zmax - z0
    verdict = "AIRBORNE!" if zmax > 0.01 else "on ground"
    return f"{label}: z0={z0:+.4f} zmax={zmax:+.4f} dz={dz:+.4f}  {verdict}"

print(f"fly mass (approx): sum of geom masses = {sum(m.geom_mass):.2e} kg" if False else
      f"total fly mass = {m.body_mass.sum():.2e} kg")
for f in (150, 218, 300):
    for amp in (0.8, 1.5, 2.5):
        for three in (False, True):
            lbl = f"f={f:3d}Hz amp={amp:.1f} 3dof={int(three)}"
            print(run(lbl, f, amp, three), flush=True)
