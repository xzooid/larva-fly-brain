"""
diagnose_saturation.py -- is T1_left's 0.769 rad a real signal or a joint-limit ceiling?

Run this instead of the full experiment: it drives ONLY T1_left with a big
fake `delta` (bypassing the brain entirely) and checks each of its 8 joints
against MuJoCo's jnt_range. If any joint's qpos lands ON its limit, that's
your answer.
"""
import os
import numpy as np
import mujoco

CANDIDATES = ["flybody/flybody/fruitfly/assets",
              "../flybody/flybody/fruitfly/assets",
              "~/flybrain/flybody/flybody/fruitfly/assets",
              "~/flybrain/flybody_assets"]
ASSETS = next(p for p in CANDIDATES if os.path.exists(os.path.expanduser(p) + "/floor.xml"))
m = mujoco.MjModel.from_xml_path(f"{ASSETS}/floor.xml")
d = mujoco.MjData(m)

LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2"]
leg = "T1_left"
act_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_{leg}") for j in LEG_JOINTS]

print(f"{'joint':16s} {'jnt_range (rad)':22s} {'range width':>12s}")
for jname, aid in zip(LEG_JOINTS, act_ids):
    jid = m.actuator_trnid[aid][0]
    lo, hi = m.jnt_range[jid]
    limited = bool(m.jnt_limited[jid])
    width = hi - lo if limited else float("inf")
    flag = "  <-- narrower than 0.769*2!" if limited and width < 1.6 else ""
    print(f"{jname:16s} limited={limited!s:6s} [{lo:+.3f}, {hi:+.3f}]  width={width:6.3f}{flag}")

# settle, then slam every T1_left joint hard in the positive direction
for _ in range(1000):
    mujoco.mj_step(m, d)
base = [float(d.qpos[m.jnt_qposadr[m.actuator_trnid[a][0]]]) for a in act_ids]

print("\ndriving T1_left joints to base + 2.0 rad (way past CLAMP=1.0)...")
for a in act_ids:
    jid = m.actuator_trnid[a][0]
    b = float(d.qpos[m.jnt_qposadr[jid]])
    d.ctrl[a] = b + 2.0
for _ in range(500):
    mujoco.mj_step(m, d)

print(f"\n{'joint':16s} {'base':>8s} {'commanded':>10s} {'actual':>8s} {'deviation':>10s} {'AT LIMIT?':>10s}")
for jname, aid, b0 in zip(LEG_JOINTS, act_ids, base):
    jid = m.actuator_trnid[aid][0]
    actual = float(d.qpos[m.jnt_qposadr[jid]])
    dev = abs(actual - b0)
    lo, hi = m.jnt_range[jid]
    at_limit = m.jnt_limited[jid] and (actual <= lo + 1e-3 or actual >= hi - 1e-3)
    print(f"{jname:16s} {b0:8.3f} {b0+2.0:10.3f} {actual:8.3f} {dev:10.3f} {'YES' if at_limit else 'no':>10s}")
