"""
body_check.py -- meet the fly: load the MuJoCo body, list its parts, drop it.
The fly (flybody, TuragaLab/DeepMind) is a 67-body adult Drosophila:
  6 legs (T1/T2/T3, left/right) -- each: coxa, femur, tibia, 4 tarsus + claw
  + head, antennae, wings, halteres, 8-segment abdomen

  * 70 "general" actuators = MUSCLES: position servos (force = gain*(ctrl - qpos)).
    We command a target joint angle via ctrl, MuJoCo does the rest.
  * 8 "adhesion" actuators = sticky claws/labrum (like real pulvilli).
  * 15 sensors: accelerometer/gyro/velocimeter (thorax) + force_tarsus (6, foot
    load) + touch_claw (6, claw contact). Sensors = AFFERENT, actuators = EFFERENT.
"""
import os
import numpy as np
import mujoco

CANDIDATES = [
    "flybody/flybody/fruitfly/assets",
    "../flybody/flybody/fruitfly/assets",
    "~/flybrain/flybody/flybody/fruitfly/assets",
]
ASSETS = next(p for p in CANDIDATES if os.path.exists(os.path.expanduser(p) + "/floor.xml"))

m = mujoco.MjModel.from_xml_path(f"{ASSETS}/floor.xml")   # fly + floor
d = mujoco.MjData(m)
print(f"model loaded: {m.nbody} bodies, {m.nq} joint-DOFs, "
      f"{m.nu} actuators, {m.nsensor} sensors")

print("\n--- MUSCLES (actuators) ---")
for i in range(m.nu):
    print(f"  {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

print("\n--- SENSE ORGANS (sensors) ---")
for i in range(m.nsensor):
    print(f"  {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)}")

print("\n--- leg joints (each of 6 legs) ---")
print("  coxa_abduct / coxa_twist / coxa / femur_twist / femur / tibia / tarsus / tarsus2")

# ---------------- drop the fly: zero control, 2 s of physics ----------------
for _ in range(1000):                    # 1000 x 0.002 s = 2 s
    mujoco.mj_step(m, d)

z = float(d.qpos[2])
print(f"\nafter 2 s: fly free-joint z = {z:.4f} (standing on floor ~0)")
print("finite check:", bool(np.isfinite(d.qpos).all()))
touch = d.sensordata[6:]                 # 6 claw touch sensors (last 6)
print("claw touch forces:", np.round(touch, 3))

# ---------------- wiggle test: lift the right T2 leg ----------------
jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "tibia_T2_right")
aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "tibia_T2_right")
q0 = float(d.qpos[m.jnt_qposadr[jid]])
print(f"\ntibia_T2_right: q0 = {q0:.3f} rad")
for target in [q0 + 1.0, q0 - 0.5, q0]:
    d.ctrl[aid] = target
    for _ in range(250):
        mujoco.mj_step(m, d)
    print(f"  commanded {target:+.3f} -> joint now {float(d.qpos[m.jnt_qposadr[jid]]):+.3f} rad")

print("\nwiggle test done: the fly's leg followed our commands")
