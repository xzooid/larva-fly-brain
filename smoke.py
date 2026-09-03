import os
os.environ.setdefault("MUJOCO_GL", "osmesa")  # software rendering: works everywhere headless.
# To use your AMD GPU instead (needs a healthy EGL stack):  MUJOCO_GL=egl python smoke.py
import mujoco, numpy as np
from PIL import Image

# RED ball on a LIGHT floor, so it can't hide from us
XML = """<mujoco><worldbody>
<body><freejoint/><geom size="0.05" type="sphere" mass="0.1" rgba="1 0.15 0.1 1"/></body>
<geom size="1 1 0.01" type="plane" pos="0 0 -0.01" rgba="0.92 0.92 0.92 1"/>
</worldbody></mujoco>"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
for _ in range(5000): mujoco.mj_step(m, d)       # 10 s of physics
assert np.isfinite(d.qpos).all(), "physics exploded!"
print("ball settled at z =", round(d.qpos[2], 4))  # expect ~0.04

renderer = mujoco.Renderer(m, 480, 640)          # Renderer(model, height, width)

# Orbit camera: looks at the ball from 1 m away, 25 deg above horizontal
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(m, cam)
cam.lookat[:] = [0, 0, 0.05]
cam.distance = 1.0
cam.azimuth = 90
cam.elevation = 25

renderer.update_scene(d, camera=cam)             # camera goes HERE in mujoco 3.11
img = renderer.render()

# --- diagnostics: prove what is actually in the frame ---
print("image stats: min %d  mean %.1f  max %d   non-black %.1f%%" % (
    img.min(), img.mean(), img.max(), 100 * (img.sum(axis=2) > 0).mean()))

Image.fromarray(img).save("smoke.png")
print("saved smoke.png", img.shape)

# Don't trust your eyes: prove the ball is on screen, pixel by pixel
redness = img[:, :, 0].astype(int) - np.maximum(img[:, :, 1], img[:, :, 2]).astype(int)
red = (redness > 40) & (img[:, :, 0] > 120)
ys, xs = np.where(red)
print("red ball pixels:", len(xs))
if len(xs):
    print("ball on screen at x [%d..%d], y [%d..%d]" % (xs.min(), xs.max(), ys.min(), ys.max()))
else:
    print("NO red pixels found -- ball not in frame!")
