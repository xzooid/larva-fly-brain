FlyBrain Project — Lab Notebook
Project

Embodied brain emulation: 3,016-neuron Drosophila larva connectome (Winding et
al., Science 2023) driving a MuJoCo fly body (flybody, TuragaLab/DeepMind).
Log
2026-08-19 — Environment bring-up (Artix Linux)

    Python venv at ~/flybrain/.venv (Python 3.14), MuJoCo 3.11, numpy, matplotlib.
    Rendering saga: EGL broken on tty; osmesa unavailable on Artix (no libOSMesa in
    mesa build); PyOpenGL needs GL platform. Working recipe:
    MUJOCO_GL=glfw xvfb-run -a python smoke.py (xorg-server-xvfb via doas).
    Rendering is optional for the science; physics verified headless.
    Smoke test: ball settles z = 0.0396 (predicted 0.04). Physics ✓.

2026-08-19 — Toy 2-neuron LIF (toy_lif.py)

    A drives B (W[B,A]=7), B whispers back (W[A,B]=1.5). Temporal summation:
    B fires after 2 A-spikes; 36/37 phase-locked; A: 90 → 123 Hz from feedback.
    Lessons: predict-before-run; state matters; recurrent excitation speeds rates.

2026-08-19 — Real connectome (explore_connectome.py)

    Data: networks.skewed.de fly_larva mirror of Winding et al. 2023.
    2,956 neurons (paper: 3,016), 116,922 connections, 352,611 synapses
    (paper: ~548k; mirror is curated — documented provenance caveat).
    Partition: 434 afferent (sensory), 1,598 intrinsic, 924 efferent (descending).

2026-08-19 — Whole-brain LIF (lif_full_brain.py)

    100 ms poke to sensory neurons → wave: sensory → intrinsic → descending.
    Gain 8 (row-normalized W): 10,600 spikes; 13 ms sensorimotor latency.
    Gain sweep: critical point between 10 and 12 (19.7k → 220.6k spikes).

2026-08-19 — Dale's law experiment (lif_dales.py)

    Literature NT table (MBON, LN = inhibitory). Result: mean E/I balance does
    NOT stabilize high gain; seizure is fluctuation-driven
    (sigma = sqrt(sum w^2 r tau_syn) crosses theta at predicted gain ~10.4).
    Balanced normalization amplifies variance -> earlier seizure. Conclusion:
    simplified LIF is fluctuation-dominated; conductance-based E/I = future work.
    Operating point for embodiment: gain 8, all-excitatory (stable regime).

Decisions / assumptions (for the write-up)

    Synapse counts → weights via row normalization × global gain (synaptic scaling).
    Dale's law table from larval literature; unknown classes default excitatory.
    Refractory period 2 ms; tau = 10 ms; tau_syn = 2 ms; theta = 1 (normalized).
    Larva brain (1st instar) + adult fly body (flybody) = proof-of-concept mismatch,
    to be disclosed honestly.

Next

    Clone flybody, inspect XML (joints/actuators/sensors).
    Input projection: body sensors → sensory neurons.
    Output projection: descending neuron rates → joint actuators.
    First demo: poke → legs move (reflex).

2026-08-19 — EMBODIMENT: brain in the fly (embodied_fly.py) ✅

    Body: floor.xml = adult fly + floor. 68 bodies, 109 DOFs, 78 actuators
    (70 position-servo "muscles", gain 0.8, biastype affine; 8 adhesive claws),
    15 sensors: accel/gyro/veloc (thorax) + 6 force_tarsus (3-vector!) + 6 touch_claw.
    sensordata layout: [0:3] accel, [3:6] gyro, [6:9] veloc, [9:27] force 6x3, [27:33] touch.
    Coupling: touch_claw (sensordata[27:33]) -> 6 seeded sensory pools (rng 42);
    DN-VNC split into 6 leg groups (rng 7) -> each leg's 8 joints;
    DN-SEZ -> head/abdomen. rate (low-pass 50 ms) -> ctrl = baseline + clamp(gain_joint*rate).
    BUGS FIXED (great teaching material):
        UNITS: brain dt must be ms (tau=10 ms), MuJoCo timestep is s (0.0001 s).
        I_ext must be zeroed every step (accumulation -> runaway).
        force_tarsus are 3-vectors -> touch sensors are sensordata[27:33], not [6:].
        Calibration window: resting rates measured pre-poke, joints held at stance.
        Single-pool poke too weak (pool->DN direct Wn sum 42 vs 106): poke right side
        (pools 1,3,5) + 400 ms duration.
    RESULT: poke -> 25,476 brain spikes / 1,225 descending spikes during poke
    (0 before, 6 after) -> legs twitch; T1_left moved 0.769 rad, T1_right 0.492,
    others < 0.16. Connectome routes response non-uniformly (emergent).
    Caveat: sensory pools are arbitrary seeded partitions (mirror lacks per-leg
    identity), so "right-side poke" is not biologically lateralized -- honest note.
    Runtime ~30 s per 4 s of sim (Python loop; vectorize later).
    GIF: MUJOCO_GL=glfw xvfb-run -a python embodied_fly.py --video
    FIX: video now captured DURING the main loop (frames list), not by re-running
    the sim afterwards (old code: silent ~30 s + no poke in the re-run).

2026-08-19 — EXPERIMENT BATTERY (experiments.py) ✅

Conditions: control / poke RIGHT / poke LEFT / poke BOTH /
poke RIGHT + KC lesion / poke RIGHT + LN lesion (6 x ~30 s).
| condition | desc@poke | best leg | dev |
| control | 0 | T3_right | 0.038 | <- negative control
| poke RIGHT | 1231 | T1_left | 0.769 |
| poke LEFT | 1706 | T1_left | 0.769 |
| poke BOTH | 5653 | T1_right | 0.988 | <- dose-like scaling
| poke RIGHT + KC les. | 1215 | T1_left | 0.769 | <- null (reflex ~KC-free)
| poke RIGHT + LN les. | 2564 | T1_left | 0.769 | <- DISINHIBITION: 2.1x
Findings:

    Negative control passes (0 desc spikes, ~0.03 rad noise).
    Input mass scales the response (1231 < 1706 < 5653).
    KC lesion ~no change -> the touch->leg reflex does NOT route through the
    mushroom-body learning circuit (sensible: KC are olfactory/learning).
    LN lesion DOUBLES motor output -> LN inhibitory interneurons act as a
    gain brake on the sensorimotor reflex (ties into the Dale's-law work: LN
    were our GABAergic class!). Strongest causal result.
    Left vs right poke: not cleanly lateralized -- pools are ARBITRARY seeded
    partitions (mirror lacks per-leg sensory identity) -> lateralization not
    resolvable; honest caveat for write-up.
    PIPELINE PARITY LESSON: experiments.py initially omitted the DN-SEZ ->
    head/abdomen output channel; tiny input differences got amplified by the
    recurrent connectome into different leg patterns (T1_left 0.03 vs 0.77!).
    Always run conditions with the FULL validated pipeline.
    sandbox note: flybody .obj assets dropped from snapshot -> copy assets to
    flybrain/flybody_assets (self-contained); scripts search it first.

2026-08-19 — SATURATION INVESTIGATION: the "identical 0.769" artifact ✅

User flagged: T1_left = 0.769 identical to 3 decimals across 4 conditions =
red flag (joint limit / clamp saturation?).
ROOT CAUSE FOUND (sat_check.py + diff + trnid check):
tendon-driven actuators (tarsus2_*) have actuator_trnid[0] = TENDON id,
which numerically collides with a JOINT id (2 = head_twist). Old joint_pos()
read m.jnt_qposadr[tendon_id] -> garbage "tarsus2 base" -> ctrl pinned to
ctrlrange edge -> wrong foot pose -> stance changed -> recurrent brain
routed differently. The identical 0.769 was a stance artifact, NOT a joint
limit (all joints in-range; delta_max < CLAMP in every corrected run).
diagnose_saturation.py's "tarsus2 actual 0.000" was the smoking gun.
FIX: read joint positions by NAME (actuator name == joint name) in
embodied_fly.py + experiments.py. Tendon actuators have no 1:1 joint;
their joints are read by name correctly.
CORRECTED BATTERY (final, publishable):
| condition | desc@poke | best leg | dev | asym |
| control | 0 | T3_right | 0.034 | -0.02|
| poke RIGHT | 1206 | T1_right | 0.490 | +0.72| ipsilateral!
| poke LEFT | 1768 | T1_right | 0.711 | +0.05| both fronts
| poke BOTH | 5641 | T1_right | 0.988 | -0.15| dose
| poke RIGHT + KC les. | 1225 | T1_right | 0.514 | +0.73| ~null (KC)
| poke RIGHT + LN les. | 2517 | T1_right | 0.539 | +0.62| 2.1x volume,
| | | | | | T2_right 6x,
| | | | | | broader recruitment

    poke RIGHT now routes IPSILATERALLY (T1_right, asym +0.72) - sensible.
    LN disinhibition SURVIVES and is cleaner (unsaturated readouts:
    T2_right 0.032 -> 0.190 ~6x; desc 1206 -> 2517 ~2.1x).
    No saturation: all joints in-range, deltas < clamp. Graded, condition-
    dependent values (0.490/0.711/0.988/0.514/0.539).
    LESSON: identical outputs across conditions = measurement bug, not biology.
    Check actuator transmission types when reading joint positions.

2026-08-19 — BRAIN VISUALIZATION (brain_viz.py) ✅

Uses _pos 2D embedding from nodes.csv: 2,956 neurons as a graph, top 3000
synapses as skeleton, neurons colored by recent spiking during a poke.
Output: brain_activity.gif (70 frames) + brain_activity_peak.png.
Verified: poke-window motion 1.35 vs pre-poke 0.49.
2026-08-19 — FLIGHT + SEIZURE EXPERIMENTS (flight_test.py, wing_sweep.py)

    Base floor.xml has zero wing fluid coefficients -> NOT flight-ready.
    flybody's flight config: fluidcoef=[1.0,0.5,1.5,1.7,1.0], wing gainprm=18,
    wing damping 0.00777, dt 1e-4 s.
    wing_sweep.py: pitch-only flap barely lifts (dz ~0.02 m); 3-DOF flapping
    (pitch + roll +90 deg + yaw +180 deg phase offsets) LAUNCHES the fly
    (z up to +3 m in 1 s). Fly mass = 9.85e-4 kg (~1000x a real fly - stylized).
    flight_test.py results:
    A) Hover attempt (3-DOF flap, 200 Hz, amp 1.0): AIRBORNE, max z = 1.46 m,
    but tumbles (tilt max 175 deg) -- no stabilization controller.
    B) Poke mid-flight: brain responds (1205 desc spikes during poke), legs
    kick; still airborne at end (z = 0.59 m), keeps tumbling. Doesn't
    balance (no feedback), doesn't crash in-window, physics finite.
    C) Seizure on ground (gain 20, poke all): 47,541 desc spikes, ALL 6 legs
    pin to ~0.99 rad (clamp saturation = convulsion), tilt 9.9 deg, body
    sinks to z -0.057. Physics finite -> convulses, does NOT die/explode.
    BUG (classic): used for s in ("left","right") in the flap block ->
    clobbered the synaptic-current array s (shadowing). Renamed to side.
    Honest limits: naive periodic flapping = rocket + tumble, not controlled
    flight; stabilization would need attitude feedback (gyro/haltere -> wing
    pitch) or flybody's RL-trained flight policies. Future work.

2026-08-19 — VIDEO QUALITY + BRAIN-LEVEL COMPARISON (user feedback)

    Camera fix: track1/2/3 are ORBITING cameras (caused the spinning flight
    video). Ground videos now use camera="side" (fixed); flight uses a chase-cam
    (MjvCamera free cam, lookat = fly pos, fixed azimuth/elevation/distance).
    brain_viz_lesions.py: the lesion comparison IN THE BRAIN (3 panels side by
    side: baseline / KC / LN, same poke). Numbers reproduce the battery:
    baseline 1196 desc@poke | KC 1217 (~null) | LN 2495 (~2x disinhibition).
    Output: brain_lesions.gif + brain_lesions_peak.png (pure matplotlib, no GL).
    Lesson: the body videos look "the same"; the difference lives in the brain.
