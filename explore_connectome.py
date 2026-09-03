"""
explore_connectome.py -- meet the real brain: the Drosophila larva connectome.
DATA: Winding et al., Science 379, 2023 (mirror: networks.skewed.de/fly_larva)
  nodes.csv : index, vids, hemisphere, homologue, cell_type, ...
  edges.csv : source, target, count, etype    (one row per connection)
    etype = synapse placement: ad = axon->dendrite (classic chemical synapse)
            aa = axon->axon, da = dendrite->axon, dd = dendrite->dendrite
THE KEY MOVE: turn the EDGE LIST into MATRIX W with W[post, pre] = synapse count
  (rows = receivers, cols = senders -- same convention as our toy brain,
   so the toy LIF loop runs unchanged on this matrix)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/fly_larva"

# 1) load nodes
nodes = []
with open(f"{DATA}/nodes.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        nodes.append({"idx": int(p[0]), "vids": int(p[1]), "hemi": p[2],
                      "cell_type": p[4]})
n = len(nodes)
print(f"neurons: {n}")

# 2) build the adjacency matrix W
W = np.zeros((n, n), dtype=np.float32)     # W[post, pre]
edge_count = 0; synapse_total = 0; etype_count = {}
with open(f"{DATA}/edges.csv") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split(",")
        pre, post, cnt, etype = int(p[0]), int(p[1]), int(p[2]), p[3]
        W[post, pre] += cnt
        edge_count += 1; synapse_total += cnt
        etype_count[etype] = etype_count.get(etype, 0) + cnt

print(f"connections: {edge_count:,}   total synapses: {synapse_total:,}")
print("synapses by placement:", {k: f"{v:,}" for k, v in sorted(etype_count.items())})
sparsity = 1.0 - np.count_nonzero(W) / (n * n)
print(f"W is {sparsity*100:.2f}% empty (a real brain is sparse)")

# 3) degrees
out_deg = np.count_nonzero(W, axis=1)
in_deg  = np.count_nonzero(W, axis=0)
print(f"\nout-degree  mean {out_deg.mean():6.1f}  max {out_deg.max()}")
print(f"in-degree   mean {in_deg.mean():6.1f}  max {in_deg.max()}")

# 4) afferent / intrinsic / efferent partition
SENSORY    = {"sensory"}
DESCENDING = {"DN-VNC", "DN-SEZ", "pre-DN-VNC", "pre-DN-SEZ"}
sens = [i for i, nd in enumerate(nodes) if nd["cell_type"] in SENSORY]
desc = [i for i, nd in enumerate(nodes) if nd["cell_type"] in DESCENDING]
intr = [i for i in range(n) if i not in set(sens) and i not in set(desc)]
print(f"\npartition:  {len(sens):4d} afferent (sensory)")
print(f"            {len(intr):4d} intrinsic (interneurons)")
print(f"            {len(desc):4d} efferent  (descending -> motor output)")

# 5) draw the brain
order = np.array(sens + intr + desc)
Wo = W[np.ix_(order, order)]
blocks = [len(sens), len(intr), len(desc)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
im = ax.imshow(np.log1p(Wo), cmap="viridis", interpolation="nearest")
ax.set_title("Whole larva brain connectome (log synapses)\nrows/cols: sensory -> interneurons -> descending")
ax.set_xlabel("presynaptic neuron (sender)")
ax.set_ylabel("postsynaptic neuron (receiver)")
x = 0
for b, label in zip(blocks, ["sensory", "interneurons", "descending"]):
    ax.axhline(x - 0.5, color="w", lw=0.5); ax.axvline(x - 0.5, color="w", lw=0.5)
    x += b
fig.colorbar(im, ax=ax, shrink=0.8, label="log(1 + synapses)")

ax = axes[1]
ax.hist(out_deg, bins=60, color="#1f77b4", alpha=0.8)
ax.set_title("Out-degree distribution")
ax.set_xlabel("postsynaptic partners (out-degree)")
ax.set_ylabel("neurons")
ax.set_yscale("log")

plt.tight_layout()
plt.savefig("connectome_overview.png", dpi=130)
print("\nsaved connectome_overview.png")
