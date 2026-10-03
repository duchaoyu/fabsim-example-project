"""4-part strategies: deviation maps of the baseline and the best runs of each strategy.
    python3 FDM/figure_4part_strategies.py [tag ...]
Reads optimisation/4part_s_<tag>_result.json and _fem_best.obj; writes figures/4part_strategies.png."""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from optimise_4part import load_off, boundary_vertices

TAGS = sys.argv[1:] or ["G", "ring90", "pw_crest_r90", "pw_crestd2"]
V, F = load_off(os.path.join(HERE, "data", "4part", "4part_tri_m.off"))
bd = boundary_vertices(F); inn = np.array([i for i in range(len(V)) if i not in bd])
fig, axs = plt.subplots(2, len(TAGS), figsize=(4.2 * len(TAGS), 8.4))
for k, t in enumerate(TAGS):
    R = json.load(open(os.path.join(HERE, "optimisation", f"4part_s_{t}_result.json")))
    X = np.array([[float(x) for x in l.split()[1:]] for l in
                  open(os.path.join(HERE, "optimisation", f"4part_s_{t}_fem_best.obj")) if l.startswith("v ")])
    d = np.linalg.norm(X - V, axis=1) * 1e3
    ax = axs[0, k]
    tc = ax.tripcolor(V[:, 0], V[:, 1], F, d, shading="gouraud", cmap="magma_r", vmin=0, vmax=20)
    for c in R["cable_paths"]:
        ax.plot(V[c, 0], V[c, 1], color="tab:green", lw=1.5)
    ax.set_title(f"{t}: {R['n_params']} params\nRMSE {R['rmse_mm']:.2f} mm, max {R['max_dev_mm']:.1f} mm", fontsize=10)
    ax.set_aspect("equal"); ax.axis("off")
    # region map coloured by sf_course
    rm = json.load(open(os.path.join(HERE, "optimisation", f"4part_s_{t}_region_map.json")))["face_regions"]
    sc = np.array([R["regions"][r]["sf_course"] for r in rm])
    ax = axs[1, k]
    pc = PolyCollection(V[F][:, :, :2], array=sc, cmap="viridis", edgecolors="none")
    pc.set_clim(1.0, max(1.1, sc.max())); ax.add_collection(pc)
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"sf_course per face ({sc.min():.3f}-{sc.max():.3f})", fontsize=10)
    plt.colorbar(pc, ax=ax, fraction=0.046)
plt.colorbar(tc, ax=axs[0, :].tolist(), fraction=0.02, label="distance to target (mm)")
out = os.path.join(HERE, "figures", "4part_strategies.png")
plt.savefig(out, dpi=130, bbox_inches="tight"); print(out)
