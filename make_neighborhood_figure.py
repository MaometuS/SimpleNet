"""
Presentation graphic: the neighborhood-pooled covariance change in
TrueSpatialLowRankGaussian.

Three panels:
  1. neighborhood = 1  -- original per-patch fit (rank-deficient)
  2. neighborhood = 3  -- tied covariance, pool the 3x3 window (interior patch)
  3. neighborhood = 3  -- at the grid border, with reflection padding

Output: neighborhood_inclusion.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

G = 9                  # representative grid (the real patch grid is 36x36 = 1296)
CENTER = (4, 4)        # an interior patch
CORNER = (0, 0)        # an edge / corner patch

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GRAYPAD = "#9aa0a6"


def draw_base_grid(ax, pad=False):
    """Light reference grid. If pad, also draw the 1-cell reflection ring."""
    lo, hi = (-1, G + 1) if pad else (0, G)
    for i in range(lo, hi + 1):
        ax.plot([lo, hi], [i, i], color="0.85", lw=0.8, zorder=0)
        ax.plot([i, i], [lo, hi], color="0.85", lw=0.8, zorder=0)
    if pad:
        ax.add_patch(mpatches.Rectangle((0, 0), G, G, fill=False,
                                        edgecolor="0.35", lw=2.0, zorder=3))
    ax.set_xlim(-2.5, G + 2.5)
    ax.set_ylim(-2.0, G + 5.0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def cell(ax, r, c, color, alpha=1.0, lw=1.0, edge=None, hatch=None, z=2):
    ax.add_patch(mpatches.Rectangle((c, r), 1, 1, facecolor=color, alpha=alpha,
                                    edgecolor=edge or color, lw=lw, hatch=hatch,
                                    zorder=z))


fig, axes = plt.subplots(1, 3, figsize=(17, 6.6))

# -- Panel 1: neighborhood = 1 -------------------------------------------
ax = axes[0]
draw_base_grid(ax)
r, c = CENTER
cell(ax, r, c, BLUE, alpha=0.85, lw=2.0, edge="#0d3b66")
ax.text(c + 0.5, r + 0.5, "p", ha="center", va="center",
        color="white", fontweight="bold", fontsize=13)
ax.set_title("neighborhood = 1\noriginal: independent per-patch fit",
             fontsize=13, fontweight="bold")
ax.text(G / 2, G + 1.7, r"$\Sigma_p$ fit from patch $p$ alone:  $N \approx 250$",
        ha="center", fontsize=11)
ax.text(G / 2, G + 2.9, r"rank $\leq 249 \;\ll\; C = 1536$   (rank-deficient)",
        ha="center", fontsize=10.5, color="#c1121f", fontweight="bold")

# -- Panel 2: neighborhood = 3, interior ---------------------------------
ax = axes[1]
draw_base_grid(ax)
r, c = CENTER
for dr in (-1, 0, 1):
    for dc in (-1, 0, 1):
        cell(ax, r + dr, c + dc, ORANGE, alpha=0.40, lw=1.0, edge="#cc6600")
cell(ax, r, c, BLUE, alpha=0.85, lw=2.0, edge="#0d3b66")
ax.text(c + 0.5, r + 0.5, "p", ha="center", va="center",
        color="white", fontweight="bold", fontsize=13)
ax.set_title("neighborhood = 3\ntied covariance: pool the 3x3 window",
             fontsize=13, fontweight="bold")
ax.text(G / 2, G + 1.7, r"$\Sigma_p$ pooled over $p$'s window:  $\approx 9N \approx 2250$",
        ha="center", fontsize=11)
ax.text(G / 2, G + 2.9, "near full rank  --  well-conditioned",
        ha="center", fontsize=10.5, color="#2a9d8f", fontweight="bold")

# -- Panel 3: neighborhood = 3, edge with reflection ---------------------
ax = axes[2]
draw_base_grid(ax, pad=True)
r, c = CORNER
for dr in (-1, 0, 1):
    for dc in (-1, 0, 1):
        rr, cc = r + dr, c + dc
        is_pad = rr < 0 or cc < 0 or rr >= G or cc >= G
        if is_pad:
            cell(ax, rr, cc, GRAYPAD, alpha=0.55, lw=1.0,
                 edge="#6b6f76", hatch="////")
        else:
            cell(ax, rr, cc, ORANGE, alpha=0.40, lw=1.0, edge="#cc6600")
cell(ax, r, c, BLUE, alpha=0.85, lw=2.0, edge="#0d3b66")
ax.text(c + 0.5, r + 0.5, "p", ha="center", va="center",
        color="white", fontweight="bold", fontsize=13)
ax.set_title("neighborhood = 3 at the border\nreflection padding",
             fontsize=13, fontweight="bold")
ax.text(G / 2, G + 1.7, "missing cells filled by reflection",
        ha="center", fontsize=11)
ax.text(G / 2, G + 2.9, r"every patch gets a full window $\Rightarrow$ uniform $k_{\mathrm{eff}}$",
        ha="center", fontsize=10.5, color="#2a9d8f", fontweight="bold")

# -- shared legend + footer ----------------------------------------------
handles = [
    mpatches.Patch(facecolor=BLUE, alpha=0.85, label=r"patch $p$  (the one being fit)"),
    mpatches.Patch(facecolor=ORANGE, alpha=0.40, label=r"neighbors pooled into $\Sigma_p$"),
    mpatches.Patch(facecolor=GRAYPAD, alpha=0.55, hatch="////",
                   label="reflection-padded cells"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
           frameon=False, bbox_to_anchor=(0.5, 0.005))

fig.suptitle("Neighborhood-pooled covariance:  what changed",
             fontsize=15, fontweight="bold", y=1.04)
fig.text(0.5, -0.04,
         r"Pooled: only $\Sigma_p$ (covariance).    "
         r"Still per-patch: $\mu_p$ (mean) and $T_p$ (threshold).",
         ha="center", fontsize=11, style="italic")

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("neighborhood_inclusion.png", dpi=200, bbox_inches="tight")
print("wrote neighborhood_inclusion.png")
