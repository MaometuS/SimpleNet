"""
Presentation graphic: the `radius` parameter -- anomaly magnitude pinned to the
fitted threshold T_p, vs. decoupled into a free curriculum.

  radius = None  : r ~ U(sqrt(T_p), sqrt(T_p) + delta)   -- a thin shell pinned
                   to the statistical threshold; T_p itself swings between fits.
  radius = R     : r ~ U(0, R)                           -- a free magnitude,
                   a curriculum from subtle (small r) to strong (large r).

Output: radius_sampling.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)

BLUE = "#1f77b4"
RED = "#d62728"

N_DATA = 320
N_ANOM = 420
SQRT_TP = 3.4        # sqrt(T_p): where the default thin shell sits
DELTA = 0.18         # shell half-width
RADIUS = 4.6         # the explicit `radius` parameter

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.0))

# shared normal-data blob
ddist = rng.normal(0, 1.0, (N_DATA, 2))

# -- Panel 1: radius = None -- pinned to sqrt(T_p) -----------------------
ax = ax1
ax.scatter(ddist[:, 0], ddist[:, 1], s=12, c=BLUE, alpha=0.45,
           label="normal features", zorder=2)
th = rng.uniform(0, 2 * np.pi, N_ANOM)
r = SQRT_TP + DELTA * rng.random(N_ANOM)
ax.scatter(r * np.cos(th), r * np.sin(th), s=15, c=RED, alpha=0.8,
           label="synthetic anomalies", zorder=3)
# mark the sqrt(T_p) shell
circ = plt.Circle((0, 0), SQRT_TP, fill=False, ls="--", lw=1.3,
                  edgecolor="0.35", zorder=4)
ax.add_patch(circ)
ax.annotate(r"$\sqrt{T_p}$", xy=(SQRT_TP * 0.71, SQRT_TP * 0.71),
            xytext=(SQRT_TP * 0.71 + 0.7, SQRT_TP * 0.71 + 0.7),
            fontsize=12, arrowprops=dict(arrowstyle="-", color="0.35"))
ax.set_title("radius = None  (default)\n"
             r"$r \sim U(\sqrt{T_p},\; \sqrt{T_p}+\delta)$",
             fontsize=12.5, fontweight="bold")
ax.text(0.5, -0.085,
        "a thin shell pinned to the statistical threshold\n"
        r"$T_p$ swung $825 \to 5.8{\times}10^4 \to 1660$ across our fits",
        transform=ax.transAxes, ha="center", fontsize=10.5)

# -- Panel 2: radius = R -- free curriculum ------------------------------
ax = ax2
ax.scatter(ddist[:, 0], ddist[:, 1], s=12, c=BLUE, alpha=0.45,
           label="normal features", zorder=2)
th = rng.uniform(0, 2 * np.pi, N_ANOM)
r = RADIUS * rng.random(N_ANOM)          # r ~ U(0, RADIUS)
sc = ax.scatter(r * np.cos(th), r * np.sin(th), s=15, c=r, cmap="plasma",
                alpha=0.85, zorder=3)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(r"shift magnitude $r$", fontsize=10)
# mark the radius extent
circ = plt.Circle((0, 0), RADIUS, fill=False, ls="--", lw=1.3,
                  edgecolor="0.35", zorder=4)
ax.add_patch(circ)
ax.annotate("radius", xy=(RADIUS * 0.71, RADIUS * 0.71),
            xytext=(RADIUS * 0.71 + 0.5, RADIUS * 0.71 + 0.5),
            fontsize=12, arrowprops=dict(arrowstyle="-", color="0.35"))
ax.set_title("radius = R  (decoupled)\n"
             r"$r \sim U(0,\; R)$",
             fontsize=12.5, fontweight="bold")
ax.text(0.5, -0.085,
        "a free magnitude, independent of $T_p$\n"
        r"a curriculum: subtle (small $r$) $\to$ strong (large $r$)",
        transform=ax.transAxes, ha="center", fontsize=10.5)

for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_xlim(-5.6, 5.6)
    ax.set_ylim(-5.6, 5.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

fig.suptitle("The radius parameter:  anomaly magnitude pinned to $T_p$, "
             "or a free curriculum",
             fontsize=15, fontweight="bold", y=1.0)
fig.text(0.5, -0.01,
         r"Default ties the shift magnitude to $\sqrt{T_p}$ -- a single "
         r"difficulty level, and $T_p$ is unstable across fits.  Setting "
         r"$radius$ decouples magnitude from the" "\n"
         r"threshold: anomalies span $0$ to $R$, giving the discriminator a "
         r"curriculum of subtle-to-strong examples at a scale you choose.",
         ha="center", fontsize=10.5, style="italic")

fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig("radius_sampling.png", dpi=200, bbox_inches="tight")
print("wrote radius_sampling.png")
