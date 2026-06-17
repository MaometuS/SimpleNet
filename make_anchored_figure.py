"""
Presentation graphic: subspace anchoring -- shift from mu vs. from a real
normal feature.

  mode = "subspace"  : x_anom = mu      + r * U_k sqrt(Lambda) v_k
  mode = "anchored"  : x_anom = x_real  + r * U_k sqrt(Lambda) v_k

Same shift r and direction; only the ANCHOR differs. The fitted (mu, Sigma) is
only an approximation of the data manifold -- when the true distribution is not
a clean Gaussian, shifting from mu sprays anomalies symmetrically around the
centroid (which may not even sit on the data). Shifting from real features
makes every anomaly "a real point, nudged off the manifold".

Output: anchored_sampling.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
MU = "#c1121f"

SHIFT = 2.5          # in-subspace shift magnitude r (same in both panels)
N_DATA = 280
N_ANOM = 95


def c_shape(n):
    """A curved, non-Gaussian data manifold: an open 'C'."""
    phi = rng.uniform(np.radians(-140), np.radians(140), n)
    R = 3.0
    x = R * np.sin(phi) + rng.normal(0, 0.26, n)
    y = R * np.cos(phi) + rng.normal(0, 0.26, n)
    return x, y


DX, DY = c_shape(N_DATA)
MU_X, MU_Y = DX.mean(), DY.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.0))

# -- Panel 1: subspace -- shift from the model center mu -----------------
ax = ax1
ax.scatter(DX, DY, s=13, c=BLUE, alpha=0.45, label="normal features", zorder=2)
th = rng.uniform(0, 2 * np.pi, N_ANOM)
ex, ey = MU_X + SHIFT * np.cos(th), MU_Y + SHIFT * np.sin(th)
for x, y in zip(ex, ey):
    ax.plot([MU_X, x], [MU_Y, y], color=ORANGE, lw=0.4, alpha=0.30, zorder=1)
ax.scatter(ex, ey, s=18, c=ORANGE, alpha=0.85, zorder=4,
           label="synthetic anomalies")
ax.scatter([MU_X], [MU_Y], s=240, c=MU, marker="*", edgecolor="black",
           lw=0.7, zorder=5, label=r"model center $\mu$")
ax.set_title("mode = subspace\n"
             r"$x_{\mathrm{anom}} = \mu + r\,U_k\sqrt{\Lambda}\,v_k$",
             fontsize=12.5, fontweight="bold")
ax.text(0.5, -0.085,
        "every shift starts from the single model center $\\mu$\n"
        r"$\Rightarrow$ a symmetric ring that ignores the manifold's shape",
        transform=ax.transAxes, ha="center", fontsize=10.5)

# -- Panel 2: anchored -- shift from real features -----------------------
ax = ax2
ax.scatter(DX, DY, s=13, c=BLUE, alpha=0.45, label="normal features", zorder=2)
idx = rng.choice(N_DATA, N_ANOM, replace=False)
th = rng.uniform(0, 2 * np.pi, N_ANOM)
ex, ey = DX[idx] + SHIFT * np.cos(th), DY[idx] + SHIFT * np.sin(th)
for ax0, ay0, x, y in zip(DX[idx], DY[idx], ex, ey):
    ax.plot([ax0, x], [ay0, y], color=GREEN, lw=0.4, alpha=0.35, zorder=1)
ax.scatter(ex, ey, s=18, c=GREEN, alpha=0.85, zorder=4,
           label="synthetic anomalies")
ax.scatter(DX[idx], DY[idx], s=20, facecolor="none", edgecolor="#0d3b66",
           lw=0.8, zorder=3, label="anchors (real features)")
ax.set_title("mode = anchored\n"
             r"$x_{\mathrm{anom}} = x_{\mathrm{real}} + r\,U_k\sqrt{\Lambda}\,v_k$",
             fontsize=12.5, fontweight="bold")
ax.text(0.5, -0.085,
        "every shift starts from a real feature\n"
        r"$\Rightarrow$ anomalies hug the true distribution",
        transform=ax.transAxes, ha="center", fontsize=10.5)

for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-6.0, 6.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

fig.suptitle("Anchoring:  shift from the model center, or from real features",
             fontsize=15, fontweight="bold", y=1.0)
fig.text(0.5, -0.01,
         r"Same shift $r$ and direction in both panels -- only the anchor differs.  "
         r"The fitted $(\mu,\Sigma)$ only approximates the manifold; when the true "
         "distribution is not a clean" "\n"
         r"Gaussian, $\mu$ may not even sit on the data.  Anchoring on real "
         r"features keeps every anomaly close to the manifold -- 'a real point, "
         r"nudged off it'.",
         ha="center", fontsize=10.5, style="italic")

fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig("anchored_sampling.png", dpi=200, bbox_inches="tight")
print("wrote anchored_sampling.png")
