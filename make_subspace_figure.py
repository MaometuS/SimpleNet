"""
Presentation graphic: PDF anomaly sampling vs. subspace anomaly sampling.

Toy geometry for the drawing: C = 3, k = 2.
  - x, y axes = U_k, the data-variance subspace (here 2-D)
  - z axis    = the orthogonal complement (here 1-D; in reality C - k = 1280-D)

Normal features form a near-flat disk in the U_k plane: large variance along
x, y; tiny variance (eps) along z.

  PDF method      : direction ~ uniform on the full 3-sphere
                    -> anomalies leave the data plane
  Subspace method : direction ~ uniform on the 2-sphere inside U_k (z = 0)
                    -> anomalies stay in the data plane

Both anomaly sets lie on the SAME Mahalanobis shell (radius r); they differ
only in which directions of that shell they occupy.

Output: subspace_sampling.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

N_DATA = 240
N_ANOM = 170
R = 3.0           # anomaly shell radius (normal-data std = 1)
EPS_STD = 0.15    # orthogonal std of the normal data (the "eps" direction)

BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"


def normal_cloud():
    return (rng.normal(0, 1.0, N_DATA),
            rng.normal(0, 1.0, N_DATA),
            rng.normal(0, EPS_STD, N_DATA))


def pdf_anomalies():
    # uniform on the full 3-sphere of radius R
    v = rng.normal(size=(N_ANOM, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= R
    return v[:, 0], v[:, 1], v[:, 2]


def subspace_anomalies():
    # uniform on the 2-sphere (a circle) INSIDE the U_k plane: z = 0
    v = rng.normal(size=(N_ANOM, 2))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= R
    return v[:, 0], v[:, 1], np.zeros(N_ANOM)


def style_axes(ax):
    lim = R * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=16, azim=-58)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(r"$U_k$", fontsize=10, labelpad=-8)
    ax.set_ylabel(r"$U_k$", fontsize=10, labelpad=-8)
    ax.set_zlabel("orthogonal\ncomplement", fontsize=9.5, labelpad=-6)
    # translucent U_k plane at z = 0  ("the data subspace")
    g = np.linspace(-lim, lim, 2)
    gx, gy = np.meshgrid(g, g)
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="0.5", alpha=0.10,
                    zorder=0, shade=False)
    # reference circle: where the Mahalanobis shell meets the U_k plane
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(th), R * np.sin(th), np.zeros_like(th),
            color="0.55", lw=1.0, ls="--", zorder=1)


fig = plt.figure(figsize=(15, 7.2))

# -- Panel 1: PDF method --------------------------------------------------
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
dx, dy, dz = normal_cloud()
ax1.scatter(dx, dy, dz, s=10, c=BLUE, alpha=0.45,
            label="normal features", depthshade=False)
axx, ayy, azz = pdf_anomalies()
ax1.scatter(axx, ayy, azz, s=16, c=RED, alpha=0.80,
            label="synthetic anomalies", depthshade=False)
style_axes(ax1)
ax1.set_title("PDF method\n"
              r"direction $u \sim$ uniform on the full $C$-sphere",
              fontsize=12.5, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

# -- Panel 2: Subspace method --------------------------------------------
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
dx, dy, dz = normal_cloud()
ax2.scatter(dx, dy, dz, s=10, c=BLUE, alpha=0.45,
            label="normal features", depthshade=False)
sx, sy, sz = subspace_anomalies()
ax2.scatter(sx, sy, sz, s=16, c=GREEN, alpha=0.80,
            label="synthetic anomalies", depthshade=False)
style_axes(ax2)
ax2.set_title("Subspace method\n"
              r"direction $v_k \sim$ uniform on the $k$-sphere inside $U_k$",
              fontsize=12.5, fontweight="bold")
ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

fig.suptitle("Subspace sampling:  same Mahalanobis shell, different directions",
             fontsize=15, fontweight="bold", y=1.02)
fig.text(0.5, 0.015,
         "Toy view: $C = 3$, $k = 2$.   Normal features (blue) form a near-flat "
         "disk in the data subspace $U_k$ — large variance in $U_k$, tiny "
         "variance $\\varepsilon$ orthogonal.\n"
         "Both anomaly sets sit on the same Mahalanobis shell of radius $r$.  "
         "PDF (red) spreads into the orthogonal complement, where real defects "
         "never go;  subspace (green) stays on the data's own axes.",
         ha="center", fontsize=10.5, style="italic")

fig.tight_layout(rect=[0, 0.07, 1, 0.96])
fig.savefig("subspace_sampling.png", dpi=200, bbox_inches="tight")
print("wrote subspace_sampling.png")
