#!/usr/bin/env python3
"""Generate clean reusable PNG diagrams for slides.

These are diagram assets, not full slide pages. They intentionally avoid big
titles, page frames, and long prose. Use them as figures inside a deck.
"""

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle, Arc


OUT = Path("experiment_diagrams")
OUT.mkdir(exist_ok=True)

INK = "#0F172A"
MUTED = "#64748B"
BLUE = "#2563EB"
GREEN = "#059669"
RED = "#DC2626"
AMBER = "#D97706"
PURPLE = "#7C3AED"
CYAN = "#0891B2"
LIGHT_BLUE = "#DBEAFE"
LIGHT_GREEN = "#D1FAE5"
LIGHT_RED = "#FEE2E2"
LIGHT_AMBER = "#FEF3C7"
PANEL = "#F8FAFC"
LINE = "#CBD5E1"

rng = np.random.default_rng(22)


def fig_ax():
    fig = plt.figure(figsize=(14, 7.875), dpi=180)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.875)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1)
    ax.set_facecolor("white")
    return fig, ax


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, facecolor="white", transparent=False, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {path}")


def label(ax, x, y, s, size=14, color=INK, weight="bold", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def note(ax, x, y, s, size=11, color=MUTED, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def arrow(ax, p0, p1, color=BLUE, lw=3, ms=18, alpha=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    ))


def card(ax, x, y, w, h, fc="white", ec=LINE, radius=0.16, lw=1.3, alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha
    )
    ax.add_patch(p)
    return p


def normal_cloud(ax, cx, cy, sx=1.35, sy=0.55, angle=-14, n=110, alpha=0.34):
    pts = rng.normal(size=(n, 2))
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    th = np.deg2rad(angle)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = pts @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    ax.scatter(pts[:, 0], pts[:, 1], s=18, c=BLUE, alpha=alpha, edgecolors="none")
    ax.add_patch(Ellipse((cx, cy), 2 * sx * 2.2, 2 * sy * 2.2, angle=angle,
                         fill=True, fc=LIGHT_BLUE, ec=BLUE, lw=2, alpha=0.24))
    return pts


def pca_manifold(ax, cx, cy, sx=1.9, sy=0.46, angle=-13, fc=LIGHT_GREEN):
    ax.add_patch(Ellipse((cx, cy), 2 * sx, 2 * sy, angle=angle,
                         fc=fc, ec=GREEN, lw=2.4, alpha=0.72))
    th = np.deg2rad(angle)
    ax.plot([cx - sx * np.cos(th), cx + sx * np.cos(th)],
            [cy - sx * np.sin(th), cy + sx * np.sin(th)],
            color=GREEN, lw=4, solid_capstyle="round")
    ax.plot([cx - sy * np.cos(th + np.pi / 2), cx + sy * np.cos(th + np.pi / 2)],
            [cy - sy * np.sin(th + np.pi / 2), cy + sy * np.sin(th + np.pi / 2)],
            color=GREEN, lw=2.4, alpha=0.55, solid_capstyle="round")


def draw_grid(ax, x0, y0, cols=9, rows=6, size=0.32, colored=(), color=RED):
    for i in range(cols):
        for j in range(rows):
            ax.add_patch(Rectangle((x0 + i * size, y0 + j * size), size, size,
                                   fill=False, ec=LINE, lw=0.85))
    for i, j in colored:
        ax.add_patch(Rectangle((x0 + i * size, y0 + j * size), size, size,
                               fc=color, ec="white", lw=0.6, alpha=0.9))


def exp01_noise():
    fig, ax = fig_ax()
    pts = normal_cloud(ax, 4.3, 4.0, sx=1.65, sy=0.72, angle=-10)
    ax.text(3.1, 5.45, "normal features", color=BLUE, fontsize=16, fontweight="bold")
    for x, y in pts[rng.choice(len(pts), 18, replace=False)]:
        theta = rng.uniform(0, 2 * np.pi)
        arrow(ax, (x, y), (x + 0.48 * np.cos(theta), y + 0.48 * np.sin(theta)), RED, lw=1.6, ms=11, alpha=0.75)
    ax.text(8.1, 4.4, r"$x_{\mathrm{fake}} = x_{\mathrm{real}} + \epsilon$", fontsize=25, color=INK)
    ax.text(8.1, 3.75, r"$\epsilon \sim \mathcal{N}(0,\sigma^2 I)$", fontsize=19, color=MUTED)
    label(ax, 4.0, 2.15, "directionless perturbations", 16, RED)
    save(fig, "exp_00_simplenet_noise.png")


def exp02_anchored_pca():
    fig, ax = fig_ax()

    # Left: before isotropic noise.
    label(ax, 2.7, 5.55, "before: random noise", 16, MUTED)
    pts = normal_cloud(ax, 2.75, 3.75, sx=1.2, sy=0.5, angle=-10, n=80, alpha=0.25)
    anchor = (2.2, 3.66)
    ax.scatter(*anchor, s=105, c=BLUE, edgecolors="white", linewidths=2.3, zorder=5)
    for theta in np.linspace(0.2, 2 * np.pi, 8, endpoint=False):
        arrow(ax, anchor, (anchor[0] + 0.72 * np.cos(theta), anchor[1] + 0.72 * np.sin(theta)),
              RED, lw=1.5, ms=10, alpha=0.55)
    ax.text(1.15, 2.38, "isotropic\nnoise", color=RED, fontsize=16, fontweight="bold", ha="center")

    # Arrow between concepts.
    arrow(ax, (4.25, 3.72), (5.35, 3.72), BLUE, lw=4.0, ms=24)
    label(ax, 4.8, 4.12, "replace", 15, BLUE)

    # Right: anchored PCA threshold.
    label(ax, 8.65, 6.25, "after: anchored PCA step", 16, MUTED)
    pca_manifold(ax, 9.0, 3.75, sx=2.25, sy=0.58, angle=-12)
    ax.add_patch(Circle((9.0, 3.75), 2.12, fill=False, ec=RED, lw=3.0, ls=(0, (7, 5))))
    ax.text(11.05, 4.85, r"threshold shell $\sqrt{T_p}$", fontsize=15, color=RED, fontweight="bold")
    anchor = (7.8, 3.5)
    fake = (10.9, 2.83)
    ax.scatter(*anchor, s=115, c=BLUE, edgecolors="white", linewidths=2.3, zorder=5)
    ax.scatter(*fake, s=115, c=RED, edgecolors="white", linewidths=2.3, zorder=5)
    arrow(ax, anchor, fake, GREEN, lw=4.2, ms=20)
    ax.text(anchor[0] - 0.72, anchor[1] - 0.55, "real\nanchor", color=BLUE, fontsize=15, fontweight="bold", ha="center")
    ax.text(fake[0] + 0.38, fake[1] + 0.2, "fake", color=RED, fontsize=16, fontweight="bold")
    ax.text(7.25, 1.2, r"$x_{\mathrm{fake},p}=x_{\mathrm{real},p}+r\,U_p\sqrt{\Lambda_p}v$",
            fontsize=21, color=INK)
    ax.text(7.25, 0.72, r"$r=\sqrt{T_p}+\delta U(0,1)$", fontsize=17, color=MUTED)
    save(fig, "exp_01_anchored_threshold.png")


def exp03_fixed_radius(name="exp_02_anchored_fixed_radius.png"):
    fig, ax = fig_ax()
    center = (4.15, 3.7)
    ax.scatter(*center, s=115, c=BLUE, edgecolors="white", linewidths=2.3, zorder=5)
    for r, c, t in [(0.75, BLUE, "subtle"), (1.5, AMBER, "medium"), (2.25, RED, "strong")]:
        ax.add_patch(Circle(center, r, fill=False, ec=c, lw=2.8))
        arrow(ax, center, (center[0] + r * 0.9, center[1] + r * 0.32), c, lw=2.3, ms=13)
        ax.text(center[0] + r * 0.95, center[1] + r * 0.48, t, color=c, fontsize=12, fontweight="bold")
    ax.text(8.0, 4.22, r"$r=\rho\,U(0,1)$", fontsize=27, color=INK)
    ax.text(8.0, 3.52, r"$\rho \in \{0.25,0.5,1,2,5\}$", fontsize=20, color=MUTED)
    label(ax, 6.95, 1.18, "decouple anomaly magnitude from the threshold", 16, AMBER)
    save(fig, name)


def exp03_patch_radius():
    fig, ax = fig_ax()
    centers = [(2.45, 4.5), (4.75, 3.45), (7.05, 2.4)]
    thresholds = [0.72, 1.05, 1.42]
    colors = [BLUE, CYAN, RED]
    names = [r"low $T_p$", r"mid $T_p$", r"high $T_p$"]

    for (cx, cy), thresh, color, name in zip(centers, thresholds, colors, names):
        ax.add_patch(Circle((cx, cy), thresh, fill=False, ec=color, lw=2.8, ls=(0, (6, 4))))
        ax.scatter(cx, cy, s=80, c=INK, edgecolors="white", linewidths=2, zorder=5)
        step = 0.55 * thresh
        arrow(ax, (cx, cy), (cx + step * 0.82, cy + step * 0.38), color, lw=3.0, ms=16)
        ax.text(cx - 0.7, cy + thresh + 0.35, name, fontsize=15, color=color, fontweight="bold")
        ax.text(cx + 0.48, cy - thresh - 0.2, r"$\sqrt{T_p}$", fontsize=13, color=color, fontweight="bold")

    ax.text(9.0, 4.72, r"$r=\rho\sqrt{T_p/C}\,U(0,1)$", fontsize=27, color=INK)
    ax.text(9.0, 3.92, r"$\rho \in \{0.25,0.5,1,2,5\}$", fontsize=20, color=MUTED)
    ax.text(9.0, 2.85, "same sweep knob,\npatch-calibrated absolute step",
            fontsize=19, color=GREEN, fontweight="bold", ha="left")
    ax.text(9.0, 1.72, "patch thresholds differ across the feature map", fontsize=15.5,
            color=MUTED, fontweight="bold", ha="left")
    save(fig, "exp_03_patch_radius_sweep.png")


def exp04_anchor_radius():
    fig, ax = fig_ax()
    # two compact comparison panels
    for cx, title, color in [(3.8, "threshold radius", RED), (10.15, "anchored radius", GREEN)]:
        label(ax, cx, 5.55, title, 17, color)
        ax.add_patch(Circle((cx, 3.45), 1.45, fill=False, ec=color, lw=2.4, ls=(0, (6, 4))))
        ax.scatter(cx, 3.45, s=75, c=INK, edgecolors="white", linewidths=2)
        ax.text(cx + 0.95, 4.65, r"$\sqrt{T_p}$", fontsize=11.5, color=color, fontweight="bold")

    A, B = (3.2, 3.62), (4.75, 3.26)
    ax.scatter(*A, s=105, c=BLUE, edgecolors="white", linewidths=2.3, zorder=5)
    ax.scatter(*B, s=105, c=AMBER, edgecolors="white", linewidths=2.3, zorder=5)
    arrow(ax, A, (2.32, 2.38), RED, lw=3.2)
    arrow(ax, B, (5.65, 2.38), RED, lw=3.2)
    ax.text(3.8, 1.84, "same boundary-scale step\nfor both anchors", fontsize=11, color=MUTED, ha="center")

    A, B = (9.55, 3.62), (11.08, 3.26)
    ax.scatter(*A, s=105, c=BLUE, edgecolors="white", linewidths=2.3, zorder=5)
    ax.scatter(*B, s=105, c=AMBER, edgecolors="white", linewidths=2.3, zorder=5)
    arrow(ax, A, (8.72, 2.38), GREEN, lw=3.2)
    arrow(ax, B, (11.38, 2.38), GREEN, lw=3.2)
    ax.text(10.15, 1.84, "step = remaining gap\nto the boundary", fontsize=11, color=MUTED, ha="center")
    ax.text(7.0, 0.95, r"$r \propto \sqrt{T_p}-\sqrt{s(x_{\mathrm{real}})}$", fontsize=23, color=GREEN, ha="center", fontweight="bold")
    save(fig, "exp_04_anchor_radius_sweep.png")


def exp05_masks(name="exp_05_sparse_random_sweep.png"):
    fig, ax = fig_ax()
    def grid(x0, y0, cells, color, name):
        size = 0.34
        for i in range(10):
            for j in range(6):
                ax.add_patch(Rectangle((x0 + i * size, y0 + j * size), size, size, fill=False, ec=LINE, lw=0.85))
        for i, j in cells:
            ax.add_patch(Rectangle((x0 + i * size, y0 + j * size), size, size, fc=color, ec="white", lw=0.7, alpha=0.9))
        label(ax, x0 + 1.7, y0 + 2.45, name, 13, color)

    grid(1.6, 3.0, [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2), (5, 2), (3, 3), (4, 3)], RED, "block mask")
    grid(5.65, 3.0, [(1, 0), (2, 4), (4, 1), (5, 5), (7, 2), (9, 4), (3, 3), (8, 0)], AMBER, "random mask")
    ax.text(9.95, 4.52, "fake loss only on\nselected patches", fontsize=20, color=INK, fontweight="bold", ha="left")
    ax.text(9.95, 3.38, "all patches fake\n-> local anomaly\nsupervision", fontsize=15, color=MUTED, ha="left")
    save(fig, name)


def exp06_gradient():
    fig, ax = fig_ax()
    # Main geometry: sampled synthetic negative is refined, then constrained.
    pca_manifold(ax, 4.5, 3.55, sx=2.6, sy=0.62, angle=-8)
    anchor = (2.75, 3.35)
    sampled = (4.55, 3.5)
    refined = (6.55, 3.82)
    ax.scatter(*anchor, s=110, c=BLUE, edgecolors="white", linewidths=2.3, zorder=5)
    ax.scatter(*sampled, s=110, c=AMBER, edgecolors="white", linewidths=2.3, zorder=5)
    ax.scatter(*refined, s=125, c=RED, edgecolors="white", linewidths=2.5, zorder=5)
    arrow(ax, anchor, sampled, AMBER, lw=3.2)
    arrow(ax, sampled, refined, RED, lw=3.2, rad=0.12)
    ax.text(anchor[0] - 0.1, anchor[1] - 0.55, "real\nanchor", color=BLUE, fontsize=15, fontweight="bold", ha="center")
    ax.text(sampled[0], sampled[1] + 0.52, "sampled\nfake", color=AMBER, fontsize=15, fontweight="bold", ha="center")
    ax.text(refined[0] + 0.32, refined[1] + 0.52, "harder\nfake", color=RED, fontsize=15, fontweight="bold", ha="center")
    ax.text(1.6, 5.55, "local PCA geometry", color=GREEN, fontsize=17, fontweight="bold", ha="left")

    # Refinement loop chips.
    xs = [8.4, 10.35, 12.3]
    labels = ["gradient\nfrom D", "project\nto Up", "clamp\nradius"]
    colors = [PURPLE, GREEN, BLUE]
    for x, t, c in zip(xs, labels, colors):
        ax.add_patch(Circle((x, 4.7), 0.78, fc="white", ec=c, lw=2.4))
        ax.text(x, 4.7, t, fontsize=13.5, color=c, fontweight="bold", ha="center", va="center")
    arrow(ax, (9.18, 4.7), (9.55, 4.7), BLUE, lw=2.8, ms=16)
    arrow(ax, (11.13, 4.7), (11.5, 4.7), BLUE, lw=2.8, ms=16)
    ax.add_patch(Arc((10.35, 3.65), 3.55, 1.28, theta1=200, theta2=340, ec=MUTED, lw=2.2, ls=(0, (5, 5))))
    ax.text(10.35, 2.75, "1-3 small constrained steps", fontsize=15, color=MUTED, ha="center")
    ax.text(8.45, 1.55, "harder negatives, still constrained", fontsize=18, color=GREEN, fontweight="bold")
    save(fig, "exp_07_gradient_refinement_sweep.png")


if __name__ == "__main__":
    exp01_noise()
    exp02_anchored_pca()
    exp03_fixed_radius("exp_02_anchored_fixed_radius.png")
    exp03_patch_radius()
    exp04_anchor_radius()
    exp05_masks("exp_05_sparse_random_sweep.png")
    exp05_masks("exp_06_sparse_block_sweep.png")
    exp06_gradient()
