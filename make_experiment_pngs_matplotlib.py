#!/usr/bin/env python3
"""Create six PNG diagrams for the feature-anomaly experiment suite."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Ellipse
import numpy as np


OUT = Path("experiment_diagrams")
OUT.mkdir(exist_ok=True)

INK = "#111827"
MUTED = "#64748B"
BLUE = "#2563EB"
GREEN = "#059669"
RED = "#DC2626"
AMBER = "#D97706"
PANEL = "#F8FAFC"
LINE = "#CBD5E1"

rng = np.random.default_rng(12)


def setup(title, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.55, 8.45, title, fontsize=26, fontweight="bold", color=INK, ha="left")
    ax.text(0.57, 8.08, subtitle, fontsize=13.5, color=MUTED, ha="left")
    return fig, ax


def panel(ax, x, y, w, h, fc=PANEL):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=LINE, facecolor=fc
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, xy1, xy2, color=BLUE, lw=3.0, mutation=18):
    ax.add_patch(FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2
    ))


def label(ax, x, y, s, color=INK, size=12, weight="bold", ha="center"):
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va="center")


def small(ax, x, y, s, color=MUTED, size=10.5, ha="center"):
    ax.text(x, y, s, color=color, fontsize=size, ha=ha, va="center", wrap=True)


def draw_cloud(ax, cx, cy, sx=1.25, sy=0.58, angle=-12, n=95):
    pts = rng.normal(size=(n, 2))
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    th = np.deg2rad(angle)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = pts @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    ax.scatter(pts[:, 0], pts[:, 1], s=18, c=BLUE, alpha=0.35, edgecolors="none")
    return pts


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def exp01_simplenet_noise():
    fig, ax = setup(
        "Experiment 0: Vanilla SimpleNet noise",
        "Baseline: isotropic Gaussian perturbations around real normal features."
    )
    panel(ax, 0.65, 0.65, 14.7, 6.65)
    pts = draw_cloud(ax, 4.35, 4.25, sx=1.35, sy=0.7, angle=-10)
    ax.add_patch(Ellipse((4.35, 4.25), 4.7, 2.2, angle=-10, fill=False, lw=2.0, ec=BLUE, alpha=0.45))
    sample = pts[rng.choice(len(pts), 12, replace=False)]
    for x, y in sample:
        theta = rng.uniform(0, 2 * np.pi)
        arrow(ax, (x, y), (x + 0.62 * np.cos(theta), y + 0.62 * np.sin(theta)), RED, lw=2.2, mutation=13)
    label(ax, 4.35, 6.0, "normal feature cloud", BLUE, 15)
    panel(ax, 8.25, 4.25, 5.9, 1.55, fc="white")
    label(ax, 8.55, 5.42, "Generator", INK, 13, ha="left")
    ax.text(8.55, 4.95, r"$x_{\mathrm{fake}} = x_{\mathrm{real}} + \epsilon$", fontsize=21, color=INK)
    ax.text(8.55, 4.55, r"$\epsilon \sim \mathcal{N}(0,\sigma^2 I)$", fontsize=16, color=MUTED)
    ax.text(
        8.35, 3.15,
        "Reasoning: this is the controlled baseline. Any geometry-aware generator must beat it under the same discriminator, backbone, and training path.",
        fontsize=12.5, color=MUTED, wrap=True
    )
    label(ax, 1.1, 1.25, "Before/after: real feature -> random direction with fixed global scale", RED, 14, ha="left")
    save(fig, "exp01_simplenet_noise.png")


def exp02_anchored_pca_threshold():
    fig, ax = setup(
        "Experiment 1: Anchored PCA with threshold radius",
        "Replace directionless noise with a real anchor plus a local PCA-subspace shift."
    )
    panel(ax, 0.65, 0.65, 14.7, 6.65)
    ax.add_patch(Ellipse((4.3, 4.25), 5.2, 1.45, angle=-18, fc="#ECFDF5", ec=GREEN, lw=2.0, alpha=0.85))
    ax.plot([2.05, 6.55], [3.55, 4.95], color=GREEN, lw=4)
    ax.plot([4.05, 4.55], [3.5, 5.0], color=GREEN, lw=3, alpha=0.55)
    ax.add_patch(Circle((4.3, 4.25), 2.05, fill=False, ec=RED, lw=2.5, ls="--"))
    anchor = (3.35, 4.05)
    fake = (6.1, 4.9)
    ax.scatter(*anchor, s=120, c=BLUE, edgecolors="white", linewidths=2.5, zorder=5)
    ax.scatter(*fake, s=120, c=RED, edgecolors="white", linewidths=2.5, zorder=5)
    arrow(ax, anchor, fake, GREEN, lw=4)
    label(ax, 3.35, 3.7, "real anchor", BLUE, 12)
    label(ax, 6.1, 5.25, "fake", RED, 12)
    small(ax, 6.55, 2.75, "threshold shell\nsqrt(Tp)", RED, 11)
    panel(ax, 8.0, 4.25, 6.45, 1.7, fc="white")
    label(ax, 8.35, 5.48, "Generator", INK, 13, ha="left")
    ax.text(8.35, 5.0, r"$x_{\mathrm{fake},p}=x_{\mathrm{real},p}+r\,U_p\sqrt{\Lambda_p}v$", fontsize=18, color=INK)
    ax.text(8.35, 4.57, r"$r=\sqrt{T_p}+\delta U(0,1)$", fontsize=16, color=MUTED)
    ax.text(8.25, 3.12, "Reasoning: anchor on the empirical normal manifold and move along directions where real patch features vary.", fontsize=12.5, color=MUTED, wrap=True)
    label(ax, 1.1, 1.25, "Before/after: random isotropic direction -> local PCA direction from a real feature", GREEN, 14, ha="left")
    save(fig, "exp02_anchored_pca_threshold.png")


def exp03_fixed_radius_curriculum():
    fig, ax = setup(
        "Experiment 2: Fixed small radius curriculum",
        "Decouple synthetic anomaly magnitude from the fitted Mahalanobis threshold."
    )
    panel(ax, 0.65, 0.65, 14.7, 6.65)
    center = (4.25, 4.25)
    for r, c, txt, pos in [(0.75, BLUE, "subtle", (5.15, 4.95)), (1.55, AMBER, "medium", (5.8, 5.65)), (2.45, RED, "strong", (6.6, 6.25))]:
        ax.add_patch(Circle(center, r, fill=False, ec=c, lw=3, alpha=0.85))
        label(ax, *pos, txt, c, 12)
    ax.scatter(*center, s=120, c=BLUE, edgecolors="white", linewidths=2.5, zorder=5)
    for r, c in [(0.7, BLUE), (1.42, AMBER), (2.22, RED)]:
        arrow(ax, center, (center[0] + r * 0.93, center[1] + r * 0.36), c, lw=3)
    panel(ax, 8.2, 4.25, 5.85, 1.55, fc="white")
    label(ax, 8.55, 5.42, "Radius override", INK, 13, ha="left")
    ax.text(8.55, 4.95, r"$r=\rho\,U(0,1)$", fontsize=21, color=INK)
    ax.text(8.55, 4.55, r"$\rho \in \{0.25,0.5,1,2,5\}$", fontsize=15.5, color=MUTED)
    ax.text(8.35, 3.15, "Reasoning: the statistical threshold may produce anomalies that are too large for localization. Sweep a task-level magnitude instead.", fontsize=12.5, color=MUTED, wrap=True)
    label(ax, 1.1, 1.25, "Before/after: threshold-pinned shell -> subtle-to-strong training curriculum", AMBER, 14, ha="left")
    save(fig, "exp03_fixed_radius_curriculum.png")


def exp04_threshold_vs_anchor_radius():
    fig, ax = setup(
        "Experiment 4: Threshold radius vs. anchored radius",
        "Threshold asks where the boundary is; anchored radius asks how far this feature is from it."
    )
    panel(ax, 0.65, 1.0, 7.05, 6.2)
    panel(ax, 8.3, 1.0, 7.05, 6.2)
    for cx, title, color in [(4.18, "threshold radius", RED), (11.82, "anchored radius", GREEN)]:
        label(ax, cx, 6.75, title, color, 17)
        ax.add_patch(Circle((cx, 4.1), 1.95, fill=False, ec=color, lw=3, ls="--"))
        ax.add_patch(Circle((cx, 4.1), 0.45, fill=True, fc="#DBEAFE", ec="none", alpha=0.75))
        ax.scatter(cx, 4.1, s=80, c=INK, edgecolors="white", linewidths=2)
        small(ax, cx + 1.35, 5.62, "boundary\nsqrt(Tp)", color)
    # threshold arrows
    A1, B1 = (3.45, 4.35), (5.65, 3.82)
    ax.scatter(*A1, s=120, c=BLUE, edgecolors="white", linewidths=2.5, zorder=5)
    ax.scatter(*B1, s=120, c=AMBER, edgecolors="white", linewidths=2.5, zorder=5)
    label(ax, A1[0], A1[1] + 0.35, "A", BLUE, 12)
    label(ax, B1[0], B1[1] + 0.35, "B", AMBER, 12)
    arrow(ax, A1, (2.25, 2.72), RED, lw=4)
    arrow(ax, B1, (6.95, 2.72), RED, lw=4)
    ax.text(4.18, 1.55, r"Both anchors get a boundary-scale step: $r=\sqrt{T_p}+\delta U(0,1)$.", fontsize=12.5, color=MUTED, ha="center", wrap=True)
    # anchored arrows
    A2, B2 = (11.12, 4.35), (13.35, 3.82)
    ax.scatter(*A2, s=120, c=BLUE, edgecolors="white", linewidths=2.5, zorder=5)
    ax.scatter(*B2, s=120, c=AMBER, edgecolors="white", linewidths=2.5, zorder=5)
    label(ax, A2[0], A2[1] + 0.35, "A", BLUE, 12)
    label(ax, B2[0], B2[1] + 0.35, "B", AMBER, 12)
    arrow(ax, A2, (9.95, 2.72), GREEN, lw=4)
    arrow(ax, B2, (13.78, 2.72), GREEN, lw=4)
    ax.text(11.82, 1.55, r"Each anchor moves by its remaining gap: $r\propto\sqrt{T_p}-\sqrt{s(x_{\mathrm{real}})}$.", fontsize=12.5, color=MUTED, ha="center", wrap=True)
    label(ax, 1.1, 0.42, "Before/after: fixed boundary shell -> adaptive near-boundary negatives", GREEN, 14, ha="left")
    save(fig, "exp04_threshold_vs_anchor_radius.png")


def draw_grid(ax, x0, y0, color, mode):
    for i in range(10):
        for j in range(6):
            ax.add_patch(Rectangle((x0 + i * 0.38, y0 + j * 0.38), 0.38, 0.38, fill=False, ec=LINE, lw=0.8))
    if mode == "block":
        cells = [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2), (5, 2), (3, 3), (4, 3)]
    else:
        cells = [(1, 0), (2, 4), (4, 1), (5, 5), (7, 2), (9, 4), (3, 3), (8, 0)]
    for i, j in cells:
        ax.add_patch(Rectangle((x0 + i * 0.38, y0 + j * 0.38), 0.38, 0.38, fc=color, ec="none", alpha=0.85))


def exp05_sparse_patch_masks():
    fig, ax = setup(
        "Experiments 5--6: Sparse patch masks",
        "Move from all-fake feature maps to localized synthetic supervision."
    )
    panel(ax, 0.65, 0.65, 14.7, 6.65)
    draw_grid(ax, 1.55, 3.25, RED, "block")
    label(ax, 3.45, 6.0, "block mask", RED, 14)
    draw_grid(ax, 6.2, 3.25, AMBER, "random")
    label(ax, 8.1, 6.0, "random mask", AMBER, 14)
    panel(ax, 10.65, 3.2, 4.15, 2.45, fc="white")
    label(ax, 10.95, 5.25, "Training change", INK, 13, ha="left")
    ax.text(10.95, 4.75, "Only selected patches become fake.", fontsize=12.2, color=INK)
    ax.text(10.95, 4.32, "Unselected patches stay real.", fontsize=12.2, color=INK)
    ax.text(10.95, 3.89, "Fake loss uses selected patches only.", fontsize=12.2, color=INK)
    ax.text(10.95, 2.55, "Reasoning: real defects are local; all-fake maps may teach global shifts instead of localization.", fontsize=12.5, color=MUTED, wrap=True)
    label(ax, 1.1, 1.25, "Before/after: every patch fake -> sparse local fake supervision", BLUE, 14, ha="left")
    save(fig, "exp05_sparse_patch_masks.png")


def exp06_gradient_refinement():
    fig, ax = setup(
        "Experiment 7: Gradient-guided PCA refinement",
        "Turn sampled anomalies into hard negatives while staying inside the local PCA geometry."
    )
    panel(ax, 0.65, 0.65, 14.7, 6.65)
    xs = [2.55, 5.65, 8.75, 11.85]
    names = ["anchored\nPCA fake", "discriminator\ngradient", "project\nto Up", "clamp\nradius"]
    for x, name in zip(xs, names):
        panel(ax, x - 1.15, 5.02, 2.3, 1.05, fc="white")
        ax.text(x, 5.55, name, fontsize=13.5, color=INK, fontweight="bold", ha="center", va="center")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 1.15, 5.55), (b - 1.15, 5.55), BLUE, lw=3)
    ax.add_patch(Ellipse((7.45, 3.0), 5.4, 1.15, angle=4, fc="#ECFDF5", ec=GREEN, lw=2.4))
    ax.plot([4.95, 9.95], [2.88, 3.13], color=GREEN, lw=4)
    ax.scatter(6.1, 2.95, s=120, c=BLUE, edgecolors="white", linewidths=2.5, zorder=5)
    ax.scatter(8.9, 3.09, s=120, c=RED, edgecolors="white", linewidths=2.5, zorder=5)
    arrow(ax, (6.1, 2.95), (8.9, 3.09), RED, lw=4)
    small(ax, 7.45, 1.95, "The update searches for harder fake samples, but projection and radius clamping prevent unrestricted adversarial drift.", MUTED, 12)
    label(ax, 1.1, 1.05, "Before/after: random fake sample -> near-boundary hard negative", GREEN, 14, ha="left")
    save(fig, "exp06_gradient_refinement.png")


if __name__ == "__main__":
    exp01_simplenet_noise()
    exp02_anchored_pca_threshold()
    exp03_fixed_radius_curriculum()
    exp04_threshold_vs_anchor_radius()
    exp05_sparse_patch_masks()
    exp06_gradient_refinement()
