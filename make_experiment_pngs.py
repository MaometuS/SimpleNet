#!/usr/bin/env python3
"""Generate six presentation PNG diagrams with LaTeX/TikZ + macOS sips."""

import subprocess
from pathlib import Path


OUT = Path("experiment_diagrams")
TEXBIN = "/Library/TeX/texbin"


PREAMBLE = r"""
\documentclass[10pt]{article}
\usepackage[paperwidth=16in,paperheight=9in,margin=0in]{geometry}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\usepackage{xcolor}
\usepackage{amsmath}
\pagestyle{empty}
\definecolor{ink}{HTML}{111827}
\definecolor{muted}{HTML}{64748B}
\definecolor{blue}{HTML}{2563EB}
\definecolor{green}{HTML}{059669}
\definecolor{red}{HTML}{DC2626}
\definecolor{amber}{HTML}{D97706}
\definecolor{panel}{HTML}{F8FAFC}
\definecolor{line}{HTML}{CBD5E1}
\tikzset{
  box/.style={rounded corners=8pt, draw=line, fill=panel, very thick},
  title/.style={font=\sffamily\bfseries\fontsize{28}{32}\selectfont, text=ink},
  subtitle/.style={font=\sffamily\fontsize{14}{18}\selectfont, text=muted},
  label/.style={font=\sffamily\bfseries\fontsize{13}{15}\selectfont, text=ink},
  small/.style={font=\sffamily\fontsize{11}{13}\selectfont, text=muted},
  arrow/.style={-{Latex[length=4mm]}, very thick},
}
\begin{document}
\begin{tikzpicture}[x=1in,y=1in]
\fill[white] (0,0) rectangle (16,9);
"""

POST = r"""
\end{tikzpicture}
\end{document}
"""


def page(title, subtitle, body):
    return PREAMBLE + rf"""
\node[title, anchor=west] at (0.65,8.45) {{{title}}};
\node[subtitle, anchor=west] at (0.67,8.08) {{{subtitle}}};
{body}
""" + POST


def write_tex(name, title, subtitle, body):
    tex = OUT / f"{name}.tex"
    tex.write_text(page(title, subtitle, body))
    return tex


DIAGRAMS = {
    "exp01_simplenet_noise": (
        "Experiment 0: Vanilla SimpleNet noise",
        "Baseline: isotropic Gaussian perturbations around real normal features.",
        r"""
\node[box, minimum width=14.7in, minimum height=6.7in, anchor=south west] at (0.65,0.65) {};
\draw[blue!15, fill=blue!7] (4.0,4.1) ellipse (2.4 and 1.25);
\node[label, blue] at (4.0,5.65) {normal feature cloud};
\foreach \x/\y in {3.0/4.2,3.6/3.6,4.2/4.3,4.7/3.8,5.0/4.6,3.4/4.8,4.0/3.4} {
  \fill[blue] (\x,\y) circle (0.055);
  \draw[arrow, red] (\x,\y) -- ++({0.55*cos(55+\x*41)},{0.55*sin(55+\y*38)});
}
\node[box, minimum width=4.7in, minimum height=1.25in, anchor=west] at (8.4,4.45) {};
\node[label, anchor=west] at (8.75,5.15) {Generator};
\node[font=\sffamily\fontsize{18}{22}\selectfont, anchor=west, text=ink] at (8.75,4.70)
  {$x_{\mathrm{fake}} = x_{\mathrm{real}} + \epsilon$};
\node[font=\sffamily\fontsize{15}{18}\selectfont, anchor=west, text=muted] at (8.75,4.28)
  {$\epsilon \sim \mathcal{N}(0,\sigma^2 I)$};
\node[small, text width=5.2in, anchor=west] at (8.55,3.25)
  {Reasoning: this is the controlled baseline. Any geometry-aware generator must beat it under the same discriminator, backbone, and training path.};
\node[label, red, anchor=west] at (1.15,1.25) {Before/after: real feature $\rightarrow$ random direction with fixed global scale};
""",
    ),
    "exp02_anchored_pca_threshold": (
        "Experiment 1: Anchored PCA with threshold radius",
        "Replace directionless noise with a real anchor plus a local PCA-subspace shift.",
        r"""
\node[box, minimum width=14.7in, minimum height=6.7in, anchor=south west] at (0.65,0.65) {};
\draw[green!20, fill=green!6, rotate around={-18:(4.4,4.2)}] (4.4,4.2) ellipse (2.65 and 0.75);
\draw[green, very thick, rotate around={-18:(4.4,4.2)}] (4.4,4.2) -- ++(2.55,0);
\draw[green, very thick, rotate around={-18:(4.4,4.2)}] (4.4,4.2) -- ++(0,0.72);
\node[label, green] at (4.4,5.55) {patch PCA geometry};
\fill[blue] (3.55,4.25) circle (0.08);
\node[label, blue] at (3.55,4.55) {real anchor};
\draw[arrow, green] (3.55,4.25) -- (6.15,3.38);
\fill[red] (6.15,3.38) circle (0.08);
\draw[red, dashed, very thick] (4.4,4.2) circle (2.1);
\node[small, red] at (6.7,5.35) {threshold shell};
\node[box, minimum width=5.8in, minimum height=1.45in, anchor=west] at (8.2,4.5) {};
\node[label, anchor=west] at (8.55,5.25) {Generator};
\node[font=\sffamily\fontsize{17}{21}\selectfont, anchor=west, text=ink] at (8.55,4.78)
  {$x_{\mathrm{fake},p}=x_{\mathrm{real},p}+r\,U_p\sqrt{\Lambda_p}v$};
\node[font=\sffamily\fontsize{14}{17}\selectfont, anchor=west, text=muted] at (8.55,4.36)
  {$r=\sqrt{T_p}+\delta U(0,1)$};
\node[small, text width=5.2in, anchor=west] at (8.55,3.15)
  {Reasoning: anchor on the empirical normal manifold and move along directions where real patch features vary.};
\node[label, green, anchor=west] at (1.15,1.25) {Before/after: random isotropic direction $\rightarrow$ local PCA direction from a real feature};
""",
    ),
    "exp03_fixed_radius_curriculum": (
        "Experiment 2: Fixed small radius curriculum",
        "Decouple synthetic anomaly magnitude from the fitted Mahalanobis threshold.",
        r"""
\node[box, minimum width=14.7in, minimum height=6.7in, anchor=south west] at (0.65,0.65) {};
\coordinate (c) at (4.25,4.2);
\fill[blue!8] (c) circle (2.5);
\draw[blue!40, very thick] (c) circle (0.75);
\draw[amber!70, very thick] (c) circle (1.55);
\draw[red!65, very thick] (c) circle (2.45);
\fill[blue] (c) circle (0.08);
\node[label] at (4.25,4.55) {anchor};
\node[small, blue] at (5.15,4.82) {subtle};
\node[small, amber] at (5.85,5.45) {medium};
\node[small, red] at (6.52,6.08) {strong};
\draw[arrow, blue] (c) -- ++(0.72,0.16);
\draw[arrow, amber] (c) -- ++(1.42,0.43);
\draw[arrow, red] (c) -- ++(2.17,0.85);
\node[box, minimum width=5.8in, minimum height=1.45in, anchor=west] at (8.25,4.55) {};
\node[label, anchor=west] at (8.6,5.28) {Radius override};
\node[font=\sffamily\fontsize{17}{21}\selectfont, anchor=west, text=ink] at (8.6,4.80)
  {$r=\rho\,U(0,1)$};
\node[font=\sffamily\fontsize{14}{17}\selectfont, anchor=west, text=muted] at (8.6,4.38)
  {$\rho \in \{0.25,0.5,1,2,5\}$};
\node[small, text width=5.2in, anchor=west] at (8.55,3.15)
  {Reasoning: the statistical threshold may produce anomalies that are too large for localization. Sweep a task-level magnitude instead.};
\node[label, amber, anchor=west] at (1.15,1.25) {Before/after: threshold-pinned shell $\rightarrow$ subtle-to-strong training curriculum};
""",
    ),
    "exp04_threshold_vs_anchor_radius": (
        "Experiment 4: Threshold radius vs. anchored radius",
        "Threshold asks where the boundary is; anchored radius asks how far this feature is from it.",
        r"""
\node[box, minimum width=6.75in, minimum height=5.8in, anchor=south west] at (0.75,1.0) {};
\node[label, red] at (4.1,6.45) {threshold radius};
\coordinate (l) at (4.1,3.75);
\draw[red, dashed, very thick] (l) circle (1.85);
\fill[blue] (3.55,3.92) circle (0.07); \node[small, blue] at (3.55,4.22) {A};
\fill[amber] (5.65,3.55) circle (0.07); \node[small, amber] at (5.65,3.85) {B};
\draw[arrow, red] (3.55,3.92) -- (2.25,2.75);
\draw[arrow, red] (5.65,3.55) -- (6.95,2.75);
\node[small, text width=5.6in, align=center] at (4.1,1.62) {Both anchors get a boundary-scale step: $r=\sqrt{T_p}+\delta U(0,1)$. Anchor location is ignored.};
\node[box, minimum width=6.75in, minimum height=5.8in, anchor=south west] at (8.5,1.0) {};
\node[label, green] at (11.85,6.45) {anchored radius};
\coordinate (r) at (11.85,3.75);
\draw[green, dashed, very thick] (r) circle (1.85);
\fill[blue] (11.25,3.92) circle (0.07); \node[small, blue] at (11.25,4.22) {A};
\fill[amber] (13.32,3.55) circle (0.07); \node[small, amber] at (13.32,3.85) {B};
\draw[arrow, green] (11.25,3.92) -- (10.08,2.72);
\draw[arrow, green] (13.32,3.55) -- (13.72,2.72);
\node[small, text width=5.6in, align=center] at (11.85,1.62) {Each anchor moves by its remaining gap: $r\propto\sqrt{T_p}-\sqrt{s(x_{\mathrm{real}})}$.};
\node[label, green, anchor=west] at (1.15,0.55) {Before/after: fixed boundary shell $\rightarrow$ adaptive near-boundary negatives};
""",
    ),
    "exp05_sparse_patch_masks": (
        "Experiments 5--6: Sparse patch masks",
        "Move from all-fake feature maps to localized synthetic supervision.",
        r"""
\node[box, minimum width=14.7in, minimum height=6.7in, anchor=south west] at (0.65,0.65) {};
\foreach \x in {0,...,9}{\foreach \y in {0,...,5}{\draw[line] (1.4+\x*0.38,2.8+\y*0.38) rectangle ++(0.38,0.38);}}
\foreach \x/\y in {2/1,3/1,4/1,2/2,3/2,4/2,5/2,3/3,4/3}{\fill[red!70] (1.4+\x*0.38,2.8+\y*0.38) rectangle ++(0.38,0.38);}
\node[label] at (3.3,5.75) {block mask};
\foreach \x in {0,...,9}{\foreach \y in {0,...,5}{\draw[line] (6.2+\x*0.38,2.8+\y*0.38) rectangle ++(0.38,0.38);}}
\foreach \x/\y in {1/0,2/4,4/1,5/5,7/2,9/4,3/3,8/0}{\fill[amber!80] (6.2+\x*0.38,2.8+\y*0.38) rectangle ++(0.38,0.38);}
\node[label] at (8.1,5.75) {random mask};
\node[box, minimum width=4.5in, minimum height=2.4in, anchor=west] at (10.6,3.0) {};
\node[label, anchor=west] at (10.95,4.9) {Training change};
\node[small, text width=3.8in, anchor=west] at (10.95,4.42) {Only selected patches become fake. Unselected patches stay equal to their real anchors. Fake loss is computed only on the selected set.};
\node[small, text width=3.8in, anchor=west] at (10.95,3.35) {Reasoning: real industrial defects are local; all-fake maps may teach global shifts instead of localization.};
\node[label, blue, anchor=west] at (1.15,1.25) {Before/after: every patch fake $\rightarrow$ sparse local fake supervision};
""",
    ),
    "exp06_gradient_refinement": (
        "Experiment 7: Gradient-guided PCA refinement",
        "Turn sampled anomalies into hard negatives while staying inside the local PCA geometry.",
        r"""
\node[box, minimum width=14.7in, minimum height=6.7in, anchor=south west] at (0.65,0.65) {};
\node[box, minimum width=2.5in, minimum height=1.0in] (a) at (2.7,4.55) {};
\node[label, align=center] at (2.7,4.68) {anchored\\PCA fake};
\node[box, minimum width=2.5in, minimum height=1.0in] (b) at (5.85,4.55) {};
\node[label, align=center] at (5.85,4.68) {discriminator\\gradient};
\node[box, minimum width=2.5in, minimum height=1.0in] (c) at (9.0,4.55) {};
\node[label, align=center] at (9.0,4.68) {project\\to $U_p$};
\node[box, minimum width=2.5in, minimum height=1.0in] (d) at (12.15,4.55) {};
\node[label, align=center] at (12.15,4.68) {clamp\\radius};
\draw[arrow, blue] (a) -- (b);
\draw[arrow, blue] (b) -- (c);
\draw[arrow, blue] (c) -- (d);
\draw[green!20, fill=green!6] (7.35,2.35) ellipse (2.65 and 0.58);
\draw[green, very thick] (5.2,2.25) -- (9.45,2.46);
\fill[blue] (6.25,2.36) circle (0.07);
\fill[red] (8.85,2.47) circle (0.07);
\draw[arrow, red] (6.25,2.36) -- (8.85,2.47);
\node[small, text width=5.2in, align=center] at (7.35,1.43) {The update searches for harder fake samples, but projection and radius clamping prevent unrestricted adversarial drift.};
\node[label, green, anchor=west] at (1.15,1.0) {Before/after: random fake sample $\rightarrow$ near-boundary hard negative};
""",
    ),
}


def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def build():
    OUT.mkdir(exist_ok=True)
    for name, (title, subtitle, body) in DIAGRAMS.items():
        tex = write_tex(name, title, subtitle, body)
        run([f"{TEXBIN}/pdflatex", "-interaction=nonstopmode", tex.name], cwd=OUT)
        pdf = OUT / f"{name}.pdf"
        png = OUT / f"{name}.png"
        run(["sips", "-s", "format", "png", pdf.as_posix(), "--out", png.as_posix()])
        run(["sips", "-z", "900", "1600", png.as_posix()])
        print(f"wrote {png}")


if __name__ == "__main__":
    build()
