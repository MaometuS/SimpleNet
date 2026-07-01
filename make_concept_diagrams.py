#!/usr/bin/env python3
"""Dependency-free SVG diagrams for the feature-anomaly presentation."""

from pathlib import Path


INK = "#111827"
MUTED = "#64748B"
BLUE = "#2563EB"
GREEN = "#059669"
RED = "#DC2626"
AMBER = "#D97706"
PANEL = "#F8FAFC"
LINE = "#CBD5E1"


def svg_header(width, height):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="{RED}" />
    </marker>
    <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="{GREEN}" />
    </marker>
    <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="{BLUE}" />
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.12"/>
    </filter>
  </defs>
'''


def text(x, y, value, size=20, color=INK, weight="400", anchor="start"):
    return f'<text x="{x}" y="{y}" font-family="Aptos, Inter, Helvetica, Arial, sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{value}</text>'


def rect(x, y, w, h, fill=PANEL, stroke=LINE, rx=14):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" filter="url(#shadow)"/>'


def line(x1, y1, x2, y2, color=BLUE, width=4, marker="arrow-blue", dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" marker-end="url(#{marker})"{dash_attr}/>'


def circle(cx, cy, r, fill="none", stroke=LINE, width=2, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def point(cx, cy, color=BLUE, label=None):
    out = [f'<circle cx="{cx}" cy="{cy}" r="8" fill="{color}" stroke="white" stroke-width="3"/>']
    if label:
        out.append(text(cx, cy - 14, label, size=15, color=INK, weight="700", anchor="middle"))
    return "\n".join(out)


def threshold_vs_anchor():
    w, h = 1500, 820
    parts = [svg_header(w, h)]
    parts.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    parts.append(text(w / 2, 54, "Threshold radius vs. anchored radius", 34, INK, "800", "middle"))
    parts.append(text(w / 2, 88, "Threshold asks: “where is the boundary?”  Anchored asks: “how far is this real feature from the boundary?”", 18, MUTED, "400", "middle"))

    # Panels.
    parts.append(rect(60, 130, 660, 590))
    parts.append(rect(780, 130, 660, 590))
    parts.append(text(390, 175, "Threshold radius", 27, RED, "800", "middle"))
    parts.append(text(1110, 175, "Anchored radius", 27, GREEN, "800", "middle"))

    for ox, color in [(390, RED), (1110, GREEN)]:
        cy = 430
        parts.append(circle(ox, cy, 190, fill="#EFF6FF", stroke="#93C5FD", width=2))
        parts.append(circle(ox, cy, 190, stroke=INK, width=3, dash="10 8"))
        parts.append(circle(ox, cy, 55, fill="#DBEAFE", stroke="none"))
        parts.append(point(ox, cy, "#0F172A", "mu"))
        parts.append(text(ox + 145, cy - 132, "normal boundary", 15, MUTED, "500"))
        parts.append(text(ox + 145, cy - 110, "sqrt(Tp)", 15, MUTED, "500"))

    # Threshold panel anchors and equal-radius arrows.
    cy = 430
    central = (330, cy + 28)
    edge = (515, cy - 42)
    parts.append(point(*central, BLUE, "A"))
    parts.append(point(*edge, AMBER, "B"))
    parts.append(line(central[0], central[1], 190, 305, RED, 5, "arrow-red"))
    parts.append(line(edge[0], edge[1], 640, 330, RED, 5, "arrow-red"))
    parts.append(text(390, 655, "Same boundary-scale step for both anchors", 18, INK, "700", "middle"))
    parts.append(text(390, 684, "r = sqrt(Tp) + delta · U(0,1)", 17, MUTED, "500", "middle"))
    parts.append(text(390, 708, "Anchor location is ignored", 16, RED, "700", "middle"))

    # Anchored panel anchors and gap arrows.
    central = (1048, cy + 28)
    edge = (1235, cy - 42)
    parts.append(point(*central, BLUE, "A"))
    parts.append(point(*edge, AMBER, "B"))
    parts.append(line(central[0], central[1], 936, 298, GREEN, 5, "arrow-green"))
    parts.append(line(edge[0], edge[1], 1276, 292, GREEN, 5, "arrow-green"))
    parts.append(text(1110, 655, "Step length depends on each anchor's gap", 18, INK, "700", "middle"))
    parts.append(text(1110, 684, "r = rho · (sqrt(Tp) - sqrt(s(xreal))) · U(0,1)", 17, MUTED, "500", "middle"))
    parts.append(text(1110, 708, "Central anchor moves more; boundary anchor moves less", 16, GREEN, "700", "middle"))

    # Bottom takeaway.
    parts.append(f'<rect x="210" y="754" width="1080" height="44" rx="22" fill="#ECFDF5" stroke="#A7F3D0"/>')
    parts.append(text(750, 783, "Anchored radius creates near-boundary negatives by moving each real feature toward its own local boundary.", 18, GREEN, "800", "middle"))
    parts.append("</svg>")
    Path("threshold_vs_anchored_radius.svg").write_text("\n".join(parts))


def experiment_before_after():
    w, h = 1800, 1280
    rows = [
        ("0", "Vanilla noise baseline", "Uncontrolled comparison", "Same path: xreal + Gaussian noise", RED),
        ("1", "Anchored PCA", "Synthetic from model geometry only", "Real anchor + local PCA shift", GREEN),
        ("2", "Fixed small radius", "Threshold magnitude may be huge", "Free radius sweep, independent of Tp", AMBER),
        ("3", "Patch radius", "One scale for every patch", "Radius normalized by sqrt(Tp / C)", BLUE),
        ("4", "Anchor radius", "Same radius regardless of anchor", "Move each anchor toward its boundary", GREEN),
        ("5", "Sparse random mask", "Every patch labeled fake", "Random local fake patches only", AMBER),
        ("6", "Sparse block mask", "Independent fake patches", "Connected defect-like regions", BLUE),
        ("7", "Gradient refinement", "Random fake samples", "Hard negatives guided by discriminator", GREEN),
    ]
    parts = [svg_header(w, h)]
    parts.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    parts.append(text(w / 2, 54, "Before / after map for the proposed experiments", 34, INK, "800", "middle"))
    parts.append(text(w / 2, 88, "Each experiment changes one mechanism so the ablation can tell a causal story.", 18, MUTED, "400", "middle"))

    x0, y0 = 70, 132
    col_id, col_exp, col_before, col_after = 80, 230, 610, 1160
    row_h = 126
    parts.append(f'<rect x="{x0}" y="{y0}" width="1660" height="58" rx="18" fill="{INK}"/>')
    parts.append(text(col_id, y0 + 38, "ID", 18, "white", "800"))
    parts.append(text(col_exp, y0 + 38, "Experiment", 18, "white", "800"))
    parts.append(text(col_before, y0 + 38, "Before", 18, "white", "800"))
    parts.append(text(col_after, y0 + 38, "After", 18, "white", "800"))

    for i, (eid, name, before, after, color) in enumerate(rows):
        y = y0 + 76 + i * row_h
        fill = "#F8FAFC" if i % 2 == 0 else "#FFFFFF"
        parts.append(f'<rect x="{x0}" y="{y}" width="1660" height="{row_h - 16}" rx="18" fill="{fill}" stroke="{LINE}"/>')
        parts.append(f'<circle cx="{col_id + 22}" cy="{y + 54}" r="26" fill="{color}"/>')
        parts.append(text(col_id + 22, y + 62, eid, 20, "white", "800", "middle"))
        parts.append(text(col_exp, y + 43, name, 22, INK, "800"))
        parts.append(text(col_exp, y + 72, "script: experiments/run_0" + eid + ("_*.sh" if eid != "0" else "_simplenet_noise.sh"), 14, MUTED, "500"))
        parts.append(f'<rect x="{col_before - 26}" y="{y + 22}" width="430" height="64" rx="14" fill="#FEF2F2" stroke="#FECACA"/>')
        parts.append(text(col_before - 6, y + 62, before, 17, "#7F1D1D", "700"))
        parts.append(line(col_before + 430, y + 54, col_after - 70, y + 54, BLUE, 3, "arrow-blue"))
        parts.append(f'<rect x="{col_after - 26}" y="{y + 22}" width="500" height="64" rx="14" fill="#ECFDF5" stroke="#A7F3D0"/>')
        parts.append(text(col_after - 6, y + 62, after, 17, "#064E3B", "700"))

    parts.append(f'<rect x="250" y="1195" width="1300" height="50" rx="25" fill="#EFF6FF" stroke="#BFDBFE"/>')
    parts.append(text(900, 1227, "Read the ladder left-to-right: direction, magnitude, locality, then hardness.", 19, BLUE, "800", "middle"))
    parts.append("</svg>")
    Path("experiment_before_after.svg").write_text("\n".join(parts))


if __name__ == "__main__":
    threshold_vs_anchor()
    experiment_before_after()
    print("wrote threshold_vs_anchored_radius.svg")
    print("wrote experiment_before_after.svg")
