#!/usr/bin/env python3
"""Create a lightweight PPTX deck using only the Python standard library."""

from html import escape
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


OUT = Path("FEATURE_ANOMALY_PRESENTATION.pptx")
SLIDE_W = 12192000
SLIDE_H = 6858000


slides = [
    {
        "title": "Feature-Level Anomaly Generation",
        "subtitle": "From SimpleNet random noise to geometry-aware synthetic defects",
        "bullets": [
            "Research question: can feature-level anomalies be better than isotropic noise?",
            "Core idea: anchor on real normal features, move in local PCA directions, calibrate radius, and localize the supervision.",
            "Deliverable: an experiment ladder that tests each design choice separately.",
        ],
    },
    {
        "title": "The Problem With Random Feature Noise",
        "bullets": [
            "SimpleNet trains a discriminator with fake features made by adding Gaussian noise to normal features.",
            "This is efficient, but directionless: every channel direction receives the same noise budget.",
            "Deep patch features are not isotropic; useful variation lives in structured, patch-specific subspaces.",
            "Random noise can be easy to classify without resembling real defects.",
        ],
    },
    {
        "title": "Where The Ideas Come From",
        "bullets": [
            "SimpleNet: feature-space synthetic negatives and discriminator training.",
            "PaDiM: patch-wise Gaussian/Mahalanobis modeling of normal features.",
            "PatchCore: real normal patch features are a strong empirical manifold.",
            "CutPaste, NSA, DRAEM, DeSTSeg: synthetic anomalies should provide local supervision.",
            "GLASS: feature anomalies can be gradient-guided hard negatives.",
            "CRAS: anomaly magnitude should be distance-aware, not fixed blindly.",
        ],
    },
    {
        "title": "Current Diagnosis",
        "bullets": [
            "A self-consistent Gaussian model is not automatically useful for SimpleNet training.",
            "Full-sphere Mahalanobis anomalies mostly move in low-variance orthogonal directions.",
            "Anchoring on real normal features and moving inside the PCA subspace improved image AUROC.",
            "Remaining gap: localization is weaker than image-level detection.",
        ],
    },
    {
        "title": "Geometry-Aware Generator",
        "bullets": [
            "For each patch p, fit a low-rank Gaussian: Sigma_p = U_p Lambda_p U_p^T + eps_p I.",
            "Generate anchored anomalies: x_fake,p = x_real,p + r U_p sqrt(Lambda_p) v.",
            "v is sampled on the k-dimensional PCA sphere.",
            "The generator controls four things: anchor, direction, radius, and spatial mask.",
        ],
    },
    {
        "title": "Experiment Ladder",
        "bullets": [
            "0. Vanilla SimpleNet noise: re-run the baseline in the same training path.",
            "1. Anchored PCA with threshold radius: reproduce the current geometry-aware baseline.",
            "2. Fixed small radius: decouple training magnitude from Mahalanobis threshold.",
            "3-4. Patch and anchor radius sweeps: test local magnitude calibration.",
            "5-6. Sparse random and block masks: test localization supervision.",
            "7. Gradient refinement: test discriminator-guided hard negatives.",
        ],
    },
    {
        "title": "Experiments 0-2: Baselines And Magnitude",
        "bullets": [
            "Exp 0 asks: what does vanilla SimpleNet noise achieve under the same code path?",
            "Exp 1 asks: does real-anchor + PCA direction beat random feature noise?",
            "Exp 2 asks: was the Mahalanobis threshold radius simply too large?",
            "Interpretation: if small radius helps localization, the earlier anomalies were too far from realistic defects.",
        ],
    },
    {
        "title": "Experiments 3-4: Radius Calibration",
        "bullets": [
            "Patch radius: r = rho sqrt(T_p / C) U(0, 1).",
            "Anchor radius: r depends on the gap between each anchor and its patch threshold.",
            "Reasoning: different patches and anchors have different normal margins.",
            "Goal: produce near-boundary negatives that are hard but plausible.",
        ],
    },
    {
        "title": "Experiments 5-6: Sparse Patch Masks",
        "bullets": [
            "Random mask: synthetic patches are scattered across the feature grid.",
            "Block mask: synthetic patches are spatially coherent connected regions.",
            "Fake loss is applied only to selected synthetic patches.",
            "Reasoning: real defects are local; all-fake feature maps may teach global shifts instead of localization.",
        ],
    },
    {
        "title": "Experiment 7: Gradient-Guided Hard Negatives",
        "bullets": [
            "Start from anchored PCA fake features.",
            "Use discriminator gradients to make them harder negatives.",
            "Project updates back into U_p and clamp Mahalanobis radius.",
            "Reasoning: random samples may miss the current decision boundary; hard negatives can tighten it.",
        ],
    },
    {
        "title": "Metrics And Diagnostics",
        "bullets": [
            "Primary: image AUROC, pixel AUROC, PRO, anomaly-pixel AUROC.",
            "Training: p_true, p_fake, fake_patch_ratio, loss.",
            "Geometry to add next: generated Mahalanobis radius, Euclidean shift norm, PCA-vs-residual energy.",
            "Success requires test metrics, not only easy fake classification during training.",
        ],
    },
    {
        "title": "How To Read Outcomes",
        "bullets": [
            "Image AUROC up, pixel metrics flat: generator helps detection but not localization.",
            "Pixel metrics up, image AUROC down: synthetic signal may be too sparse or subtle.",
            "High p_fake, weak test metrics: fake anomalies are too easy or unrealistic.",
            "Gradient refinement hurts: adversarial directions may not match real defects.",
            "Block masks beat random masks: spatial coherence matters.",
        ],
    },
    {
        "title": "Expected Contribution",
        "bullets": [
            "A controlled study of feature-level anomaly synthesis beyond random noise.",
            "A generator family that is real-anchored, patch-adaptive, radius-calibrated, and optionally hard-negative guided.",
            "Ablations that identify whether direction, magnitude, sparsity, or refinement matters most.",
            "Thesis: synthetic feature anomalies should respect the local geometry of normal features.",
        ],
    },
]


def textbox(x, y, w, h, text, size=2800, bold=False, color="1F2937"):
    runs = []
    for line in text.split("\n"):
        runs.append(
            f"""
            <a:r>
              <a:rPr lang="en-US" sz="{size}" {'b="1"' if bold else ''}>
                <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
              </a:rPr>
              <a:t>{escape(line)}</a:t>
            </a:r>
            <a:br/>"""
        )
    body = "".join(runs)
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{textbox.next_id}" name="TextBox {textbox.next_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/>
        <a:lstStyle/>
        <a:p>{body}</a:p>
      </p:txBody>
    </p:sp>"""


textbox.next_id = 10


def bullet_box(x, y, w, h, bullets):
    paras = []
    for bullet in bullets:
        paras.append(
            f"""
            <a:p>
              <a:pPr marL="342900" indent="-171450">
                <a:buChar char="•"/>
              </a:pPr>
              <a:r>
                <a:rPr lang="en-US" sz="2300">
                  <a:solidFill><a:srgbClr val="111827"/></a:solidFill>
                </a:rPr>
                <a:t>{escape(bullet)}</a:t>
              </a:r>
            </a:p>"""
        )
    textbox.next_id += 1
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{textbox.next_id}" name="Bullets {textbox.next_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square"/>
        <a:lstStyle/>
        {''.join(paras)}
      </p:txBody>
    </p:sp>"""


def slide_xml(slide):
    textbox.next_id = 10
    title = textbox(650000, 420000, 10900000, 720000, slide["title"], size=3600, bold=True, color="0F172A")
    subtitle = ""
    if "subtitle" in slide:
        textbox.next_id += 1
        subtitle = textbox(650000, 1120000, 10900000, 480000, slide["subtitle"], size=2300, color="475569")
    bullets = bullet_box(900000, 1750000 if subtitle else 1450000, 10300000, 4300000, slide["bullets"])
    footer = textbox(650000, 6400000, 10400000, 220000, "SimpleNet feature-anomaly research", size=1200, color="64748B")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr>
      {title}
      {subtitle}
      {bullets}
      {footer}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def write_pptx():
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(slides) + 1)
    )
    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    sld_ids = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1)
    )
    presentation = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(slides)+1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""
    pres_rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, len(slides) + 1)
    )
    pres_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {pres_rels}
  <Relationship Id="rId{len(slides)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
</Relationships>"""
    slide_master = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""
    slide_layout = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld>
</p:sldLayout>"""
    theme = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="SimpleNet Theme">
  <a:themeElements>
    <a:clrScheme name="SimpleNet"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="F8FAFC"/></a:lt1><a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="E2E8F0"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="059669"/></a:accent2><a:accent3><a:srgbClr val="DC2626"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="EA580C"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="SimpleNet"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="SimpleNet"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
  </a:themeElements>
</a:theme>"""
    app = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{len(slides)}</Slides>
</Properties>"""
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Feature-Level Anomaly Generation</dc:title>
  <dc:creator>SimpleNet research project</dc:creator>
  <cp:keywords>anomaly detection; SimpleNet; feature synthesis</cp:keywords>
</cp:coreProperties>"""

    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/app.xml", app)
        z.writestr("docProps/core.xml", core)
        z.writestr("ppt/presentation.xml", presentation)
        z.writestr("ppt/_rels/presentation.xml.rels", pres_rels)
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""")
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""")
        z.writestr("ppt/theme/theme1.xml", theme)
        for i, slide in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""")


if __name__ == "__main__":
    write_pptx()
    print(f"Wrote {OUT}")
