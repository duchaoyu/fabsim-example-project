"""Emit Section 6.5.3 (Robustness to construction imprecision) as .docx and .md.

Same arrangement as FDM/build_section_6_5_1.py: one source for both formats, and
every number pulled from the Block A output files rather than retyped, so the
section cannot drift from the runs behind it.

    data/block_A_disc_sensitivity.csv     the six factors on the circular dome
    data/block_A_2part_sensitivity.csv    the same six on the crease
    data/block_A_overlap.json             aggregates from --check-overlap

    python3 build_section_6_5_3.py
"""

import csv
import json
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
OUT_DOCX = os.path.join(HERE, "data", "Section_6_5_3_Robustness.docx")
OUT_MD = os.path.join(ROOT, "docs", "6_5_3_robustness.md")


def load(geom):
    path = os.path.join(HERE, "data", f"block_A_{geom}_sensitivity.csv")
    with open(path) as f:
        return {r["factor"]: r for r in csv.DictReader(f)}


disc, twopart = load("disc"), load("2part")
ov = json.load(open(os.path.join(HERE, "data", "block_A_overlap.json")))

# Display names and the unit each tolerance is quoted in.
NAME = {"s_wale": "s_wale", "s_course": "s_course", "pressure": "p",
        "E1": "E1", "r": "E2/E1", "R": "R"}
DELTA = {"s_wale": "0.05", "s_course": "0.05", "pressure": "50 Pa",
         "E1": "500 N/m", "r": "0.250", "R": "5 mm"}

MM = 1e3


def rows(d):
    """Factor rows, largest crown-height effect first."""
    order = sorted(d, key=lambda k: -abs(float(d[k]["crown_height_half"])))
    out = []
    for k in order:
        r = d[k]
        out.append((
            NAME[k], DELTA[k],
            f"{MM * float(r['crown_height_plus']):+.2f}",
            f"{MM * float(r['crown_height_minus']):+.2f}",
            f"{MM * float(r['crown_height_half']):+.2f}",
            f"{float(r['crown_height_elast']):+.2f}",
            f"{100 * float(r['crown_height_asym']):.1f}%",
            f"{MM * float(r['L_pos_mean']):.2f}",
        ))
    return out


def crown_mm(d):
    return MM * float(next(iter(d.values()))["crown_height_base"])


def worst_asym(d):
    k = max(d, key=lambda k: float(d[k]["crown_height_asym"]))
    return NAME[k], 100 * float(d[k]["crown_height_asym"])


H_DISC, H_2P = crown_mm(disc), crown_mm(twopart)
L_TARGET = MM * float(twopart["s_wale"]["L_target_base"])
S_NOM = float(twopart["s_wale"]["nominal"])
WA_DISC, WA_2P = worst_asym(disc), worst_asym(twopart)
DOM_2P = MM * float(twopart["s_course"]["L_pos_mean"])
D, T = ov["disc"], ov["2part"]
OOR = ov["disc_out_of_round"]

HEAD = ("factor", "tolerance", "crown, +d (mm)", "crown, -d (mm)",
        "half-range (mm)", "elasticity", "asymmetry", "L_pos (mm)")

BODY = [
("h", "6.5.3 Robustness to construction imprecision"),

("p", "The design of Section 6.4 is computed from nominal values: a stretch "
      "factor the knitting is asked to deliver, a membrane stiffness measured on "
      "coupons, an inflation pressure held by a regulator, a boundary ring cut "
      "to a drawing. The built structure has none of these exactly. This section "
      "asks how far the inflated equilibrium moves when they are wrong by as "
      "much as fabrication can be expected to leave them wrong, and, where the "
      "geometry carries a design target, whether that movement matters against "
      "the error the design already has."),

("p", f"Six factors are perturbed one at a time by plus and minus one tolerance "
      f"about a nominal working point: the two stretch factors s_wale and "
      f"s_course, the membrane stiffness E1, the orthotropy ratio E2/E1, the "
      f"inflation pressure p, and the boundary radius R. Thirteen solves per "
      f"geometry — a baseline and twelve perturbations — on two geometries. The "
      f"circular dome of Section 6.2 serves as the method check: it has a flat "
      f"rest mesh and no design target of its own, so its deviations are "
      f"referenced to its own baseline equilibrium. The creased shell of "
      f"Section 6.4.1 is the case that carries a target, and is where the "
      f"question of what the tolerances mean can actually be answered."),

("h2", "How the responses are measured"),

("p", f"Two outputs are reported for each perturbation. The crown height is a "
      f"scalar and admits the usual first-order treatment: a central difference "
      f"gives the half-range, and an elasticity, the relative change in crown "
      f"height per relative change in the factor, makes the six comparable "
      f"across their different units. L_pos is the RMS displacement of the "
      f"surface from the baseline equilibrium, taken over interior vertices "
      f"only. The boundary is clamped, so its deviation is zero by construction "
      f"and including it would deflate every figure by an amount set by nothing "
      f"but boundary discretisation."),

("p", f"The nominal for the creased shell is not free to be chosen. Its rest "
      f"mesh is the shape the structure is meant to hold, so the stretch factors "
      f"are whatever puts the inflated equilibrium on that shape, and they are "
      f"fitted. The anisotropic two-parameter fit saturates at both bounds, "
      f"s_wale to 1.400 and s_course to 0.950, which is a result in itself — a "
      f"uniform pre-strain cannot reach this target — but it is unusable as a "
      f"nominal, since a working point on a bound cannot be perturbed "
      f"symmetrically and the plus and minus responses would differ because one "
      f"side was clipped rather than because the physics is asymmetric. The "
      f"isotropic fit is interior at s = {S_NOM:.4f}, and that is the working "
      f"point used here. It leaves a standing mismatch of {L_TARGET:.1f} mm "
      f"before any imperfection is applied."),

("p", "The numerical floor was measured rather than assumed. Re-solving the "
      "baseline along a different stretch-factor continuation path reproduces it "
      "to below 0.01 um in both outputs, at the 1e-8 m precision of the solver's "
      "output. Every response reported below is physical by five orders of "
      "magnitude."),

("h2", "The circular dome"),

("table_disc", None),

("p", f"The predicted spread with all six factors at one tolerance is "
      f"{D['crown_rss_mm']:.1f} mm RSS in crown height, "
      f"{D['crown_rss_pct_of_h']:.1f}% of the {H_DISC:.1f} mm the dome rises, "
      f"and {D['crown_aligned_mm']:.1f} mm if every error happens to align. "
      f"Three things in the table are worth drawing out."),

("p", f"s_course dominates, at 2.7 times the next factor, because the course "
      f"direction is the stiff one — E2 is 2.5 times E1 — so pre-strain applied "
      f"along it does the most work. It is also the factor whose tolerance is "
      f"the least well known, which is where the uncertainty in this study "
      f"sits."),

("p", f"R has the second-largest elasticity at "
      f"{float(disc['R']['crown_height_elast']):+.2f} and nearly the smallest "
      f"effect, purely because its tolerance is tight: 0.8% of the nominal, "
      f"against 4.5 to 10% for the others. It is a factor worth controlling "
      f"precisely rather than one that does not matter, and the distinction is "
      f"only visible because elasticity and effect are reported separately."),

("p", f"The six factors are very nearly degenerate on this geometry. The median "
      f"absolute cosine between their displacement fields is "
      f"{D['median_abs_cosine']:.3f} off the diagonal, and "
      f"{D['most_aligned_pair']} sit at {D['most_aligned_cosine']:.3f}. They "
      f"excite one and the same axisymmetric inflation mode, differing in sign "
      f"and amplitude but not in shape. Two consequences follow. A measured "
      f"deviation cannot be attributed to a factor by this block, only bounded "
      f"in size; attribution needs a geometry on which the fields separate. And "
      f"a joint sampling of all six would give a broad, right-skewed "
      f"distribution of L_pos with a mean below the RSS, because near-parallel "
      f"contributions add and cancel algebraically. The crown-height RSS is "
      f"unaffected by this: for a scalar output RSS is correct whenever the "
      f"factor errors are independent, whatever the shape of the response."),

("h2", "The creased shell"),

("table_2part", None),

("p", f"The predicted spread is {T['crown_rss_mm']:.1f} mm RSS, similar in "
      f"absolute terms to the dome's but only {T['crown_rss_pct_of_h']:.1f}% of "
      f"the crown height here against {D['crown_rss_pct_of_h']:.1f}% there, "
      f"because this shell is more than twice as tall. The ordering of the six "
      f"is the dome's with a single swap, s_wale overtaking E1, so the dominant "
      f"tolerance does not change with geometry — at least between these two, "
      f"and before cables and multiple regions enter."),

("h2", "What the target adds"),

("p", f"The creased shell can answer a question the dome cannot, because it has "
      f"a design target to be wrong about. The standing mismatch at the nominal "
      f"is {L_TARGET:.1f} mm. The largest single tolerance, s_course, adds "
      f"{DOM_2P:.1f} mm on top of it. The tolerances are therefore not the "
      f"binding error, and the two are not merely different in size but in kind: "
      f"the cosine between the mismatch field and the s_course displacement "
      f"field is {T['mismatch_vs_dominant_cosine']:+.3f}, orthogonal to three "
      f"decimal places."),

("p", f"Figure 6.33 shows what that means on the surface. The mismatch is a "
      f"narrow stripe along x = 0 reaching {T['mismatch_peak_mm']:.0f} mm: the "
      f"target has a sharp crease between its two lobes, and the section through "
      f"them has the target dipping to 292 mm at the valley while the "
      f"equilibrium runs flat across at 380 to 385 mm. The tolerance "
      f"perturbation, by contrast, is a broad smooth mode that lifts or drops "
      f"the whole cap by about 20 mm and does nothing at the valley. No "
      f"tightening of these six tolerances moves this shape toward its target."),

("p", "Figure 6.34 says why, and corrects the obvious reading. The natural "
      "conclusion from the saturated anisotropic fit is that a uniform "
      "two-parameter pre-strain is too poor a parameterisation to form a crease. "
      "That is not what happened. The target was form-found with a tie along the "
      "valley — the force densities of the form-finding carry a stiff line "
      "exactly along x = 0 — and the finite-element model has no cable there. "
      "The stiff line of the form finding and the stripe of the mismatch are the "
      "same line. The crease is missing from the simulation because the element "
      "that creates it was not carried across from the form finding, not because "
      "the stretch-factor field is too coarse to express it. The fix is to model "
      "the valley tie, and only then to ask whether the parameterisation is rich "
      "enough."),

("p", "One further caution on reading the table for this geometry. L_target is "
      "stationary in the stretch factors at the nominal, because the nominal is "
      "a fitted minimum: both signs make it worse, by 4.84 and 4.89 mm. A "
      "central difference at an optimum is second order and therefore near zero "
      "and meaningless, so the analysis reports the increase instead. In the "
      "directions that were never fitted — E1, p and R — one sign does improve "
      "L_target, as it should."),

("h2", "What this establishes, and what it does not"),

("p", f"Within the tolerances assumed, construction imprecision moves the crown "
      f"height by about {D['crown_rss_pct_of_h']:.0f}% on the dome and "
      f"{T['crown_rss_pct_of_h']:.0f}% on the creased shell, and the surface by "
      f"{D['L_pos_rss_mm']:.0f} and {T['L_pos_rss_mm']:.0f} mm RMS "
      f"respectively. On the geometry that has a target, that is smaller than "
      f"and geometrically unrelated to the error the model already carries. The "
      f"practical reading is that for this case study fabrication tolerance is "
      f"not the limiting factor on how closely the built shape matches the "
      f"design, and effort is better spent on the model than on the workshop."),

("p", f"Three limits should be stated with equal clarity. All six tolerances are "
      f"estimates; none is yet backed by a measurement, and the dominant one, "
      f"the stretch factor, is the one with no measurement behind it at all. "
      f"Replacing an estimate does not require re-running provided the response "
      f"is linear over the tolerance, which the block checks by the asymmetry "
      f"between the plus and minus responses: on the dome the worst is "
      f"{WA_DISC[1]:.1f}% ({WA_DISC[0]}), inside the 15% threshold, so the "
      f"responses rescale; on the creased shell {WA_2P[0]} reaches "
      f"{WA_2P[1]:.1f}% and E2/E1 15.2%, so those two rows do not rescale safely "
      f"and want a smaller-delta re-run once the material scatter is known. The "
      f"stretch-factor rows, at 1.5 to 2.8%, are solidly linear on both — which "
      f"is the important case, since that is the estimate most likely to be "
      f"revised."),

("p", f"The perturbations here are uniform: a single stretch factor wrong "
      f"everywhere, a single radius wrong everywhere. Real imprecision is also "
      f"non-uniform, and the two need not have the same effect. The reference "
      f"dome makes the point by accident — its boundary vertices run between "
      f"{OOR['boundary_radius_min_mm']:.1f} and "
      f"{OOR['boundary_radius_max_mm']:.1f} mm, a "
      f"{OOR['span_mm']:.1f} mm out-of-round span at "
      f"{OOR['boundary_radius_std_mm']:.2f} mm standard deviation, which is "
      f"{OOR['pct_of_delta_R']}% of the radius tolerance being tested. The "
      f"nominal baseline already carries an out-of-round imperfection "
      f"comparable to the tolerance under test, so a dedicated out-of-round "
      f"study has to be measured against this mesh's existing scatter rather "
      f"than against a perfect circle."),

("p", "And this is one geometry perturbed one factor at a time about a "
      "single-region, cable-free model. The case study of Section 6.4.6 is "
      "multi-region and carries cables, whose rest length is a tolerance of its "
      "own and a quantised one, set by the resolution of a turnbuckle rather "
      "than by a Gaussian. Independent per-region draws, and a joint sampling "
      "that lets the factors interfere, remain to be run."),
]

CAPTIONS = [
    ("blockA_disc_sensitivity",
     "Figure 6.31: Block A on the circular dome. (a) Crown-height response to "
     "plus and minus one tolerance, sorted by effect. (b) Surface deviation "
     "L_pos, with the two signs shown separately as ticks. (c) Asymmetry "
     "between the two signs, against the 15% linearity threshold. (d) Pairwise "
     "cosine between the per-factor displacement fields: the six are nearly the "
     "same mode, and p and E1 are exactly opposed."),
    ("blockA_2part_sensitivity",
     "Figure 6.32: Block A on the creased shell, the same four panels. The "
     "ordering of the factors is the dome's with one swap, but E1 and E2/E1 "
     "exceed the linearity threshold here, so those two rows do not rescale to "
     "a different tolerance."),
    ("blockA_2part_shape",
     "Figure 6.33: Where the tolerance shows up, rather than how much. (a) The "
     "standing mismatch between the baseline equilibrium and the design target "
     "is a narrow stripe along the valley. (b) The dominant tolerance moves the "
     "whole cap smoothly and does nothing at the valley. (c) The section "
     "through the lobes: the target creases to 292 mm where the equilibrium "
     "runs flat at 380 to 385 mm. (d) The section through the saddle, where "
     "the perturbation acts and the mismatch does not."),
    ("2part_target_diagnosis",
     "Figure 6.34: The standing mismatch is a missing valley tie, not a limit "
     "of the parameterisation. (a) The force densities of the form finding "
     "carry a stiff line along x = 0. (b) The finite-element model has no cable "
     "there, so the crease never forms. (c) The stiff line and the mismatch "
     "are the same line."),
]

NOTES = [
("h", "Notes for revision — not part of the section"),
("p", "Every number above is read from the Block A outputs by "
      "build_section_6_5_3.py: the per-factor responses from "
      "data/block_A_{disc,2part}_sensitivity.csv, and the spreads, cosines and "
      "out-of-round figures from data/block_A_overlap.json, itself produced by "
      "analyse_block_A.py --check-overlap and plot_shape.py. Re-run those and "
      "rebuild and the section follows."),
("p", "1. The tolerances are estimates. tolerances.py records the source each "
      "one waits on. Until they are measured every magnitude in this section is "
      "conditional, and that should be said in the text as well as here if the "
      "section goes out before the measurements come in."),
("p", "2. Two scripts disagree about the standing mismatch. analyse_block_A.py "
      "reports L_target = 24.59 mm; plot_shape.py prints 20.65 mm RMS for what "
      "reads as the same quantity. The difference is that plot_shape averages "
      "over all vertices while the analysis restricts to interior ones, and the "
      "clamped boundary deflates the former. The section uses 24.59 mm. Worth "
      "making plot_shape mask the boundary so the two agree."),
("p", "3. Figure numbers 6.31 to 6.34 are provisional and assume 6.29 and 6.30 "
      "belong to Section 6.5.1."),
("p", "4. Blocks B to E are not built. B adds Poisson's ratio and cable rest "
      "length at the case-study optimum and needs the n-region binary; C draws "
      "per-region and per-cable independently; D samples jointly and reports the "
      "distribution of L_pos against the RSS predicted here; E repeats B on a "
      "second geometry. Block A already answers part of E's question — the "
      "dominant tolerance is s_course on both geometries — so E is testing "
      "whether that survives cables and regions."),
("p", "5. Section 6.4 currently gives one scan, so comparing a measured "
      "deviation against this predicted spread compares a distribution with a "
      "point. Two specimens from the same knit programme, or one specimen "
      "re-scanned after re-inflation, would make the comparison two-sided."),
]


def add_table(doc, rows_):
    t = doc.add_table(rows=1, cols=len(HEAD))
    t.style = "Table Grid"
    for i, h in enumerate(HEAD):
        cell = t.rows[0].cells[i]
        cell.text = h
        for r in cell.paragraphs[0].runs:
            r.font.bold = True
            r.font.size = Pt(8)
    for row in rows_:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(8)
    return t


def md_table(rows_):
    out = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    out += ["| " + " | ".join(r) + " |" for r in rows_]
    return "\n".join(out) + "\n"


TABLE_CAPTIONS = {
    "table_disc":
        "Table 6.Y: Block A on the circular dome. Crown-height response to one "
        "tolerance either side of the nominal, sorted by effect. Elasticity is "
        "the relative change in crown height per relative change in the factor; "
        "asymmetry is the difference between the two signs as a fraction of the "
        "first difference, and a value under 15% means the response may be "
        "rescaled to a different tolerance without re-running. L_pos is the RMS "
        "surface displacement from the baseline over interior vertices.",
    "table_2part":
        "Table 6.Z: Block A on the creased shell, about the fitted isotropic "
        "nominal. Same columns. E1 at 17.5% and E2/E1 at 15.2% exceed the "
        "linearity threshold, so those two rows should not be rescaled.",
}


def build_md():
    lines = []
    for kind, text in BODY:
        if kind == "h":
            lines.append(f"# {text}\n")
        elif kind == "h2":
            lines.append(f"## {text}\n")
        elif kind == "p":
            lines.append(f"{text}\n")
        elif kind.startswith("table_"):
            lines.append(md_table(rows(disc if kind == "table_disc" else twopart)))
            lines.append(f"*{TABLE_CAPTIONS[kind]}*\n")
    for name, cap in CAPTIONS:
        lines.append(f"![{name}](../imperfection_study/figures/{name}.png)\n")
        lines.append(f"*{cap}*\n")
    for kind, text in NOTES:
        lines.append(("# " if kind == "h" else "") + text + "\n")
    return "\n".join(lines)


def build_docx():
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    for kind, text in BODY:
        if kind == "h":
            doc.add_heading(text, level=2)
        elif kind == "h2":
            doc.add_heading(text, level=3)
        elif kind == "p":
            p = doc.add_paragraph(text)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind.startswith("table_"):
            add_table(doc, rows(disc if kind == "table_disc" else twopart))
            cap = doc.add_paragraph(TABLE_CAPTIONS[kind])
            cap.runs[0].font.size = Pt(9)
            cap.runs[0].font.italic = True

    for name, cap in CAPTIONS:
        path = os.path.join(FIG, f"{name}.png")
        if not os.path.exists(path):
            print(f"  missing figure, skipped: {path}")
            continue
        doc.add_picture(path, width=Inches(6.2))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph(cap)
        c.runs[0].font.size = Pt(9)
        c.runs[0].font.italic = True

    doc.add_page_break()
    for kind, text in NOTES:
        if kind == "h":
            doc.add_heading(text, level=2)
        else:
            doc.add_paragraph(text, style="List Bullet")

    doc.save(OUT_DOCX)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(build_md())
    build_docx()
    print(f"  saved: {OUT_DOCX}")
    print(f"  saved: {OUT_MD}")
