"""Draw figure 1, the study design. Writes results/figures/figure1_study_design.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "DejaVu Sans"

PX = 200.0                  # layout units: pixels at 200 px per inch (traced from the v6 figure)
W, H = 1300, 1455
fig = plt.figure(figsize=(W / PX, H / PX))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

INK = "#222222"; SUB = "#444444"; LINE = "#5a5a5a"; LW = 0.8
STY = {
    "public":  ("#dce6f2", "#3b6aa0"),
    "plain":   ("#ffffff", "#8a8a8a"),
    "dp":      ("#dcefe0", "#2e7d4f"),
    "private": ("#f6dcda", "#a8322d"),
    "attack":  ("#fbebd6", "#b8761a"),
    "label":   ("#f2f2f2", "#8a8a8a"),
}

def box(x0, y0, x1, y1, kind, title, sub=None, tsize=8, ssize=7, dashed=False):
    fc, ec = STY[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=9",
                                fc=fc, ec=ec, lw=1.4, ls=(0, (4, 2.5)) if dashed else "-"))
    cy = (y0 + y1) / 2
    if sub:
        ax.text((x0 + x1) / 2, cy - 10, title, ha="center", va="center", fontsize=tsize, weight="bold", color=INK)
        ax.text((x0 + x1) / 2, cy + 13, sub, ha="center", va="center", fontsize=ssize, color=SUB)
    else:
        ax.text((x0 + x1) / 2, cy, title, ha="center", va="center", fontsize=tsize, weight="bold", color=INK)

def path(pts, arrow=True):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=LINE, lw=LW, solid_capstyle="butt", solid_joinstyle="miter")
    if arrow:
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ax.annotate("", xy=(xb, yb), xytext=(xa + (xb - xa) * 0.9, ya + (yb - ya) * 0.9),
                    arrowprops=dict(arrowstyle="-|>", color=LINE, lw=LW, mutation_scale=7, shrinkA=0, shrinkB=0))

def note(x, y, s, ha="left"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=7, style="italic", color=SUB)

def ptitle(y, letter, s):
    ax.text(40, y, f"{letter}. {s}", ha="left", va="center", fontsize=10, weight="bold", color=INK)

# ---------------- a. calibration ----------------
ptitle(25, "a", "Cross-dataset public-statistic calibration")
box(40, 115, 315, 200, "public", "Public source data", "separate dataset")
box(448, 115, 775, 200, "plain", "Per-feature statistics", "range, variance, mean")
box(448, 275, 775, 360, "plain", "Noise scale per feature", "by rule")
box(932, 265, 1272, 360, "dp", "Clip and add noise once", "DP-protected features")
box(40, 405, 315, 490, "private", "Protected dataset", "private clinical data")
path([(315, 157), (448, 157)])
path([(611, 200), (611, 275)])
path([(775, 157), (884, 157), (884, 298), (932, 298)])
path([(775, 318), (932, 318)])
path([(315, 448), (884, 448), (884, 338), (932, 338)])
note(40, 235, "public statistics cost no privacy budget")
note(872, 221, "clip bounds", ha="right")
note(443, 470, "features: the only noised path out")

# ---------------- b. evaluation ----------------
oy = -60
ptitle(560, "b", "Evaluation factorial")
box(27, 810 + oy, 252, 912 + oy, "dp", "DP-protected", "features")
box(27, 975 + oy, 252, 1077 + oy, "label", "Labels", "public, not noised", dashed=True)
clf = [("KNN", 697), ("Naïve Bayes", 804), ("Random forest", 910), ("Logistic regression", 1017)]
for name, y in clf:
    box(358, y + oy, 648, y + 79 + oy, "plain", name)
    path([(305, y + 40 + oy), (358, y + 40 + oy)])
ax.plot([305, 305], [737 + oy, 1057 + oy], color=LINE, lw=LW)
path([(252, 861 + oy), (305, 861 + oy)], arrow=False)
path([(252, 1026 + oy), (305, 1026 + oy)], arrow=False)
for name, y in clf:
    ax.plot([648, 718], [y + 40 + oy] * 2, color=LINE, lw=LW)
ax.plot([718, 718], [737 + oy, 1132 + oy], color=LINE, lw=LW)
for name, y in [("Membership inference", 748), ("Reconstruction", 860), ("Attribute inference", 972)]:
    box(757, y + oy, 1110, y + 73 + oy, "attack", name)
    path([(718, y + 36 + oy), (757, y + 36 + oy)])
    ax.plot([1110, 1131], [y + 36 + oy] * 2, color=LINE, lw=LW)
ax.plot([1131, 1131], [784 + oy, 1008 + oy], color=LINE, lw=LW)
path([(1131, 896 + oy), (1148, 896 + oy)])
box(1148, 860 + oy, 1273, 933 + oy, "plain", "Privacy")
path([(718, 1132 + oy), (1148, 1132 + oy)])
note(955, 1113 + oy, "held-out AUROC (no attack)", ha="center")
box(1148, 1100 + oy, 1273, 1165 + oy, "plain", "Utility")

# ---------------- c. folds ----------------
cy0 = 1165
ptitle(cy0, "c", "Patient-disjoint fold split")
x0, x1 = 330, 1273
cw = (x1 - x0) / 10
for k in range(10):
    ax.text(x0 + cw * (k + 0.5), cy0 + 50, str(k + 1), ha="center", va="center", fontsize=7, color=SUB)
ax.text(x0 - 12, cy0 + 50, "fold", ha="right", va="center", fontsize=7, style="italic", color=SUB)
rows = [
    ("Utility models", [(1, 8, "plain", "train", False), (9, 9, "plain", "validate", False), (10, 10, "dp", "test", False)]),
    ("Membership inference", [(1, 4, "private", "target model (members)", False),
                              (5, 8, "attack", "shadow models, non-members", False),
                              (9, 9, "label", "unused", True), (10, 10, "dp", "test", False)]),
]
g = 4
for r, (lab, spans) in enumerate(rows):
    ya, yb = cy0 + 75 + r * 95, cy0 + 150 + r * 95
    ax.text(27, (ya + yb) / 2, lab, ha="left", va="center", fontsize=8, weight="bold", color=INK)
    for a, b, kind, txt, dashed in spans:
        xa, xb = x0 + cw * (a - 1) + g, x0 + cw * b - g
        size = 7.5 if b > a else 6.5
        fc, ec = STY[kind]
        ax.add_patch(FancyBboxPatch((xa, ya), xb - xa, yb - ya, boxstyle="round,pad=0,rounding_size=7",
                                    fc=fc, ec=ec, lw=1.4, ls=(0, (4, 2.5)) if dashed else "-"))
        ax.text((xa + xb) / 2, (ya + yb) / 2, txt, ha="center", va="center", fontsize=size,
                weight="bold" if b > a else "normal", color=INK)

from pathlib import Path
out = Path(__file__).resolve().parents[1] / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "figure1_study_design.png", dpi=600)
