#!/usr/bin/env python3
"""Schematic figures for the master report. All are data-independent concept drawings.
Fig A  the lift - a bare data model becomes a portable, self-describing semantic model
Fig B  reconciliation - two lifted models: correspondences, a rejected cognate, the residual,
                             and the thin reference as a flat identity bridge
Fig C  cross-domain - Meridian <-> Cascade at one seam; a thin descriptive toehold unlocks it
Fig D  observability - the overloaded legacy alarm lifted and decomposed against the NMOP ladder
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, SURFACE = "#9a958d", "#fbfaf8"
INK = "#14314f"


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / name).name)


def box(ax, x, y, w, h, fc, ec, lw=1.4, pad=0.08, r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def chip(ax, cx, cy, w, h, text, fc, ec, fs=8.6, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight=weight, color=tc)


# --- Figure A: the lift -----------------------------------------------------------------------
def fig_lift():
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 9.6); ax.axis("off"); ax.set_facecolor(SURFACE)

    # ---------- LEFT: data model ----------
    box(ax, 0.3, 3.3, 2.5, 3.1, "#f3f1ee", GREY, lw=1.5)
    ax.text(1.55, 6.05, "data model", ha="center", fontsize=11, weight="bold", color=INK)
    ax.text(1.55, 5.62, "(schema + records)", ha="center", fontsize=8.2, color="#666", style="italic")
    for j, t in enumerate(["term:  och_grade", "rec:  {id:0x7a, v:3}", "rec:  {id:0x7b, v:1}"]):
        ax.text(0.5, 5.1 - j * 0.44, t, ha="left", fontsize=7.8, color="#555", family="monospace")
    ax.text(0.5, 3.7, "- a fixed, partial picture", ha="left", fontsize=7.8, color="#8a8681", style="italic")

    # ---------- MIDDLE: the lift = cognition, with its supports surrounding it ----------
    ax.add_patch(FancyBboxPatch((3.05, 2.1), 2.5, 4.5, boxstyle="round,pad=0.08,rounding_size=0.06",
                                fc="#fcf7f3", ec=ORANGE, lw=1.2, linestyle=(0, (4, 3))))
    ax.text(4.3, 6.3, "supports to the lift", ha="center", fontsize=8.4, weight="bold", color=ORANGE)
    ax.text(4.3, 6.02, "(aids to cognition)", ha="center", fontsize=7.4, color=ORANGE, style="italic")
    for y, t in [(5.55, "· definitions"), (5.2, "· worked examples")]:
        ax.text(4.3, y, t, ha="center", fontsize=7.9, color="#7a5a45")
    # the arrow itself is the lift - a cognitive act - with the supports above and below it
    ax.text(4.3, 4.58, "the lift", ha="center", fontsize=10.5, weight="bold", color=INK)
    ax.add_patch(FancyArrowPatch((2.95, 4.26), (5.85, 4.26), arrowstyle="-|>",
                                 mutation_scale=24, color=BLUE, lw=3.0))
    ax.text(4.3, 3.9, "= a cognitive act", ha="center", fontsize=8.2, weight="bold", color=BLUE)
    for y, t in [(3.35, "· a canonical example"), (3.0, "· linked reference (opt.)")]:
        ax.text(4.3, y, t, ha="center", fontsize=7.9, color="#7a5a45")

    # ---------- RIGHT: the semantic model - one object, three parts ----------
    box(ax, 5.95, 0.85, 6.75, 6.5, "#ffffff", BLUE, lw=1.7)
    ax.text(9.3, 7.02, "ad hoc semantic model", ha="center", fontsize=11.5, weight="bold", color=INK)

    # Part 1: ontology (incl lexicon), schematic + concrete
    box(ax, 6.25, 3.7, 6.15, 2.85, "#eef4fc", "#9fc0e8", lw=1.2)
    ax.text(6.45, 6.28, "ontology  (including lexicon)", ha="left", fontsize=9.6, weight="bold", color=INK)
    ax.text(6.45, 5.92, "schematic - concepts, kinds, relations", ha="left", fontsize=7.6,
            color="#4a6580", style="italic")
    chip(ax, 11.91, 5.92, 0.82, 0.32, "TBox", BLUE, BLUE, fs=7.8, weight="bold", tc="white")
    nodes = {"service": (7.9, 5.35), "port": (9.9, 5.5), "link": (11.6, 5.2)}
    for a, b in [("service", "port"), ("port", "link"), ("service", "link")]:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.plot([x1, x2], [y1, y2], color="#b9cbe0", lw=1.1, zorder=1)
    for t, (nx, ny) in nodes.items():
        chip(ax, nx, ny, 1.15, 0.5, t, "#eaf1fb", BLUE, fs=8.2)
    ax.text(6.45, 4.55, "concrete - the individuals that populate them", ha="left", fontsize=7.6,
            color="#4a6580", style="italic")
    chip(ax, 11.91, 4.55, 0.82, 0.32, "ABox", BLUE, BLUE, fs=7.8, weight="bold", tc="white")
    for (nx, ny) in nodes.values():
        ax.scatter([nx - 0.28, nx, nx + 0.28], [4.15, 4.15, 4.15], s=16, color="#7fa8d6", zorder=2)

    # Part 2: pragmatics
    box(ax, 6.25, 2.45, 6.15, 1.05, "#eafbf3", AQUA, lw=1.2)
    ax.text(6.45, 3.15, "pragmatics", ha="left", fontsize=9.4, weight="bold", color=INK)
    ax.text(6.45, 2.78, "contextual information - use · authority · context", ha="left", fontsize=7.8,
            color="#2a6b52")

    # Part 3: provenance
    box(ax, 6.25, 1.15, 6.15, 1.05, "#f0eeeb", GREY, lw=1.2)
    ax.text(6.45, 1.85, "provenance", ha="left", fontsize=9.4, weight="bold", color=INK)
    ax.text(6.45, 1.48, "who asserted it · by what method · how firmly", ha="left", fontsize=7.8,
            color="#5a564f")

    ax.text(9.3, 0.4, "self-describing  →  portable: this is what cognition consumes, with no "
            "pre-agreed standard", ha="center", fontsize=8.8, weight="bold", color=ORANGE)
    ax.set_title("The lift - cognition turns a data model into an ad hoc semantic model "
                 "(ontology, pragmatics, provenance)", fontsize=11, y=1.01)
    save(fig, "fig_master_lift.png")


# --- Figure B: reconciliation over lifted models ----------------------------------------------
def fig_reconcile():
    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9.6); ax.axis("off"); ax.set_facecolor(SURFACE)

    lx, rx = 2.0, 10.0
    corr = [("service", "tunnel"), ("demarcation", "termination"),
            ("committed-rate", "committed-rate")]
    cys = [7.9, 6.85, 5.8]

    ax.text(lx, 8.9, "lifted model A", ha="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(rx, 8.9, "lifted model B", ha="center", fontsize=10.5, weight="bold", color=INK)

    # thin reference bridge (spans the correspondence rows); title above the box
    box(ax, 5.2, 5.35, 1.6, 3.15, "#fdf1ea", ORANGE, lw=1.5)
    ax.text(6.0, 8.62, "thin reference", ha="center", fontsize=9.2, weight="bold", color=ORANGE)
    ax.text(6.0, 8.28, "(flat identity bridge)", ha="center", fontsize=7.4, color=ORANGE, style="italic")

    for (a, b), y in zip(corr, cys):
        chip(ax, lx, y, 2.5, 0.72, a, "#eaf1fb", BLUE, fs=8.6)
        chip(ax, rx, y, 2.5, 0.72, b, "#eaf1fb", BLUE, fs=8.6)
        ax.add_patch(FancyBboxPatch((5.5, y - 0.15), 1.0, 0.3, boxstyle="round,pad=0.01",
                                    fc="white", ec=ORANGE, lw=0.9))
        ax.add_patch(FancyArrowPatch((lx + 1.3, y), (5.48, y), arrowstyle="-", color=AQUA, lw=1.9))
        ax.add_patch(FancyArrowPatch((6.52, y), (rx - 1.3, y), arrowstyle="-", color=AQUA, lw=1.9))
    ax.text(6.0, 4.95, "resolved correspondences - drawn on grounded evidence, "
            "bound through the reference", ha="center", fontsize=8.3, color="#1b8f63", weight="bold")

    # rejected false cognate - clear below the reference box
    yR = 3.85
    chip(ax, lx, yR, 2.5, 0.72, "grade\n(protection class)", "#fdf1ea", ORANGE, fs=7.8)
    chip(ax, rx, yR, 2.5, 0.72, "grade\n(class of service)", "#fdf1ea", ORANGE, fs=7.8)
    ax.add_patch(FancyArrowPatch((lx + 1.3, yR), (rx - 1.3, yR), arrowstyle="-", color=ORANGE,
                                 lw=1.6, linestyle=(0, (4, 3))))
    ax.text(6.0, yR + 0.02, "✗", ha="center", va="center", fontsize=15, color=ORANGE, weight="bold")
    ax.text(6.0, yR - 0.62, "false cognate rejected - same surface word, different kind",
            ha="center", fontsize=8.0, color=ORANGE, weight="bold")

    # natives with no counterpart -> correctly-unmatched bucket (descriptor inside the bucket, clear of arrows)
    yN = 2.55
    chip(ax, lx, yN, 2.5, 0.6, "vlan  (native)", "#f0eeeb", GREY, fs=7.8, tc="#5a564f")
    chip(ax, rx, yN, 2.5, 0.6, "wavelength  (native)", "#f0eeeb", GREY, fs=7.8, tc="#5a564f")
    for sx in (lx + 1.3, rx - 1.3):
        ax.add_patch(FancyArrowPatch((sx, yN - 0.1), (6.0, 1.66), arrowstyle="-|>", mutation_scale=10,
                                     color=GREY, lw=1.1))
    box(ax, 4.15, 0.35, 3.7, 1.25, "#f0eeeb", GREY, lw=1.3)
    ax.text(6.0, 1.28, "left unbound", ha="center", fontsize=9.0, weight="bold", color="#5a564f")
    ax.text(6.0, 0.93, "no counterpart · correctly unmatched", ha="center", fontsize=6.9,
            color="#6a655e", style="italic")
    ax.text(6.0, 0.57, "unconfirmed · the residual, referred onward", ha="center", fontsize=6.9,
            color="#8a857d", style="italic")

    ax.set_title("Reconciliation over two lifted models - grounded correspondences bound through a "
                 "thin reference, a rejected cognate, and what is honestly left unbound",
                 fontsize=10.6, y=1.02)
    save(fig, "fig_master_reconcile.png")


# --- Figure C: cross-domain, instantiated -----------------------------------------------------
def fig_crossdomain():
    fig, ax = plt.subplots(figsize=(11.6, 5.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off"); ax.set_facecolor(SURFACE)

    ax.text(2.0, 8.4, "Meridian - transport OSS", ha="center", fontsize=10.2, weight="bold", color=INK)
    ax.text(10.0, 8.4, "Cascade - IP / VPN", ha="center", fontsize=10.2, weight="bold", color=INK)
    ax.text(6.0, 8.62, "(both home-grown; no public standard beneath either)", ha="center",
            fontsize=8.3, color="#777", style="italic")

    pairs = [   # (meridian, cascade, kind)
        ("circuit", "underlay", "bind"),
        ("hand-off", "attachment", "bind"),
        ("rate  (committed payload)", "rate  (committed payload)", "pin"),
        ("latency  (bound)", "latency  (bound)", "pin"),
        ("protection  (path failure)", "protection  (path failure)", "pin"),
        ("grade  (protection class)", "grade  (class of service)", "reject"),
    ]
    ys = [7.4, 6.35, 5.3, 4.25, 3.2, 1.95]
    lx, rx = 2.0, 10.0
    for (ml, cl, kind), y in zip(pairs, ys):
        ec = ORANGE if kind == "reject" else BLUE
        fc = "#fdf1ea" if kind == "reject" else "#eaf1fb"
        chip(ax, lx, y, 2.9, 0.72, ml, fc, ec, fs=7.7)
        chip(ax, rx, y, 2.9, 0.72, cl, fc, ec, fs=7.7)
        if kind == "reject":
            ax.add_patch(FancyArrowPatch((lx + 1.5, y), (rx - 1.5, y), arrowstyle="-", color=ORANGE,
                                         lw=1.5, linestyle=(0, (4, 3))))
            ax.text(6.0, y + 0.02, "✗", ha="center", va="center", fontsize=13, color=ORANGE, weight="bold")
        else:
            ax.add_patch(FancyArrowPatch((lx + 1.5, y), (rx - 1.5, y), arrowstyle="-", color=AQUA, lw=1.8))
            tag = "same object,\ntwo names" if kind == "bind" else "one meaning,\npinned"
            ax.text(6.0, y + 0.34, tag, ha="center", va="center", fontsize=6.7, color="#6a6a6a")

    ax.text(6.0, 0.95, "constructed reference: a single descriptive field (label · class · "
            "definition · example) unlocks the strong agent - a bare shared pointer does not",
            ha="center", fontsize=8.4, color=ORANGE, weight="bold")
    ax.text(6.0, 0.45, "the seam - five bindings to make, one look-alike ('grade') to reject",
            ha="center", fontsize=8.2, color="#5a564f", style="italic")
    ax.set_title("Standard-free, instantiated - one order across the Meridian/Cascade seam, "
                 "with no standard to appeal to", fontsize=10.8, y=1.02)
    save(fig, "fig_master_crossdomain.png")


# --- Figure D: observability, instantiated ----------------------------------------------------
def fig_observability():
    fig, ax = plt.subplots(figsize=(11.6, 6.1))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9.4); ax.axis("off"); ax.set_facecolor(SURFACE)

    # legacy overloaded alarm
    box(ax, 0.4, 3.3, 3.0, 3.0, "#fdf1ea", ORANGE, lw=1.6)
    ax.text(1.9, 6.0, "legacy  ALARM", ha="center", fontsize=10.2, weight="bold", color=INK)
    ax.text(1.9, 5.55, "one object, bundling:", ha="center", fontsize=8.2, color="#555", style="italic")
    for j, t in enumerate(["· event", "· undesirable state", "· fixed severity", "· probable-cause"]):
        ax.text(1.15, 5.1 - j * 0.42, t, ha="left", fontsize=8.3, color="#555")

    # "lift + decompose" label - the three coloured arrows below carry the mapping itself
    ax.text(4.3, 6.5, "lift + decompose", ha="center", fontsize=7.8, color="#777")

    # NMOP ladder rungs (targets)
    rungs = [
        ("alarm  (a State)", AQUA, "#eafbf3", "correct core"),
        ("fault", AQUA, "#eafbf3", "+ decomposition"),
        ("anomaly  (a deviation)", ORANGE, "#fdf1ea", "TRAP: not a state"),
        ("symptom · problem · cause · incident", BLUE, "white", ""),
    ]
    x0 = 5.15
    ys = [6.55, 5.5, 4.45, 3.3]
    for (t, ec, fc, note), y in zip(rungs, ys):
        w = 4.2 if "symptom" not in t else 5.7
        chip(ax, x0 + w / 2, y, w, 0.72, t, fc, ec, fs=8.4,
             weight="bold" if ec in (AQUA, ORANGE) else "normal")
        if note:
            ax.text(x0 + w + 0.35, y, note, ha="left", va="center", fontsize=7.6,
                    color=ec if ec != AQUA else "#1b8f63", weight="bold" if "TRAP" in note else "normal")
    # decomposition - three clean arrows straight from the ALARM box to the NMOP rungs, no mid-air junction
    ax.add_patch(FancyArrowPatch((3.55, 5.5), (x0 + 0.03, ys[0] - 0.1), arrowstyle="-|>",
                                 mutation_scale=13, color=AQUA, lw=1.6))          # -> alarm (State)
    ax.add_patch(FancyArrowPatch((3.55, 4.9), (x0 + 0.03, ys[1]), arrowstyle="-|>",
                                 mutation_scale=13, color=AQUA, lw=1.6))          # -> fault
    ax.add_patch(FancyArrowPatch((3.55, 4.3), (x0 + 0.03, ys[2] + 0.05), arrowstyle="-|>",
                                 mutation_scale=13, color=ORANGE, lw=1.5, linestyle=(0, (3, 2))))  # -> anomaly (trap)

    ax.text(7.3, 7.5, "NMOP  (RFC 9940 ladder)", ha="left", fontsize=9.6, weight="bold", color=INK)

    # anomaly-semantics annotations chip (the lifted content)
    box(ax, 4.9, 0.55, 6.7, 1.7, "#eaf1fb", BLUE, lw=1.4)
    ax.text(8.25, 1.9, "anomaly-semantics annotations - the lifted content the verdict runs on",
            ha="center", fontsize=8.2, weight="bold", color=INK)
    ann = "concern · confidence · plane · pattern · lifecycle · season   →   act / watch / suppress"
    ax.text(8.25, 1.15, ann, ha="center", fontsize=8.0, color="#33506e")

    ax.text(1.9, 2.4, "an alarm is\nnot an anomaly", ha="center", fontsize=8.6, color=ORANGE,
            weight="bold")
    ax.set_title("Observability, instantiated - the overloaded legacy alarm lifted and decomposed "
                 "one-to-many into the NMOP ladder; the deep cognate is alarm ≠ anomaly",
                 fontsize=10.6, y=1.02)
    save(fig, "fig_master_observability.png")


# --- Figure E: cognitive load is capability-signed (cross-setting synthesis) ------------------
def fig_effort():
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    settings = ["intent\n(scenario 2)", "observability\n(scenario 4)"]
    data = {"intent\n(scenario 2)": {"sol": 150, "mini": 860, "nano": 5400},
            "observability\n(scenario 4)": {"sol": 60, "mini": 520, "nano": 1200}}
    models = ["sol", "mini", "nano"]
    colM = {"sol": BLUE, "mini": ORANGE, "nano": AQUA}
    labM = {"sol": "sol (strong)", "mini": "mini (mid)", "nano": "nano (weak)"}
    x = range(len(settings)); w = 0.26
    for k, mdl in enumerate(models):
        xs = [i + (k - 1) * w for i in x]
        ys = [data[s][mdl] for s in settings]
        ax.bar(xs, ys, width=w, color=colM[mdl], label=labM[mdl], edgecolor="white")
        for xi, yv in zip(xs, ys):
            ax.text(xi, yv * 1.07, f"{yv:,}", ha="center", va="bottom", fontsize=8, color="#555")
    ax.set_yscale("log"); ax.set_ylim(35, 16000)
    ax.set_xticks(list(x)); ax.set_xticklabels(settings, fontsize=9.5)
    ax.set_ylabel("reasoning tokens to decide\n(mean, log scale)", fontsize=9)
    ax.set_title("Cognitive load rises as capability falls: across scenarios the weak agent spends "
                 "~20–35×\nthe strong agent's effort - to reach lower accuracy, not higher",
                 fontsize=10.3)
    # ratio brackets
    for i, s in enumerate(settings):
        r = data[s]["nano"] / data[s]["sol"]
        ax.text(i, 13000, f"nano/sol ≈ {r:.0f}×", ha="center", fontsize=8.6, color="#14314f", weight="bold")
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.28))
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig_master_effort.png")


if __name__ == "__main__":
    fig_lift()
    fig_reconcile()
    fig_crossdomain()
    fig_observability()
    fig_effort()
