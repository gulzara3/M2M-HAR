"""Copy the final paper figures, numbered as in the manuscript, into figures/.

Usage (from the repository root, after the two analysis scripts):
    python analysis/collect_figures.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "paper_analysis" / "figs"
D = ROOT / "analysis" / "diagrams"
FIGS = [  # (manuscript number, short name, source stem)
    (1, "framework", D / "fig_framework"),
    (2, "encoder", D / "fig_encoder"),
    (3, "extraction", A / "Fig2_extraction"),
    (4, "probes", A / "FigP_probes"),
    (5, "verification", A / "FigV_verification"),
    (6, "conditions", A / "Fig4_conditions"),
    (7, "forest", A / "FigF_forest"),
    (8, "defects", A / "Fig5_defects"),
    (9, "fewshot", A / "Fig6_fewshot"),
    (10, "value_reliability", A / "FigQ_value_reliability"),
]
out = ROOT / "figures"
out.mkdir(exist_ok=True)
for n, name, src in FIGS:
    for ext in ("png", "pdf"):
        f = Path(f"{src}.{ext}")
        if f.exists():
            shutil.copy(f, out / f"Fig{n:02d}_{name}.{ext}")
            print("figures/" + f"Fig{n:02d}_{name}.{ext}")
