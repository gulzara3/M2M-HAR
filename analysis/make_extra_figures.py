"""Additional paper figures (probe decision space, verification by model, E3 forest plot, value vs reliability).

Usage (from the repository root, after make_paper_figures.py):
    python analysis/make_extra_figures.py results/notebook_outputs results/paper_analysis results/paper_analysis/figs

Writes FigP_probes, FigV_verification, FigF_forest, FigQ_value_reliability (PDF + PNG) and kbs_fig_stats.json.
"""
import json, glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(sys.argv[1]); P7 = Path(sys.argv[2]); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": INK2,
    "axes.linewidth": 0.8, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig); print("figure", name)

DSN = {"mhealth": "MHEALTH", "uschad": "USC-HAD", "harth": "HARTH", "motionsense": "MotionSense", "hapt": "HAPT", "pamap2": "PAMAP2",
       "wisdm19": "WISDM-19", "dsads": "DSADS", "realdisp": "REALDISP"}
AE = pd.read_csv(R / "audit/audit_extracted.csv"); AG = pd.read_csv(R / "audit/audit_gold.csv")
GOLD = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(R / "gold/*.json"))}
T2 = pd.read_csv(R / "tables/T2_extraction_field_scores.csv")
TE3 = pd.read_csv(P7 / "tables/T_E3_pairwise.csv"); TE4 = pd.read_csv(P7 / "tables/T_E4_costs.csv")
out = {}

# ---------------- Fig: probe decision space ----------------
acc = AG[AG.field == "accelerometer_unit"].copy(); gyr = AG[AG.field == "gyroscope_unit"].copy()
acc["norm"] = acc.data.str.extract(r"≈([\d.]+)").astype(float); gyr["p99"] = gyr.data.str.extract(r"≈([\d.]+)").astype(float)
acc = acc.dropna(subset=["norm"]).sort_values("norm"); gyr = gyr.dropna(subset=["p99"]).sort_values("p99")
out["probe_acc"] = dict(zip(acc.dataset, acc.norm)); out["probe_gyr"] = dict(zip(gyr.dataset, gyr.p99))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.55))
ax = axes[0]
for lo, hi, lab, c in [(0.6, 1.5, "g", PAL[0]), (6, 14, "m/s²", PAL[2]), (600, 1500, "milli-g", PAL[3])]:
    ax.axvspan(lo, hi, color=c, alpha=0.13, lw=0); ax.text(np.sqrt(lo * hi), len(acc) - 0.25, lab, ha="center", va="bottom", fontsize=7.5, color=c, weight="bold")
for i, (_, r) in enumerate(acc.iterrows()):
    c = PAL[0] if r.documented == "g" else PAL[2]
    ax.scatter(r.norm, i, s=34, color=c, edgecolor="white", lw=0.6, zorder=3)
    ax.text(r.norm * 1.25, i, f"{r.norm:.2f}", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(range(len(acc))); ax.set_yticklabels([DSN[d] for d in acc.dataset]); ax.set_xscale("log"); ax.set_xlim(0.3, 3000)
ax.set_ylim(-0.7, len(acc) + 0.5); ax.set_xlabel("resting acceleration magnitude (native units, log)")
ax.set_title("a  Accelerometer-unit probe (gravity at rest)", loc="left")
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=PAL[0], label="documented g"), Line2D([0], [0], marker="o", ls="", color=PAL[2], label="documented m/s²")],
          loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6.3)
ax = axes[1]
ax.axvspan(20, 2000, color=PAL[3], alpha=0.12, lw=0); ax.axvline(20, color=INK2, lw=0.8, ls=(0, (3, 2)))
ax.text(2.2, len(gyr) - 0.35, "→ rad/s", fontsize=7.5, color=PAL[0], weight="bold"); ax.text(140, len(gyr) - 0.35, "→ deg/s", fontsize=7.5, color=PAL[3], weight="bold", ha="center")
for i, (_, r) in enumerate(gyr.iterrows()):
    ok = r.status == "confirmed"; c = PAL[0] if r.documented == "rad/s" else PAL[3]
    ax.scatter(r.p99, i, s=34, color=c, edgecolor=INK if not ok else "white", lw=1.2 if not ok else 0.6, zorder=3, marker="o" if ok else "X")
    ax.text(r.p99 * 1.3, i, f"{r.p99:.1f}" + ("  doc: deg/s ≠ probe" if not ok else ""), va="center", fontsize=6.5, color=PAL[7] if not ok else INK2)
ax.set_yticks(range(len(gyr))); ax.set_yticklabels([DSN[d] for d in gyr.dataset]); ax.set_xscale("log"); ax.set_xlim(0.5, 2000)
ax.set_ylim(-0.7, len(gyr) + 0.4); ax.set_xlabel("99th percentile of |ω| per axis (native units, log)")
ax.set_title("b  Gyroscope-unit probe (threshold 20)", loc="left")
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=PAL[0], label="documented rad/s"), Line2D([0], [0], marker="o", ls="", color=PAL[3], label="documented deg/s"),
                   Line2D([0], [0], marker="X", ls="", color=PAL[3], mec=INK, label="contradicted")], loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, fontsize=6.3, columnspacing=0.8, handletextpad=0.2)
save(fig, "FigP_probes")

# ---------------- Fig: verification, pooled + per family ----------------
def unit_of(s): return s.split(" ")[0] if isinstance(s, str) else None
rows = []
for _, r in AE[AE.field.isin(["sampling_rate_hz", "accelerometer_unit", "gyroscope_unit"])].iterrows():
    g = GOLD[r.dataset][r.field]
    if g is None: continue
    if r.field == "sampling_rate_hz":
        if isinstance(g, list): continue
        try: d = float(r.documented)
        except Exception: d = None
        before = d is not None and abs(d - g) / g <= 0.1
        after = (abs(float(r.data) - g) / g <= 0.1) if r.status == "contradicted" else before
    else:
        before = r.documented == g; after = (unit_of(r.data) == g) if r.status == "contradicted" else before
    rows.append(dict(model=r.model, variant=r.variant, dataset=r.dataset, field=r.field, status=r.status, verifiable=r.status != "unverifiable", before=before, after=after))
PV = pd.DataFrame(rows); PVv = PV[PV.verifiable]
pv = PVv.groupby("field").agg(n=("before", "size"), before=("before", "mean"), after=("after", "mean")).reindex(["accelerometer_unit", "sampling_rate_hz", "gyroscope_unit"])
fam = PVv.groupby("model").agg(before=("before", "mean"), after=("after", "mean"), n=("before", "size"))
famu = PVv[PVv.field == "accelerometer_unit"].groupby("model").agg(before=("before", "mean"), after=("after", "mean"), n=("before", "size"))
out["verif_by_field"] = pv.round(4).to_dict(orient="index"); out["verif_by_family_all"] = fam.round(4).to_dict(orient="index"); out["verif_by_family_accunit"] = famu.round(4).to_dict(orient="index")
out["verif_overall"] = dict(before=float(PVv.before.mean()), after=float(PVv.after.mean()), n=int(len(PVv)))
MN = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen2.5-14B-Instruct": "Qwen2.5-14B", "Mistral-7B-Instruct-v0.3": "Mistral-7B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "gemma-2-9b-it": "Gemma-2-9B"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.45))
ax = axes[0]; x = np.arange(len(pv)); w = 0.36
ax.bar(x - w / 2 - 0.01, pv.before * 100, w, color=PAL[1], label="extracted (LLM)")
ax.bar(x + w / 2 + 0.01, pv.after * 100, w, color=PAL[2], label="extracted + verified")
for i, (b, a) in enumerate(zip(pv.before, pv.after)):
    ax.text(i - w / 2, b * 100 + 2, f"{b*100:.0f}", ha="center", fontsize=6.6); ax.text(i + w / 2, a * 100 + 2, f"{a*100:.0f}", ha="center", fontsize=6.6)
ax.set_xticks(x); ax.set_xticklabels([f"{t}\n(n = {n})" for t, n in zip(["Acc. unit", "Sampling rate", "Gyro. unit"], pv.n)], fontsize=7)
ax.set_ylim(0, 115); ax.set_ylabel("agreement with gold (%)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, fontsize=6.8)
ax.set_title("a  By field (pooled)", loc="left", y=1.18)
ax = axes[1]; order = famu.sort_values("before").index
for i, m in enumerate(order):
    b, a = famu.loc[m, "before"] * 100, famu.loc[m, "after"] * 100
    ax.plot([b, a], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(b, i, color=PAL[1], s=40, zorder=3); ax.scatter(a, i, color=PAL[2], s=40, zorder=3)
    ax.text(b - 3, i, f"{b:.0f}", ha="right", va="center", fontsize=6.6)
ax.set_yticks(range(len(order))); ax.set_yticklabels([MN.get(m, m) for m in order]); ax.set_xlim(20, 108); ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlabel("accelerometer-unit agreement with gold (%)"); ax.set_title("b  Accelerometer unit by model", loc="left", y=1.18)
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=PAL[1], label="extracted"), Line2D([0], [0], marker="o", ls="", color=PAL[2], label="verified")], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
save(fig, "FigV_verification")

# ---------------- Fig: E3 forest plot ----------------
fig, ax = plt.subplots(figsize=(7.2, 2.35))
lab = {"M4 gold − M0 none": "Gold vs none (M4 − M0)", "M3 verified − M0 none": "Verified vs none (M3 − M0)", "M3 verified − M2 extracted": "Verified vs extracted (M3 − M2)",
       "M4 gold − M3 verified": "Gold vs verified (M4 − M3)", "M1 generic − M0 none": "Channel text only (M1 − M0)"}
T = TE3.iloc[::-1].reset_index(drop=True)
for i, r in T.iterrows():
    c = PAL[2] if "M3 verified − M2" in r.comparison else (PAL[0] if "M4 gold − M3" in r.comparison else INK2)
    ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=2, solid_capstyle="round"); ax.scatter(r.delta_pts, i, color=c, s=42, zorder=3, marker="D")
    fm = lambda v: f"{v:.2f}" if abs(v) < 0.1 else f"{v:.1f}"
    ax.text(41, i, f"{r.delta_pts:+.1f}  [{fm(r.ci_lo)}, {fm(r.ci_hi)}]", va="center", fontsize=7, color=INK)
    ax.text(62, i, f"{r.p_holm:.2f}", va="center", fontsize=7, color=INK)
    dsv = [r[f"Δ {DSN[k]}"] for k in ["uschad", "harth", "motionsense", "mhealth"]]
    for j, v in enumerate(dsv): ax.scatter(v, i + (j - 1.5) * 0.09, s=8, color=PAL[[0, 1, 3, 4][j]], alpha=0.8, zorder=2, marker="o")
ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(range(len(T))); ax.set_yticklabels([lab[c] for c in T.comparison])
ax.set_xlim(-40, 70); ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
ax.text(41, len(T) - 0.35, "Δ (points) [95% CI]", fontsize=7, weight="bold"); ax.text(62, len(T) - 0.35, "Holm p", fontsize=7, weight="bold")
ax.set_xlabel("difference in zero-shot accuracy (points)"); ax.set_ylim(-0.6, len(T) + 0.1)
ax.legend(handles=[Line2D([0], [0], marker="D", ls="-", color=INK2, label="mean over 12 (data set × seed) pairs"),
                   *[Line2D([0], [0], marker="o", ls="", ms=3.5, color=PAL[[0, 1, 3, 4][j]], label=DSN[k]) for j, k in enumerate(["uschad", "harth", "motionsense", "mhealth"])]],
          loc="lower center", bbox_to_anchor=(0.36, -0.52), ncol=5, fontsize=6.6)
save(fig, "FigF_forest")

# ---------------- Fig: value vs reliability ----------------
ext = T2[["sampling_rate_hz", "accelerometer_unit", "activities", "sensor_placements"]].mean()
E4c = TE4[TE4.severity == 1.0].set_index("defect")
pts = [("Activity-label semantics", "activities", "label", None, "file-verifiable"),
       ("Accelerometer unit", "accelerometer_unit", "unit", "accelerometer_unit", "signal-verifiable"),
       ("Sampling rate", "sampling_rate_hz", "rate", None, "signal-verifiable"),
       ("Sensor placement", "sensor_placements", "placement", None, "not independently verifiable")]
CC = {"signal-verifiable": PAL[2], "file-verifiable": PAL[0], "not independently verifiable": PAL[4]}
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ax.axhspan(10, 36, xmin=0, xmax=1, color=PAL[7], alpha=0.05, lw=0)
for name, f, d, vf, cls in pts:
    x = ext[f] * 100; y = E4c.loc[d, "cost"]; lo, hi = E4c.loc[d, "ci_lo"], E4c.loc[d, "ci_hi"]
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=CC[cls], ms=7, capsize=2.5, lw=1.2, zorder=3)
    if vf is not None:
        xa = pv.loc[vf, "after"] * 100
        ax.annotate("", (xa, y), (x + 1.2, y), arrowprops=dict(arrowstyle="-|>", color=PAL[2], lw=1.3, ls="--"))
        ax.scatter(xa, y, s=55, facecolor="white", edgecolor=PAL[2], lw=1.5, zorder=4)
    dy = {"Activity-label semantics": 2.2, "Accelerometer unit": 2.4, "Sampling rate": 2.2, "Sensor placement": -3.6}[name]
    ax.text(x, y + dy, name, ha="center", fontsize=7.2, color=INK, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6), zorder=5)
    out.setdefault("value_reliability", {})[name] = dict(extraction=float(x), cost=float(y), ci=[float(lo), float(hi)])
ax.axhline(0, color=INK2, lw=0.8); ax.set_xlim(35, 108); ax.set_ylim(-7, 38)
ax.set_xlabel("automatic acquisition accuracy (%)\n(LLM extraction, mean of 10 runs; ○ after signal verification)")
ax.set_ylabel("cost of a 100% defect (points of\nzero-shot accuracy, 95% CI)")
ax.text(36, 34.5, "high-value fields", fontsize=6.8, color=PAL[7], style="italic")
ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=c, label=k) for k, c in CC.items()], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.5, title="verifiability", title_fontsize=6.8)
save(fig, "FigQ_value_reliability")

json.dump(out, open(OUT / "kbs_fig_stats.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:3000])
