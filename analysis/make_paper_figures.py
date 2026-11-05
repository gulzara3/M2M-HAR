"""Regenerate the paper's statistics, tables and data figures from the raw notebook outputs.

Usage (from the repository root):
    python analysis/make_paper_figures.py results/notebook_outputs results/paper_analysis

Inputs : <notebook_outputs>/{results,tables,audit,gold}/...  (written by the notebook)
Outputs: <out>/figs/*.pdf|png, <out>/tables/*.csv, <out>/stats.json
No number in the paper is typed by hand: every value comes from this script or make_extra_figures.py.
(Fig1_pipeline produced here is an early schematic; the paper's Fig. 1 is analysis/diagrams/fig_framework.html.)
"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

R = Path(sys.argv[1]); OUT = Path(sys.argv[2]); (OUT / "figs").mkdir(parents=True, exist_ok=True); (OUT / "tables").mkdir(exist_ok=True)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc3ee", "#2a78d6", "#123d73"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": INK2,
    "axes.linewidth": 0.8, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(OUT / "figs" / f"{name}.{ext}")
    plt.close(fig); print("figure", name)

COND = ["M0_none", "M1_generic", "M2_extracted", "M3_verified", "M4_gold"]
CL = {"M0_none": "M0 none", "M1_generic": "M1 generic", "M2_extracted": "M2 extracted", "M3_verified": "M3 verified", "M4_gold": "M4 gold"}
MK = {"M0_none": "o", "M1_generic": "s", "M2_extracted": "^", "M3_verified": "D", "M4_gold": "P"}
DSN = {"mhealth": "MHEALTH", "uschad": "USC-HAD", "harth": "HARTH", "motionsense": "MotionSense", "hapt": "HAPT", "pamap2": "PAMAP2",
       "wisdm19": "WISDM-19", "dsads": "DSADS", "realdisp": "REALDISP"}
TEST = ["uschad", "harth", "motionsense", "mhealth"]
FIELDS = ["sampling_rate_hz", "accelerometer_unit", "gyroscope_unit", "n_participants", "activities", "sensor_placements", "modalities", "devices", "protocol_type", "setting"]
FL = {"sampling_rate_hz": "Sampling\nrate", "accelerometer_unit": "Acc.\nunit", "gyroscope_unit": "Gyro.\nunit", "n_participants": "Partici-\npants",
      "activities": "Activity\nlabels", "sensor_placements": "Place-\nments", "modalities": "Modal-\nities", "devices": "Devices", "protocol_type": "Protocol\ntype", "setting": "Setting"}

E3 = pd.read_csv(R / "results/E3_runs.csv"); E4 = pd.read_csv(R / "results/E4_runs.csv")
T2 = pd.read_csv(R / "tables/T2_extraction_field_scores.csv"); T8 = pd.read_csv(R / "tables/T8_label_alignment.csv")
AE = pd.read_csv(R / "audit/audit_extracted.csv"); AG = pd.read_csv(R / "audit/audit_gold.csv")
GOLD = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(R / "gold/*.json"))}
T1 = pd.read_csv(R / "tables/T1_datasets_overview.csv")
stats = {}

def signflip(d, n=20000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float); obs = d.mean()
    null = (d[None, :] * rng.choice([-1, 1], (n, len(d)))).mean(1); return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))
def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float); b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj

# ======================= Fig 1: pipeline schematic =======================
fig, ax = plt.subplots(figsize=(7.2, 3.7)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
def box(x, y, w, h, title, sub, color, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fill, ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.4, weight="bold", color=INK, linespacing=1.15)
    ax.text(x + w / 2, y + 2.0, sub, ha="center", va="bottom", fontsize=6.2, color=INK2, linespacing=1.3)
def arrow(x0, y0, x1, y1, color=INK2):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=0, shrinkB=0))
H = 15
box(0.5, 24, 18, 18, "Documentation", "README, landing\npage, paper PDF", PAL[0])
box(25, 24, 20, 18, "1  Grounded\nextraction", "4-bit open LLM,\nretrieval, JSON schema", PAL[0])
box(51.5, 24, 20, 18, "2  Signal\nverification", "rate: timestamps\nunit: |gravity|\nlabels, IDs: files", PAL[2])
box(78, 24, 21.5, 18, "Harmonisation\nrecord", "units, rate, placements,\nlabel semantics", PAL[2], fill="#f2faf6")
box(51.5, 48, 20, 10, "Raw data files", "IMU streams, native units", PAL[2])
box(78, 2, 21.5, H, "3  Compile", "unit + rate normalisation,\nchannel text, prototypes", PAL[0])
box(40, 2, 31.5, H, "4  Cross-data set encoder", "channel-token transformer\nconditioned on channel text;\nzero/few-shot via prototypes", PAL[0])
box(0.5, 2, 33, H, "5  Causal measurement", "M0 none, M1 generic, M2 extracted,\nM3 verified, M4 gold;\ninjected defects -> cost curves", PAL[1], fill="#fdf3ee")
arrow(19.3, 33, 24.2, 33); arrow(45.8, 33, 50.7, 33); arrow(72.3, 33, 77.2, 33)
arrow(61.5, 47.5, 61.5, 42.8, PAL[2]); ax.text(62.8, 45.1, "probes read the data", fontsize=6.0, color=PAL[2], va="center")
arrow(88.75, 23.2, 88.75, 17.8); arrow(77.2, 9.5, 72.3, 9.5); arrow(39.2, 9.5, 34.3, 9.5)
save(fig, "Fig1_pipeline")

# ======================= Fig 2: extraction accuracy heatmap (fixed row/bar alignment) =======================
T2 = T2.copy(); T2["label"] = T2.model.str.replace("-Instruct", "") + " · " + T2.variant
T2 = T2.sort_values(["model", "variant"], ascending=[True, True]).reset_index(drop=True)
mat = T2[FIELDS].values
fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.27 * len(T2) + 1.25), gridspec_kw=dict(width_ratios=[5.2, 1.0], wspace=0.05))
ax = axes[0]; im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(FIELDS))); ax.set_xticklabels([FL[f] for f in FIELDS], fontsize=6.6)
ax.set_yticks(range(len(T2))); ax.set_yticklabels(T2.label, fontsize=7); ax.grid(False)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.4, color="white" if mat[i, j] > 0.62 else INK)
for x in (2.5, 4.5, 6.5): ax.axvline(x, color="white", lw=2.2)
for x0, x1, t in ((-0.5, 2.5, "signal-verifiable"), (2.5, 4.5, "file-verifiable"), (4.5, 6.5, "layout"), (6.5, 9.5, "not verifiable")):
    ax.text((x0 + x1) / 2, -0.75, t, ha="center", va="bottom", fontsize=6.4, color=INK2)
for s in ax.spines.values(): s.set_visible(False)
axb = axes[1]; y = np.arange(len(T2)); axb.barh(y, T2.composite, color=PAL[0], height=0.55)
axb.set_ylim(len(T2) - 0.5, -0.5); axb.set_xlim(0, 1); axb.set_yticks([]); axb.set_xticks([0, 0.5, 1]); axb.set_title("Composite", fontsize=7.5, pad=12)
for i, v in enumerate(T2.composite): axb.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=6.6)
cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01); cb.ax.tick_params(labelsize=6); cb.set_label("accuracy / set-F1", fontsize=6.6)
save(fig, "Fig2_extraction")

# ======================= Physical-field verification (recomputed) =======================
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
        verifiable = r.status != "unverifiable"
    else:
        before = r.documented == g; after = (unit_of(r.data) == g) if r.status == "contradicted" else before; verifiable = r.status != "unverifiable"
    rows.append(dict(model=r.model, variant=r.variant, dataset=r.dataset, field=r.field, status=r.status, verifiable=verifiable, before=before, after=after))
PV = pd.DataFrame(rows); PV.to_csv(OUT / "tables/verification_physical_rows.csv", index=False)
pv = PV[PV.verifiable].groupby("field").agg(n=("before", "size"), before=("before", "mean"), after=("after", "mean")).reindex(["accelerometer_unit", "sampling_rate_hz", "gyroscope_unit"])
flags = PV[PV.status == "contradicted"]; stats["verif_flags"] = int(len(flags)); stats["verif_probe_agrees_gold"] = float(flags.after.mean())
stats["verif_physical_before"] = float(PV[PV.verifiable].before.mean()); stats["verif_physical_after"] = float(PV[PV.verifiable].after.mean())
stats["verif_by_field"] = pv.round(3).to_dict(orient="index")
# share of contradiction flags by probe class (all fields)
grp = {"sampling_rate_hz": "signal", "accelerometer_unit": "signal", "gyroscope_unit": "signal", "activities": "file", "n_participants": "file", "sensor_placements": "layout", "modalities": "layout"}
ae2 = AE[AE.field.isin(grp)].copy(); ae2["cls"] = ae2.field.map(grp); ae2 = ae2[ae2.status != "unverifiable"]
stats["contradiction_rate_by_class"] = ae2.groupby("cls").apply(lambda g: float((g.status == "contradicted").mean())).round(3).to_dict()
stats["contradiction_n_by_class"] = ae2.groupby("cls").size().to_dict()

fig, ax = plt.subplots(figsize=(3.5, 2.5)); x = np.arange(len(pv)); w = 0.36
ax.bar(x - w / 2 - 0.01, pv.before * 100, w, color=PAL[1], label="extracted (LLM)")
ax.bar(x + w / 2 + 0.01, pv.after * 100, w, color=PAL[2], label="extracted + verified")
for i, (b, a, n) in enumerate(zip(pv.before, pv.after, pv.n)):
    ax.text(i - w / 2, b * 100 + 2, f"{b*100:.0f}", ha="center", fontsize=6.6); ax.text(i + w / 2, a * 100 + 2, f"{a*100:.0f}", ha="center", fontsize=6.6)
ax.set_xticks(x); ax.set_xticklabels([f"{t}\n(n = {n})" for t, n in zip(["Accelerometer unit", "Sampling rate", "Gyroscope unit"], pv.n)], fontsize=7); ax.set_ylim(0, 112); ax.set_ylabel("agreement with gold (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
save(fig, "Fig3_verification")

# ======================= E3 per data set with seed points =======================
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.5), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, ds in zip(axes, TEST + ["all"]):
    for ci, c in enumerate(COND):
        if ds == "all": vals = E3[E3.condition == c].groupby("seed").zeroshot_acc.mean().values
        else: vals = E3[(E3.condition == c) & (E3.dataset == ds)].sort_values("seed").zeroshot_acc.values
        ax.scatter(np.full(len(vals), ci) + np.array([-0.12, 0, 0.12])[:len(vals)], vals * 100, s=11, color=PAL[ci], marker=MK[c], alpha=0.55, linewidths=0)
        ax.plot([ci - 0.3, ci + 0.3], [vals.mean() * 100] * 2, color=PAL[ci], lw=2.4, solid_capstyle="round")
    ax.set_title(DSN.get(ds, "Mean of 4"), fontsize=8); ax.set_xticks(range(5)); ax.set_xticklabels(["M0", "M1", "M2", "M3", "M4"], fontsize=6.8); ax.set_xlim(-0.6, 4.6)
    if ds in ("uschad", "harth", "mhealth"): ax.axhline(100 / 12, color=INK2, lw=0.7, ls=(0, (3, 2)))
    if ds == "motionsense": ax.axhline(100 / 6, color=INK2, lw=0.7, ls=(0, (3, 2)))
axes[0].set_ylabel("zero-shot accuracy (%)"); axes[0].set_ylim(0, 80)
axes[0].text(4.5, 100 / 12 + 1.2, "chance", fontsize=6, color=INK2, ha="right")
handles = [Line2D([0], [0], color=PAL[i], marker=MK[c], lw=2.2, ms=4, label=CL[c]) for i, c in enumerate(COND)]
fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
save(fig, "Fig4_conditions")

# E3 statistics
P = E3.pivot_table(index=["dataset", "seed"], columns="condition", values="zeroshot_acc")
pairs = [("M0_none", "M4_gold"), ("M0_none", "M3_verified"), ("M2_extracted", "M3_verified"), ("M3_verified", "M4_gold"), ("M0_none", "M1_generic")]
rows = []
for a, b in pairs:
    d = (P[b] - P[a]).dropna(); lo, hi = boot(d.values)
    rows.append(dict(comparison=f"{CL[b]} − {CL[a]}", delta_pts=d.mean() * 100, ci_lo=lo * 100, ci_hi=hi * 100, p=signflip(d.values),
                     **{f"Δ {DSN[k]}": d.xs(k, level=0).mean() * 100 for k in TEST}))
TE3 = pd.DataFrame(rows); TE3["p_holm"] = holm(TE3.p.values); TE3.round(3).to_csv(OUT / "tables/T_E3_pairwise.csv", index=False)
summ = E3.groupby("condition").agg(zs_acc=("zeroshot_acc", "mean"), zs_f1=("zeroshot_f1", "mean"), fs1=("fewshot1_acc", "mean"), fs10=("fewshot10_acc", "mean")).reindex(COND)
summ["seed_sd_within_ds"] = E3.groupby(["condition", "dataset"]).zeroshot_acc.std().groupby("condition").mean().reindex(COND)
summ["seed_sd_of_mean"] = E3.groupby(["condition", "seed"]).zeroshot_acc.mean().groupby("condition").std().reindex(COND)
for k in TEST: summ[f"acc_{k}"] = E3[E3.dataset == k].groupby("condition").zeroshot_acc.mean().reindex(COND)
summ.round(4).to_csv(OUT / "tables/T_E3_summary.csv"); stats["E3_summary"] = summ.round(4).to_dict(orient="index"); stats["E3_pairwise"] = TE3.round(4).to_dict(orient="records")

# ======================= E4 cost curves with CIs + per-data set heatmap =======================
base = E3[E3.condition == "M4_gold"][["dataset", "seed", "zeroshot_acc"]].rename(columns={"zeroshot_acc": "b"})
m = E4.merge(base, on=["dataset", "seed"]); m["cost"] = (m.b - m.zeroshot_acc) * 100
rows = []
for (k, s), g in m.groupby(["defect", "severity"]):
    lo, hi = boot(g.cost.values); rows.append(dict(defect=k, severity=s, cost=g.cost.mean(), ci_lo=lo, ci_hi=hi, p=signflip(g.cost.values),
                                                   datasets_hurt=int((g.groupby("dataset").cost.mean() > 0).sum())))
TE4 = pd.DataFrame(rows); TE4["p_holm"] = holm(TE4.p.values); TE4.round(4).to_csv(OUT / "tables/T_E4_costs.csv", index=False); stats["E4"] = TE4.round(4).to_dict(orient="records")
DEF = ["label", "unit", "axis", "rate", "placement"]
DL = {"label": "wrong activity-label text", "unit": "wrong accelerometer unit", "axis": "wrong axis labels", "rate": "wrong sampling rate (×2)", "placement": "wrong body placement"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.35))
ax = axes[0]
for di, k in enumerate(DEF):
    s = TE4[TE4.defect == k].sort_values("severity"); xs = np.r_[0, s.severity * 100] + (di - 2) * 1.1
    ys = np.r_[0, s.cost]; lo = np.r_[0, s.ci_lo]; hi = np.r_[0, s.ci_hi]
    ax.fill_between(xs, lo, hi, color=PAL[di], alpha=0.10, lw=0)
    ax.plot(xs, ys, color=PAL[di], lw=2, marker="o", ms=4, label=DL[k])
    for x_, y_, p_ in zip(xs[1:], ys[1:], s.p_holm):
        if p_ < 0.05: ax.text(x_, y_ + 1.6, "*" if p_ >= 0.01 else "**", ha="center", fontsize=8, color=PAL[di], weight="bold")
ax.axhline(0, color=INK2, lw=0.8); ax.set_xlim(-4, 106); ax.set_xticks([0, 25, 50, 100])
ax.set_xlabel("training data sets / channels carrying the defect (%)"); ax.set_ylabel("zero-shot accuracy lost (points)")
ax.legend(loc="upper left", fontsize=6.6); ax.set_title("a  Dose–response (mean, 95% bootstrap CI)", loc="left", fontsize=8.5)
ax.text(105, -9.5, "* p<.05  ** p<.01 (Holm)", ha="right", fontsize=6.2, color=INK2)
ax2 = axes[1]; H = m[m.severity == 1.0].pivot_table(index="defect", columns="dataset", values="cost", aggfunc="mean").reindex(DEF)[TEST]
lim = 50; cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f2f0", "#e34948"])
im = ax2.imshow(H.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]): ax2.text(j, i, f"{H.values[i, j]:+.0f}", ha="center", va="center", fontsize=7, color=INK)
ax2.set_xticks(range(len(TEST))); ax2.set_xticklabels([DSN[k] for k in TEST], rotation=30, ha="right"); ax2.set_yticks(range(len(DEF)))
ax2.set_yticklabels(["label text", "acc. unit", "axis", "rate", "placement"]); ax2.grid(False)
for s_ in ax2.spines.values(): s_.set_visible(False)
ax2.set_title("b  Cost at 100% by held-out data set", loc="left", fontsize=8.5)
cb = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.03); cb.set_label("points lost", fontsize=6.6); cb.ax.tick_params(labelsize=6)
save(fig, "Fig5_defects")

# ======================= Few-shot =======================
fig, ax = plt.subplots(figsize=(3.5, 2.6))
cols = [("zeroshot_acc", 0), ("fewshot1_acc", 1), ("fewshot10_acc", 10)]
for ci, c in enumerate(COND):
    sub = E3[E3.condition == c]; ys = [sub[k].mean() * 100 for k, _ in cols]
    ax.plot([0, 1, 2], ys, color=PAL[ci], marker=MK[c], lw=2, ms=4.5, label=CL[c])
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["zero-shot", "1 % labelled", "10 % labelled"]); ax.set_ylabel("accuracy (%), mean of 4 held-out sets"); ax.set_ylim(20, 90)
ax.legend(loc="lower right", fontsize=6.6)
save(fig, "Fig6_fewshot")

# ======================= Tables for the manuscript =======================
t1 = T1.copy(); t1["rate_hz"] = [GOLD[k]["sampling_rate_hz"] for k in t1.dataset]; t1["acc_unit"] = [GOLD[k]["accelerometer_unit"] for k in t1.dataset]
t1["gyro_unit"] = [GOLD[k]["gyroscope_unit"] or "—" for k in t1.dataset]; t1["placements"] = ["; ".join(GOLD[k]["sensor_placements"]) for k in t1.dataset]
t1.to_csv(OUT / "tables/T1_datasets.csv", index=False)
stats["E5"] = T8.set_index("method")["mean"].round(4).to_dict()
stats["E5_excl_realdisp"] = T8.set_index("method").drop(columns=["mean", "realdisp"]).mean(1).round(4).to_dict()
stats["E1"] = T2.set_index("label")[FIELDS + ["composite"]].round(3).to_dict(orient="index")
stats["n_eval_windows"] = int(E3.groupby("dataset").n_windows.first().sum())
stats["gold_audit_contradictions"] = AG[AG.status == "contradicted"][["dataset", "field", "documented", "data"]].to_dict(orient="records")
stats["gold_rate_verifiable"] = int(((AG.field == "sampling_rate_hz") & (AG.status != "unverifiable")).sum())
json.dump(stats, open(OUT / "stats.json", "w"), indent=1, default=str); print(json.dumps({k: stats[k] for k in ["verif_physical_before", "verif_physical_after", "verif_flags", "verif_probe_agrees_gold", "contradiction_rate_by_class", "E5_excl_realdisp"]}, indent=1))
print(TE3.round(3).to_string()); print(TE4.round(3).to_string())
