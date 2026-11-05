"""Auto-generated, read-only export of notebooks/M2M_Metadata_to_Model.ipynb (percent format).

The notebook is the canonical source; regenerate this file with `python scripts/export_notebook.py`.
"""

# %% [markdown]
# # Metadata-to-Model (M2M): grounded harmonisation metadata for cross-data set HAR
#
# **One notebook, five experiments, publication-ready figures and tables.**
#
# | Stage | What it does | Experiment |
# |---|---|---|
# | 1. Data | Downloads up to 10 public IMU HAR data sets, parses them into one session format | — |
# | 2. Gold records | Loads author-provided *harmonisation records* (schema = HAR metadata ontology) | — |
# | 3. Documentation | Builds a documentation bundle per data set (README + landing page + paper) | — |
# | 4. Extraction | Open-weight 7–14B LLMs (4-bit) fill the schema from documentation, with retrieval + JSON repair | **E1** |
# | 5. Verification | Deterministic probes check every field against the raw data → documentation–data audit | **E2** |
# | 6. Harmonisation | Records → unit conversion, resampling, 5-s windows, channel descriptions, label prototypes | — |
# | 7. Model | Compact heterogeneity-aware channel-token transformer (HALO-style conditioning) | — |
# | 8. Conditions | Train under M0 none / M1 generic / M2 extracted / M3 verified / M4 gold, 3 seeds, held-out data sets | **E3** |
# | 9. Defects | Inject placement / unit / rate / axis / label defects into gold metadata → cost curves | **E4** |
# | 10. Labels | Ontology-constrained LLM label alignment vs WordNet synonyms vs raw names | **E5** |
# | 11. Report | Figures (PDF+PNG, 300 dpi), tables (CSV, LaTeX, Markdown), `summary.json`, `RESULTS.md` | — |
#
# **Modes.** `MODE = "SMOKE"` runs the whole pipeline in minutes on CPU with three synthetic data sets and a mock
# extractor — use it first to check the environment. `MODE = "FULL"` runs everything on a Colab **L4** (24 GB).
# Budget: downloads 1–2 h, extraction 2–6 h, training 30–80 GPU-hours spread over sessions. Every stage caches to
# `WORK_DIR` (mount Google Drive!) and is resumable; re-running a cell skips finished work.
#
# **Honesty notes.** (i) The gold records shipped here were written by the authors from the data sets' documentation
# and must be independently checked by a second annotator before publication (Cohen's κ is computed if you add a
# second record file). (ii) Models, prompts and seeds are logged to `runs/manifest.jsonl` for the reproducibility
# statement. (iii) Nothing in the report is hand-typed: every number in `RESULTS.md` is read from the result files.

# %% [markdown]
# **Repository version.** This is the notebook that produced the results in `results/notebook_outputs/`
# (v7 of the experiment). Relative to the first full run (v5), the M3 *verified* condition falls back to the
# verified record's documented sampling rate for sessions without timestamps (it previously fell back to 50 Hz,
# which affected DSADS and USC-HAD); the three M3 runs were re-trained after this fix and every other run is
# identical. Each trained encoder is also saved to `runs/<run>.pt`. See `CHANGELOG.md`.
#
# **Quick start.** Set `MODE = "SMOKE"` in the configuration cell for a CPU-only check (a few minutes, synthetic
# data, mock extractor). For the full study set `MODE = "FULL"`, run on a GPU with at least 24 GB (Colab L4),
# paste a Hugging Face token into `HF_TOKEN` for the gated models (never commit it), and choose *Run all*.
# Every stage caches to `WORK_DIR` and is resumable.

# %% [markdown]
# ## 0. Configuration

# %%
# ======================= CONFIGURATION — edit this cell only =======================
MODE = "FULL"            # "SMOKE" (CPU, minutes, synthetic) or "FULL" (L4, real data sets)
WORK_DIR = "/content/drive/MyDrive/M2M"   # persistent cache; set to "./m2m_work" if not using Drive
MOUNT_DRIVE = True       # mount Google Drive at /content/drive (Colab only)
KEEP_RAW_ON_DRIVE = True # True: raw downloads live under WORK_DIR/raw (needs ~15 GB on Drive, survives restarts); False: ephemeral /content disk
HF_TOKEN = ""            # <<< PASTE YOUR HUGGING FACE TOKEN HERE, e.g. "hf_abc123...". Leave "" to run Qwen only.

# Data sets (keys of DATASET_REGISTRY below). Comment out any you cannot download.
TRAIN_DATASETS = ["hapt", "pamap2", "wisdm19", "dsads", "realdisp", "hhar"]
TEST_DATASETS  = ["mhealth", "uschad", "harth", "motionsense"]
INCLUDE_OPPORTUNITY = False   # 2 GB download; set True to add OPPORTUNITY to TRAIN_DATASETS
MAX_WINDOWS_PER_DATASET = 20000   # cap to keep L4 runs ~1.5–3 h each

# Extraction (E1)
LLM_MODELS = ["Qwen/Qwen2.5-7B-Instruct",          # ungated
              "mistralai/Mistral-7B-Instruct-v0.3",# gated (accept licence on HF, set HF_TOKEN)
              "meta-llama/Llama-3.1-8B-Instruct",  # gated
              "google/gemma-2-9b-it",              # gated
              "Qwen/Qwen2.5-14B-Instruct"]         # ungated, ~9.5 GB in 4-bit; runs last
RETRIEVAL_TOPK = 6
EXTRACTION_VARIANTS = ["zeroshot", "retrieval"]   # "zeroshot" = whole doc truncated; "retrieval" = top-k chunks

# Training (E3/E4)
SEEDS = [0, 1, 2]
EPOCHS = 25
BATCH_SIZE = 256
TARGET_HZ = 30
WINDOW_SEC = 5.0
STRIDE_SEC = 2.5
FEWSHOT_FRACS = [0.01, 0.10]
CONDITIONS = ["M0_none", "M1_generic", "M2_extracted", "M3_verified", "M4_gold"]
DEFECTS = ["placement", "unit", "rate", "axis", "label"]
DEFECT_SEVERITIES = [0.25, 0.5, 1.0]
EXTRACTED_MODEL_FOR_M2 = "Qwen/Qwen2.5-7B-Instruct"   # which model's records feed M2/M3

# Text encoder for channel descriptions and label prototypes
TEXT_ENCODER = "sentence-transformers/all-mpnet-base-v2"   # final-run encoder (768-d); all-MiniLM-L6-v2 is ~3x faster for pilots

# Figures
FIG_DPI = 300
# ===================================================================================

# %% [markdown]
# ## 1. Environment

# %%
import os, sys, json, re, io, zipfile, time, math, random, hashlib, shutil, glob, warnings, subprocess, textwrap
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if MODE == "SMOKE":
    WORK_DIR = "./m2m_smoke"
    MOUNT_DRIVE = False
if IN_COLAB and MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
WORK = Path(WORK_DIR); WORK.mkdir(parents=True, exist_ok=True)
# Raw downloads stay on the ephemeral Colab disk (they are only needed until sessions/ and docs/ exist); Drive keeps the small derived files.
RAW_DIR = Path(str(WORK / "raw") if (KEEP_RAW_ON_DRIVE or not IN_COLAB) else "/content/m2m_raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
for sub in ["sessions", "docs", "gold", "extracted", "audit", "windows", "runs", "figures", "tables", "results"]:
    (WORK / sub).mkdir(exist_ok=True)
print("Work dir:", WORK.resolve(), "| raw dir:", RAW_DIR, "| mode:", MODE, "| colab:", IN_COLAB)

def pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
if MODE == "FULL":
    pip_install(["transformers>=4.44", "accelerate", "bitsandbytes", "json-repair",
                 "pypdf", "beautifulsoup4", "requests", "scipy", "scikit-learn", "umap-learn", "nltk", "pyarrow", "tqdm"])
else:
    pip_install(["json-repair", "scipy", "scikit-learn", "tqdm", "pyarrow"])

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import signal as spsignal
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", DEVICE,
      "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(s)

def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
def jload(path):
    return json.loads(Path(path).read_text())
def log_manifest(**kw):
    kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(WORK / "runs" / "manifest.jsonl", "a") as f: f.write(json.dumps(kw, default=str) + "\n")

# ---- Publication plotting style (validated categorical palette; sequential = single hue) ----
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", ["#eef4fc", "#9cc3ee", "#2a78d6", "#123d73"])
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": FIG_DPI, "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": "#52514e", "axes.linewidth": 0.8,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(WORK / "figures" / f"{name}.{ext}")
    print("saved figure", name)
def save_table(df, name, caption="", index=False, float_fmt="%.3f"):
    df.to_csv(WORK / "tables" / f"{name}.csv", index=index)
    (WORK / "tables" / f"{name}.md").write_text(df.to_markdown(index=index, floatfmt=float_fmt.replace("%", "")))
    (WORK / "tables" / f"{name}.tex").write_text(df.to_latex(index=index, caption=caption, label=f"tab:{name}",
                                                             float_format=float_fmt, escape=True))
    print("saved table", name)

# %% [markdown]
# ## 2. Data sets: registry, download, unified session format
#
# Every data set is parsed into the same structure — a list of **sessions** (one subject, one contiguous activity),
# each an `N × C` float array in *native* units, plus the channel list the loader used. Native units are kept on
# purpose: the verification probes (Section 5) must be able to catch documentation that says `g` when the data is
# in m/s². Harmonisation happens later, driven by a *record*, never hard-coded.

# %%
@dataclass
class Channel:
    modality: str      # accelerometer | gyroscope | magnetometer
    axis: str          # x | y | z
    placement: str     # body location, free text normalised later
    unit: str          # g | m/s^2 | rad/s | deg/s | milli-g | a.u.

@dataclass
class Session:
    subject: str
    label: str                 # raw label string as the data set names it
    data: np.ndarray           # [N, C] float32, native units
    rate_hz: Optional[float]   # per-session rate when measurable from timestamps, else None
    timestamps: Optional[np.ndarray] = None   # seconds, if the data set has them

def ch_block(placement, unit_acc, unit_gyro=None, modalities=("accelerometer", "gyroscope")):
    out = []
    for m in modalities:
        u = unit_acc if m == "accelerometer" else (unit_gyro or "rad/s")
        for ax in "xyz": out.append(Channel(m, ax, placement, u))
    return out

UCI = "https://archive.ics.uci.edu/static/public/{id}/{slug}.zip"
DATASET_REGISTRY = {
 "hapt":        dict(name="HAPT (UCI-HAR raw with postural transitions)", uci_id=341,
                     slug="smartphone+based+recognition+of+human+activities+and+postural+transitions", rate=50,
                     page="https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions"),
 "pamap2":      dict(name="PAMAP2", uci_id=231, slug="pamap2+physical+activity+monitoring", rate=100,
                     page="https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring"),
 "mhealth":     dict(name="MHEALTH", uci_id=319, slug="mhealth+dataset", rate=50,
                     page="https://archive.ics.uci.edu/dataset/319/mhealth+dataset"),
 "wisdm19":     dict(name="WISDM 2019 (phone accelerometer)", uci_id=507,
                     slug="wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset", rate=20,
                     page="https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset"),
 "dsads":       dict(name="DSADS (Daily and Sports Activities)", uci_id=256, slug="daily+and+sports+activities", rate=25,
                     page="https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities"),
 "realdisp":    dict(name="REALDISP (ideal placement)", uci_id=305, slug="realdisp+activity+recognition+dataset", rate=50,
                     page="https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset"),
 "hhar":        dict(name="HHAR (Heterogeneity HAR, phone accelerometer)", uci_id=344, slug="heterogeneity+activity+recognition", rate=None,
                     page="https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition"),
 "harth":       dict(name="HARTH", uci_id=779, slug="harth", rate=50,
                     page="https://archive.ics.uci.edu/dataset/779/harth"),
 "opportunity": dict(name="OPPORTUNITY", uci_id=226, slug="opportunity+activity+recognition", rate=30,
                     page="https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition"),
 "motionsense": dict(name="MotionSense", url="https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip",
                     rate=50, page="https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/README.md"),
 "uschad":      dict(name="USC-HAD", url="http://sipi.usc.edu/had/USC-HAD.zip", rate=100,
                     page="http://sipi.usc.edu/had/"),
}
for k, v in DATASET_REGISTRY.items():
    if "uci_id" in v: v["url"] = UCI.format(id=v["uci_id"], slug=v["slug"])
if INCLUDE_OPPORTUNITY and "opportunity" not in TRAIN_DATASETS: TRAIN_DATASETS.append("opportunity")

def download(url, dest, retries=3):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0: return dest
    import requests
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
                r.raise_for_status(); tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20): f.write(chunk)
                tmp.rename(dest); return dest
        except Exception as e:
            print(f"  download attempt {i+1} failed: {e}"); time.sleep(3)
    raise RuntimeError(f"could not download {url}")

def fetch_zip(key):
    """Download and extract (recursively, for zips-in-zips) into raw/<key>/. Returns the extraction root."""
    spec = DATASET_REGISTRY[key]; root = RAW_DIR / key
    if (root / ".done").exists(): return root
    root.mkdir(parents=True, exist_ok=True)
    z = download(spec["url"], root / "download.zip")
    with zipfile.ZipFile(z) as zf: zf.extractall(root)
    for inner in list(root.rglob("*.zip")):
        if inner.name == "download.zip": continue
        try:
            with zipfile.ZipFile(inner) as zf: zf.extractall(inner.parent)
        except zipfile.BadZipFile: pass
    (root / ".done").write_text("ok"); return root

def find(root, pattern):
    hits = sorted(Path(root).rglob(pattern)); return hits

def segments_from_labels(labels, min_len):
    """Yield (start, end, label) for contiguous runs of a label array, skipping None/'null'."""
    labels = np.asarray(labels, dtype=object); n = len(labels); i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]: j += 1
        if labels[i] not in (None, "null", "0", 0, "nan") and (j - i) >= min_len:
            yield i, j, str(labels[i])
        i = j

def rate_from_timestamps(t):
    t = np.asarray(t, dtype=float); d = np.diff(t); d = d[(d > 0) & np.isfinite(d)]
    return float(1.0 / np.median(d)) if len(d) > 10 else None

# %% [markdown]
# ### 2.1 Loaders (one per data set). Each returns `(sessions, channels)`.

# %%
def load_hapt(root):
    base = find(root, "RawData")[0]
    acts = {int(l.split()[0]): l.split()[1].lower().replace("_", " ") for l in (base.parent / "activity_labels.txt").read_text().splitlines() if l.strip()}
    lab = np.loadtxt(base / "labels.txt", dtype=int)  # exp, user, act, start, end
    sessions = []
    for exp, user, act, s, e in lab:
        acc = np.loadtxt(base / f"acc_exp{exp:02d}_user{user:02d}.txt")
        gyr = np.loadtxt(base / f"gyro_exp{exp:02d}_user{user:02d}.txt")
        x = np.hstack([acc[s:e+1], gyr[s:e+1]]).astype(np.float32)
        if len(x) >= 50: sessions.append(Session(f"u{user}", acts[act], x, None))
    return sessions, ch_block("waist (smartphone)", "g", "rad/s")

PAMAP2_ACTS = {1:"lying",2:"sitting",3:"standing",4:"walking",5:"running",6:"cycling",7:"Nordic walking",9:"watching TV",
               10:"computer work",11:"car driving",12:"ascending stairs",13:"descending stairs",16:"vacuum cleaning",
               17:"ironing",18:"folding laundry",19:"house cleaning",20:"playing soccer",24:"rope jumping"}
def load_pamap2(root):
    files = find(root, "Protocol/subject*.dat"); sessions = []
    cols = {"hand": (4, 10), "chest": (21, 27), "ankle": (38, 44)}  # acc16 start, gyro start (0-indexed)
    for f in files:
        df = pd.read_csv(f, sep=r"\s+", header=None).interpolate(limit_direction="both")
        t = df[0].values; act = df[1].values.astype(int)
        parts = []
        for p, (a, g) in cols.items(): parts += [df.iloc[:, a:a+3].values, df.iloc[:, g:g+3].values]
        x = np.hstack(parts).astype(np.float32)
        for s, e, lab in segments_from_labels([PAMAP2_ACTS.get(a) for a in act], 200):
            sessions.append(Session(f.stem, lab, x[s:e], rate_from_timestamps(t[s:e]), t[s:e]))
    ch = ch_block("wrist (dominant arm)", "m/s^2") + ch_block("chest", "m/s^2") + ch_block("ankle (dominant side)", "m/s^2")
    return sessions, ch

MHEALTH_ACTS = {1:"standing still",2:"sitting and relaxing",3:"lying down",4:"walking",5:"climbing stairs",6:"waist bends forward",
                7:"frontal elevation of arms",8:"knees bending (crouching)",9:"cycling",10:"jogging",11:"running",12:"jump front and back"}
def load_mhealth(root):
    sessions = []
    for f in find(root, "mHealth_subject*.log"):
        a = np.loadtxt(f); lab = a[:, 23].astype(int)
        x = np.hstack([a[:, 0:3], a[:, 5:8], a[:, 8:11], a[:, 14:17], a[:, 17:20]]).astype(np.float32)
        for s, e, l in segments_from_labels([MHEALTH_ACTS.get(v) for v in lab], 100):
            sessions.append(Session(f.stem, l, x[s:e], None))
    ch = ch_block("chest", "m/s^2", modalities=("accelerometer",)) + ch_block("left ankle", "m/s^2", "deg/s") + ch_block("right lower arm", "m/s^2", "deg/s")
    return sessions, ch

WISDM_ACTS = {"A":"walking","B":"jogging","C":"stairs","D":"sitting","E":"standing","F":"typing","G":"brushing teeth","H":"eating soup",
              "I":"eating chips","J":"eating pasta","K":"drinking from cup","L":"eating sandwich","M":"kicking (soccer ball)",
              "O":"playing catch with tennis ball","P":"dribbling (basketball)","Q":"writing","R":"clapping","S":"folding clothes"}
def load_wisdm19(root):
    sessions = []
    for f in find(root, "raw/phone/accel/data_*_accel_phone.txt"):
        rows = [l.strip().rstrip(";").split(",") for l in f.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows, columns=["subj", "act", "t", "x", "y", "z"])
        df[["t", "x", "y", "z"]] = df[["t", "x", "y", "z"]].astype(float)
        for act, g in df.groupby("act", sort=False):
            g = g.sort_values("t"); x = g[["x", "y", "z"]].values.astype(np.float32); t = g["t"].values / 1e9
            if len(x) >= 100 and act in WISDM_ACTS:
                sessions.append(Session(f"s{g.subj.iloc[0]}", WISDM_ACTS[act], x, rate_from_timestamps(t), t))
    return sessions, ch_block("front trouser pocket (smartphone)", "m/s^2", modalities=("accelerometer",))

DSADS_ACTS = ["sitting","standing","lying on back","lying on right side","ascending stairs","descending stairs","standing in elevator still",
              "moving around in elevator","walking in parking lot","walking on treadmill (flat, 4 km/h)","walking on treadmill (15 deg incline)",
              "running on treadmill (8 km/h)","exercising on stepper","exercising on cross trainer","cycling on exercise bike (horizontal)",
              "cycling on exercise bike (vertical)","rowing","jumping","playing basketball"]
def load_dsads(root):
    sessions = []; base = [p for p in find(root, "a01") if p.is_dir()][0].parent
    for a in range(1, 20):
        for p in range(1, 9):
            segs = sorted((base / f"a{a:02d}" / f"p{p}").glob("s*.txt"))
            if not segs: continue
            x = np.vstack([np.loadtxt(s, delimiter=",") for s in segs])
            keep = []
            for u in range(5): keep += list(range(u*9, u*9+6))   # acc3 + gyro3 per unit, drop mag
            sessions.append(Session(f"p{p}", DSADS_ACTS[a-1], x[:, keep].astype(np.float32), None))
    ch = []
    for pl in ["torso", "right arm", "left arm", "right leg", "left leg"]: ch += ch_block(pl, "m/s^2")
    return sessions, ch

REALDISP_ACTS = ["walking","jogging","running","jump up","jump front and back","jump sideways","jump legs/arms open/closed","jump rope",
 "trunk twist (arms outstretched)","trunk twist (elbows bent)","waist bends forward","waist rotation","waist bends (reach foot with opposite hand)",
 "reach heels backwards","lateral bend","lateral bend with arm up","repetitive forward stretching","upper trunk and lower body opposite twist",
 "lateral elevation of arms","frontal elevation of arms","frontal hand claps","frontal crossing of arms","shoulders high-amplitude rotation",
 "shoulders low-amplitude rotation","arms inner rotation","knees (alternating) to the breast","heels (alternating) to the backside",
 "knees bending (crouching)","knees (alternating) bending forward","rotation on the knees","rowing","elliptical bike","cycling"]
REALDISP_PLACES = ["right lower arm","right upper arm","back","left upper arm","left lower arm","right calf","right thigh","left thigh","left calf"]
def load_realdisp(root):
    sessions = []
    for f in find(root, "subject*_ideal.log"):
        a = np.loadtxt(f); t = a[:, 0] + a[:, 1] / 1e6; lab = a[:, 119].astype(int)
        keep = []
        for s in range(9): keep += list(range(2 + s*13, 2 + s*13 + 6))
        x = a[:, keep].astype(np.float32)
        for s, e, l in segments_from_labels([REALDISP_ACTS[v-1] if 1 <= v <= 33 else None for v in lab], 100):
            sessions.append(Session(f.stem.split("_")[0], l, x[s:e], rate_from_timestamps(t[s:e]), t[s:e]))
    ch = []
    for pl in REALDISP_PLACES: ch += ch_block(pl, "m/s^2")
    return sessions, ch

def load_hhar(root, max_rows=6_000_000):
    f = find(root, "Phones_accelerometer.csv")[0]
    parts = []
    for chunk in pd.read_csv(f, usecols=["Creation_Time", "x", "y", "z", "User", "Device", "gt"], nrows=max_rows, chunksize=500_000,
                             dtype={"x": "float32", "y": "float32", "z": "float32", "User": "category", "Device": "category", "gt": "category"}):
        parts.append(chunk[chunk.gt.notna() & (chunk.gt.astype(str) != "null")])
    df = pd.concat(parts, ignore_index=True); del parts
    sessions = []
    for (u, d, g), grp in df.groupby(["User", "Device", "gt"], sort=False):
        grp = grp.sort_values("Creation_Time"); t = grp.Creation_Time.values / 1e9
        x = grp[["x", "y", "z"]].values.astype(np.float32)
        if len(x) >= 200: sessions.append(Session(f"{u}_{d}", {"bike":"cycling","sit":"sitting","stand":"standing","walk":"walking",
                                                                "stairsup":"ascending stairs","stairsdown":"descending stairs"}[g],
                                                  x, rate_from_timestamps(t), t))
    return sessions, ch_block("waist (smartphone in pouch)", "m/s^2", modalities=("accelerometer",))

HARTH_ACTS = {1:"walking",2:"running",3:"shuffling",4:"ascending stairs",5:"descending stairs",6:"standing",7:"sitting",8:"lying",
              13:"cycling (sit)",14:"cycling (stand)",130:"cycling (sit, inactive)",140:"cycling (stand, inactive)"}
def load_harth(root):
    sessions = []
    for f in find(root, "S0*.csv"):
        df = pd.read_csv(f); t = pd.to_datetime(df["timestamp"]).astype("int64").values / 1e9
        x = df[["back_x","back_y","back_z","thigh_x","thigh_y","thigh_z"]].values.astype(np.float32)
        for s, e, l in segments_from_labels([HARTH_ACTS.get(v) for v in df["label"].values], 100):
            sessions.append(Session(f.stem, l, x[s:e], rate_from_timestamps(t[s:e]), t[s:e]))
    ch = ch_block("lower back", "g", modalities=("accelerometer",)) + ch_block("right thigh", "g", modalities=("accelerometer",))
    return sessions, ch

MS_ACTS = {"dws":"walking downstairs","ups":"walking upstairs","wlk":"walking","jog":"jogging","sit":"sitting","std":"standing"}
def load_motionsense(root):
    sessions = []
    for d in sorted([p for p in find(root, "*_*") if p.is_dir() and p.name[:3] in MS_ACTS]):
        for f in sorted(d.glob("sub_*.csv")):
            df = pd.read_csv(f)
            acc = df[["userAcceleration.x","userAcceleration.y","userAcceleration.z"]].values + df[["gravity.x","gravity.y","gravity.z"]].values
            gyr = df[["rotationRate.x","rotationRate.y","rotationRate.z"]].values
            sessions.append(Session(f.stem, MS_ACTS[d.name[:3]], np.hstack([acc, gyr]).astype(np.float32), None))
    return sessions, ch_block("front trouser pocket (smartphone)", "g", "rad/s")

USC_ACTS = {1:"walking forward",2:"walking left",3:"walking right",4:"walking upstairs",5:"walking downstairs",6:"running forward",
            7:"jumping up",8:"sitting",9:"standing",10:"sleeping",11:"elevator up",12:"elevator down"}
def load_uschad(root):
    from scipy.io import loadmat
    sessions = []
    for f in find(root, "Subject*/a*t*.mat"):
        m = loadmat(f); x = m["sensor_readings"].astype(np.float32)
        a = int(re.search(r"a(\d+)t", f.name).group(1))
        sessions.append(Session(f.parent.name, USC_ACTS[a], x, None))
    return sessions, ch_block("front right hip", "g", "deg/s")

def load_opportunity(root):
    sessions = []
    for f in find(root, "S*-ADL*.dat") + find(root, "S*-Drill.dat"):
        a = np.loadtxt(f); t = a[:, 0] / 1000.0
        keep = []
        for s in range(5): keep += list(range(37 + s*13, 37 + s*13 + 6))   # BACK, RUA, RLA, LUA, LLA: acc3 + gyro3
        x = a[:, keep]; x = pd.DataFrame(x).interpolate(limit_direction="both").values.astype(np.float32)
        loco = {1:"standing",2:"walking",4:"sitting",5:"lying"}
        for s, e, l in segments_from_labels([loco.get(int(v)) for v in a[:, 243]], 60):
            sessions.append(Session(f.name.split("-")[0], l, x[s:e], rate_from_timestamps(t[s:e]), t[s:e]))
    ch = []
    for pl in ["back", "right upper arm", "right lower arm", "left upper arm", "left lower arm"]: ch += ch_block(pl, "milli-g", "deg/s")
    return sessions, ch

LOADERS = dict(hapt=load_hapt, pamap2=load_pamap2, mhealth=load_mhealth, wisdm19=load_wisdm19, dsads=load_dsads, realdisp=load_realdisp,
               hhar=load_hhar, harth=load_harth, motionsense=load_motionsense, uschad=load_uschad, opportunity=load_opportunity)

# %% [markdown]
# ### 2.2 SMOKE-mode synthetic data sets (three "data sets" with known, deliberately heterogeneous metadata)

# %%
SYN_SPECS = {
  "syn_a": dict(rate=50, unit="g", gunit="rad/s", places=["wrist", "ankle"], acts=["walking", "running", "sitting", "standing", "ascending stairs"], n_subj=6),
  "syn_b": dict(rate=100, unit="m/s^2", gunit="deg/s", places=["chest"], acts=["walking", "jogging", "lying", "standing", "cycling"], n_subj=5),
  "syn_c": dict(rate=25, unit="m/s^2", gunit="rad/s", places=["waist", "thigh"], acts=["walking", "running", "sitting", "descending stairs"], n_subj=4),
}
CANON_FREQ = {"walking": 1.8, "running": 2.8, "jogging": 2.5, "sitting": 0.0, "standing": 0.0, "lying": 0.0, "ascending stairs": 1.4,
              "descending stairs": 1.6, "cycling": 1.2}
def make_synthetic(key, seed=0):
    rng = np.random.default_rng(seed + hash(key) % 1000); sp = SYN_SPECS[key]
    scale = 9.81 if sp["unit"] == "m/s^2" else 1.0; gscale = 57.3 if sp["gunit"] == "deg/s" else 1.0
    sessions = []
    for s in range(sp["n_subj"]):
        for act in sp["acts"]:
            n = int(sp["rate"] * rng.uniform(40, 70)); t = np.arange(n) / sp["rate"]
            f0 = CANON_FREQ[act]; parts = []
            for pi, pl in enumerate(sp["places"]):
                amp = (0.3 if f0 > 0 else 0.02) * (1.5 if "ankle" in pl or "thigh" in pl else 1.0)
                grav = np.array([0, 0, 1.0]) if act not in ("lying",) else np.array([1.0, 0, 0])
                acc = grav[None, :] + amp * np.stack([np.sin(2*np.pi*f0*t + pi), np.cos(2*np.pi*f0*t), np.sin(4*np.pi*f0*t)], 1) + 0.03*rng.standard_normal((n, 3))
                gyr = amp * 6 * np.stack([np.cos(2*np.pi*f0*t), np.sin(2*np.pi*f0*t + 1), np.cos(4*np.pi*f0*t)], 1) + 0.02*rng.standard_normal((n, 3))
                parts += [acc * scale, gyr * gscale]
            sessions.append(Session(f"s{s}", act, np.hstack(parts).astype(np.float32), None))
    ch = []
    for pl in sp["places"]: ch += ch_block(pl, sp["unit"], sp["gunit"])
    return sessions, ch
SYN_DOCS = {
 "syn_a": "SynA data set. Six volunteers wore two inertial units, one on the wrist and one on the ankle. Each unit recorded tri-axial acceleration in g and angular velocity in rad/s at 50 Hz. Activities: walking, running, sitting, standing and ascending stairs, performed in a laboratory under a scripted protocol.",
 "syn_b": "SynB corpus. A single chest-mounted IMU sampled at 100 Hz. Acceleration is given in m/s^2, gyroscope in deg/s. Five subjects performed walking, jogging, lying, standing and cycling in a gym (experimental protocol).",
 "syn_c": "SynC. Four participants, sensors at the waist and the thigh, 25 Hz, acceleration in m/s^2 and gyroscope in rad/s. Labels: walking, running, sitting, descending stairs. Free-living observational recording.",
}

# %% [markdown]
# ### 2.3 Build the session store

# %%
def build_sessions(key):
    out = WORK / "sessions" / f"{key}.pkl"
    if out.exists():
        import pickle; return pickle.load(open(out, "rb"))
    print(f"[{key}] loading ...")
    if MODE == "SMOKE": sessions, channels = make_synthetic(key)
    else:
        root = fetch_zip(key); sessions, channels = LOADERS[key](root)
    store = dict(sessions=sessions, channels=[asdict(c) for c in channels])
    import pickle; pickle.dump(store, open(out, "wb"))
    print(f"[{key}] {len(sessions)} sessions, {len(channels)} channels, {len(set(s.label for s in sessions))} labels, "
          f"{len(set(s.subject for s in sessions))} subjects")
    return store

if MODE == "SMOKE":
    TRAIN_DATASETS, TEST_DATASETS = ["syn_a", "syn_b"], ["syn_c"]
    for k in SYN_SPECS: DATASET_REGISTRY[k] = dict(name=k, rate=SYN_SPECS[k]["rate"])
ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS
STORES = {}
for k in ALL_DATASETS:
    try: STORES[k] = build_sessions(k)
    except Exception as e:
        print(f"[{k}] FAILED: {e!r} — dropping this data set");
for k in list(ALL_DATASETS):
    if k not in STORES:
        if k in TRAIN_DATASETS: TRAIN_DATASETS.remove(k)
        if k in TEST_DATASETS: TEST_DATASETS.remove(k)
ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS
print("Data sets available:", ALL_DATASETS)

overview = pd.DataFrame([dict(dataset=k, name=DATASET_REGISTRY[k]["name"], sessions=len(v["sessions"]),
                              subjects=len(set(s.subject for s in v["sessions"])), labels=len(set(s.label for s in v["sessions"])),
                              channels=len(v["channels"]), placements=len(set(c["placement"] for c in v["channels"])),
                              hours=round(sum(len(s.data) for s in v["sessions"]) / (DATASET_REGISTRY[k]["rate"] or 100) / 3600, 1),
                              split="train" if k in TRAIN_DATASETS else "test") for k, v in STORES.items()])
save_table(overview, "T1_datasets_overview", "Data sets used in this study.")
overview

# %% [markdown]
# ## 3. Harmonisation records: schema, canonical activity taxonomy, gold records
#
# The **harmonisation record** is the paper's central object: the subset of a data set's metadata that a cross-data
# set model needs. Its fields are a projection of the canonical HAR metadata ontology onto what the model consumes.
# The gold records below were written by the authors from the official documentation of each data set; the data
# probes in Section 5 check them too, so author mistakes surface rather than hide.

# %%
RECORD_FIELDS = ["sampling_rate_hz", "n_participants", "devices", "sensor_placements", "modalities",
                 "accelerometer_unit", "gyroscope_unit", "activities", "protocol_type", "setting"]
FIELD_HELP = {
 "sampling_rate_hz": "Nominal sampling frequency of the inertial sensors in Hz (a number; use the modal value if several).",
 "n_participants": "Number of human subjects/participants/volunteers who contributed data (integer).",
 "devices": "Recording devices or sensor products (e.g. 'Samsung Galaxy S II smartphone', 'Xsens MTx', 'Shimmer2', 'Axivity AX3').",
 "sensor_placements": "Body locations where sensors were worn (e.g. 'waist', 'chest', 'right wrist', 'ankle', 'front trouser pocket').",
 "modalities": "Sensor modalities recorded, from: accelerometer, gyroscope, magnetometer, heart rate, ECG, temperature, other.",
 "accelerometer_unit": "Unit of the accelerometer values in the data files: one of 'g', 'm/s^2', 'milli-g', or null if not stated.",
 "gyroscope_unit": "Unit of the gyroscope values: one of 'rad/s', 'deg/s', or null if not stated / no gyroscope.",
 "activities": "The list of activity class names exactly as the data set labels them.",
 "protocol_type": "One of 'experimental' (scripted activities), 'observational' (free-living), 'survey', or null.",
 "setting": "One of 'laboratory', 'free-living', 'both', or null.",
}

CANONICAL_ACTIVITIES = ["walking","running","jogging","sitting","standing","lying","ascending stairs","descending stairs","cycling",
    "nordic walking","jumping","rope jumping","vacuum cleaning","ironing","folding laundry","house cleaning","computer work","watching tv",
    "car driving","playing sport","elevator","exercise (upper body)","exercise (lower body)","rowing","eating","drinking","writing/typing",
    "brushing teeth","clapping","transition","sleeping","shuffling","other"]

def _m(prefixes, canon):
    return {p: canon for p in prefixes}
CANON_MAP = {}   # dataset -> {raw label -> canonical}
CANON_MAP["hapt"] = {"walking":"walking","walking upstairs":"ascending stairs","walking downstairs":"descending stairs","sitting":"sitting",
    "standing":"standing","laying":"lying","stand to sit":"transition","sit to stand":"transition","sit to lie":"transition","lie to sit":"transition",
    "stand to lie":"transition","lie to stand":"transition"}
CANON_MAP["pamap2"] = {"lying":"lying","sitting":"sitting","standing":"standing","walking":"walking","running":"running","cycling":"cycling",
    "Nordic walking":"nordic walking","watching TV":"watching tv","computer work":"computer work","car driving":"car driving",
    "ascending stairs":"ascending stairs","descending stairs":"descending stairs","vacuum cleaning":"vacuum cleaning","ironing":"ironing",
    "folding laundry":"folding laundry","house cleaning":"house cleaning","playing soccer":"playing sport","rope jumping":"rope jumping"}
CANON_MAP["mhealth"] = {"standing still":"standing","sitting and relaxing":"sitting","lying down":"lying","walking":"walking","climbing stairs":"ascending stairs",
    "waist bends forward":"exercise (upper body)","frontal elevation of arms":"exercise (upper body)","knees bending (crouching)":"exercise (lower body)",
    "cycling":"cycling","jogging":"jogging","running":"running","jump front and back":"jumping"}
CANON_MAP["wisdm19"] = {"walking":"walking","jogging":"jogging","stairs":"ascending stairs","sitting":"sitting","standing":"standing","typing":"writing/typing",
    "brushing teeth":"brushing teeth","eating soup":"eating","eating chips":"eating","eating pasta":"eating","drinking from cup":"drinking",
    "eating sandwich":"eating","kicking (soccer ball)":"playing sport","playing catch with tennis ball":"playing sport","dribbling (basketball)":"playing sport",
    "writing":"writing/typing","clapping":"clapping","folding clothes":"folding laundry"}
CANON_MAP["dsads"] = {"sitting":"sitting","standing":"standing","lying on back":"lying","lying on right side":"lying","ascending stairs":"ascending stairs",
    "descending stairs":"descending stairs","standing in elevator still":"elevator","moving around in elevator":"elevator","walking in parking lot":"walking",
    "walking on treadmill (flat, 4 km/h)":"walking","walking on treadmill (15 deg incline)":"walking","running on treadmill (8 km/h)":"running",
    "exercising on stepper":"exercise (lower body)","exercising on cross trainer":"exercise (lower body)","cycling on exercise bike (horizontal)":"cycling",
    "cycling on exercise bike (vertical)":"cycling","rowing":"rowing","jumping":"jumping","playing basketball":"playing sport"}
CANON_MAP["realdisp"] = {a: ("walking" if a=="walking" else "jogging" if a=="jogging" else "running" if a=="running" else "rope jumping" if a=="jump rope"
    else "jumping" if a.startswith("jump") else "rowing" if a=="rowing" else "cycling" if a in ("cycling","elliptical bike")
    else "exercise (lower body)" if any(w in a for w in ("knees","heels","rotation on the knees")) else "exercise (upper body)") for a in REALDISP_ACTS}
CANON_MAP["hhar"] = {"cycling":"cycling","sitting":"sitting","standing":"standing","walking":"walking","ascending stairs":"ascending stairs","descending stairs":"descending stairs"}
CANON_MAP["harth"] = {"walking":"walking","running":"running","shuffling":"shuffling","ascending stairs":"ascending stairs","descending stairs":"descending stairs",
    "standing":"standing","sitting":"sitting","lying":"lying","cycling (sit)":"cycling","cycling (stand)":"cycling","cycling (sit, inactive)":"cycling","cycling (stand, inactive)":"cycling"}
CANON_MAP["motionsense"] = {"walking downstairs":"descending stairs","walking upstairs":"ascending stairs","walking":"walking","jogging":"jogging","sitting":"sitting","standing":"standing"}
CANON_MAP["uschad"] = {"walking forward":"walking","walking left":"walking","walking right":"walking","walking upstairs":"ascending stairs","walking downstairs":"descending stairs",
    "running forward":"running","jumping up":"jumping","sitting":"sitting","standing":"standing","sleeping":"sleeping","elevator up":"elevator","elevator down":"elevator"}
CANON_MAP["opportunity"] = {"standing":"standing","walking":"walking","sitting":"sitting","lying":"lying"}
for k in SYN_SPECS: CANON_MAP[k] = {a: a for a in SYN_SPECS[k]["acts"]}

GOLD_EXTRA = {   # fields the loader cannot know
 "hapt":        dict(n_participants=30, devices=["Samsung Galaxy S II smartphone"], protocol_type="experimental", setting="laboratory"),
 "pamap2":      dict(n_participants=9, devices=["Colibri wireless IMU", "BM-CS5SR heart-rate monitor"], protocol_type="experimental", setting="both"),
 "mhealth":     dict(n_participants=10, devices=["Shimmer2 wearable sensors"], protocol_type="experimental", setting="laboratory"),
 "wisdm19":     dict(n_participants=51, devices=["Google Nexus 5/5X or Samsung Galaxy S5 smartphone", "LG G Watch"], protocol_type="experimental", setting="laboratory"),
 "dsads":       dict(n_participants=8, devices=["Xsens MTx"], protocol_type="experimental", setting="both"),
 "realdisp":    dict(n_participants=17, devices=["Xsens MTx"], protocol_type="experimental", setting="laboratory"),
 "hhar":        dict(n_participants=9, devices=["LG Nexus 4", "Samsung Galaxy S3", "Samsung Galaxy S3 mini", "Samsung Galaxy S+", "LG G Watch", "Samsung Galaxy Gear"],
                     protocol_type="experimental", setting="laboratory", sampling_rate_hz=[50, 100, 150, 200]),
 "harth":       dict(n_participants=22, devices=["Axivity AX3"], protocol_type="observational", setting="free-living"),
 "motionsense": dict(n_participants=24, devices=["Apple iPhone 6s"], protocol_type="experimental", setting="laboratory"),
 "uschad":      dict(n_participants=14, devices=["MotionNode"], protocol_type="experimental", setting="laboratory"),
 "opportunity": dict(n_participants=4, devices=["Xsens MTx IMU", "InertiaCube3", "Sun SPOT", "Bluetooth accelerometers"], protocol_type="experimental", setting="laboratory"),
 "syn_a": dict(n_participants=6, devices=["generic IMU"], protocol_type="experimental", setting="laboratory"),
 "syn_b": dict(n_participants=5, devices=["generic IMU"], protocol_type="experimental", setting="laboratory"),
 "syn_c": dict(n_participants=4, devices=["generic IMU"], protocol_type="observational", setting="free-living"),
}

def gold_record(key):
    st = STORES[key]; ch = st["channels"]; extra = dict(GOLD_EXTRA[key])
    rec = dict(dataset=key, name=DATASET_REGISTRY[key]["name"],
               sampling_rate_hz=extra.pop("sampling_rate_hz", DATASET_REGISTRY[key]["rate"]),
               devices=extra["devices"], n_participants=extra["n_participants"],
               sensor_placements=sorted(set(c["placement"] for c in ch)),
               modalities=sorted(set(c["modality"] for c in ch)),
               accelerometer_unit=next((c["unit"] for c in ch if c["modality"] == "accelerometer"), None),
               gyroscope_unit=next((c["unit"] for c in ch if c["modality"] == "gyroscope"), None),
               activities=sorted(set(s.label for s in st["sessions"])),
               protocol_type=extra["protocol_type"], setting=extra["setting"],
               channels=ch, canonical_map={l: CANON_MAP[key].get(l, "other") for l in set(s.label for s in st["sessions"])})
    return rec

GOLD = {}
for k in ALL_DATASETS:
    GOLD[k] = gold_record(k); jdump(GOLD[k], WORK / "gold" / f"{k}.json")
print("Gold records written for", list(GOLD))

# %% [markdown]
# ## 4. Documentation bundles and LLM extraction (E1)

# %%
def html_to_text(html):
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "nav", "footer"]): t.decompose()
        return re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n"))
    except Exception: return re.sub(r"<[^>]+>", " ", html)

def pdf_to_text(path):
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    except Exception as e: return ""

def build_docs(key):
    out = WORK / "docs" / f"{key}.txt"
    if out.exists(): return out.read_text()
    if MODE == "SMOKE": out.write_text(SYN_DOCS[key]); return SYN_DOCS[key]
    parts, prov = [], []
    import requests
    page = DATASET_REGISTRY[key].get("page")
    if page:
        try:
            r = requests.get(page, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            txt = r.text if page.endswith(".md") else html_to_text(r.text)
            parts.append(f"=== LANDING PAGE: {page} ===\n{txt}"); prov.append(page)
        except Exception as e: print("  landing page failed:", e)
    root = RAW_DIR / key
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.stat().st_size < 400_000 and (re.search(r"readme|description|info|about|documentation", f.name, re.I)
                                                         or f.suffix.lower() in (".pdf",) ):
            txt = pdf_to_text(f) if f.suffix.lower() == ".pdf" else f.read_text(errors="ignore")
            if len(txt.strip()) > 100:
                parts.append(f"=== FILE: {f.relative_to(root)} ===\n{txt}"); prov.append(str(f.relative_to(root)))
    text = "\n\n".join(parts) if parts else f"(no documentation found for {key})"
    out.write_text(text); jdump(prov, WORK / "docs" / f"{key}.provenance.json")
    return text

DOCS = {k: build_docs(k) for k in ALL_DATASETS}
# Raw downloads are no longer needed from here on: sessions/ and docs/ hold everything. Free the disk.
if MODE == "FULL" and not KEEP_RAW_ON_DRIVE and all((WORK / "docs" / f"{k}.txt").exists() and (WORK / "sessions" / f"{k}.pkl").exists() for k in ALL_DATASETS):
    shutil.rmtree(RAW_DIR, ignore_errors=True); print("raw downloads removed from the ephemeral disk (sessions and docs are cached on Drive)")
pd.DataFrame([dict(dataset=k, doc_chars=len(v), doc_words=len(v.split())) for k, v in DOCS.items()])

# %%
# ---- Text embeddings (sentence-transformers; deterministic hashed fallback keeps SMOKE mode offline) ----
class TextEncoder:
    """Sentence embeddings with plain `transformers` (mean pooling + L2 norm = what SentenceTransformer does for
    MiniLM/mpnet). Avoids the sentence-transformers / transformers version clash. Deterministic hashed fallback
    keeps SMOKE mode offline."""
    def __init__(self, name=TEXT_ENCODER):
        self.dim = 384; self.model = None; self.cache = {}
        if MODE == "FULL":
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tok = AutoTokenizer.from_pretrained(name); self.model = AutoModel.from_pretrained(name).to(DEVICE).eval()
                self.dim = self.model.config.hidden_size
            except Exception as e: print("text encoder unavailable, using hashed fallback:", e)
    @torch.no_grad()
    def _encode(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i+bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            h = self.model(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
            z = (h * m).sum(1) / m.sum(1).clamp(min=1e-9); out.append(F.normalize(z, dim=-1).float().cpu().numpy())
        return np.concatenate(out)
    def __call__(self, texts):
        texts = [str(t) for t in texts]
        if self.model is not None:
            todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
            if todo:
                for t, z in zip(todo, self._encode(todo)): self.cache[t] = z
            return np.stack([self.cache[t] for t in texts]).astype(np.float32)
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9/^]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16); out[i, h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
            n = np.linalg.norm(out[i]); out[i] = out[i] / n if n > 0 else out[i]
        return out
TEXT = TextEncoder()

def chunk_text(text, words=220, overlap=40):
    w = text.split(); chunks = []
    for i in range(0, max(1, len(w)), words - overlap):
        chunks.append(" ".join(w[i:i + words]))
        if i + words >= len(w): break
    return chunks

FIELD_GROUPS = {
 "acquisition": ["sampling_rate_hz", "n_participants", "devices", "modalities"],
 "placement_units": ["sensor_placements", "accelerometer_unit", "gyroscope_unit"],
 "activities": ["activities"],
 "protocol": ["protocol_type", "setting"],
}
GROUP_QUERIES = {
 "acquisition": "sampling rate Hz frequency number of subjects participants volunteers device sensor model accelerometer gyroscope magnetometer",
 "placement_units": "sensor worn on body location wrist chest ankle waist pocket placement units g m/s^2 rad/s deg/s acceleration gyroscope values",
 "activities": "activities performed list of activity classes labels walking sitting standing",
 "protocol": "data collection protocol scripted free-living laboratory experiment observational",
}

def retrieve(doc_text, group, k=RETRIEVAL_TOPK):
    chunks = chunk_text(doc_text)
    if len(chunks) <= k: return chunks
    q = TEXT([GROUP_QUERIES[group]]); e = TEXT(chunks)
    idx = np.argsort(-(e @ q[0]))[:k]
    return [chunks[i] for i in sorted(idx)]

def build_prompt(group, context):
    schema = {f: FIELD_HELP[f] for f in FIELD_GROUPS[group]}
    return (f"You extract metadata about a human activity recognition (HAR) data set from its documentation.\n"
            f"Return ONLY a JSON object with exactly these keys (use null when the documentation does not say):\n"
            f"{json.dumps(schema, indent=1)}\n\nDOCUMENTATION:\n\"\"\"\n{context}\n\"\"\"\n\nJSON:")

# ---- LLM backend ----
class LLM:
    def __init__(self, name):
        self.name = name
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        tok_kw = dict(token=HF_TOKEN or None)
        self.tok = AutoTokenizer.from_pretrained(name, **tok_kw)
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        import gc; gc.collect(); torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto",
                                                          max_memory={0: "21GiB", "cpu": "24GiB"} if DEVICE == "cuda" else None, **tok_kw)
        self.model.eval()
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=400):
        # tokenize=False → plain string → ordinary tokenisation: stable across transformers 4.x and 5.x
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    def close(self):
        import gc; del self.model; gc.collect(); torch.cuda.empty_cache()

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    raw = m.group(0) if m else text
    try: return json.loads(raw)
    except Exception:
        try:
            from json_repair import repair_json; return json.loads(repair_json(raw))
        except Exception: return {}

def normalise_record(rec):
    out = {}
    for f in RECORD_FIELDS:
        v = rec.get(f)
        if f == "sampling_rate_hz":
            if isinstance(v, str):
                nums = re.findall(r"\d+(?:\.\d+)?", v); v = float(nums[0]) if nums else None
            elif isinstance(v, list): v = [float(x) for x in v if isinstance(x, (int, float))] or None
        if f == "n_participants":
            if isinstance(v, str):
                nums = re.findall(r"\d+", v); v = int(nums[0]) if nums else None
            elif isinstance(v, float): v = int(v)
        if f in ("devices", "sensor_placements", "modalities", "activities"):
            if isinstance(v, str): v = [x.strip() for x in re.split(r"[,;\n]", v) if x.strip()]
            v = [str(x).strip().lower() for x in (v or [])]
        if f in ("accelerometer_unit", "gyroscope_unit", "protocol_type", "setting") and isinstance(v, str):
            v = v.strip().lower().replace("m/s2", "m/s^2").replace("m/s²", "m/s^2").replace("meters per second squared", "m/s^2")
            v = {"degrees per second": "deg/s", "°/s": "deg/s", "radians per second": "rad/s", "mg": "milli-g"}.get(v, v)
        out[f] = v
    return out

def mock_extract(gold, variant, seed=0):
    """SMOKE-mode stand-in for an LLM: copies gold and injects realistic errors (worse for 'zeroshot')."""
    rng = np.random.default_rng(seed); p = 0.35 if variant == "zeroshot" else 0.15; rec = {}
    for f in RECORD_FIELDS:
        v = gold[f]
        if rng.random() < p:
            if f == "sampling_rate_hz": v = (v or 50) * rng.choice([0.5, 2.0])
            elif f == "n_participants": v = (v or 5) + int(rng.integers(1, 4))
            elif f in ("devices", "sensor_placements", "modalities", "activities"): v = list(v)[:-1] if len(v) > 1 else v
            elif f == "accelerometer_unit": v = "g" if v == "m/s^2" else "m/s^2"
            elif f == "gyroscope_unit": v = "deg/s" if v == "rad/s" else "rad/s"
            else: v = None
        rec[f] = v
    return normalise_record(rec)

def extract_all():
    results = {}   # (model, variant) -> {dataset: record}
    if MODE == "SMOKE":
        for mi, model in enumerate(["mock-llm-A", "mock-llm-B"]):
            for variant in EXTRACTION_VARIANTS:
                results[(model, variant)] = {k: mock_extract(GOLD[k], variant, seed=mi * 10 + hash(k) % 7) for k in ALL_DATASETS}
        return results
    for model in LLM_MODELS:
        gated = any(s in model for s in ("meta-llama", "mistralai", "google/gemma"))
        if gated and not HF_TOKEN: print("skip gated model (no HF_TOKEN):", model); continue
        tag = model.replace("/", "__"); todo = [(v, k) for v in EXTRACTION_VARIANTS for k in ALL_DATASETS
                                              if not (WORK / "extracted" / f"{tag}__{v}__{k}.json").exists()]
        if todo:
            print(f"loading {model} ...");
            try: llm = LLM(model)
            except Exception as e: print("  failed to load:", e); continue
            for variant, k in tqdm(todo, desc=tag):
                rec, raw_outputs = {}, {}
                for group in FIELD_GROUPS:
                    ctx = "\n...\n".join(retrieve(DOCS[k], group)) if variant == "retrieval" else " ".join(DOCS[k].split()[:5000])
                    out = llm.generate(build_prompt(group, ctx)); raw_outputs[group] = out
                    rec.update(parse_json(out))
                rec = normalise_record(rec); rec["_raw"] = raw_outputs
                jdump(rec, WORK / "extracted" / f"{tag}__{variant}__{k}.json")
                log_manifest(stage="extraction", model=model, variant=variant, dataset=k)
            llm.close()
        for variant in EXTRACTION_VARIANTS:
            results[(model, variant)] = {k: jload(WORK / "extracted" / f"{tag}__{variant}__{k}.json") for k in ALL_DATASETS
                                         if (WORK / "extracted" / f"{tag}__{variant}__{k}.json").exists()}
    return results

EXTRACTED = extract_all()
print("extraction runs:", [f"{m} / {v}: {len(r)} data sets" for (m, v), r in EXTRACTED.items()])

# %% [markdown]
# ### 4.1 Scoring extraction against gold (field-level)

# %%
PLACE_SYNONYMS = {"hand": "wrist", "arm": "wrist", "forearm": "lower arm", "lower back": "back", "hip": "waist", "belt": "waist",
                  "pocket": "trouser pocket", "trousers": "trouser", "leg": "thigh", "shin": "calf", "torso": "chest"}
def norm_tokens(s):
    s = re.sub(r"\(.*?\)", " ", str(s).lower()); s = re.sub(r"[^a-z0-9/^ ]", " ", s)
    toks = [PLACE_SYNONYMS.get(t, t) for t in s.split() if t not in ("the", "a", "an", "on", "of", "and", "sensor", "sensors", "mounted", "worn")]
    return " ".join(toks)
def fuzzy_set_f1(pred, gold, thr=0.75):
    import difflib
    pred = [norm_tokens(p) for p in (pred or []) if str(p).strip()]; gold = [norm_tokens(g) for g in (gold or [])]
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    def match(a, b): return a == b or a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= thr
    tp_p = sum(any(match(p, g) for g in gold) for p in pred); tp_g = sum(any(match(g, p) for p in pred) for g in gold)
    prec, rec = tp_p / len(pred), tp_g / len(gold)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
def score_field(f, pred, gold):
    if f == "sampling_rate_hz":
        gs = gold if isinstance(gold, list) else [gold]; ps = pred if isinstance(pred, list) else [pred]
        ps = [p for p in ps if isinstance(p, (int, float))]
        if not ps: return 0.0
        return float(np.mean([any(abs(p - g) / max(g, 1e-6) <= 0.10 for g in gs) for p in ps]))
    if f == "n_participants": return float(pred == gold) if pred is not None else 0.0
    if f in ("devices", "sensor_placements", "modalities", "activities"): return fuzzy_set_f1(pred, gold)
    return float(pred == gold) if pred is not None else 0.0

def score_run(records):
    rows = []
    for k, rec in records.items():
        for f in RECORD_FIELDS:
            rows.append(dict(dataset=k, field=f, score=score_field(f, rec.get(f), GOLD[k][f])))
    return pd.DataFrame(rows)

E1 = []
for (model, variant), recs in EXTRACTED.items():
    df = score_run(recs); df["model"] = model.split("/")[-1]; df["variant"] = variant; E1.append(df)
E1 = pd.concat(E1, ignore_index=True)
T2 = E1.pivot_table(index=["model", "variant"], columns="field", values="score", aggfunc="mean")[RECORD_FIELDS]
T2["composite"] = T2.mean(axis=1); T2 = T2.reset_index()
save_table(T2, "T2_extraction_field_scores", "E1: field-level extraction accuracy (mean over data sets) by model and prompting variant.")
T2.round(3)

# %%
# Figure 2: extraction heatmap (sequential single-hue), plus composite bars
fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(T2) + 2.2), gridspec_kw=dict(width_ratios=[4, 1.2]))
mat = T2.set_index(T2.model + " · " + T2.variant)[RECORD_FIELDS]
im = axes[0].imshow(mat.values, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
axes[0].set_xticks(range(len(RECORD_FIELDS))); axes[0].set_xticklabels([f.replace("_", "\n") for f in RECORD_FIELDS], fontsize=7)
axes[0].set_yticks(range(len(mat))); axes[0].set_yticklabels(mat.index); axes[0].grid(False)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]; axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "#0b0b0b")
axes[0].set_title("E1  Field-level extraction accuracy vs gold (mean over data sets)")
cb = fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02); cb.set_label("accuracy / set-F1")
axes[1].barh(range(len(mat)), T2["composite"].values, color=PALETTE[0], height=0.6)
axes[1].set_yticks([]); axes[1].set_xlim(0, 1); axes[1].invert_yaxis(); axes[1].set_xlabel("composite"); axes[1].set_title("Composite")
for i, v in enumerate(T2["composite"].values): axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)
axes[0].invert_yaxis(); fig.tight_layout(); savefig(fig, "F2_extraction_heatmap"); plt.show()

# %% [markdown]
# ## 5. Data-grounded verification probes and the documentation–data audit (E2)
#
# Each probe reads the raw files and returns a *data value*; a field is **confirmed** if the record agrees,
# **contradicted** if it does not, **unverifiable** if the data cannot speak to it. Confident contradictions are
# auto-corrected to produce the *verified* record (condition M3).

# %%
def probe_rate(store):
    rates = [s.rate_hz for s in store["sessions"] if s.rate_hz]
    return (float(np.median(rates)), sorted(set(round(r / 10) * 10 for r in rates))) if rates else (None, None)

def probe_acc_unit(store):
    ch = store["channels"]; groups = {}
    for i, c in enumerate(ch):
        if c["modality"] == "accelerometer": groups.setdefault(c["placement"], []).append(i)
    mags = []
    for s in store["sessions"][: 400]:
        for pl, idx in groups.items():
            if len(idx) != 3: continue
            v = s.data[:, idx]; n = min(len(v), 2000); v = v[:n]
            norm = np.linalg.norm(v, axis=1)
            w = max(10, n // 20); stds = pd.Series(norm).rolling(w).std().values; means = pd.Series(norm).rolling(w).mean().values
            ok = ~np.isnan(stds)
            if ok.sum() > 0: mags.append(float(means[ok][np.argmin(stds[ok])]))
    if not mags: return None, None
    m = float(np.median(mags))
    unit = "g" if 0.6 < m < 1.5 else "m/s^2" if 6 < m < 14 else "milli-g" if 600 < m < 1500 else None
    return unit, m

def probe_gyro_unit(store):
    idx = [i for i, c in enumerate(store["channels"]) if c["modality"] == "gyroscope"]
    if not idx: return None, None
    q = [np.percentile(np.abs(s.data[:, idx]), 99) for s in store["sessions"][:400]]
    p99 = float(np.median(q)); return ("deg/s" if p99 > 20 else "rad/s"), p99

def probe_placements(store): return sorted(set(c["placement"] for c in store["channels"]))
def probe_modalities(store): return sorted(set(c["modality"] for c in store["channels"]))
def probe_labels(store): return sorted(set(s.label for s in store["sessions"]))
def probe_participants(store): return len(set(s.subject for s in store["sessions"]))

def verify_record(key, rec):
    """Returns (verified_record, audit_rows)."""
    st = STORES[key]; ver = dict(rec); rows = []
    def row(field, status, doc, data, conf, corrected=False):
        rows.append(dict(dataset=key, field=field, status=status, documented=doc, data=data, confidence=conf, corrected=corrected))
    # sampling rate
    r_med, r_set = probe_rate(st); doc = rec.get("sampling_rate_hz")
    if r_med is None: row("sampling_rate_hz", "unverifiable", doc, None, "none")
    else:
        docs = doc if isinstance(doc, list) else [doc]; docs = [d for d in docs if isinstance(d, (int, float))]
        ok = bool(docs) and all(any(abs(d - r) / r <= 0.10 for r in (r_set or [r_med])) for d in docs)
        if ok: row("sampling_rate_hz", "confirmed", doc, r_med, "high")
        else:
            ver["sampling_rate_hz"] = r_set if (r_set and len(r_set) > 1) else r_med; row("sampling_rate_hz", "contradicted", doc, r_med, "high", True)
    # accelerometer unit
    u, m = probe_acc_unit(st); doc = rec.get("accelerometer_unit")
    if u is None: row("accelerometer_unit", "unverifiable", doc, m, "none")
    elif doc == u: row("accelerometer_unit", "confirmed", doc, f"{u} (|a|≈{m:.2f})", "high")
    else: ver["accelerometer_unit"] = u; row("accelerometer_unit", "contradicted", doc, f"{u} (|a|≈{m:.2f})", "high", True)
    # gyroscope unit (weak probe: amplitude heuristic)
    u, p = probe_gyro_unit(st); doc = rec.get("gyroscope_unit")
    if u is None: row("gyroscope_unit", "unverifiable", doc, None, "none")
    elif doc == u: row("gyroscope_unit", "confirmed", doc, f"{u} (p99≈{p:.1f})", "medium")
    else: ver["gyroscope_unit"] = u; row("gyroscope_unit", "contradicted", doc, f"{u} (p99≈{p:.1f})", "medium", True)
    # participants
    n = probe_participants(st); doc = rec.get("n_participants")
    if doc == n: row("n_participants", "confirmed", doc, n, "high")
    else: ver["n_participants"] = n; row("n_participants", "contradicted", doc, n, "high", True)
    # placements / modalities (channel layout)
    for f, probe in (("sensor_placements", probe_placements), ("modalities", probe_modalities)):
        data = probe(st); f1 = fuzzy_set_f1(rec.get(f), data)
        if f1 >= 0.8: row(f, "confirmed", rec.get(f), data, "medium")
        else: ver[f] = data; row(f, "contradicted", rec.get(f), data, "medium", True)
    # label set
    data = probe_labels(st); f1 = fuzzy_set_f1(rec.get("activities"), data)
    if f1 >= 0.9: row("activities", "confirmed", len(rec.get("activities") or []), len(data), "high")
    else: ver["activities"] = data; row("activities", "contradicted", len(rec.get("activities") or []), len(data), "high", True)
    for f in ("devices", "protocol_type", "setting"): row(f, "unverifiable", rec.get(f), None, "none")
    return ver, rows

AUDIT_GOLD = []; VERIFIED_GOLD = {}
for k in ALL_DATASETS:
    VERIFIED_GOLD[k], rows = verify_record(k, GOLD[k]); AUDIT_GOLD += rows
AUDIT_GOLD = pd.DataFrame(AUDIT_GOLD)
AUDIT_GOLD.to_csv(WORK / "audit" / "audit_gold.csv", index=False)

AUDIT_EXTRACTED = []; VERIFIED_EXTRACTED = {}
for (model, variant), recs in EXTRACTED.items():
    VERIFIED_EXTRACTED[(model, variant)] = {}
    for k, rec in recs.items():
        v, rows = verify_record(k, rec); VERIFIED_EXTRACTED[(model, variant)][k] = v
        for r in rows: r.update(model=model.split("/")[-1], variant=variant)
        AUDIT_EXTRACTED += rows
AUDIT_EXTRACTED = pd.DataFrame(AUDIT_EXTRACTED); AUDIT_EXTRACTED.to_csv(WORK / "audit" / "audit_extracted.csv", index=False)

verifiable = AUDIT_GOLD[AUDIT_GOLD.status != "unverifiable"]
T3 = (AUDIT_GOLD.groupby("field").status.value_counts().unstack(fill_value=0).reindex(RECORD_FIELDS).fillna(0).astype(int).reset_index())
save_table(T3, "T3_audit_gold_by_field", "E2: documentation–data audit of the gold (author-written) records, by field.")
n_contra_ds = AUDIT_GOLD[AUDIT_GOLD.status == "contradicted"].dataset.nunique()
print(f"Gold records: {n_contra_ds}/{len(ALL_DATASETS)} data sets have at least one documentation–data contradiction; "
      f"{(verifiable.status=='contradicted').mean():.1%} of verifiable fields contradicted.")

# How much does verification repair extraction errors?  (score before vs after, verifiable fields only)
def score_records(recs):
    return score_run(recs).assign(model="x")
E2_REPAIR = []
for key_mv, recs in EXTRACTED.items():
    before = score_run(recs); after = score_run(VERIFIED_EXTRACTED[key_mv])
    vf = ["sampling_rate_hz", "accelerometer_unit", "gyroscope_unit", "n_participants", "sensor_placements", "modalities", "activities"]
    E2_REPAIR.append(dict(model=key_mv[0].split("/")[-1], variant=key_mv[1],
                          before=before[before.field.isin(vf)].score.mean(), after=after[after.field.isin(vf)].score.mean()))
E2_REPAIR = pd.DataFrame(E2_REPAIR); E2_REPAIR["gain"] = E2_REPAIR.after - E2_REPAIR.before
save_table(E2_REPAIR, "T4_verification_repair", "E2: accuracy of extracted records on verifiable fields before and after data-grounded verification.")
E2_REPAIR.round(3)

# %%
# Figure 1: audit matrix for gold records (glyphs carry the status; colour is secondary)
STATUS_COL = {"confirmed": "#1baf7a", "contradicted": "#e34948", "unverifiable": "#c3c2b7"}
STATUS_GLYPH = {"confirmed": "✓", "contradicted": "✗", "unverifiable": "·"}
piv = AUDIT_GOLD.pivot(index="dataset", columns="field", values="status").reindex(columns=RECORD_FIELDS)
fig, ax = plt.subplots(figsize=(10, 0.42 * len(piv) + 1.8))
for i, ds in enumerate(piv.index):
    for j, f in enumerate(RECORD_FIELDS):
        s = piv.loc[ds, f]; ax.add_patch(plt.Rectangle((j, i), 1, 1, color=STATUS_COL[s], alpha=0.85 if s != "unverifiable" else 0.5, ec="white", lw=1.5))
        ax.text(j + 0.5, i + 0.5, STATUS_GLYPH[s], ha="center", va="center", fontsize=11, color="white" if s != "unverifiable" else "#52514e")
ax.set_xlim(0, len(RECORD_FIELDS)); ax.set_ylim(len(piv), 0); ax.set_xticks(np.arange(len(RECORD_FIELDS)) + 0.5)
ax.set_xticklabels([f.replace("_", "\n") for f in RECORD_FIELDS], fontsize=7.5); ax.set_yticks(np.arange(len(piv)) + 0.5)
ax.set_yticklabels([DATASET_REGISTRY[d]["name"].split(" (")[0] for d in piv.index]); ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=STATUS_COL[s], label=f"{STATUS_GLYPH[s]}  {s}") for s in STATUS_COL], loc="upper center",
          bbox_to_anchor=(0.5, -0.18), ncol=3)
ax.set_title("E2  Documentation–data audit of the harmonisation records (gold)")
fig.tight_layout(); savefig(fig, "F1_audit_matrix"); plt.show()

# %% [markdown]
# ## 6. Harmonisation driven by a record: units → resampling → windows → channel descriptions
#
# Nothing here is data set specific. Whatever the record says is what the model gets — which is exactly how a wrong
# record hurts.

# %%
ACC_FACTOR = {"g": 1.0, "m/s^2": 1 / 9.81, "milli-g": 1e-3, None: 1 / 9.81}        # → g
GYR_FACTOR = {"rad/s": 1.0, "deg/s": np.pi / 180, None: 1.0}                      # → rad/s
DEFAULT_RATE = 50.0

def resample_to(x, src_hz, dst_hz):
    if src_hz is None or abs(src_hz - dst_hz) < 1e-6: return x
    n_out = int(round(len(x) * dst_hz / src_hz))
    if n_out < 2: return None
    t_src = np.arange(len(x)) / src_hz; t_dst = np.arange(n_out) / dst_hz
    return np.stack([np.interp(t_dst, t_src, x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)

def harmonise_windows(key, acc_unit, gyr_unit, rate, cap=MAX_WINDOWS_PER_DATASET, seed=0, use_session_rate=False):
    """Returns X [N, C, T], labels list, subjects list. `rate` may be a number, a list (→ use per-session measured rate), or None."""
    st = STORES[key]; ch = st["channels"]; T = int(WINDOW_SEC * TARGET_HZ); S = int(STRIDE_SEC * TARGET_HZ)
    scale = np.array([ACC_FACTOR.get(acc_unit, 1/9.81) if c["modality"] == "accelerometer" else GYR_FACTOR.get(gyr_unit, 1.0) / 5.0
                      if c["modality"] == "gyroscope" else 1.0 for c in ch], np.float32)
    Xs, ys, subs = [], [], []
    for s in st["sessions"]:
        if isinstance(rate, list) or use_session_rate: src = s.rate_hz or (float(np.median(rate)) if isinstance(rate, list) else (float(rate) if rate else DEFAULT_RATE))  # v6 fix: fall back to the record's rate, not 50 Hz
        else: src = float(rate) if rate else DEFAULT_RATE
        x = resample_to(s.data * scale, src, TARGET_HZ)
        if x is None or len(x) < T: continue
        for i in range(0, len(x) - T + 1, S):
            Xs.append(x[i:i+T].T); ys.append(s.label); subs.append(s.subject)
    X = np.stack(Xs).astype(np.float32); ys = np.array(ys); subs = np.array(subs)
    X = np.nan_to_num(np.clip(X, -20, 20))
    if len(X) > cap:
        rng = np.random.default_rng(seed); keep = []
        for lab in np.unique(ys):
            idx = np.where(ys == lab)[0]; k = max(20, int(cap * len(idx) / len(ys))); keep += list(rng.choice(idx, min(k, len(idx)), replace=False))
        keep = np.array(sorted(keep)); X, ys, subs = X[keep], ys[keep], subs[keep]
    return X, ys, subs

def channel_descriptions(key, condition, rec=None, defect=None):
    """Per-channel natural-language descriptions under a metadata condition (+ optional defect spec)."""
    ch = [dict(c) for c in STORES[key]["channels"]]
    if condition == "M0_none": return ["IMU channel"] * len(ch)
    if condition == "M1_generic": return [f"{c['modality']} {c['axis']}-axis" for c in ch]
    if condition in ("M2_extracted", "M3_verified"):
        places = rec.get("sensor_placements") or []; acc_u = rec.get("accelerometer_unit") or "unknown unit"; gyr_u = rec.get("gyroscope_unit") or "unknown unit"
        import difflib
        def best(p):
            cands = [(difflib.SequenceMatcher(None, norm_tokens(p), norm_tokens(q)).ratio(), q) for q in places]
            return max(cands)[1] if cands and max(cands)[0] >= 0.5 else "unknown body location"
        return [f"{c['modality']} {c['axis']}-axis, {best(c['placement'])}, {acc_u if c['modality']=='accelerometer' else gyr_u}" for c in ch]
    # M4 gold (optionally with a defect)
    if defect:
        kind, sev, rng = defect["kind"], defect["severity"], defect["rng"]
        n = len(ch); idx = rng.choice(n, int(round(sev * n)), replace=False)
        if kind == "placement":
            pool = [p for p in ["wrist", "chest", "ankle", "waist", "thigh", "upper arm", "back", "head", "front trouser pocket"]]
            for i in idx: ch[i]["placement"] = rng.choice([p for p in pool if p not in ch[i]["placement"]])
        if kind == "axis":
            for i in idx: ch[i]["axis"] = rng.choice([a for a in "xyz" if a != ch[i]["axis"]])
    return [f"{c['modality']} {c['axis']}-axis, {c['placement']}, {c['unit']}" for c in ch]

def record_for(key, condition, rec_source=None):
    if condition == "M4_gold": return GOLD[key]
    if condition == "M3_verified": return rec_source[key] if rec_source and key in rec_source else VERIFIED_GOLD[key]
    if condition == "M2_extracted": return rec_source[key] if rec_source and key in rec_source else GOLD[key]
    return dict(GOLD[key], accelerometer_unit=None, gyroscope_unit=None, sampling_rate_hz=None)   # M0/M1: model knows nothing

# %% [markdown]
# ## 7. The metadata-conditioned encoder (compact, HALO-style channel conditioning)

# %%
class ChannelTokenEncoder(nn.Module):
    def __init__(self, text_dim, d=128, n_tokens=10, layers=3, heads=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 64, 7, stride=2, padding=3), nn.GELU(), nn.Conv1d(64, d, 5, stride=2, padding=2), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(n_tokens); self.n_tokens = n_tokens
        self.pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02); self.chan_proj = nn.Linear(text_dim, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers); self.out = nn.Linear(d, text_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    def forward(self, x, cemb, cmask):
        B, C, T = x.shape
        h = self.stem(x.reshape(B * C, 1, T)); h = self.pool(h).transpose(1, 2)            # [B*C, n_tok, d]
        h = h.reshape(B, C, self.n_tokens, -1) + self.pos[None, None] + self.chan_proj(cemb)[:, :, None, :]
        h = h.reshape(B, C * self.n_tokens, -1); kpm = ~cmask.repeat_interleave(self.n_tokens, dim=1)
        h = self.tr(h, src_key_padding_mask=kpm)
        m = (~kpm).float()[..., None]; z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.out(z), dim=-1)

@dataclass
class DSBundle:
    key: str; X: np.ndarray; y: np.ndarray; subjects: np.ndarray; cemb: np.ndarray; canon: np.ndarray

def build_bundle(key, condition, rec_source=None, defect=None, seed=0):
    rec = record_for(key, condition, rec_source)
    acc_u, gyr_u, rate = rec.get("accelerometer_unit"), rec.get("gyroscope_unit"), rec.get("sampling_rate_hz")
    if defect and defect["kind"] == "unit" and defect["hit"]: acc_u = "g" if acc_u == "m/s^2" else "m/s^2"
    if defect and defect["kind"] == "rate" and defect["hit"] and isinstance(rate, (int, float)): rate = rate * 2 if rate < 60 else rate / 2
    X, y, subs = harmonise_windows(key, acc_u, gyr_u, rate, seed=seed, use_session_rate=(condition == "M3_verified"))
    desc = channel_descriptions(key, condition, rec, defect if defect and defect["kind"] in ("placement", "axis") else None)
    cemb = TEXT(desc)
    canon = np.array([GOLD[key]["canonical_map"].get(l, "other") for l in y])
    return DSBundle(key, X, y, subs, cemb, canon)

def collate(bundles_batch):
    """bundles_batch: list of (X[C,T], cemb[C,D], target_idx) with possibly different C."""
    Cmax = max(b[0].shape[0] for b in bundles_batch); T = bundles_batch[0][0].shape[1]; D = bundles_batch[0][1].shape[1]
    X = np.zeros((len(bundles_batch), Cmax, T), np.float32); E = np.zeros((len(bundles_batch), Cmax, D), np.float32)
    M = np.zeros((len(bundles_batch), Cmax), bool); Y = np.zeros(len(bundles_batch), np.int64)
    for i, (x, e, yi) in enumerate(bundles_batch):
        c = x.shape[0]; X[i, :c] = x; E[i, :c] = e; M[i, :c] = True; Y[i] = yi
    return torch.from_numpy(X), torch.from_numpy(E), torch.from_numpy(M), torch.from_numpy(Y)

@torch.no_grad()
def embed(model, bundle, bs=512):
    model.eval(); out = []
    for i in range(0, len(bundle.X), bs):
        X = torch.from_numpy(bundle.X[i:i+bs]).to(DEVICE); C = X.shape[1]
        E = torch.from_numpy(bundle.cemb)[None].repeat(len(X), 1, 1).to(DEVICE); M = torch.ones(len(X), C, dtype=torch.bool, device=DEVICE)
        out.append(model(X, E, M).cpu().numpy())
    return np.concatenate(out)

def bootstrap_ci(y_true, y_pred, n=1000, seed=0):
    rng = np.random.default_rng(seed); accs, f1s = [], []
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); N = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, N, N); accs.append(accuracy_score(y_true[idx], y_pred[idx])); f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return (np.percentile(accs, 2.5), np.percentile(accs, 97.5)), (np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))

def evaluate(model, test_bundle, seed=0):
    """Zero-shot (prototypes = the test data set's own label names) and few-shot linear probes."""
    Z = embed(model, test_bundle); labels = sorted(set(test_bundle.y)); P = TEXT(labels)
    pred = np.array(labels)[np.argmax(Z @ P.T, 1)]
    acc = accuracy_score(test_bundle.y, pred); f1 = f1_score(test_bundle.y, pred, average="macro"); (acc_ci, f1_ci) = bootstrap_ci(test_bundle.y, pred)
    res = dict(zeroshot_acc=acc, zeroshot_f1=f1, zeroshot_acc_lo=acc_ci[0], zeroshot_acc_hi=acc_ci[1], zeroshot_f1_lo=f1_ci[0], zeroshot_f1_hi=f1_ci[1], n_windows=len(Z))
    rng = np.random.default_rng(seed)
    for frac in FEWSHOT_FRACS:
        tr = []
        for lab in labels:
            idx = np.where(test_bundle.y == lab)[0]; k = max(2, int(round(frac * len(idx)))); tr += list(rng.choice(idx, min(k, len(idx)), replace=False))
        tr = np.array(tr); te = np.setdiff1d(np.arange(len(Z)), tr)
        if len(set(test_bundle.y[tr])) < 2: continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], test_bundle.y[tr]); p = clf.predict(Z[te])
        res[f"fewshot{int(frac*100)}_acc"] = accuracy_score(test_bundle.y[te], p); res[f"fewshot{int(frac*100)}_f1"] = f1_score(test_bundle.y[te], p, average="macro")
    return res, Z, pred

# %% [markdown]
# ## 8. Experiments E3 (metadata conditions), E4 (defect injection), E5 (label alignment)
#
# One function runs a configuration end to end and caches its result as JSON in `runs/`; re-running the notebook
# never repeats finished work. In SMOKE mode epochs and seeds are reduced automatically.

# %%
if MODE == "SMOKE":
    EPOCHS, SEEDS, DEFECT_SEVERITIES, BATCH_SIZE = 3, [0], [0.5, 1.0], 64
EXTRACTED_SOURCE = EXTRACTED.get((EXTRACTED_MODEL_FOR_M2, "retrieval")) or (next(iter(EXTRACTED.values())) if EXTRACTED else {})
_missing = [k for k in ALL_DATASETS if k not in EXTRACTED_SOURCE]
if _missing: print("WARNING: no extracted record for", _missing, "— M2/M3 fall back to gold for these data sets (report this!)")
VERIFIED_SOURCE = VERIFIED_EXTRACTED.get((EXTRACTED_MODEL_FOR_M2, "retrieval")) or (next(iter(VERIFIED_EXTRACTED.values())) if VERIFIED_EXTRACTED else {})

def run_config(name, condition, seed, defect_kind=None, severity=0.0, label_texts_override=None, save_embeddings=False):
    out = WORK / "runs" / f"{name}.json"
    if out.exists(): return jload(out)
    rng = np.random.default_rng(1000 + seed)
    hit_ds = set(rng.choice(TRAIN_DATASETS, int(round(severity * len(TRAIN_DATASETS))), replace=False)) if defect_kind in ("unit", "rate") else set()
    train_bundles = []
    for k in TRAIN_DATASETS:
        defect = None
        if defect_kind: defect = dict(kind=defect_kind, severity=severity, rng=np.random.default_rng(seed * 7 + hash(k) % 97), hit=(k in hit_ds))
        src = EXTRACTED_SOURCE if condition == "M2_extracted" else VERIFIED_SOURCE if condition == "M3_verified" else None
        train_bundles.append(build_bundle(k, condition, src, defect, seed))
    canon_space = sorted(set(np.concatenate([b.canon for b in train_bundles])))
    label_texts = list(canon_space)
    if defect_kind == "label":   # corrupt the text of a fraction of training prototypes with a *different* canonical name
        n = int(round(severity * len(label_texts))); idx = rng.choice(len(label_texts), n, replace=False)
        for i in idx: label_texts[i] = rng.choice([c for c in CANONICAL_ACTIVITIES if c != canon_space[i]])
    if label_texts_override: label_texts = [label_texts_override.get(c, c) for c in canon_space]
    # canon index must align: targets are indices into canon_space; prototypes are label_texts (same order)
    for b in train_bundles: b.canon = np.array([canon_space.index(c) for c in b.canon])
    model, losses = train_encoder(train_bundles, label_texts, seed, tag=name)
    torch.save(dict(state=model.state_dict(), label_texts=label_texts, canon_space=canon_space), WORK / "runs" / f"{name}.pt")  # v6: keep checkpoints for later held-out evaluation
    res = dict(name=name, condition=condition, seed=seed, defect=defect_kind, severity=severity, losses=losses, per_dataset={})
    for k in TEST_DATASETS:
        src = EXTRACTED_SOURCE if condition == "M2_extracted" else VERIFIED_SOURCE if condition == "M3_verified" else None
        tb = build_bundle(k, condition, src, None, seed); r, Z, pred = evaluate(model, tb, seed)
        res["per_dataset"][k] = r
        if save_embeddings: np.savez_compressed(WORK / "runs" / f"{name}__{k}_emb.npz", Z=Z, y=tb.y, canon=tb.canon, pred=pred)
    res["mean_zeroshot_acc"] = float(np.mean([v["zeroshot_acc"] for v in res["per_dataset"].values()]))
    res["mean_zeroshot_f1"] = float(np.mean([v["zeroshot_f1"] for v in res["per_dataset"].values()]))
    jdump(res, out); log_manifest(stage="train", **{k: v for k, v in res.items() if k not in ("per_dataset", "losses")})
    return res

def train_encoder(train_bundles, label_texts, seed, epochs=EPOCHS, tag="run"):
    set_seed(seed); protos = torch.from_numpy(TEXT(label_texts)).to(DEVICE)
    model = ChannelTokenEncoder(protos.shape[1]).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-2)
    items = [(b.X[i], b.cemb, int(b.canon[i])) for b in train_bundles for i in range(len(b.X))]
    steps = max(1, len(items) // BATCH_SIZE); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * steps)
    rng = np.random.default_rng(seed); losses = []
    pbar = tqdm(range(epochs), desc=f"train {tag}", leave=False)
    for ep in pbar:
        model.train(); order = rng.permutation(len(items)); tot = 0
        for s in range(steps):
            batch = [items[j] for j in order[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]]
            if not batch: continue
            X, E, M, Y = [t.to(DEVICE) for t in collate(batch)]
            logits = model.logit_scale.exp() * model(X, E, M) @ protos.T
            loss = F.cross_entropy(logits, Y); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        losses.append(tot / steps); pbar.set_postfix(loss=f"{losses[-1]:.3f}")
    return model, losses

# ---------------- E3: metadata conditions ----------------
E3_RUNS = []
for cond in CONDITIONS:
    for seed in SEEDS:
        E3_RUNS.append(run_config(f"E3_{cond}_s{seed}", cond, seed, save_embeddings=(cond in ("M0_none", "M4_gold") and seed == SEEDS[0])))
def runs_to_df(runs):
    rows = []
    for r in runs:
        for k, v in r["per_dataset"].items():
            rows.append(dict(name=r["name"], condition=r["condition"], seed=r["seed"], defect=r.get("defect"), severity=r.get("severity", 0), dataset=k, **v))
    return pd.DataFrame(rows)
E3 = runs_to_df(E3_RUNS); E3.to_csv(WORK / "results" / "E3_runs.csv", index=False)

metric_cols = [c for c in E3.columns if c.endswith("_acc") or c.endswith("_f1")]
T5 = E3.groupby("condition")[["zeroshot_acc", "zeroshot_f1"] + [c for c in metric_cols if c.startswith("fewshot")]].agg(["mean", "std"])
T5.columns = [f"{a}_{b}" for a, b in T5.columns]; T5 = T5.reindex(CONDITIONS).reset_index()
save_table(T5, "T5_conditions_summary", "E3: zero-shot and few-shot performance on held-out data sets under five metadata conditions (mean ± sd over seeds and data sets).")
T5.round(3)

# Paired comparisons between conditions (per data set × seed), Holm-corrected sign-flip permutation tests
def holm(pvals):
    order = np.argsort(pvals); adj = np.empty(len(pvals)); m = len(pvals); running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i]); adj[i] = min(1.0, running)
    return adj
pairs = [("M0_none", "M4_gold"), ("M1_generic", "M4_gold"), ("M2_extracted", "M4_gold"), ("M3_verified", "M4_gold"), ("M2_extracted", "M3_verified"), ("M0_none", "M3_verified")]
rows = []
piv = E3.pivot_table(index=["dataset", "seed"], columns="condition", values="zeroshot_acc")
for a, b in pairs:
    if a in piv and b in piv:
        d = (piv[b] - piv[a]).dropna().values; rng = np.random.default_rng(0)
        obs = d.mean(); null = [np.mean(d * rng.choice([-1, 1], len(d))) for _ in range(5000)]
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1); rows.append(dict(comparison=f"{b} − {a}", mean_delta=obs, p_raw=p))
T6 = pd.DataFrame(rows)
if len(T6): T6["p_holm"] = holm(T6.p_raw.values)
save_table(T6, "T6_condition_tests", "E3: paired sign-flip permutation tests between metadata conditions (zero-shot accuracy; Holm-corrected).")
T6

# ---------------- E4: defect injection on gold metadata ----------------
E4_RUNS = []
for kind in DEFECTS:
    for sev in DEFECT_SEVERITIES:
        for seed in SEEDS:
            E4_RUNS.append(run_config(f"E4_{kind}_{int(sev*100)}_s{seed}", "M4_gold", seed, defect_kind=kind, severity=sev))
E4 = runs_to_df(E4_RUNS); E4.to_csv(WORK / "results" / "E4_runs.csv", index=False)
base = E3[E3.condition == "M4_gold"].groupby(["dataset", "seed"]).zeroshot_acc.mean().rename("base_acc").reset_index()
E4m = E4.merge(base, on=["dataset", "seed"]); E4m["cost"] = E4m.base_acc - E4m.zeroshot_acc
T7 = E4m.groupby(["defect", "severity"]).cost.agg(["mean", "std", "count"]).reset_index()
T7.insert(0, "defect_label", T7.defect.map({"placement": "wrong placement", "unit": "wrong accelerometer unit", "rate": "wrong sampling rate", "axis": "wrong axis labels", "label": "wrong label synonym"}))
save_table(T7, "T7_defect_costs", "E4: zero-shot accuracy lost (points) when a given fraction of the gold metadata carries a defect.")
T7.round(3)

# ---------------- E5: label alignment methods ----------------
def wordnet_align(label):
    toks = set(norm_tokens(label).split()); best, score = "other", 0
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try: wn.synsets("walk")
        except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
        def lemmas(t):
            out = {t}
            for s in wn.synsets(t): out |= {l.name().lower().replace("_", " ") for l in s.lemmas()}
            return out
        exp = set().union(*[lemmas(t) for t in toks]) if toks else set()
    except Exception: exp = toks
    for c in CANONICAL_ACTIVITIES:
        ct = set(norm_tokens(c).split()); ov = len(exp & ct) + 2 * len(toks & ct)
        if ov > score: best, score = c, ov
    return best
def embedding_align(labels):
    P = TEXT(CANONICAL_ACTIVITIES); L = TEXT(labels); return [CANONICAL_ACTIVITIES[i] for i in np.argmax(L @ P.T, 1)]
def llm_align(labels, llm=None):
    if llm is None: return None
    prompt = ("Map each human-activity label to exactly one canonical class from this list:\n" + json.dumps(CANONICAL_ACTIVITIES) +
              "\nLabels:\n" + json.dumps(labels) + "\nReturn ONLY a JSON object {label: canonical_class}.")
    out = parse_json(llm.generate(prompt, 600)); return [out.get(l, "other") if out.get(l) in CANONICAL_ACTIVITIES else "other" for l in labels]

E5_rows = []; llm5 = None
if MODE == "FULL":
    try: llm5 = LLM(EXTRACTED_MODEL_FOR_M2)
    except Exception as e: print("E5: LLM unavailable:", e)
for k in ALL_DATASETS:
    labels = GOLD[k]["activities"]; gold = [GOLD[k]["canonical_map"][l] for l in labels]
    methods = {"wordnet": [wordnet_align(l) for l in labels], "embedding": embedding_align(labels)}
    la = llm_align(labels, llm5)
    if la: methods["llm_constrained"] = la
    if MODE == "SMOKE": methods["llm_constrained"] = [g if np.random.default_rng(hash(l) % 11).random() > 0.1 else "other" for l, g in zip(labels, gold)]
    for m, pred in methods.items():
        E5_rows.append(dict(dataset=k, method=m, accuracy=np.mean([p == g for p, g in zip(pred, gold)]), n_labels=len(labels)))
if llm5: llm5.close()
E5 = pd.DataFrame(E5_rows); T8 = E5.pivot_table(index="method", columns="dataset", values="accuracy"); T8["mean"] = T8.mean(1); T8 = T8.reset_index()
save_table(T8, "T8_label_alignment", "E5: accuracy of automatic raw-label → canonical-class alignment against the gold mapping.")
T8.round(3)

# %% [markdown]
# ## 9. Figures for the paper

# %%
COND_LABEL = {"M0_none": "M0 none", "M1_generic": "M1 generic", "M2_extracted": "M2 extracted", "M3_verified": "M3 verified", "M4_gold": "M4 gold"}
# ---- Figure 3: metadata conditions, per held-out data set and overall ----
agg = E3.groupby(["dataset", "condition"]).agg(acc=("zeroshot_acc", "mean"), sd=("zeroshot_acc", "std")).reset_index()
datasets = TEST_DATASETS + ["all"]; allm = E3.groupby("condition").agg(acc=("zeroshot_acc", "mean"), sd=("zeroshot_acc", "std")).reset_index().assign(dataset="all")
agg = pd.concat([agg, allm]); fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 2, 3.6)); w = 0.8 / len(CONDITIONS)
for ci, cond in enumerate(CONDITIONS):
    sub = agg[agg.condition == cond].set_index("dataset").reindex(datasets)
    xs = np.arange(len(datasets)) + (ci - (len(CONDITIONS) - 1) / 2) * w
    ax.bar(xs, sub.acc.values, w * 0.92, color=PALETTE[ci], label=COND_LABEL[cond], yerr=sub.sd.fillna(0).values, capsize=2, error_kw=dict(lw=0.8, color="#52514e"))
ax.set_xticks(range(len(datasets))); ax.set_xticklabels([DATASET_REGISTRY[d]["name"].split(" (")[0] if d != "all" else "All held-out" for d in datasets])
ax.set_ylabel("zero-shot accuracy"); ax.set_ylim(0, 1); ax.set_title("E3  Held-out zero-shot accuracy under five metadata conditions (mean ± sd over seeds)")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14)); fig.tight_layout(); savefig(fig, "F3_conditions"); plt.show()

# ---- Figure 4: defect cost curves ----
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for di, kind in enumerate(DEFECTS):
    sub = T7[T7.defect == kind].sort_values("severity")
    if sub.empty: continue
    xs = [0] + list(sub.severity * 100); ys = [0] + list(sub["mean"] * 100); sd = [0] + list(sub["std"].fillna(0) * 100)
    ax.errorbar(xs, ys, yerr=sd, marker="o", ms=5, lw=2, capsize=2, color=PALETTE[di], label=sub.defect_label.iloc[0])
ax.axhline(0, color="#52514e", lw=0.8); ax.set_xlabel("share of training metadata carrying the defect (%)"); ax.set_ylabel("zero-shot accuracy lost (points)")
ax.set_title("E4  The accuracy cost of documentation defects"); ax.set_xlim(-3, 105); ax.legend(loc="upper left")
fig.tight_layout(); savefig(fig, "F4_defect_costs"); plt.show()

# ---- Figure 5: few-shot curves per condition ----
fs_cols = [("zeroshot_acc", 0)] + [(f"fewshot{int(f*100)}_acc", f * 100) for f in FEWSHOT_FRACS if f"fewshot{int(f*100)}_acc" in E3]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for ci, cond in enumerate(CONDITIONS):
    sub = E3[E3.condition == cond]; ys = [sub[c].mean() for c, _ in fs_cols]; sds = [sub[c].std() for c, _ in fs_cols]
    ax.errorbar([x for _, x in fs_cols], ys, yerr=sds, marker="o", ms=5, lw=2, capsize=2, color=PALETTE[ci], label=COND_LABEL[cond])
ax.set_xlabel("labelled windows of the held-out data set (%)"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
ax.set_title("E3  From zero-shot to few-shot under each metadata condition"); ax.legend(); fig.tight_layout(); savefig(fig, "F5_fewshot"); plt.show()

# ---- Figure 6: embedding maps for one held-out data set, M0 vs M4 ----
def project2d(Z, seed=0):
    try:
        import umap; return umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
    except Exception:
        from sklearn.manifold import TSNE; return TSNE(n_components=2, random_state=seed, init="pca", perplexity=min(30, max(5, len(Z) // 50))).fit_transform(Z)
k0 = TEST_DATASETS[0]; fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=False, sharey=False)
for ax, cond in zip(axes, ["M0_none", "M4_gold"]):
    f = WORK / "runs" / f"E3_{cond}_s{SEEDS[0]}__{k0}_emb.npz"
    if not f.exists(): continue
    d = np.load(f, allow_pickle=True); Z, canon = d["Z"], d["canon"]; sel = np.random.default_rng(0).choice(len(Z), min(2500, len(Z)), replace=False)
    P = project2d(Z[sel]); cls = [c for c, _ in pd.Series(canon[sel]).value_counts().items()][:8]
    for ci, c in enumerate(cls):
        m = canon[sel] == c; ax.scatter(P[m, 0], P[m, 1], s=6, color=PALETTE[ci], label=c, alpha=0.8, linewidths=0)
    ax.set_title(f"{COND_LABEL[cond]} — {DATASET_REGISTRY[k0]['name'].split(' (')[0]}"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
axes[-1].legend(markerscale=2.5, loc="center left", bbox_to_anchor=(1.0, 0.5)); fig.suptitle("Window embeddings of a held-out data set, coloured by activity", y=1.02)
fig.tight_layout(); savefig(fig, "F6_embeddings"); plt.show()

# ---- Figure 7: label alignment accuracy ----
fig, ax = plt.subplots(figsize=(5, 3.2)); meth = [m for m in ["wordnet", "embedding", "llm_constrained"] if m in set(E5.method)]
vals = [E5[E5.method == m].accuracy.mean() for m in meth]; sds = [E5[E5.method == m].accuracy.std() for m in meth]
ax.bar(range(len(meth)), vals, 0.6, color=PALETTE[:len(meth)], yerr=sds, capsize=3, error_kw=dict(lw=0.8, color="#52514e"))
for i, v in enumerate(vals): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(len(meth))); ax.set_xticklabels(["WordNet synonyms", "Sentence embedding", "Ontology-constrained LLM"][:len(meth)]); ax.set_ylim(0, 1.05)
ax.set_ylabel("mapping accuracy vs gold"); ax.set_title("E5  Raw label → canonical class alignment"); fig.tight_layout(); savefig(fig, "F7_label_alignment"); plt.show()

# ---- Figure 8: verification repair (before/after) ----
fig, ax = plt.subplots(figsize=(5.2, 3.2)); xs = np.arange(len(E2_REPAIR))
ax.bar(xs - 0.18, E2_REPAIR.before, 0.36, color=PALETTE[1], label="extracted"); ax.bar(xs + 0.18, E2_REPAIR.after, 0.36, color=PALETTE[2], label="extracted + verified")
ax.set_xticks(xs); ax.set_xticklabels([f"{m}\n{v}" for m, v in zip(E2_REPAIR.model, E2_REPAIR.variant)], fontsize=7); ax.set_ylim(0, 1.05)
ax.set_ylabel("accuracy on verifiable fields"); ax.set_title("E2  Data-grounded verification repairs extraction errors"); ax.legend(loc="lower right")
fig.tight_layout(); savefig(fig, "F8_verification_repair"); plt.show()

# %% [markdown]
# ## 10. Results summary (`summary.json`, `RESULTS.md`) and export

# %%
def g(cond, col="zeroshot_acc"): return float(E3[E3.condition == cond][col].mean())
gain_gold = g("M4_gold") - g("M0_none"); gain_ver = g("M3_verified") - g("M0_none"); gain_ext = g("M2_extracted") - g("M0_none")
summary = dict(
    mode=MODE, datasets_train=TRAIN_DATASETS, datasets_test=TEST_DATASETS, n_windows_total=int(E3.groupby("dataset").n_windows.first().sum()),
    E1_best=T2.sort_values("composite", ascending=False).iloc[0][["model", "variant", "composite"]].to_dict(),
    E1_hardest_fields=T2[RECORD_FIELDS].mean().sort_values().index[:3].tolist(),
    E2_datasets_with_contradiction=int(n_contra_ds), E2_contradiction_rate=float((verifiable.status == "contradicted").mean()),
    E2_repair_gain_mean=float(E2_REPAIR.gain.mean()),
    E3_zeroshot={c: g(c) for c in CONDITIONS}, E3_gain_gold_over_none=gain_gold, E3_gain_verified_over_none=gain_ver, E3_gain_extracted_over_none=gain_ext,
    E3_recovery_verified=float(gain_ver / gain_gold) if gain_gold > 0.01 else None, E3_recovery_extracted=float(gain_ext / gain_gold) if gain_gold > 0.01 else None,
    E4_cost_at_100pct={k: float(T7[(T7.defect == k) & (T7.severity == max(DEFECT_SEVERITIES))]["mean"].mean()) for k in DEFECTS if k in set(T7.defect)},
    E5_alignment={m: float(E5[E5.method == m].accuracy.mean()) for m in set(E5.method)},
    seeds=SEEDS, epochs=EPOCHS, target_hz=TARGET_HZ, window_sec=WINDOW_SEC, text_encoder=TEXT_ENCODER, device=DEVICE)
jdump(summary, WORK / "results" / "summary.json")
pct = lambda x: f"{100*x:.1f}"
md = f"""# M2M results ({MODE} mode, {time.strftime('%Y-%m-%d')})

**Data.** {len(TRAIN_DATASETS)} training data sets ({', '.join(TRAIN_DATASETS)}) and {len(TEST_DATASETS)} held-out ({', '.join(TEST_DATASETS)}); {summary['n_windows_total']:,} evaluation windows of {WINDOW_SEC:.0f} s at {TARGET_HZ} Hz.

**E1 Extraction.** Best system: {summary['E1_best']['model']} / {summary['E1_best']['variant']} with composite field accuracy {summary['E1_best']['composite']:.3f}. Hardest fields: {', '.join(summary['E1_hardest_fields'])}.

**E2 Audit.** {n_contra_ds} of {len(ALL_DATASETS)} data sets have at least one documentation–data contradiction in the gold records; {pct(summary['E2_contradiction_rate'])}% of verifiable fields are contradicted. Verification raised extracted-record accuracy on verifiable fields by {pct(summary['E2_repair_gain_mean'])} points on average.

**E3 Metadata conditions (zero-shot accuracy, mean over held-out data sets and seeds).** """ + "; ".join(f"{COND_LABEL[c]} {pct(g(c))}%" for c in CONDITIONS) + f""".
Gold metadata adds {pct(gain_gold)} points over no metadata; verified automatic metadata adds {pct(gain_ver)} points and unverified extraction {pct(gain_ext)} points""" + (f" (recovery of the gold gain: {pct(summary['E3_recovery_verified'])}% verified, {pct(summary['E3_recovery_extracted'])}% unverified)." if summary['E3_recovery_verified'] is not None else " (gold gain too small to express as a recovery ratio).") + f"""

**E4 Defect costs at {int(max(DEFECT_SEVERITIES)*100)}% severity (points of zero-shot accuracy).** """ + "; ".join(f"{k}: {pct(v)}" for k, v in summary['E4_cost_at_100pct'].items()) + f""".

**E5 Label alignment accuracy.** """ + "; ".join(f"{m}: {pct(v)}%" for m, v in summary['E5_alignment'].items()) + """.

Figures: F1 audit matrix, F2 extraction heatmap, F3 conditions, F4 defect costs, F5 few-shot, F6 embeddings, F7 label alignment, F8 verification repair (PDF + PNG in `figures/`). Tables T1–T8 in `tables/` (CSV, Markdown, LaTeX).
"""
(WORK / "results" / "RESULTS.md").write_text(md); print(md)

# %%
# Zip everything the paper needs (figures, tables, results, audit, gold records, manifest)
zip_path = WORK / f"M2M_paper_assets_{MODE}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ["figures", "tables", "results", "audit", "gold", "extracted"]:
        for f in (WORK / sub).glob("*"):
            if f.is_file(): zf.write(f, f"{sub}/{f.name}")
    if (WORK / "runs" / "manifest.jsonl").exists(): zf.write(WORK / "runs" / "manifest.jsonl", "runs/manifest.jsonl")
print("Paper assets:", zip_path)
if IN_COLAB:
    from google.colab import files; files.download(str(zip_path))

# %% [markdown]
# ## 11. Outputs
#
# The cell above writes `M2M_paper_assets_<MODE>.zip`, containing every figure (PDF + PNG), every table (CSV /
# Markdown / LaTeX), `results/summary.json`, `results/RESULTS.md`, the audit files, the gold and extracted records
# and the run manifest. Unzip it into `results/notebook_outputs/` and run the scripts in `analysis/` to regenerate
# the paper's statistics, tables and figures (see `docs/reproducibility.md`).
#
# **Settings used for the paper:** `TEXT_ENCODER = "sentence-transformers/all-mpnet-base-v2"`, `SEEDS = [0, 1, 2]`,
# `EPOCHS = 25`, `MAX_WINDOWS_PER_DATASET = 20000`, all five LLMs in `LLM_MODELS`. To report inter-annotator
# agreement, have a second annotator write their own `gold/*.json` with the same schema.
