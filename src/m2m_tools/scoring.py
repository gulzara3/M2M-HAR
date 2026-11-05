"""Record normalisation and field-level scoring against gold records (paper Eq. 18).

Verbatim logic from the experiment notebook (Section 4.1).

Known property (disclosed in the paper, Section 6.1): ``score_field`` returns 0 when the
prediction is ``None``, even if the gold value is also ``None``. For the gyroscope unit this caps
the score at 7/9 because two datasets have no gyroscope.
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, List

import numpy as np

from .schema import RECORD_FIELDS

PLACE_SYNONYMS = {"hand": "wrist", "arm": "wrist", "forearm": "lower arm", "lower back": "back", "hip": "waist", "belt": "waist",
                  "pocket": "trouser pocket", "trousers": "trouser", "leg": "thigh", "shin": "calf", "torso": "chest"}
_STOP = ("the", "a", "an", "on", "of", "and", "sensor", "sensors", "mounted", "worn")


def norm_tokens(s) -> str:
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    s = re.sub(r"[^a-z0-9/^ ]", " ", s)
    return " ".join(PLACE_SYNONYMS.get(t, t) for t in s.split() if t not in _STOP)


def fuzzy_set_f1(pred, gold, thr: float = 0.75) -> float:
    """Set-F1 where items match by equality, containment or string-similarity ratio >= ``thr``."""
    pred = [norm_tokens(p) for p in (pred or []) if str(p).strip()]
    gold = [norm_tokens(g) for g in (gold or [])]
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    def match(a, b):
        return a == b or a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= thr

    tp_p = sum(any(match(p, g) for g in gold) for p in pred)
    tp_g = sum(any(match(g, p) for p in pred) for g in gold)
    prec, rec = tp_p / len(pred), tp_g / len(gold)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def score_field(f: str, pred, gold) -> float:
    if f == "sampling_rate_hz":
        gs = gold if isinstance(gold, list) else [gold]
        ps = pred if isinstance(pred, list) else [pred]
        ps = [p for p in ps if isinstance(p, (int, float))]
        if not ps:
            return 0.0
        return float(np.mean([any(abs(p - g) / max(g, 1e-6) <= 0.10 for g in gs) for p in ps]))
    if f == "n_participants":
        return float(pred == gold) if pred is not None else 0.0
    if f in ("devices", "sensor_placements", "modalities", "activities"):
        return fuzzy_set_f1(pred, gold)
    return float(pred == gold) if pred is not None else 0.0


def composite_score(pred_record: Dict, gold_record: Dict) -> float:
    """Mean of the ten field scores."""
    return float(np.mean([score_field(f, pred_record.get(f), gold_record.get(f)) for f in RECORD_FIELDS]))


def normalise_record(rec: Dict) -> Dict:
    """Normalise raw LLM JSON output: numeric strings, lists, and a fixed list of unit synonyms."""
    out = {}
    for f in RECORD_FIELDS:
        v = rec.get(f)
        if f == "sampling_rate_hz":
            if isinstance(v, str):
                nums = re.findall(r"\d+(?:\.\d+)?", v)
                v = float(nums[0]) if nums else None
            elif isinstance(v, list):
                v = [float(x) for x in v if isinstance(x, (int, float))] or None
        if f == "n_participants":
            if isinstance(v, str):
                nums = re.findall(r"\d+", v)
                v = int(nums[0]) if nums else None
            elif isinstance(v, float):
                v = int(v)
        if f in ("devices", "sensor_placements", "modalities", "activities"):
            if isinstance(v, str):
                v = [x.strip() for x in re.split(r"[,;\n]", v) if x.strip()]
            v = [str(x).strip().lower() for x in (v or [])]
        if f in ("accelerometer_unit", "gyroscope_unit", "protocol_type", "setting") and isinstance(v, str):
            v = v.strip().lower().replace("m/s2", "m/s^2").replace("m/s²", "m/s^2").replace("meters per second squared", "m/s^2")
            v = {"degrees per second": "deg/s", "°/s": "deg/s", "radians per second": "rad/s", "mg": "milli-g"}.get(v, v)
        out[f] = v
    return out
