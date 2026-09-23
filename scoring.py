"""
评分层：原始指标 → 0-100 子分 → 恐慌轴 / 估值轴 → 综合分 → 定投倍数（纯函数）。
分越高 = 市场越恐慌/越便宜 = 越建议加仓。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_FILE = Path(__file__).parent / "config.json"

FEAR_KEYS = ["qvix", "breadth", "rsi", "boll_z", "turnover", "margin"]


def load_config(path=CONFIG_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def interp(x, anchors):
    """Piecewise-linear interpolation across anchors; clamps beyond endpoints."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if x <= anchors[0][0]:
        return float(anchors[0][1])
    if x >= anchors[-1][0]:
        return float(anchors[-1][1])
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return float(anchors[-1][1])


def interp_series(series, anchors):
    xs = np.array([a[0] for a in anchors], dtype=float)
    ys = np.array([a[1] for a in anchors], dtype=float)
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.interp(vals, xs, ys)
    out[np.isnan(vals)] = np.nan
    return pd.Series(out, index=series.index)


def weighted_mean(frame, weights):
    """按权重平均；缺失的分项自动剔除并重新归一化。"""
    num = pd.Series(0.0, index=frame.index)
    den = pd.Series(0.0, index=frame.index)
    for key, w in weights.items():
        if key not in frame or float(w) == 0.0:
            continue
        s = frame[key]
        num = num + s.fillna(0.0) * float(w)
        den = den + s.notna() * float(w)
    return num / den.replace(0.0, np.nan)


def sub_scores(ind, config):
    a = config["anchors"]
    out = pd.DataFrame(index=ind.index)
    for key in FEAR_KEYS + ["dd", "erp"]:
        out[key] = interp_series(ind[key], a[key]) if key in ind else np.nan
    return out


def score_frame(ind, config, fear_weights=None, axis_weights=None):
    """Return DataFrame with sub-scores, axes, composite and base multiplier."""
    subs = sub_scores(ind, config)
    fw = fear_weights or config["fear_weights"]
    aw = axis_weights or config["axis_weights"]
    out = subs.add_prefix("s_")
    out["fear"] = weighted_mean(subs, fw)
    out["value"] = weighted_mean(subs, config["value_axis"])
    out["composite"] = weighted_mean(out[["fear", "value"]],
                                     {"fear": aw["fear"], "value": aw["value"]})
    out["multiplier"] = out["composite"].map(lambda s: multiplier_of(s, config))
    return out


def multiplier_of(score, config):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return np.nan
    bands = config["multiplier_bands"]
    for lo, hi, _label, mult in bands:
        if lo <= score < hi:
            return float(mult)
    return float(bands[-1][3])


def multiplier_label(score, config):
    bands = config["multiplier_bands"]
    for lo, hi, label, _mult in bands:
        if lo <= score < hi:
            return label
    return bands[-1][2]
