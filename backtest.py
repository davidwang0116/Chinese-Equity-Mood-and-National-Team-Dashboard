"""
回测与权重探索（只读 data/，无网络）。

输出:
  reports/backtest_summary.json  机器可读
  reports/backtest_report.md     人读报告（中文）

内容:
  1. 各子指标有效性：与未来 20/60/120 日收益的秩相关（IC）及五分位收益
  2. 情绪定投模拟（与纳指版 grid 一致：倍数由外部资金补足，XIRR + 平均成本对比）
  3. 权重网格：前半段选参（样本内），后半段检验（样本外），防止过拟合
  4. 救市资金事件研究 + 三种角色（观察 / 验证 / 并列）对比，给出结论
"""
import datetime
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from indicators import build_indicators, load_data
from national_team import apply_overlay, build_national_team
from scoring import FEAR_KEYS, load_config, multiplier_of, score_frame, sub_scores

BASE = Path(__file__).parent
REPORTS = BASE / "reports"
HORIZONS = [20, 60, 120]


# ── stats helpers ────────────────────────────────────────────────────────────

def rank_ic(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 60:
        return math.nan
    return float(np.corrcoef(x[m].rank(), y[m].rank())[0, 1])


def quintile_returns(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 100:
        return None
    q = pd.qcut(x[m].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    return {int(k): round(float(v) * 100, 2) for k, v in y[m].groupby(q, observed=True).mean().items()}


def xirr_equal(dates, final_equity, contributions):
    t = pd.to_datetime(pd.Index(dates))
    years = (t - t[0]).days.to_numpy(dtype=float) / 365.25
    c = np.asarray(contributions, dtype=float)

    def npv(r):
        disc = np.power(1.0 + r, years)
        return float(-(c / disc).sum() + final_equity / disc[-1])

    lo, hi = -0.99, 5.0
    if npv(lo) * npv(hi) > 0:
        return math.nan
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def simulate(price, mult):
    """
    与纳指版 grid 回测一致：每个倍数由外部资金补足并当日买入（无现金池约束）。
    普通定投每日买 1 单位；情绪定投每日买 mult 单位。用各自实际现金流算 XIRR，
    并比较平均持仓成本（越低越好）。
    price: asset close; mult: multiplier already shifted (actionable today).
    """
    p = price.to_numpy(dtype=float)
    m = mult.to_numpy(dtype=float)
    n = len(p)
    plain_c = np.ones(n)
    plain_sh = np.cumsum(plain_c / p)
    mood_sh = np.cumsum(m / p)
    plain_eq = plain_sh * p
    mood_eq = mood_sh * p
    dates = price.index
    plain_irr = xirr_equal(dates, plain_eq[-1], plain_c)
    mood_irr = xirr_equal(dates, mood_eq[-1], m)
    plain_avg_cost = plain_c.sum() / plain_sh[-1]
    mood_avg_cost = m.sum() / mood_sh[-1]
    return {
        "start": dates[0], "end": dates[-1], "days": n,
        "plain_irr_pct": round(plain_irr * 100, 2),
        "mood_irr_pct": round(mood_irr * 100, 2),
        "excess_irr_pct": round((mood_irr - plain_irr) * 100, 2),
        "avg_cost_vs_plain_pct": round((mood_avg_cost / plain_avg_cost - 1.0) * 100, 2),
        "capital_vs_plain": round(float(m.sum() / plain_c.sum()), 3),
        "mood_profit_pct": round(float(mood_eq[-1] / m.sum() - 1.0) * 100, 2),
        "plain_profit_pct": round(float(plain_eq[-1] / plain_c.sum() - 1.0) * 100, 2),
        "avg_multiplier": round(float(np.nanmean(m)), 3),
        "days_1_5x_plus": int((m >= 1.5).sum()),
        "days_2x": int((m >= 2.0).sum()),
    }


# ── grid ─────────────────────────────────────────────────────────────────────

def fear_profiles(config, ic_first_half):
    base = config["fear_weights"]
    profiles = {"config": dict(base), "equal": {k: 1.0 for k in FEAR_KEYS}}
    for k in FEAR_KEYS:
        w = {j: 0.6 / (len(FEAR_KEYS) - 1) for j in FEAR_KEYS}
        w[k] = 0.4
        profiles[f"heavy_{k}"] = w
    pos = {k: max(ic_first_half.get(k, 0.0) or 0.0, 0.0) for k in FEAR_KEYS}
    if sum(pos.values()) > 0:
        profiles["ic_weighted_first_half"] = pos
    return profiles


def run_backtest(data=None, reports_dir=REPORTS):
    config = load_config()
    data = data if data is not None else load_data()
    ind = build_indicators(data, config)
    nt = build_national_team(ind, data, config)

    if data.get("asset") is not None:
        asset = data["asset"]["close"].astype(float).reindex(ind.index).ffill(limit=5)
        asset_name = "510300 后复权（含分红）"
    else:
        asset = ind["close"]
        asset_name = "沪深300 价格指数（不含分红）"

    base_scores = score_frame(ind, config)
    fwd = {h: asset.shift(-h) / asset - 1.0 for h in HORIZONS}

    # 可回测区间：综合分与资产价格都有值
    valid = base_scores["composite"].notna() & asset.notna()
    dates = ind.index[valid]
    if len(dates) < 500:
        raise RuntimeError(f"可回测样本太少: {len(dates)} 天")
    mid = dates[len(dates) // 2]
    first = dates[dates < mid]
    second = dates[dates >= mid]

    # 1) 指标有效性
    factor_cols = {k: base_scores[f"s_{k}"] for k in FEAR_KEYS + ["dd", "erp"]}
    factor_cols["fear_axis"] = base_scores["fear"]
    factor_cols["composite"] = base_scores["composite"]
    factor_cols["nt_intensity"] = nt["intensity"]
    factor_cols["congestion_raw_inv"] = -ind["congestion"]
    ic, ic_first = {}, {}
    for name, s in factor_cols.items():
        ic[name] = {
            "coverage_start": s.dropna().index.min() if s.notna().any() else None,
            **{f"ic_{h}d": round(rank_ic(s, fwd[h]), 3) for h in HORIZONS},
            "quintile_fwd60_pct": quintile_returns(s, fwd[60]),
        }
        ic_first[name] = rank_ic(s.loc[first], fwd[60].loc[first])

    # 2+3) 网格
    profiles = fear_profiles(config, ic_first)
    axis_opts = [0.3, 0.4, 0.5, 0.6]
    modes = ["observe", "confirm", "parallel"]
    value_opts = config.get("value_axis_candidates") or {"config": config["value_axis"]}
    subs = sub_scores(ind, config)
    rows = []
    for (pname, fw), (vname, vw), af in itertools.product(profiles.items(), value_opts.items(), axis_opts):
        if True:
            sc = score_frame(ind, config, fear_weights=fw, value_weights=vw, subs=subs,
                             axis_weights={"fear": af, "value": 1.0 - af})
            for mode in modes:
                comp, mult = apply_overlay(sc["composite"], sc["multiplier"], nt, config, mode)
                if mult is None:
                    mult = comp.map(lambda s: multiplier_of(s, config))
                sig = mult.shift(1)  # 收盘信号次日执行
                res = {"profile": pname, "value_axis": vname, "fear_axis_weight": af, "nt_mode": mode}
                for label, ds in [("full", dates), ("first_half", first), ("second_half", second)]:
                    ds = ds[sig.loc[ds].notna().to_numpy()]
                    r = simulate(asset.loc[ds], sig.loc[ds])
                    res[f"{label}_excess_irr"] = r["excess_irr_pct"]
                    res[f"{label}_cost_vs_plain"] = r["avg_cost_vs_plain_pct"]
                    res[f"{label}_avg_mult"] = r["avg_multiplier"]
                rows.append(res)
    grid = pd.DataFrame(rows)
    grid["robust"] = grid[["first_half_excess_irr", "second_half_excess_irr"]].min(axis=1)
    best_is = grid.sort_values("first_half_excess_irr", ascending=False).head(10)
    best_robust = grid.sort_values("robust", ascending=False).head(10)

    # 当前配置的完整结果
    cur = score_frame(ind, config)
    comp, mult = apply_overlay(cur["composite"], cur["multiplier"], nt, config)
    if mult is None:
        mult = comp.map(lambda s: multiplier_of(s, config))
    sig = mult.shift(1)
    ds = dates[sig.loc[dates].notna().to_numpy()]
    current = simulate(asset.loc[ds], sig.loc[ds])
    last_5y = ds[ds >= (pd.Timestamp(ds[-1]) - pd.DateOffset(years=5)).strftime("%Y-%m-%d")]
    current_5y = simulate(asset.loc[last_5y], sig.loc[last_5y])

    # 4) 救市资金
    nt_study = national_team_study(ind, nt, asset, fwd, grid)

    summary = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "asset": asset_name,
        "sample": {"start": dates[0], "end": dates[-1], "days": len(dates),
                   "split_date": mid},
        "factor_ic": ic,
        "current_config": {"full": current, "last_5y": current_5y,
                           "multiplier_counts": {str(k): int(v) for k, v in
                                                 sig.loc[ds].value_counts().sort_index().items()}},
        "grid_top_in_sample": best_is.to_dict(orient="records"),
        "grid_top_robust": best_robust.to_dict(orient="records"),
        "grid_value_axis_summary": grid.groupby("value_axis")[
            ["first_half_excess_irr", "second_half_excess_irr", "full_excess_irr", "full_cost_vs_plain"]
        ].median().round(3).to_dict(orient="index"),
        "grid_mode_summary": grid.groupby("nt_mode")[
            ["first_half_excess_irr", "second_half_excess_irr", "full_excess_irr"]
        ].median().round(3).to_dict(orient="index"),
        "fear_profiles": profiles,
        "national_team": nt_study,
    }
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    grid.to_csv(reports_dir / "backtest_grid.csv", index=False)
    (reports_dir / "backtest_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def national_team_study(ind, nt, asset, fwd, grid):
    """事件研究：救市足迹出现后的表现，对比同样承压但无足迹的日子。"""
    active = nt["active"].fillna(False)
    prev = active.shift(1).rolling(10, min_periods=1).max().fillna(0).astype(bool)
    starts = active & ~prev
    stress = nt["stress"].fillna(False)
    fwd_min20 = (asset[::-1].rolling(20, min_periods=1).min()[::-1].shift(-1) / asset - 1.0)
    past_low = asset.rolling(21, min_periods=5).min()

    events = []
    for d in ind.index[starts.to_numpy()]:
        e = {"date": d, "intensity": round(float(nt.at[d, "intensity"]), 1),
             "dd": round(float(ind.at[d, "dd"]), 2),
             "has_flow": bool(nt.at[d, "has_flow"])}
        for h in HORIZONS:
            v = fwd[h].get(d)
            e[f"fwd_{h}d_pct"] = None if pd.isna(v) else round(float(v) * 100, 2)
        v = fwd_min20.get(d)
        e["further_drop_20d_pct"] = None if pd.isna(v) else round(float(v) * 100, 2)
        # 事件日距离近 20 日最低点多少天：>0 表示低点已出现（滞后确认）
        window = asset.loc[:d].tail(21)
        e["days_after_20d_low"] = int(len(window) - 1 - int(np.argmin(window.to_numpy())))
        events.append(e)
    ev = pd.DataFrame(events)

    def agg(mask):
        out = {"days": int(mask.sum())}
        for h in HORIZONS:
            r = fwd[h][mask].dropna()
            out[f"fwd_{h}d_mean_pct"] = round(float(r.mean()) * 100, 2) if len(r) else None
            out[f"fwd_{h}d_hit_rate"] = round(float((r > 0).mean()), 3) if len(r) else None
        return out

    post = pd.Series(ind.index >= "2020-01-01", index=ind.index)
    comparison = {
        "active_days": agg(active),
        "stress_days_without_footprint": agg(stress & ~active),
        "active_days_pre2020": agg(active & ~post),
        "stress_no_footprint_pre2020": agg(stress & ~active & ~post),
        "active_days_2020_on": agg(active & post),
        "stress_no_footprint_2020_on": agg(stress & ~active & post),
        "all_days": agg(pd.Series(True, index=ind.index)),
    }
    if len(ev):
        timing = {
            "median_days_after_20d_low": float(ev["days_after_20d_low"].median()),
            "share_low_already_in": round(float((ev["days_after_20d_low"] > 0).mean()), 3),
            "median_further_drop_20d_pct": float(ev["further_drop_20d_pct"].dropna().median())
            if ev["further_drop_20d_pct"].notna().any() else None,
        }
    else:
        timing = {}

    med = grid.groupby("nt_mode")[["first_half_excess_irr", "second_half_excess_irr"]].median()
    verdict = decide_role(comparison, timing, med)
    return {"events": events, "comparison": comparison, "timing": timing,
            "mode_median_excess": med.round(3).to_dict(orient="index"),
            "verdict": verdict}


def decide_role(comparison, timing, med):
    """根据证据给出救市资金指标的角色建议。"""
    notes = []
    a, b = comparison["active_days"], comparison["stress_days_without_footprint"]
    edge = None
    if a.get("fwd_60d_mean_pct") is not None and b.get("fwd_60d_mean_pct") is not None:
        edge = a["fwd_60d_mean_pct"] - b["fwd_60d_mean_pct"]
        notes.append(f"足迹日 vs 同样承压无足迹日 60日前瞻收益差 {edge:+.2f}pct（全样本）")
    for era, ka, kb in [("2020 前", "active_days_pre2020", "stress_no_footprint_pre2020"),
                        ("2020 起", "active_days_2020_on", "stress_no_footprint_2020_on")]:
        x, y = comparison.get(ka, {}), comparison.get(kb, {})
        if x.get("fwd_60d_mean_pct") is not None and y.get("fwd_60d_mean_pct") is not None:
            notes.append(f"{era}：足迹日 {x['days']} 天，60日收益 {x['fwd_60d_mean_pct']:+.2f}% "
                         f"vs 对照 {y['fwd_60d_mean_pct']:+.2f}%")
    lagging = timing.get("share_low_already_in", 0) >= 0.5
    if timing:
        notes.append(f"{timing.get('share_low_already_in', 0):.0%} 的事件发生在近20日低点之后"
                     f"（中位 {timing.get('median_days_after_20d_low')} 天）")
    recent = None
    x, y = comparison.get("active_days_2020_on", {}), comparison.get("stress_no_footprint_2020_on", {})
    if x.get("fwd_60d_mean_pct") is not None and y.get("fwd_60d_mean_pct") is not None:
        recent = x["fwd_60d_mean_pct"] - y["fwd_60d_mean_pct"]
    role = "observe"
    if "observe" in med.index:
        obs = med.loc["observe"]
        gains = {m: (med.loc[m] - obs).to_dict() for m in med.index if m != "observe"}
        for m, g in gains.items():
            notes.append(f"{m} 相对 observe 的超额IRR中位变化：样本内 {g['first_half_excess_irr']:+.3f}，"
                         f"样本外 {g['second_half_excess_irr']:+.3f}")
        par = gains.get("parallel")
        con = gains.get("confirm")
        # 并列：两段都显著改善才考虑
        if par and min(par["first_half_excess_irr"], par["second_half_excess_irr"]) > 0.02:
            role = "parallel"
        # 验证：近期（2020 起）足迹后收益明显好于对照，且验证模式在样本外不拖累
        elif con and recent is not None and recent > 2.0 and con["second_half_excess_irr"] >= 0:
            role = "confirm"
    if role == "confirm":
        notes.append("2020 年后足迹的前瞻收益显著好于同样承压的对照，而 2015-16 的早期救市失败；"
                     "当前制度下（汇金定位为类平准基金）作为“验证”上调一档，但事件数少，需持续跟踪")
    if lagging:
        notes.append("足迹多出现在低点附近或之后，更像“托底确认”而非领先信号")
    else:
        notes.append("足迹多出现在当日/近期低点当天，本身不预示后续不再下跌（2015-16 仍续跌约 17%）")
    role_cn = {"observe": "观察", "confirm": "验证", "parallel": "并列"}[role]
    return {"recommended_mode": role, "recommended_mode_cn": role_cn, "evidence": notes}


# ── report ───────────────────────────────────────────────────────────────────

def _tbl(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)


def render_report(s):
    cur = s["current_config"]
    lines = [
        "# 回测报告 · A股情绪定投 + 救市资金",
        f"\n生成时间 {s['generated_at']}｜标的 {s['asset']}｜样本 {s['sample']['start']} → "
        f"{s['sample']['end']}（{s['sample']['days']} 天，样本内/外分界 {s['sample']['split_date']}）\n",
        "## 1. 子指标有效性（分越高=越恐慌；IC>0 表示恐慌时未来收益更高）\n",
        _tbl([{"指标": k, **{c: v.get(c) for c in ["coverage_start", "ic_20d", "ic_60d", "ic_120d"]},
               "五分位60日收益%(1低→5高)": v.get("quintile_fwd60_pct")}
              for k, v in s["factor_ic"].items()],
             ["指标", "coverage_start", "ic_20d", "ic_60d", "ic_120d", "五分位60日收益%(1低→5高)"]),
        "\n## 2. 当前 config 定投表现\n",
        _tbl([{"区间": "全样本", **cur["full"]}, {"区间": "近5年", **cur["last_5y"]}],
             ["区间", "start", "end", "plain_irr_pct", "mood_irr_pct", "excess_irr_pct",
              "avg_cost_vs_plain_pct", "capital_vs_plain", "days_1_5x_plus", "days_2x"]),
        f"\n倍数分布：{cur['multiplier_counts']}\n",
        "\n## 3. 权重网格（前半段选参，后半段检验）\n",
        "### 稳健性排名（min(样本内, 样本外) 超额IRR）\n",
        _tbl(s["grid_top_robust"], ["profile", "value_axis", "fear_axis_weight", "nt_mode",
                                    "first_half_excess_irr", "second_half_excess_irr", "full_excess_irr",
                                    "full_cost_vs_plain", "full_avg_mult"]),
        "\n### 估值轴构成对比（各组合中位数）\n",
        _tbl([{"value_axis": k, **v} for k, v in s["grid_value_axis_summary"].items()],
             ["value_axis", "first_half_excess_irr", "second_half_excess_irr", "full_excess_irr",
              "full_cost_vs_plain"]),
        "\n### 样本内排名\n",
        _tbl(s["grid_top_in_sample"], ["profile", "value_axis", "fear_axis_weight", "nt_mode", "first_half_excess_irr",
                                       "second_half_excess_irr", "full_excess_irr"]),
        "\n## 4. 救市资金\n",
        f"**建议角色：{s['national_team']['verdict']['recommended_mode_cn']}"
        f"（{s['national_team']['verdict']['recommended_mode']}）**\n",
        "".join(f"- {n}\n" for n in s["national_team"]["verdict"]["evidence"]),
        "\n### 足迹日 vs 对照\n",
        _tbl([{"组": k, **v} for k, v in s["national_team"]["comparison"].items()],
             ["组", "days", "fwd_20d_mean_pct", "fwd_60d_mean_pct", "fwd_120d_mean_pct",
              "fwd_60d_hit_rate"]),
        "\n### 事件列表（每段首日）\n",
        _tbl(s["national_team"]["events"][-40:],
             ["date", "intensity", "dd", "has_flow", "days_after_20d_low", "further_drop_20d_pct",
              "fwd_20d_pct", "fwd_60d_pct", "fwd_120d_pct"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    out = run_backtest()
    print(json.dumps({k: out[k] for k in ["sample", "current_config"]}, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(out["national_team"]["verdict"], ensure_ascii=False, indent=2))
