"""
救市资金 / 超大资金识别（纯计算，无网络 IO）。
"National team" (Central Huijin / CSF) and very-large-money detection.

公开可得的免费数据无法直接看到汇金/证金的交易，只能从“足迹”推断。四个足迹：

1. etf_flow  宽基 ETF 份额净申购（份额变化 × 收盘价）的 z 值。
   汇金主要通过申购沪深300/上证50/中证500/1000/科创50 等宽基 ETF 入场，
   份额暴增是最直接的证据。上交所可按日回填（2023 起），深市只能逐日累积。
2. etf_amount 宽基 ETF 成交额占两市成交额比例，相对 60 日中位数的倍数。
   历史最长（2012 起），可覆盖 2015 救市、2016 熔断、2018、2024-02、2024-10、2025-04。
3. intraday_reversal 沪深300 盘中深跌后收在日内高位（“尾盘拉升 / 护盘”）。
4. large_small_divergence 小盘（中证1000）大跌而沪深300 明显抗跌（“托权重”）。

2-4 仅在“市场承压”时计分（回撤较深 / 近5日大跌 / 当日大跌），避免把牛市放量当成救市。
"""
import numpy as np
import pandas as pd

from scoring import interp_series


def stress_mask(ind):
    """市场承压：回撤 ≥ 8%、近5日跌 ≥ 4%，或当日跌 ≥ 2%。"""
    return (ind["dd"] <= -8.0) | (ind["ret_5d"] <= -4.0) | (ind["ret_1d"] <= -2.0)


def etf_flow_series(etf_daily, etf_shares, index):
    """宽基 ETF 篮子每日净申购金额（元）。仅使用相邻交易日均有份额的数据。"""
    if etf_daily is None or etf_shares is None or not len(etf_shares):
        return pd.Series(np.nan, index=index), pd.Series(0, index=index)
    shares = (etf_shares.assign(date=pd.to_datetime(etf_shares["date"]).dt.strftime("%Y-%m-%d"))
              .pivot_table(index="date", columns="code", values="shares", aggfunc="last")
              .reindex(index))
    closes = (etf_daily.assign(date=pd.to_datetime(etf_daily["date"]).dt.strftime("%Y-%m-%d"))
              .pivot_table(index="date", columns="code", values="close", aggfunc="last")
              .reindex(index))
    closes = closes.reindex(columns=shares.columns)
    dshares = shares.diff()  # NaN when either adjacent day missing
    flow = dshares * closes
    n = flow.notna().sum(axis=1)
    total = flow.sum(axis=1, min_count=1)
    return total, n


def etf_amount_ratio(etf_daily, market_amount, index, baseline_days):
    """宽基 ETF 成交额 / 两市成交额，相对过去 N 日中位数的倍数（基线不含当日）。"""
    if etf_daily is None or market_amount is None:
        return pd.Series(np.nan, index=index)
    amt = (etf_daily.assign(date=pd.to_datetime(etf_daily["date"]).dt.strftime("%Y-%m-%d"))
           .groupby("date")["amount"].sum().reindex(index))
    share = amt / market_amount.replace(0.0, np.nan)
    base = share.shift(1).rolling(baseline_days, min_periods=20).median()
    return share / base


def intraday_reversal(csi300):
    """盘中较前收最深跌 ≥1.5% 时，收盘在日内区间的位置（0=最低，1=最高）；否则 0。"""
    prev = csi300["close"].shift(1)
    dip = (csi300["low"] / prev - 1.0) * 100.0
    rng = (csi300["high"] - csi300["low"]).replace(0.0, np.nan)
    clv = ((csi300["close"] - csi300["low"]) / rng).clip(0.0, 1.0)
    return clv.where(dip <= -1.5, 0.0).fillna(0.0)


def large_small_divergence(csi300, csi1000, index):
    """中证1000 当日跌 ≥1.5% 时，沪深300 相对抗跌的百分点；否则 0。"""
    if csi1000 is None:
        return pd.Series(np.nan, index=index)
    r300 = csi300["close"].pct_change() * 100.0
    r1000 = csi1000["close"].reindex(index).pct_change() * 100.0
    div = (r300 - r1000).clip(lower=0.0)
    return div.where(r1000 <= -1.5, 0.0).reindex(index)


def build_national_team(ind, data, config):
    """Return DataFrame with raw footprints, sub-scores and intensity (0-100)."""
    nt = config["national_team"]
    a = nt["anchors"]
    index = ind.index
    csi300 = data["csi300"].reindex(index)
    stress = stress_mask(ind)

    flow, n_codes = etf_flow_series(data.get("etf_daily"), data.get("etf_shares"), index)
    fstd = flow.rolling(int(nt["flow_baseline_days"]), min_periods=20).std().shift(1)
    flow_z = (flow / fstd.replace(0.0, np.nan)).where(flow.notna())

    amount_x = etf_amount_ratio(data.get("etf_daily"), ind.get("market_amount"),
                                index, int(nt["amount_baseline_days"]))
    reversal = intraday_reversal(csi300)
    divergence = large_small_divergence(csi300, data.get("csi1000"), index)

    out = pd.DataFrame(index=index)
    out["stress"] = stress
    out["etf_flow"] = flow
    out["etf_flow_codes"] = n_codes
    out["etf_flow_z"] = flow_z
    out["etf_amount_x"] = amount_x
    out["reversal"] = reversal
    out["divergence"] = divergence

    # 份额净申购是直接证据，不需要承压门控；其余三项只在承压日计分
    out["s_etf_flow"] = interp_series(flow_z.clip(lower=0.0), a["etf_flow_z"])
    out["s_etf_amount"] = interp_series(amount_x, a["etf_amount_x"]).where(stress, 0.0).where(amount_x.notna())
    out["s_intraday_reversal"] = interp_series(reversal, a["reversal"]).where(stress, 0.0)
    out["s_large_small_divergence"] = (interp_series(divergence, a["divergence_pct"])
                                       .where(stress, 0.0).where(divergence.notna()))

    w = nt["weights"]
    cols = {"etf_flow": "s_etf_flow", "etf_amount": "s_etf_amount",
            "intraday_reversal": "s_intraday_reversal",
            "large_small_divergence": "s_large_small_divergence"}
    num = pd.Series(0.0, index=index)
    den = pd.Series(0.0, index=index)
    for key, col in cols.items():
        s = out[col]
        num = num + s.fillna(0.0) * float(w[key])
        den = den + s.notna() * float(w[key])
    out["intensity"] = (num / den.replace(0.0, np.nan))
    # 缺少份额数据时，其余足迹权重被放大；标记一下口径
    out["has_flow"] = out["s_etf_flow"].notna()
    thr = float(nt["active_threshold"])
    out["active"] = out["intensity"] >= thr
    out["active_5d"] = out["active"].rolling(5, min_periods=1).max().astype(bool)
    return out


def apply_overlay(composite, base_mult, nt_frame, config, mode=None):
    """
    把救市资金信号并入定投倍数。三种角色：
      observe  仅观察，不影响倍数
      confirm  验证：近5日出现救市足迹且综合分已偏恐慌时，倍数上调一档
      parallel 并列：救市强度作为一个分项按权重并入综合分
    Returns (composite_adj, multiplier_adj).
    """
    nt = config["national_team"]
    mode = mode or nt["mode"]
    if mode == "observe" or nt_frame is None:
        return composite, base_mult
    if mode == "confirm":
        boost = nt_frame["active_5d"].reindex(composite.index).fillna(False) & (
            composite >= float(nt["confirm_min_composite"]))
        mult = (base_mult + boost * float(nt["confirm_step"])).clip(upper=float(nt["confirm_cap"]))
        return composite, mult.where(boost, base_mult)
    if mode == "parallel":
        w = float(nt["parallel_weight"])
        inten = nt_frame["intensity"].reindex(composite.index)
        adj = composite.where(inten.isna(), (1.0 - w) * composite + w * inten)
        return adj, None
    raise ValueError(f"unknown national_team.mode: {mode}")
