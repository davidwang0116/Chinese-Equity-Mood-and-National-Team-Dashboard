"""
纯计算层：从 data/ 快照构建逐日原始指标（无网络 IO）。
Pure computation: builds the daily raw-indicator frame from data/ snapshots.

所有指标只使用当日及以前的数据（trailing），回测时信号再顺延一日执行。
All indicators are trailing-only; the backtest executes a close signal on the next day.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


# ── loading ──────────────────────────────────────────────────────────────────

def _read(path, **kw):
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, **kw)
    return df if len(df) else None


def _by_date(df, cols=None):
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return df[cols] if cols else df


def load_data(data_dir=DATA):
    d = Path(data_dir)
    out = {
        "csi300": _by_date(_read(d / "index_csi300.csv")),
        "csi1000": _by_date(_read(d / "index_csi1000.csv")),
        "sse": _by_date(_read(d / "index_sse.csv")),
        "szse": _by_date(_read(d / "index_szse.csv")),
        "asset": _by_date(_read(d / "asset_510300_hfq.csv")),
        "qvix": _by_date(_read(d / "qvix.csv")),
        "margin": _by_date(_read(d / "margin.csv")),
        "erp": _by_date(_read(d / "erp.csv")),
        "pe": _by_date(_read(d / "pe_csi300.csv")),
        "congestion": _by_date(_read(d / "congestion.csv")),
        "high_low": _by_date(_read(d / "legu_high_low_hs300.csv")),
        "market_flow": _by_date(_read(d / "market_fund_flow.csv")),
        "etf_daily": _read(d / "etf_daily.csv", dtype={"code": str}),
        "etf_shares": _read(d / "etf_shares.csv", dtype={"code": str}),
        "breadth_closes": None,
    }
    panel = d / "breadth_closes.csv.gz"
    if panel.exists():
        p = pd.read_csv(panel, index_col="date")
        p.index = pd.to_datetime(p.index).strftime("%Y-%m-%d")
        out["breadth_closes"] = p.sort_index()
    return out


# ── primitives ───────────────────────────────────────────────────────────────

def rsi(close, days):
    """Wilder RSI（与通达信 SMA(X,N,1) 口径一致）。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / days, adjust=False, min_periods=days).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / days, adjust=False, min_periods=days).mean()
    rs = gain / loss.replace(0.0, math.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((loss == 0.0) & (gain > 0.0), 100.0)
    return out.mask((loss == 0.0) & (gain == 0.0), 50.0)


def boll_z(close, days):
    """价格偏离布林中轨的标准差倍数；±2 即布林上下轨。"""
    ma = close.rolling(days, min_periods=days).mean()
    sd = close.rolling(days, min_periods=days).std(ddof=0)
    return (close - ma) / sd.replace(0.0, math.nan)


def drawdown(close, lookback):
    peak = close.rolling(lookback, min_periods=1).max()
    return ((close / peak - 1.0) * 100.0).clip(upper=0.0)


def breadth_above_ma(panel, ma_days, min_names=50):
    """成分股收盘价站上 N 日均线的比例（%）。"""
    ma = panel.rolling(ma_days, min_periods=ma_days).mean()
    valid = ma.notna() & panel.notna()
    above = (panel > ma) & valid
    n = valid.sum(axis=1)
    pct = above.sum(axis=1) / n.replace(0, np.nan) * 100.0
    return pct.where(n >= min_names)


# ── build ────────────────────────────────────────────────────────────────────

def build_indicators(data, config):
    """Return a DataFrame indexed by CSI300 trading dates with raw indicators."""
    p = config["indicators"]
    idx = data["csi300"]
    if idx is None:
        raise ValueError("缺少沪深300指数数据 data/index_csi300.csv")
    close = idx["close"].astype(float)
    df = pd.DataFrame(index=idx.index)
    df["close"] = close
    df["ret_1d"] = close.pct_change() * 100.0
    df["ret_5d"] = close.pct_change(5) * 100.0
    df["dd"] = drawdown(close, int(p["dd_lookback_days"]))
    df["rsi"] = rsi(close, int(p["rsi_days"]))
    df["boll_z"] = boll_z(close, int(p["boll_days"]))

    def align(series, limit=5):
        # 低频/延迟源向前填充，但最多 5 个交易日，避免数据源停更后用陈旧值
        return (series.reindex(df.index.union(series.index)).sort_index()
                .ffill(limit=limit).reindex(df.index))

    # 市场成交额热度：沪深两市（上证综指 + 深证成指口径）成交额短期 / 长期均值
    if data.get("sse") is not None and data.get("szse") is not None:
        amt = (data["sse"]["amount"].astype(float)
               .add(data["szse"]["amount"].astype(float), fill_value=0.0))
        amt = amt.reindex(df.index)
        df["market_amount"] = amt
        short = amt.rolling(int(p["turnover_short_days"]), min_periods=3).mean()
        long_ = amt.rolling(int(p["turnover_long_days"]), min_periods=120).mean()
        df["turnover"] = short / long_
    else:
        df["market_amount"] = np.nan
        df["turnover"] = np.nan

    if data.get("qvix") is not None:
        df["qvix"] = align(data["qvix"]["qvix"].astype(float))
    else:
        df["qvix"] = np.nan

    if data.get("breadth_closes") is not None:
        panel = data["breadth_closes"]
        b = breadth_above_ma(panel, int(p["breadth_ma_days"]))
        df["breadth"] = b.reindex(df.index)
        df["breadth_ma60"] = breadth_above_ma(panel, 60).reindex(df.index)
    else:
        df["breadth"] = np.nan
        df["breadth_ma60"] = np.nan

    if data.get("margin") is not None:
        fin = data["margin"]["fin_balance"].astype(float).dropna()
        chg = fin.pct_change(int(p["margin_change_days"])) * 100.0
        df["margin_balance"] = align(fin)
        df["margin"] = align(chg)
    else:
        df["margin_balance"] = np.nan
        df["margin"] = np.nan

    df["erp"] = align(data["erp"]["erp"].astype(float)) if data.get("erp") is not None else np.nan
    if data.get("pe") is not None and "ttmPe" in data["pe"].columns:
        df["pe_ttm"] = align(data["pe"]["ttmPe"].astype(float))
    else:
        df["pe_ttm"] = np.nan
    if data.get("congestion") is not None:
        df["congestion"] = align(data["congestion"]["congestion"].astype(float))
    else:
        df["congestion"] = np.nan
    if data.get("high_low") is not None:
        hl = data["high_low"]
        df["net_new_high_120"] = align((hl["high120"] - hl["low120"]).astype(float))
    else:
        df["net_new_high_120"] = np.nan

    return df
