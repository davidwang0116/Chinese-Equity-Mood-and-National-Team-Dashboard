"""合成数据：模拟 fetch_all.py 的输出格式，使整条流水线可离线测试。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _walk(rng, n, drift, vol, start=100.0, shock=None):
    r = rng.normal(drift, vol, n)
    if shock is not None:
        for i, v in shock.items():
            r[i] += v
    return start * np.exp(np.cumsum(r))


def make_data_dir(tmp: Path, n=1400, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n).strftime("%Y-%m-%d")
    crash = {i: -0.03 for i in range(700, 740)}          # 模拟一次急跌
    rebound = {i: 0.02 for i in range(741, 760)}
    close = _walk(rng, n, 0.0003, 0.012, 4000, {**crash, **rebound})
    small = _walk(rng, n, 0.0002, 0.016, 6000, {i: -0.045 for i in range(700, 740)})

    def ohlc(c):
        prev = np.r_[c[0], c[:-1]]
        low = np.minimum(prev, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
        high = np.maximum(prev, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
        return pd.DataFrame({"date": dates, "open": prev, "high": high, "low": low, "close": c,
                             "volume": rng.uniform(1e8, 2e8, n), "amount": rng.uniform(3e11, 5e11, n)})

    tmp.mkdir(parents=True, exist_ok=True)
    ohlc(close).to_csv(tmp / "index_csi300.csv", index=False)
    ohlc(small).to_csv(tmp / "index_csi1000.csv", index=False)
    ohlc(close * 0.8).to_csv(tmp / "index_sse.csv", index=False)
    ohlc(close * 2.5).to_csv(tmp / "index_szse.csv", index=False)
    pd.DataFrame({"date": dates, "close": close / 1000}).to_csv(tmp / "asset_510300_hfq.csv", index=False)

    vol = pd.Series(np.log(close)).diff().rolling(20).std().bfill().to_numpy() * np.sqrt(252) * 100
    pd.DataFrame({"date": dates, "qvix": vol + 5}).to_csv(tmp / "qvix.csv", index=False)
    fin = 1.5e12 * close / close[0]
    pd.DataFrame({"date": dates, "fin_balance": fin, "fin_buy": fin * 0.05}).to_csv(tmp / "margin.csv", index=False)
    pd.DataFrame({"date": dates, "erp": 5 - close / close.max() * 2, "erp_ma": 4.0}).to_csv(tmp / "erp.csv", index=False)

    codes = [f"{600000 + i:06d}" for i in range(60)]
    panel = pd.DataFrame({c: close * rng.uniform(0.5, 2) * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
                          for c in codes}, index=pd.Index(dates, name="date"))
    panel.to_csv(tmp / "breadth_closes.csv.gz", compression="gzip")

    etf_rows, share_rows = [], []
    for code in ["510300", "510050"]:
        px = close / 1000
        amt = rng.uniform(2e9, 4e9, n)
        amt[700:740] *= 4                                     # 救市放量
        shares = 1e10 + np.cumsum(rng.normal(0, 2e7, n))
        shares[700:740] += np.cumsum(np.full(40, 5e8))         # 份额暴增
        shares[740:] += 40 * 5e8
        etf_rows.append(pd.DataFrame({"date": dates, "code": code, "open": px, "high": px, "low": px,
                                      "close": px, "volume": amt / px, "amount": amt}))
        share_rows.append(pd.DataFrame({"date": dates[400:], "code": code,
                                        "shares": shares[400:], "source": "sse"}))
    pd.concat(etf_rows).to_csv(tmp / "etf_daily.csv", index=False)
    pd.concat(share_rows).to_csv(tmp / "etf_shares.csv", index=False)
    return tmp


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return make_data_dir(tmp_path_factory.mktemp("data"))
