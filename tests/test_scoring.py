import numpy as np
import pandas as pd
import pytest

from indicators import boll_z, breadth_above_ma, drawdown, rsi
from national_team import apply_overlay
from scoring import interp, interp_series, load_config, multiplier_of, weighted_mean

CFG = load_config()


def test_interp_clamps_and_midpoints():
    a = CFG["anchors"]["dd"]
    assert interp(0, a) == 40.0
    assert interp(5, a) == 40.0
    assert interp(-50, a) == 100.0
    assert interp(-15, a) == pytest.approx(72.0)


def test_interp_series_matches_scalar():
    s = pd.Series([-3.0, np.nan, -25.0])
    out = interp_series(s, CFG["anchors"]["dd"])
    assert out[0] == pytest.approx(interp(-3.0, CFG["anchors"]["dd"]))
    assert np.isnan(out[1])


def test_fear_anchors_direction():
    a = CFG["anchors"]
    assert interp(35, a["qvix"]) > interp(15, a["qvix"])          # 高波动 = 恐慌
    assert interp(15, a["breadth"]) > interp(70, a["breadth"])    # 参与度低 = 恐慌
    assert interp(25, a["rsi"]) > interp(75, a["rsi"])            # 超卖 = 恐慌
    assert interp(-2.2, a["boll_z"]) > interp(2.2, a["boll_z"])
    assert interp(0.6, a["turnover"]) > interp(2.0, a["turnover"])  # 地量 = 恐慌
    assert interp(-6, a["margin"]) > interp(6, a["margin"])        # 去杠杆 = 恐慌


def test_weighted_mean_renormalises_missing():
    f = pd.DataFrame({"a": [100.0, np.nan], "b": [0.0, 50.0]})
    out = weighted_mean(f, {"a": 0.5, "b": 0.5})
    assert out.tolist() == [50.0, 50.0]


def test_multiplier_bands():
    assert multiplier_of(59.9, CFG) == 1.0
    assert multiplier_of(60.0, CFG) == 1.5
    assert multiplier_of(80.0, CFG) == 2.0
    assert multiplier_of(100.0, CFG) == 2.0


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(1, 2, 60))
    assert rsi(up, 14).iloc[-1] == 100.0
    noisy = pd.Series(100 + np.random.default_rng(1).normal(0, 1, 300).cumsum())
    r = rsi(noisy, 14).dropna()
    assert r.between(0, 100).all()


def test_boll_z_sign():
    s = pd.Series([10.0] * 19 + [9.0, 8.0])
    assert boll_z(s, 20).iloc[-1] < -2


def test_drawdown_nonpositive():
    s = pd.Series([1, 2, 3, 1.5, 3.3])
    dd = drawdown(s, 252)
    assert (dd <= 0).all()
    assert dd.iloc[3] == pytest.approx(-50.0)


def test_breadth_above_ma():
    idx = pd.RangeIndex(10)
    panel = pd.DataFrame({f"s{i}": np.arange(10.0) * (1 if i < 60 else -1) + 100 for i in range(100)}, index=idx)
    b = breadth_above_ma(panel, 3, min_names=50)
    assert b.iloc[-1] == pytest.approx(60.0)
    assert np.isnan(b.iloc[0])


def test_overlay_modes():
    comp = pd.Series([50.0, 70.0, 85.0])
    base = comp.map(lambda s: multiplier_of(s, CFG))
    nt = pd.DataFrame({"active_5d": [True, True, False], "intensity": [90.0, 90.0, 10.0]})
    c, m = apply_overlay(comp, base, nt, CFG, "observe")
    assert m.tolist() == [1.0, 1.5, 2.0]
    c, m = apply_overlay(comp, base, nt, CFG, "confirm")
    assert m.tolist() == [1.0, 2.0, 2.0]      # 50<55 不加；70 → 1.5+0.5；85 无足迹
    c, m = apply_overlay(comp, base, nt, CFG, "parallel")
    assert m is None
    assert c[0] == pytest.approx(0.85 * 50 + 0.15 * 90)
