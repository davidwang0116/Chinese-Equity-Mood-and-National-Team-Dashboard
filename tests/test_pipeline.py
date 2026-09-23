"""端到端：合成数据 → 指标 → 评分 → 救市资金 → 回测 → 渲染。"""
import json

import numpy as np

from backtest import run_backtest, simulate
from indicators import load_data
from model import run_model
from render import render_dashboard
from scoring import load_config


def test_model_runs_and_scores_in_range(data_dir):
    frame = run_model(data=load_data(data_dir))
    comp = frame["composite_final"].dropna()
    assert len(comp) > 500
    assert comp.between(0, 100).all()
    assert frame["multiplier_final"].dropna().isin([1.0, 1.5, 2.0]).all()
    for col in ["qvix", "breadth", "rsi", "boll_z", "turnover", "margin", "dd"]:
        assert frame[col].notna().sum() > 500, col


def test_national_team_detects_synthetic_rescue(data_dir):
    frame = run_model(data=load_data(data_dir))
    crash = frame.iloc[705:740]
    calm = frame.iloc[500:650]
    assert crash["nt_intensity"].mean() > calm["nt_intensity"].mean() + 20
    assert crash["nt_active"].any()
    assert (crash["nt_etf_flow"] > 0).mean() > 0.8


def test_simulate_external_capital():
    import pandas as pd
    px = pd.Series([1.0, 1.0, 1.0], index=["2020-01-01", "2020-01-02", "2020-01-03"])
    r = simulate(px, pd.Series([0.5, 0.5, 2.0], index=px.index))
    assert abs(r["excess_irr_pct"]) < 1e-6          # 价格不变，无超额
    assert r["avg_cost_vs_plain_pct"] == 0.0
    falling = pd.Series([2.0, 1.5, 1.0], index=px.index)
    r = simulate(falling, pd.Series([1.0, 1.0, 2.0], index=px.index))
    assert r["avg_cost_vs_plain_pct"] < 0            # 低位多买 → 成本更低


def test_backtest_and_render(data_dir, tmp_path):
    data = load_data(data_dir)
    summary = run_backtest(data=data, reports_dir=tmp_path)
    assert (tmp_path / "backtest_report.md").exists()
    assert summary["national_team"]["verdict"]["recommended_mode"] in {"observe", "confirm", "parallel"}
    assert np.isfinite(summary["current_config"]["full"]["excess_irr_pct"])
    json.dumps(summary, default=str)

    frame = run_model(data=data)
    html = render_dashboard(frame, load_config(), {"indexes": {"ok": True, "info": {"last": "x"}}}, summary)
    assert "<svg" in html and "救市资金" in html
