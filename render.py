"""HTML 仪表盘渲染（自包含 SVG，无外部依赖）。"""
import html
import json
import math

import pandas as pd

CARD_DEFS = [
    # key, 名称, 说明, 原始值格式, 子分列
    ("dd", "沪深300 回撤", "距 252 日高点（估值轴）", "{:.2f}%", "s_dd"),
    ("erp", "股债利差", "沪深300 1/PE − 10年国债（估值轴）", "{:.2f}%", "s_erp"),
    ("qvix", "QVIX 恐慌指数", "300ETF 期权隐含波动率", "{:.2f}", "s_qvix"),
    ("breadth", "市场参与度", "沪深300成分股站上200日线占比", "{:.1f}%", "s_breadth"),
    ("rsi", "RSI(14)", "沪深300 超买超卖", "{:.1f}", "s_rsi"),
    ("boll_z", "布林带偏离", "距20日中轨标准差倍数", "{:+.2f}σ", "s_boll_z"),
    ("turnover", "成交热度", "两市成交额 5日/250日", "{:.2f}×", "s_turnover"),
    ("margin", "融资余额变化", "20日变化率", "{:+.2f}%", "s_margin"),
]

REF_DEFS = [
    ("pe_ttm", "沪深300 PE(TTM)", "乐咕乐股", "{:.2f}"),
    ("congestion", "大盘拥挤度", "乐咕乐股", "{:.3f}"),
    ("net_new_high_120", "120日新高−新低", "沪深300成分股家数", "{:+.0f}"),
    ("breadth_ma60", "站上60日线占比", "短期参与度", "{:.1f}%"),
]

NT_DEFS = [
    ("etf_flow", "宽基ETF净申购", "份额变化×价格", "s_etf_flow"),
    ("etf_amount_x", "ETF成交占比倍数", "承压日才计分", "s_etf_amount"),
    ("reversal", "盘中深跌后收高", "承压日才计分", "s_intraday_reversal"),
    ("divergence", "托权重抗跌", "300 相对 1000 超额(pct)", "s_large_small_divergence"),
]


def _fmt(fmt, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return fmt.format(v)


def _yi(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v / 1e8:+.1f} 亿"


def line_chart(dates, series, w=300, h=110, color="var(--c1)", overlay=None, bands=None,
               y_fmt="{:.0f}", zero=False):
    vals = [v for v in series if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return f'<div class="nodata" style="height:{h}px">暂无历史数据</div>'
    lo, hi = min(vals), max(vals)
    if bands:
        lo, hi = min(lo, min(bands)), max(hi, max(bands))
    if zero:
        lo, hi = min(lo, 0), max(hi, 0)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    ML, MR, MT, MB = 30, 6, 6, 16
    cw, ch = w - ML - MR, h - MT - MB
    n = len(series)

    def x(i):
        return ML + i / max(n - 1, 1) * cw

    def y(v):
        return MT + ch - (v - lo) / (hi - lo) * ch

    parts = []
    for b in bands or []:
        parts.append(f'<line x1="{ML}" x2="{w - MR}" y1="{y(b):.1f}" y2="{y(b):.1f}" class="grid"/>'
                     f'<text x="{ML - 3}" y="{y(b):.1f}" class="ax" text-anchor="end" '
                     f'dominant-baseline="middle">{y_fmt.format(b)}</text>')
    if zero:
        parts.append(f'<line x1="{ML}" x2="{w - MR}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>')
    if overlay:
        ov = [v for v in overlay if v is not None and not math.isnan(v)]
        if len(ov) > 1:
            olo, ohi = min(ov), max(ov)
            orng = (ohi - olo) or 1.0
            pts = " ".join(f"{x(i):.1f},{MT + ch - (v - olo) / orng * ch:.1f}"
                           for i, v in enumerate(overlay) if v is not None and not math.isnan(v))
            parts.append(f'<polyline points="{pts}" class="overlay"/>')
    segs, cur = [], []
    for i, v in enumerate(series):
        if v is None or math.isnan(v):
            if cur:
                segs.append(cur)
                cur = []
        else:
            cur.append(f"{x(i):.1f},{y(v):.1f}")
    if cur:
        segs.append(cur)
    for s in segs:
        parts.append(f'<polyline points="{" ".join(s)}" fill="none" stroke="{color}" '
                     f'stroke-width="1.8" stroke-linejoin="round"/>')
    last_i = max(i for i, v in enumerate(series) if v is not None and not math.isnan(v))
    parts.append(f'<circle cx="{x(last_i):.1f}" cy="{y(series[last_i]):.1f}" r="3.2" fill="{color}"/>')
    parts.append(f'<text x="{ML}" y="{h - 3}" class="ax">{dates[0][2:]}</text>'
                 f'<text x="{w - MR}" y="{h - 3}" class="ax" text-anchor="end">{dates[-1][2:]}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>')


def bar_chart(dates, values, w=620, h=120):
    vals = [0.0 if v is None or math.isnan(v) else v for v in values]
    if not any(vals):
        return f'<div class="nodata" style="height:{h}px">暂无 ETF 份额数据（上交所回填中 / 深市逐日累积）</div>'
    m = max(abs(v) for v in vals) or 1.0
    ML, MT, MB = 6, 6, 16
    cw, ch = w - 2 * ML, h - MT - MB
    mid = MT + ch / 2
    bw = cw / len(vals)
    parts = [f'<line x1="{ML}" x2="{w - ML}" y1="{mid}" y2="{mid}" class="zero"/>']
    for i, v in enumerate(vals):
        bh = abs(v) / m * (ch / 2)
        cls = "pos" if v > 0 else "neg"
        yv = mid - bh if v > 0 else mid
        parts.append(f'<rect x="{ML + i * bw + 0.5:.1f}" y="{yv:.1f}" width="{max(bw - 1, 1):.1f}" '
                     f'height="{bh:.1f}" class="{cls}"><title>{dates[i]} {v / 1e8:+.1f}亿</title></rect>')
    parts.append(f'<text x="{ML}" y="{h - 3}" class="ax">{dates[0][2:]}</text>'
                 f'<text x="{w - ML}" y="{h - 3}" class="ax" text-anchor="end">{dates[-1][2:]}</text>'
                 f'<text x="{w - ML}" y="{MT + 8}" class="ax" text-anchor="end">±{m / 1e8:.0f}亿</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def _series(frame, col, n):
    s = frame[col].tail(n) if col in frame else pd.Series(dtype=float)
    return list(s.index), [None if pd.isna(v) else float(v) for v in s.to_numpy()]


def _score_cls(s):
    if s is None or math.isnan(s):
        return "muted"
    return "hot" if s >= 80 else "warm" if s >= 60 else "cool" if s < 35 else "mid"


def render_dashboard(frame, config, fetch_report=None, backtest=None):
    last_date = frame["composite"].last_valid_index()
    row = frame.loc[last_date]
    comp = float(row["composite_final"])
    mult = float(row["multiplier_final"])
    from scoring import multiplier_label
    label = multiplier_label(comp, config)
    mode = config["national_team"]["mode"]
    mode_cn = {"observe": "观察", "confirm": "验证", "parallel": "并列"}[mode]
    nt_int = row.get("nt_intensity")
    nt_active = bool(row.get("nt_active_5d", False))

    # 指标卡片
    cards = []
    for key, name, desc, fmt, scol in CARD_DEFS:
        dates, vals = _series(frame, key, 130)
        score = row.get(scol)
        anchors = config["anchors"].get(key, [])
        bands = [a[0] for a in anchors][1:-1] if anchors else None
        cards.append(f"""
<div class="card">
  <div class="card-h"><span class="name">{name}</span><span class="tag {_score_cls(score)}">{_fmt('{:.0f}', score)} 分</span></div>
  <div class="raw">{_fmt(fmt, row.get(key))}</div>
  <div class="desc">{desc}</div>
  {line_chart(dates, vals, bands=bands, y_fmt='{:g}')}
</div>""")

    refs = "".join(
        f'<div class="ref"><div class="ref-n">{n}</div><div class="ref-v">{_fmt(fmt, row.get(k))}</div>'
        f'<div class="desc">{d}</div></div>' for k, n, d, fmt in REF_DEFS)

    # 救市资金
    nt_items = []
    for key, name, desc, scol in NT_DEFS:
        raw = row.get(f"nt_{key}")
        raw_s = _yi(raw) if key == "etf_flow" else _fmt("{:.2f}", raw)
        nt_items.append(f'<div class="ref"><div class="ref-n">{name}</div><div class="ref-v">{raw_s}</div>'
                        f'<div class="desc">{desc}｜子分 {_fmt("{:.0f}", row.get("nt_" + scol))}</div></div>')
    nd, nv = _series(frame, "nt_intensity", 250)
    fd, fv = _series(frame, "nt_etf_flow", 90)
    flow_note = "" if bool(row.get("nt_has_flow", False)) else \
        '<div class="note">当日无份额数据，强度由其余三项足迹按权重归一化得出。</div>'
    events = []
    if backtest:
        for e in backtest.get("national_team", {}).get("events", [])[-8:][::-1]:
            events.append(f"<tr><td>{e['date']}</td><td>{e['intensity']}</td><td>{e['dd']}%</td>"
                          f"<td>{_fmt('{:+.1f}%', e.get('fwd_20d_pct'))}</td>"
                          f"<td>{_fmt('{:+.1f}%', e.get('fwd_60d_pct'))}</td></tr>")
    verdict = ""
    if backtest:
        v = backtest.get("national_team", {}).get("verdict", {})
        verdict = (f'<div class="note">回测建议角色：<b>{v.get("recommended_mode_cn", "—")}</b>；'
                   + "；".join(html.escape(x) for x in v.get("evidence", [])) + "</div>")

    # 综合走势
    cd, cv = _series(frame, "composite_final", 250)
    _, px = _series(frame, "close", 250)
    fear_d, fear_v = _series(frame, "fear", 250)

    # 数据新鲜度
    fr_rows = ""
    for k, v in (fetch_report or {}).items():
        ok = v.get("ok")
        info = v.get("info", {})
        last = info.get("last", "") if isinstance(info, dict) else ""
        if not last and isinstance(info, dict):
            lasts = [v2.get("last") for v2 in info.values() if isinstance(v2, dict) and v2.get("last")]
            last = max(lasts) if lasts else ""
        fr_rows += (f'<tr><td>{k}</td><td class="{"ok" if ok else "bad"}">{"✓" if ok else "✗"}</td>'
                    f'<td>{html.escape(str(last))}</td><td>{html.escape(str(v.get("at", "")))}</td>'
                    f'<td class="err">{html.escape(str(v.get("error", ""))[:120])}</td></tr>')

    bt = ""
    if backtest:
        c = backtest["current_config"]["full"]
        c5 = backtest["current_config"]["last_5y"]
        bt = (f'<div class="note">回测（{c["start"]}→{c["end"]}，{html.escape(backtest["asset"])}）：'
              f'情绪定投 IRR {c["mood_irr_pct"]}% vs 普通定投 {c["plain_irr_pct"]}%（超额 {c["excess_irr_pct"]:+}pct，'
              f'平均成本 {c["avg_cost_vs_plain_pct"]:+}%，资金量 {c["capital_vs_plain"]}×）；近5年超额 {c5["excess_irr_pct"]:+}pct。'
              f'详见 reports/backtest_report.md</div>')

    fw = config["fear_weights"]
    aw = config["axis_weights"]
    weights_txt = " + ".join(f"{k} {v:.0%}" for k, v in fw.items())
    value_txt = " + ".join(f"{k} {v:.0%}" for k, v in config["value_axis"].items() if v)

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A股情绪与救市资金</title>
<style>
:root {{ --bg:#f7f7f5; --panel:#fff; --ink:#1f2328; --muted:#6b7280; --line:#e5e7eb;
  --c1:#b45309; --c2:#2563eb; --pos:#dc2626; --neg:#16a34a; --hot:#dc2626; --warm:#ea580c; --mid:#6b7280; --cool:#16a34a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111315; --panel:#1b1e22; --ink:#e6e6e6; --muted:#9aa0a6;
  --line:#2c3035; --c1:#f59e0b; --c2:#60a5fa; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif }}
.wrap {{ max-width:1180px; margin:0 auto; padding:20px 16px 40px }}
h1 {{ font-size:20px; margin:0 0 4px }} h2 {{ font-size:15px; margin:26px 0 10px }}
.sub {{ color:var(--muted); font-size:12px }}
.hero {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:14px }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px }}
.kpi .v {{ font-size:28px; font-weight:650 }} .kpi .l {{ color:var(--muted); font-size:12px }}
.grid4 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 12px 6px }}
.card-h {{ display:flex; justify-content:space-between; align-items:center }}
.name {{ font-weight:600 }} .raw {{ font-size:22px; font-weight:650; margin-top:4px }}
.desc {{ color:var(--muted); font-size:11px }}
.tag {{ font-size:11px; padding:1px 8px; border-radius:99px; color:#fff }}
.tag.hot {{ background:var(--hot) }} .tag.warm {{ background:var(--warm) }} .tag.mid {{ background:var(--mid) }}
.tag.cool {{ background:var(--cool) }} .tag.muted {{ background:var(--line); color:var(--muted) }}
.refs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px }}
.ref {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px }}
.ref-n {{ font-size:12px; color:var(--muted) }} .ref-v {{ font-size:18px; font-weight:600 }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px }}
.note {{ color:var(--muted); font-size:12px; margin-top:8px }}
svg .grid {{ stroke:var(--line); stroke-dasharray:4 3 }} svg .zero {{ stroke:var(--muted); stroke-width:.6 }}
svg .ax {{ fill:var(--muted); font-size:9px }} svg .overlay {{ fill:none; stroke:var(--c2); stroke-width:1.2; opacity:.55 }}
svg .pos {{ fill:var(--pos) }} svg .neg {{ fill:var(--neg) }}
.nodata {{ display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:11px }}
table {{ width:100%; border-collapse:collapse; font-size:12px }} td,th {{ border-bottom:1px solid var(--line); padding:4px 6px; text-align:left }}
td.ok {{ color:var(--neg) }} td.bad {{ color:var(--pos) }} td.err {{ color:var(--muted); font-size:11px }}
.badge {{ display:inline-block; padding:2px 10px; border-radius:99px; font-size:12px; font-weight:600;
  background:{'var(--pos)' if nt_active else 'var(--line)'}; color:{'#fff' if nt_active else 'var(--muted)'} }}
.scroll {{ overflow-x:auto }}
</style></head><body><div class="wrap">
<h1>A股情绪与救市资金仪表盘</h1>
<div class="sub">交易日 {last_date}｜分越高 = 越恐慌/越便宜 = 越建议加仓｜不构成投资建议</div>

<div class="hero">
  <div class="kpi"><div class="l">综合评分</div><div class="v">{comp:.1f}</div><div class="l">{label}</div></div>
  <div class="kpi"><div class="l">建议定投倍数</div><div class="v">{mult:.1f}×</div><div class="l">救市资金角色：{mode_cn}</div></div>
  <div class="kpi"><div class="l">恐慌轴 / 估值轴</div><div class="v">{row['fear']:.0f} / {row['value']:.0f}</div>
    <div class="l">权重 {aw['fear']:.0%} / {aw['value']:.0%}</div></div>
  <div class="kpi"><div class="l">救市资金强度</div><div class="v">{_fmt('{:.0f}', nt_int)}</div>
    <div class="l"><span class="badge">{'近5日有救市足迹' if nt_active else '未见明显足迹'}</span></div></div>
</div>

<h2>情绪与技术指标</h2>
<div class="grid4">{''.join(cards)}</div>
<div class="note">恐慌轴 = {weights_txt}；估值轴 = {value_txt}（缺失分项自动剔除并重新归一化）。</div>

<h2>救市资金 / 超大资金</h2>
<div class="refs">{''.join(nt_items)}</div>
<div class="grid4" style="margin-top:12px">
  <div class="panel"><div class="name">救市强度（近一年）</div>
    {line_chart(nd, nv, bands=[config['national_team']['active_threshold']], color='var(--pos)')}</div>
  <div class="panel" style="grid-column: span 2"><div class="name">宽基ETF 每日净申购（近90日）</div>
    {bar_chart(fd, fv)}</div>
</div>
{flow_note}{verdict}
{'<div class="panel scroll" style="margin-top:12px"><table><tr><th>最近事件</th><th>强度</th><th>回撤</th><th>后20日</th><th>后60日</th></tr>' + ''.join(events) + '</table></div>' if events else ''}

<h2>综合评分走势（蓝线：沪深300，归一化）</h2>
<div class="grid4">
  <div class="panel" style="grid-column: 1 / -1">{line_chart(cd, cv, w=900, h=160, color='var(--c1)', overlay=px, bands=[60, 80])}</div>
</div>
{bt}

<h2>估值与参考（不参与评分）</h2>
<div class="refs">{refs}</div>

<h2>数据源状态</h2>
<div class="panel scroll"><table><tr><th>数据源</th><th>状态</th><th>最新日期</th><th>抓取时间</th><th>错误</th></tr>{fr_rows}</table></div>
</div></body></html>"""


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
