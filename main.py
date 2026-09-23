"""
入口：读取 data/ 快照 → 计算 → 生成 docs/index.html（GitHub Pages）。

    python main.py            # 生成仪表盘
    python main.py --fetch    # 先联网抓取（需要能访问国内数据源）
    python main.py --serve    # 生成后在本地打开
"""
import argparse
import http.server
import json
import os
import threading
import webbrowser
from pathlib import Path

from model import run_model
from render import load_json, render_dashboard
from scoring import load_config

BASE = Path(__file__).parent
DOCS = BASE / "docs"


def build():
    config = load_config()
    frame = run_model(config=config)
    fetch_report = load_json(BASE / "data" / "fetch_report.json")
    backtest = load_json(BASE / "reports" / "backtest_summary.json")
    html = render_dashboard(frame, config, fetch_report, backtest)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(html, encoding="utf-8")

    last = frame["composite"].last_valid_index()
    r = frame.loc[last]
    latest = {
        "date": last,
        "composite": round(float(r["composite_final"]), 2),
        "multiplier": float(r["multiplier_final"]),
        "fear": round(float(r["fear"]), 2),
        "value": round(float(r["value"]), 2),
        "nt_intensity": None if r["nt_intensity"] != r["nt_intensity"] else round(float(r["nt_intensity"]), 1),
        "nt_active_5d": bool(r["nt_active_5d"]),
        "raw": {k: (None if r[k] != r[k] else round(float(r[k]), 4))
                for k in ["dd", "qvix", "breadth", "rsi", "boll_z", "turnover", "margin", "erp"]},
    }
    (DOCS / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["close", "dd", "qvix", "breadth", "rsi", "boll_z", "turnover", "margin", "fear", "value",
            "composite", "nt_intensity", "nt_etf_flow", "composite_final", "multiplier_final"]
    frame[cols].tail(750).round(4).to_csv(DOCS / "history.csv")
    print(json.dumps(latest, ensure_ascii=False, indent=2))
    return DOCS / "index.html"


def serve(path):
    os.chdir(path.parent)
    httpd = http.server.HTTPServer(("localhost", 8766), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://localhost:8766/{path.name}"
    webbrowser.open(url)
    print(f"仪表盘运行中: {url}  (Ctrl+C 退出)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        import fetch_all
        fetch_all.main()
    out = build()
    if args.serve:
        serve(out)
