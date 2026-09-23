"""
数据层：唯一需要联网的脚本。抓取所有原始数据并写入 data/ 快照。
Data layer — the only networked script. Writes raw snapshots into data/.

评分、回测、渲染只读 data/，因此可以在无网络环境（如沙盒）里运行。
Each source is independent: a failure is recorded in data/fetch_report.json
and the previous snapshot is kept.

Usage:
    python fetch_all.py              # 全部数据源
    python fetch_all.py qvix margin  # 仅指定数据源
"""
import datetime
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
CONFIG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
REPORT_FILE = DATA / "fetch_report.json"

# akshare 调用多数不带 timeout，统一设置 socket 超时避免卡死。
socket.setdefaulttimeout(40)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ak():
    import akshare as ak
    return ak


def retry(fn, *args, tries=3, wait=3.0, **kwargs):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — network layer, any error retried
            last = e
            if i < tries - 1:
                time.sleep(wait * (2 ** i))
    raise last


def first_ok(*attempts):
    """attempts: (label, callable). Return (label, df) of first non-empty result."""
    errors = []
    for label, fn in attempts:
        try:
            df = fn()
            if df is not None and len(df):
                return label, df
            errors.append(f"{label}: empty")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label}: {type(e).__name__}: {e}")
    raise RuntimeError(" | ".join(errors))


def _norm_date(s):
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def merge_save(path, new, keys):
    """Upsert `new` into CSV at `path` keyed by `keys`; newer rows win."""
    new = new.copy()
    if path.exists():
        old = pd.read_csv(path, dtype={k: str for k in keys})
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=keys, keep="last").sort_values(keys)
    new.to_csv(path, index=False)
    return new


def today_str():
    return datetime.date.today().strftime("%Y%m%d")


def _prefixed(code):
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


# ── sources ──────────────────────────────────────────────────────────────────

def _index_daily(symbol):
    """symbol like 'sh000300'. Eastmoney → Tencent fallback."""
    ak = _ak()
    start = CONFIG["fetch"]["history_start"]

    def em():
        df = retry(ak.stock_zh_index_daily_em, symbol=symbol, start_date=start, end_date="20500101")
        return df[["date", "open", "high", "low", "close", "volume", "amount"]]

    def tx():
        df = retry(ak.stock_zh_a_hist_tx, symbol=symbol, start_date=start, end_date="20500101")
        df = df.reset_index(drop=True)
        return df[["date", "open", "high", "low", "close", "volume", "amount"]]

    label, df = first_ok(("eastmoney", em), ("tencent", tx))
    df["date"] = _norm_date(df["date"])
    return label, df.dropna(subset=["date", "close"])


def fetch_indexes():
    out = {}
    for name, symbol in [("csi300", "sh000300"), ("csi1000", "sh000852"),
                         ("sse", "sh000001"), ("szse", "sz399001")]:
        label, df = _index_daily(symbol)
        merge_save(DATA / f"index_{name}.csv", df, ["date"])
        out[name] = {"source": label, "rows": len(df), "last": df["date"].iloc[-1]}
    return out


def fetch_qvix():
    ak = _ak()
    label, df = first_ok(
        ("optbbs-300etf", lambda: retry(ak.index_option_300etf_qvix)),
        ("optbbs-50etf", lambda: retry(ak.index_option_50etf_qvix)),
    )
    df = df[["date", "close"]].rename(columns={"close": "qvix"})
    df["date"] = _norm_date(df["date"])
    df = df.dropna()
    merge_save(DATA / "qvix.csv", df, ["date"])
    return {"source": label, "rows": len(df), "last": df["date"].iloc[-1]}


def fetch_asset():
    """510300 后复权收盘：回测投资标的（含分红）。"""
    ak = _ak()
    start = CONFIG["fetch"]["history_start"]
    label, df = first_ok(
        ("eastmoney-hfq", lambda: retry(ak.fund_etf_hist_em, symbol="510300",
                                       start_date=start, end_date="20500101", adjust="hfq")),
    )
    df = df.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]
    df["date"] = _norm_date(df["date"])
    df.to_csv(DATA / "asset_510300_hfq.csv", index=False)
    return {"source": label, "rows": len(df), "last": df["date"].iloc[-1]}


def _stock_hist(code, start):
    ak = _ak()

    def em():
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start,
                                end_date="20500101", adjust="hfq", timeout=20)
        return df.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]

    def tx():
        df = ak.stock_zh_a_hist_tx(symbol=_prefixed(code), start_date=start,
                                   end_date="20500101", adjust="hfq", timeout=20)
        return df.reset_index(drop=True)[["date", "close"]]

    _, df = first_ok(("eastmoney", lambda: retry(em, tries=2)), ("tencent", lambda: retry(tx, tries=2)))
    df["date"] = _norm_date(df["date"])
    return df.dropna()


def fetch_breadth():
    """沪深300 成分股后复权收盘面板（用于计算长期均线之上占比）。"""
    ak = _ak()
    index_code = CONFIG["fetch"]["breadth_index"]
    sleep = float(CONFIG["fetch"]["stock_request_sleep"])
    label, cons = first_ok(
        ("csindex", lambda: retry(ak.index_stock_cons_csindex, symbol=index_code)),
        ("sina", lambda: retry(ak.index_stock_cons_sina, symbol=index_code)),
    )
    code_col = "成分券代码" if "成分券代码" in cons.columns else "code"
    name_col = "成分券名称" if "成分券名称" in cons.columns else "name"
    cons = cons[[code_col, name_col]].rename(columns={code_col: "code", name_col: "name"})
    cons["code"] = cons["code"].astype(str).str[-6:].str.zfill(6)
    cons.to_csv(DATA / "breadth_constituents.csv", index=False)

    panel_path = DATA / "breadth_closes.csv.gz"
    panel = pd.read_csv(panel_path, index_col="date") if panel_path.exists() else pd.DataFrame()
    panel.columns = [str(c).zfill(6) for c in panel.columns]
    full_start = CONFIG["fetch"]["history_start"]

    frames, failed = {}, []
    for i, code in enumerate(cons["code"]):
        if code in panel.columns and panel[code].notna().any():
            last = panel[code].dropna().index.max()
            start = (pd.Timestamp(last) - pd.Timedelta(days=15)).strftime("%Y%m%d")
        else:
            start = full_start
        try:
            df = _stock_hist(code, start)
            frames[code] = df.set_index("date")["close"]
        except Exception as e:  # noqa: BLE001
            failed.append(f"{code}: {e}"[:160])
        time.sleep(sleep)
        if (i + 1) % 50 == 0:
            print(f"    breadth {i + 1}/{len(cons)}", flush=True)

    if frames:
        upd = pd.DataFrame(frames)
        panel = upd.combine_first(panel) if len(panel) else upd
    keep = [c for c in panel.columns if c in set(cons["code"])]
    panel = panel[keep].sort_index()
    panel.index.name = "date"
    panel.round(4).to_csv(panel_path, compression="gzip")
    return {"source": label, "constituents": len(cons), "fetched": len(frames),
            "failed": len(failed), "failed_sample": failed[:5],
            "rows": len(panel), "last": str(panel.index.max())}


def fetch_high_low():
    ak = _ak()
    df = retry(ak.stock_a_high_low_statistics, symbol="hs300")
    df["date"] = _norm_date(df["date"])
    merge_save(DATA / "legu_high_low_hs300.csv", df, ["date"])
    return {"source": "legulegu", "rows": len(df), "last": df["date"].iloc[-1]}


def fetch_margin():
    ak = _ak()

    def em():
        df = retry(ak.stock_margin_account_info)
        df = df.rename(columns={"日期": "date", "融资余额": "fin_balance", "融资买入额": "fin_buy"})
        return df[["date", "fin_balance", "fin_buy"]]

    def sse():
        df = retry(ak.stock_margin_sse, start_date="20100101", end_date=today_str())
        df = df.rename(columns={"信用交易日期": "date", "融资余额": "fin_balance", "融资买入额": "fin_buy"})
        return df[["date", "fin_balance", "fin_buy"]]

    label, df = first_ok(("eastmoney-all", em), ("sse-only", sse))
    df["date"] = _norm_date(df["date"].astype(str))
    df["source"] = label
    df = df.dropna(subset=["date"])
    df.to_csv(DATA / "margin.csv", index=False)
    return {"source": label, "rows": len(df), "last": df["date"].max()}


def fetch_valuation():
    ak = _ak()
    out = {}
    try:
        erp = retry(ak.stock_ebs_lg).rename(
            columns={"日期": "date", "股债利差": "erp", "股债利差均线": "erp_ma"})
        erp["date"] = _norm_date(erp["date"])
        erp = erp[["date", "erp", "erp_ma"]]
        # 乐咕返回的是小数（0.05 = 5%），统一为百分数。
        if erp["erp"].abs().median() < 0.5:
            erp[["erp", "erp_ma"]] *= 100.0
        merge_save(DATA / "erp.csv", erp, ["date"])
        out["erp"] = {"source": "legulegu", "rows": len(erp), "last": erp["date"].iloc[-1]}
    except Exception as e:  # noqa: BLE001
        out["erp"] = {"error": str(e)[:300]}
    try:
        pe = retry(ak.stock_index_pe_lg, symbol="沪深300")
        pe["date"] = _norm_date(pe["date"])
        cols = [c for c in ["date", "close", "ttmPe", "middleTTMPe", "addTtmPe"] if c in pe.columns]
        pe = pe[cols]
        merge_save(DATA / "pe_csi300.csv", pe, ["date"])
        out["pe"] = {"source": "legulegu", "rows": len(pe), "last": pe["date"].iloc[-1]}
    except Exception as e:  # noqa: BLE001
        out["pe"] = {"error": str(e)[:300]}
    try:
        cg = retry(ak.stock_a_congestion_lg)
        cg["date"] = _norm_date(cg["date"])
        merge_save(DATA / "congestion.csv", cg[["date", "congestion"]], ["date"])
        out["congestion"] = {"source": "legulegu", "rows": len(cg), "last": cg["date"].iloc[-1]}
    except Exception as e:  # noqa: BLE001
        out["congestion"] = {"error": str(e)[:300]}
    if all("error" in v for v in out.values()):
        raise RuntimeError(json.dumps(out, ensure_ascii=False))
    return out


def fetch_market_flow():
    """东方财富 大盘主力/超大单净流入（仅近 ~120 个交易日，逐日累积）。"""
    ak = _ak()
    df = retry(ak.stock_market_fund_flow)
    df = df.rename(columns={
        "日期": "date", "主力净流入-净额": "main_net",
        "超大单净流入-净额": "super_net", "大单净流入-净额": "large_net",
    })
    df["date"] = _norm_date(df["date"])
    df = df[["date", "main_net", "super_net", "large_net"]]
    merge_save(DATA / "market_fund_flow.csv", df, ["date"])
    return {"source": "eastmoney", "rows": len(df), "last": df["date"].iloc[-1]}


def fetch_etf_daily():
    ak = _ak()
    start = CONFIG["fetch"]["history_start"]
    rows, status = [], {}
    for code in CONFIG["national_team_etfs"]:
        def em(code=code):
            df = retry(ak.fund_etf_hist_em, symbol=code, start_date=start,
                       end_date="20500101", adjust="")
            return df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                                      "收盘": "close", "成交额": "amount", "成交量": "volume"})

        def sina(code=code):
            df = retry(ak.fund_etf_hist_sina, symbol=_prefixed(code))
            if "amount" not in df.columns:
                df["amount"] = df["volume"] * df["close"]
            return df

        try:
            label, df = first_ok(("eastmoney", em), ("sina", sina))
            df["date"] = _norm_date(df["date"])
            df["code"] = code
            rows.append(df[["date", "code", "open", "high", "low", "close", "volume", "amount"]])
            status[code] = label
        except Exception as e:  # noqa: BLE001
            status[code] = f"error: {e}"[:200]
        time.sleep(0.3)
    if not rows:
        raise RuntimeError(json.dumps(status, ensure_ascii=False))
    df = pd.concat(rows, ignore_index=True)
    merge_save(DATA / "etf_daily.csv", df, ["date", "code"])
    return {"per_code": status, "rows": len(df), "last": df["date"].max()}


def fetch_etf_shares():
    """ETF 份额：上交所按日回填 + 东方财富当日快照（含深市）。"""
    ak = _ak()
    codes = set(CONFIG["national_team_etfs"])
    path = DATA / "etf_shares.csv"
    have = pd.read_csv(path, dtype={"code": str}) if path.exists() else pd.DataFrame(
        columns=["date", "code", "shares", "source"])
    out = {}

    # 1) 当日快照（沪深均有），数据日期以接口返回为准
    try:
        spot = retry(ak.fund_etf_spot_em)
        spot = spot[spot["代码"].astype(str).isin(codes)]
        date_col = spot["数据日期"] if "数据日期" in spot.columns else pd.Series(
            [datetime.date.today()] * len(spot), index=spot.index)
        snap = pd.DataFrame({
            "date": _norm_date(date_col.astype(str)),
            "code": spot["代码"].astype(str),
            "shares": pd.to_numeric(spot["最新份额"], errors="coerce"),
            "source": "eastmoney-spot",
        }).dropna()
        have = pd.concat([have, snap], ignore_index=True)
        out["spot_rows"] = len(snap)
    except Exception as e:  # noqa: BLE001
        out["spot_error"] = str(e)[:200]

    # 2) 上交所历史回填（每次运行最多 N 个交易日，从新到旧）
    idx_path = DATA / "index_csi300.csv"
    if idx_path.exists():
        start = pd.Timestamp(CONFIG["fetch"]["etf_share_backfill_start"]).strftime("%Y-%m-%d")
        trade_days = pd.read_csv(idx_path)["date"]
        trade_days = sorted([d for d in trade_days if d >= start], reverse=True)
        sse_codes = {c for c in codes if c.startswith("5")}
        done = set(have.loc[have["source"] == "sse", "date"])
        todo = [d for d in trade_days if d not in done]
        cap = int(CONFIG["fetch"]["etf_share_backfill_max_per_run"])
        got, errors = 0, []
        for d in todo[:cap]:
            try:
                df = retry(ak.fund_etf_scale_sse, date=d.replace("-", ""), tries=2)
                df = df[df["基金代码"].astype(str).isin(sse_codes)]
                if len(df):
                    have = pd.concat([have, pd.DataFrame({
                        "date": d, "code": df["基金代码"].astype(str),
                        "shares": pd.to_numeric(df["基金份额"], errors="coerce"),
                        "source": "sse"})], ignore_index=True)
                    got += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{d}: {e}"[:120])
                if len(errors) >= 8 and got == 0:
                    break
            time.sleep(0.3)
        out.update({"sse_backfilled_days": got, "sse_remaining": max(len(todo) - got, 0),
                    "sse_errors": errors[:3]})

    if not len(have):
        raise RuntimeError(json.dumps(out, ensure_ascii=False))
    # 同一日同一代码优先保留交易所口径
    have["prio"] = (have["source"] == "sse").astype(int)
    have = (have.sort_values(["date", "code", "prio"])
                .drop_duplicates(["date", "code"], keep="last")
                .drop(columns="prio"))
    have.to_csv(path, index=False)
    out.update({"rows": len(have), "last": have["date"].max()})
    return out


SOURCES = {
    "indexes": fetch_indexes,
    "asset": fetch_asset,
    "qvix": fetch_qvix,
    "margin": fetch_margin,
    "valuation": fetch_valuation,
    "high_low": fetch_high_low,
    "market_flow": fetch_market_flow,
    "etf_daily": fetch_etf_daily,
    "etf_shares": fetch_etf_shares,   # 依赖 indexes 的交易日
    "breadth": fetch_breadth,
}


def main(selected=None):
    DATA.mkdir(exist_ok=True)
    report = json.loads(REPORT_FILE.read_text(encoding="utf-8")) if REPORT_FILE.exists() else {}
    for name, fn in SOURCES.items():
        if selected and name not in selected:
            continue
        t0 = time.time()
        print(f"[fetch] {name} …", flush=True)
        try:
            info = fn()
            report[name] = {"ok": True, "at": datetime.datetime.now().isoformat(timespec="seconds"),
                            "seconds": round(time.time() - t0, 1), "info": info}
            print(f"  ok ({time.time() - t0:.1f}s): {json.dumps(info, ensure_ascii=False)[:400]}", flush=True)
        except Exception as e:  # noqa: BLE001
            prev = report.get(name, {})
            report[name] = {"ok": False, "at": datetime.datetime.now().isoformat(timespec="seconds"),
                            "error": f"{type(e).__name__}: {e}"[:600],
                            "last_ok": prev.get("at") if prev.get("ok") else prev.get("last_ok")}
            print(f"  FAILED: {e}", flush=True)
            traceback.print_exc(limit=2)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for k, v in report.items() if v.get("ok"))
    print(f"[fetch] done: {ok}/{len(report)} sources ok")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
