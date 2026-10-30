# -*- coding: utf-8 -*-
"""fetch_universe.py — 抓取模块一候选股票池的日线数据

候选池来源
----------
    llm_web_ask/output/stocks.xlsx 的「共识」sheet
    三个平台（deepseek 657 / 豆包 280 / 元宝 200）合并去重后共 464 只，
    已按「出现平台数降序、平均序号升序」排好名次。
    --top 200 即取前 200 名（109 只三平台共识 + 91 只双平台）。

行情来源
--------
    东方财富日线接口，一次请求返回全历史，无需登录与 key。

    复权口径默认用**后复权**（--fqt 2），这是实测后的选择：
    东财的前复权用的是减法调整，高分红股（陕煤 601225、兖矿 600188）早期价格会被
    减成负数——601225 有 406 行负价、600188 有 727 行负价，收益率完全没法算。
    后复权是乘法调整，不会出现负价，且收益率序列与真实持有收益一致。
    代价是价格绝对值被放大（茅台显示 8399 而非实际 1350），做因子/回测无影响；
    若要真实成交价，用 --fqt 0 重新抓一份。

输出
----
    data/universe/<代码>.csv     每只一个文件，含完整 OHLCV + 涨跌幅/换手率
    data/universe/_index.csv     清单：代码/名称/排名/平台数/行数/起止日期/状态
    data/universe/_report.txt    抓取报告

用法
----
    D:\\anaconda3\\python.exe fetch_universe.py                  # 前 200 只
    D:\\anaconda3\\python.exe fetch_universe.py --top 464        # 全部 464 只
    D:\\anaconda3\\python.exe fetch_universe.py --list           # 只看池子不抓
    D:\\anaconda3\\python.exe fetch_universe.py --fqt 0          # 不复权（对齐模块二）
    D:\\anaconda3\\python.exe fetch_universe.py --workers 4      # 并发（默认 4）
    D:\\anaconda3\\python.exe fetch_universe.py --force          # 已有文件也重抓
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
POOL_XLSX = PROJECT / "llm_web_ask" / "output" / "stocks.xlsx"
OUT_DIR = ROOT / "data" / "universe"

API_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json,text/plain,*/*",
    "Connection": "close",
}

# fields2: f51 日期 f52 开 f53 收 f54 高 f55 低 f56 量 f57 额
#          f58 振幅 f59 涨跌幅 f60 涨跌额 f61 换手率
FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
# 东财返回顺序：日期/开/收/高/低/量/额/振幅/涨跌幅/涨跌额/换手率
RAW_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount",
               "amplitude", "pct_chg", "change", "turnover"]
# 落盘时调整为标准 K 线顺序，便于其他工具直接读
COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount",
           "amplitude", "pct_chg", "change", "turnover"]


# ──────────────────────────────────────────────────────────────
#  候选池
# ──────────────────────────────────────────────────────────────
def load_pool(path: Path, top: Optional[int]) -> pd.DataFrame:
    """读「共识」sheet，返回排好序的候选池。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到候选池文件：{path}")
    df = pd.read_excel(path, sheet_name="共识", dtype={"代码": str})
    df["代码"] = df["代码"].astype(str).str.strip().str.zfill(6)
    df = df[df["代码"].str.fullmatch(r"\d{6}")].copy()

    if "排名" in df.columns:
        df = df.sort_values("排名")
    elif "出现平台数" in df.columns:
        df = df.sort_values(["出现平台数", "平均序号"], ascending=[False, True])
    df = df.reset_index(drop=True)

    if top:
        df = df.head(int(top)).copy()
    return df


def to_secid(code: str) -> Optional[str]:
    """6 位代码 → 东财 secid。沪市 1.，深市/创业板/北交所 0.。"""
    if code.startswith("6"):
        return f"1.{code}"
    if code.startswith(("0", "2", "3", "4", "8")):
        return f"0.{code}"
    return None


# ──────────────────────────────────────────────────────────────
#  抓取
# ──────────────────────────────────────────────────────────────
def http_json(url: str, retries: int = 3, timeout: int = 20) -> dict:
    last: Optional[Exception] = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "ignore"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(str(last))


def fetch_one(code: str, beg: str, end: str, fqt: int) -> Tuple[pd.DataFrame, str]:
    """抓单只股票的日线，返回 (DataFrame, 名称)。"""
    secid = to_secid(code)
    if secid is None:
        return pd.DataFrame(), ""
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": FIELDS2,
        "klt": "101",            # 日线
        "fqt": str(fqt),         # 0 不复权 / 1 前复权 / 2 后复权
        "beg": beg,
        "end": end,
    }
    payload = http_json(f"{API_KLINE}?{urllib.parse.urlencode(params)}")
    data = payload.get("data")
    if not data or not data.get("klines"):
        return pd.DataFrame(), (data or {}).get("name", "") or ""

    rows = [line.split(",") for line in data["klines"]]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in RAW_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df[COLUMNS], str(data.get("name") or "")


def validate_df(df: pd.DataFrame) -> List[str]:
    """落盘前的数据校验：后复权下不应出现非正价、跳空、缺列。"""
    issues: List[str] = []
    neg = int((df["close"] <= 0).sum())
    if neg:
        issues.append(f"非正收盘价 {neg} 行")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            issues.append(f"缺列 {col}")
    if len(df) > 1:
        ret = df["close"].pct_change().abs()
        jumps = int((ret > 0.25).sum())
        if jumps:
            issues.append(f"单日变动超 25% 共 {jumps} 处（多为新股首日/重组复牌，需人工确认）")
    return issues


def run_crawl(pool: pd.DataFrame, beg: str, end: str, fqt: int,
              workers: int, force: bool, out: Path) -> Dict[str, str]:
    """抓取并落盘，返回 {代码: 名称}。"""
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for _, r in pool.iterrows():
        code = r["代码"]
        fp = out / f"{code}.csv"
        if fp.exists() and not force:
            continue
        tasks.append((code, r.get("名称"), fp))

    total = len(tasks)
    if total == 0:
        print("  所有文件已存在，跳过抓取（--force 可强制重抓）")
        return {}

    print(f"  待抓取 {total} 只，并发 {workers} ...")
    done = 0
    names: Dict[str, str] = {}
    issues_log: List[str] = []

    def work(code: str):
        df, name = fetch_one(code, beg, end, fqt)
        return code, df, name

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(work, c): (c, n, fp) for c, n, fp in tasks}
        for fut in as_completed(futures):
            code, pool_name, fp = futures[fut]
            done += 1
            try:
                _, df, api_name = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [{done}/{total}] {code} 失败 {str(exc)[:60]}")
                continue
            if df.empty:
                print(f"  [{done}/{total}] {code} 无数据")
                continue
            if api_name:
                names[code] = api_name
            problems = validate_df(df)
            if problems:
                issues_log.append(f"{code} {'；'.join(problems)}")
                print(f"  [{done}/{total}] {code} 数据异常：{'；'.join(problems)}")
            df.to_csv(fp, index=False, encoding="utf-8-sig")
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] ... 最近 {code} {len(df)} 行")
            time.sleep(0.02)
    if issues_log:
        out.joinpath("_data_issues.txt").write_text("\n".join(issues_log), encoding="utf-8")
        print(f"  数据异常 {len(issues_log)} 只，明细见 _data_issues.txt")
    return names


# ──────────────────────────────────────────────────────────────
#  清单与报告
# ──────────────────────────────────────────────────────────────
def build_index(pool: pd.DataFrame, out: Path,
                name_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """扫描输出目录，生成清单（含已存在的老文件）。名称优先用接口返回值。"""
    old_names: Dict[str, str] = {}
    old_path = out / "_index.csv"
    if old_path.exists():
        try:
            o = pd.read_csv(old_path, dtype={"代码": str})
            old_names = {c: str(n) for c, n in zip(o["代码"], o["名称"].fillna("")) if n}
        except Exception:  # noqa: BLE001
            pass

    rows = []
    for _, r in pool.iterrows():
        code = r["代码"]
        fp = out / f"{code}.csv"
        name = (name_map or {}).get(code) or old_names.get(code) or r.get("名称")
        if pd.isna(name):
            name = ""
        item = {
            "代码": code,
            "名称": name,
            "排名": r.get("排名", np.nan),
            "出现平台数": r.get("出现平台数", np.nan),
            "文件": fp.name if fp.exists() else "",
            "行数": 0,
            "起始日": pd.NaT,
            "结束日": pd.NaT,
            "状态": "缺失",
        }
        if fp.exists():
            try:
                d = pd.read_csv(fp, usecols=["date"])
                item["行数"] = len(d)
                if len(d):
                    item["起始日"] = pd.to_datetime(d["date"].iloc[0])
                    item["结束日"] = pd.to_datetime(d["date"].iloc[-1])
                item["状态"] = "OK" if len(d) >= 250 else "数据不足"
            except Exception:  # noqa: BLE001
                item["状态"] = "读取失败"
        rows.append(item)
    return pd.DataFrame(rows)


def write_report(idx: pd.DataFrame, fqt: int, beg: str, elapsed: float,
                 n_issues: int = 0) -> str:
    ok = idx[idx["状态"] == "OK"]
    bad = idx[idx["状态"] != "OK"]
    lines = [
        "=" * 72,
        "模块一候选池 · 日线数据抓取报告",
        "=" * 72,
        f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}   耗时 {elapsed:.0f}s",
        f"复权口径：{'不复权' if fqt == 0 else ('前复权' if fqt == 1 else '后复权')}"
        f"   起始 {beg}",
        f"候选池 {len(idx)} 只：成功 {len(ok)}，异常 {len(bad)}",
        f"数据校验：{n_issues} 只存在价格或跳空异常"
        + ("（明细见 _data_issues.txt）" if n_issues else ""),
        "",
    ]
    if len(ok):
        lines.append(f"交易日行数：中位 {int(ok['行数'].median())}，"
                     f"最少 {int(ok['行数'].min())}，最多 {int(ok['行数'].max())}")
        lines.append(f"数据范围：{ok['起始日'].min():%Y-%m-%d} ~ {ok['结束日'].max():%Y-%m-%d}")
        stale = ok[ok["结束日"] < ok["结束日"].max() - pd.Timedelta(days=5)]
        if len(stale):
            lines.append(f"最新数据落后 5 天以上的：{len(stale)} 只（停牌或已退市）")
            lines.append("  " + "、".join(stale["代码"].head(12).tolist()))
    if len(bad):
        lines.append("")
        lines.append("异常清单：")
        for _, r in bad.head(20).iterrows():
            lines.append(f"  {r['代码']}  {r['状态']}  {r['文件']}")
        if len(bad) > 20:
            lines.append(f"  ...另有 {len(bad)-20} 只")
    lines.append("=" * 72)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("fetch_universe", description="抓取模块一候选池的日线数据")
    ap.add_argument("--pool", default=str(POOL_XLSX), help="候选池 xlsx 路径")
    ap.add_argument("--top", type=int, default=200, help="取前 N 只（默认 200，0=全部）")
    ap.add_argument("--beg", default="20180101")
    ap.add_argument("--end", default="20500101")
    ap.add_argument("--fqt", type=int, default=2, choices=[0, 1, 2],
                    help="0 不复权 / 1 前复权 / 2 后复权（默认，前复权会出现负价）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true", help="已有文件也重抓")
    ap.add_argument("--list", action="store_true", help="只打印候选池，不抓取")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    t0 = time.time()

    pool = load_pool(Path(args.pool), args.top or None)
    print("=" * 72)
    print(f"模块一候选池：{len(pool)} 只"
          f"（来源 {Path(args.pool).name}「共识」sheet）")
    print("=" * 72)
    if len(pool) and "出现平台数" in pool.columns:
        vc = pool["出现平台数"].value_counts().sort_index(ascending=False)
        print("  平台共识构成：" + "  ".join(f"{int(k)}平台 {v}只" for k, v in vc.items()))
    print("  前 5 名：" + "、".join(
        f"{r['代码']}{'' if pd.isna(r.get('名称')) else r['名称']}"
        for _, r in pool.head(5).iterrows()))

    if args.list:
        print("\n" + pool[["排名", "代码", "名称", "出现平台数"]].head(30).to_string(index=False))
        print(f"\n（共 {len(pool)} 只，未抓取数据）")
        return 0

    print()
    names = run_crawl(pool, args.beg, args.end, args.fqt, args.workers, args.force, out)

    idx = build_index(pool, out, names)
    idx.to_csv(out / "_index.csv", index=False, encoding="utf-8-sig")
    issues_fp = out / "_data_issues.txt"
    n_issues = len(issues_fp.read_text(encoding="utf-8").splitlines()) if issues_fp.exists() else 0
    report = write_report(idx, args.fqt, args.beg, time.time() - t0, n_issues)
    (out / "_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"输出目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
