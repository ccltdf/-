# -*- coding: utf-8 -*-
"""signal_check.py — 协变量到底有没有预测力（可行性体检）

背景
----
fetch_data.py 证明了协变量拿得到。但「拿得到」不等于「能预测」。这个脚本回答三个问题：

  1. 这些协变量与 A 股当日收益的相关性有多强？（IC）
  2. 这个相关性体现在开盘跳空上，还是日内走势上？
  3. 按这个信号交易，扣掉成本还剩多少？

为什么必须拆开看
----------------
A 股 T 日收益可分解为两段：
    gap   = 开盘价(T) / 收盘价(T-1) - 1     隔夜跳空，9:25 集合竞价一次性定价
    intra = 收盘价(T) / 开盘价(T)   - 1     9:30 之后的日内走势
    full  = 收盘价(T) / 收盘价(T-1) - 1     = gap 与 intra 的复合

如果协变量的信息全在 gap 里，那意味着 9:00 得出的判断在 9:25 就被市场消化完毕，
任何「开盘后再进场」的做法都拿不到这块收益——这是决定方案形态的关键事实。

用法
----
    D:\\anaconda3\\python.exe signal_check.py
    D:\\anaconda3\\python.exe signal_check.py --bench cn_hs300 --start 2021-03-01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

GROUP_ORDER = [
    ("us_index", "美股指数"), ("us_sector", "美股板块"), ("jp_index", "日股"),
    ("kr_index", "韩股"), ("hk_index", "港股"), ("cn_industry", "A股行业"),
    ("cn_index", "A股指数"),
]
PREFIX_GROUP = [
    ("us_dji", "us_index"), ("us_spx", "us_index"), ("us_ndx", "us_index"),
    ("sec_", "us_sector"), ("jp_", "jp_index"), ("kr_", "kr_index"),
    ("hk_", "hk_index"), ("ind_", "cn_industry"), ("cn_", "cn_index"),
]


def group_of(col: str) -> str:
    for prefix, grp in PREFIX_GROUP:
        if col.startswith(prefix):
            return grp
    return "other"


def load_panel(bench: str, start: str) -> Tuple[pd.DataFrame, List[str]]:
    panel = pd.read_csv(DATA_DIR / "panel.csv", parse_dates=["date"])
    targets = pd.read_csv(DATA_DIR / "targets.csv", parse_dates=["date"])
    px = pd.read_csv(DATA_DIR / "raw" / f"{bench}.csv", parse_dates=["date"]).set_index("date")

    px["gap"] = px["open"] / px["close"].shift(1) - 1.0
    px["intra"] = px["close"] / px["open"] - 1.0
    px["full"] = px["close"] / px["close"].shift(1) - 1.0

    df = panel.merge(px[["gap", "intra", "full"]].reset_index(), on="date")
    df = df.merge(targets, on="date", how="left")
    df = df[df["date"] >= start].reset_index(drop=True)

    feats = [c for c in panel.columns if c != "date"]
    return df, feats


def ic_table(df: pd.DataFrame, feats: List[str], target: str) -> List[Tuple[str, float, float, int]]:
    rows = []
    for c in feats:
        s = df[[c, target]].dropna()
        if len(s) < 200:
            continue
        ic, _ = spearmanr(s[c], s[target])
        recent = s.tail(min(500, len(s)))
        ic2, _ = spearmanr(recent[c], recent[target])
        rows.append((c, float(ic), float(ic2), len(s)))
    rows.sort(key=lambda r: -abs(r[1]))
    return rows


def hit_rate(signal: pd.Series, target: pd.Series) -> float:
    m = target.notna() & (target.abs() > 1e-9) & signal.notna()
    if not m.any():
        return float("nan")
    return float((np.sign(signal[m]) == np.sign(target[m])).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("signal_check")
    ap.add_argument("--bench", default="cn_szzs", help="目标指数 key（见 data/raw）")
    ap.add_argument("--start", default="2021-03-01", help="起始日（美容护理板块 2021-02 才上市）")
    ap.add_argument("--cost", type=float, default=0.001, help="单边成本（默认 10bp）")
    args = ap.parse_args(argv)

    df, feats = load_panel(args.bench, args.start)
    lines: List[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    out("=" * 78)
    out(f"协变量预测力体检   目标指数 = {args.bench}   样本 {len(df)} 日")
    out(f"区间 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
    out("=" * 78)

    # ── 1. 目标拆解 ──
    out("\n[1] 当日收益拆解（%）：")
    out(f"    gap   隔夜跳空  标准差 {df['gap'].std()*100:.3f}  均值 {df['gap'].mean()*100:+.4f}")
    out(f"    intra 日内走势  标准差 {df['intra'].std()*100:.3f}  均值 {df['intra'].mean()*100:+.4f}")
    out(f"    full  全天收益  标准差 {df['full'].std()*100:.3f}  均值 {df['full'].mean()*100:+.4f}")

    # ── 2. IC 表 ──
    out("\n[2] 秩相关 IC（协变量 T-1 收益 vs 当日目标）")
    out(f"    {'协变量':22s}{'gap':>10s}{'intra':>10s}{'full':>10s}")
    ic_gap = dict((c, v) for c, v, _, _ in ic_table(df, feats, "gap"))
    ic_int = dict((c, v) for c, v, _, _ in ic_table(df, feats, "intra"))
    ic_full = dict((c, v) for c, v, _, _ in ic_table(df, feats, "full"))
    ranked = sorted(ic_full.items(), key=lambda kv: -abs(kv[1]))
    for c, _ in ranked[:12]:
        out(f"    {c:22s}{ic_gap.get(c, np.nan):10.4f}{ic_int.get(c, np.nan):10.4f}{ic_full[c]:10.4f}")

    out("\n[3] 分组平均 |IC|")
    out(f"    {'分组':10s}{'|IC| gap':>11s}{'|IC| intra':>12s}{'|IC| full':>11s}")
    for gkey, gname in GROUP_ORDER:
        cs = [c for c in feats if group_of(c) == gkey]
        if not cs:
            continue
        g = np.nanmean([abs(ic_gap[c]) for c in cs if c in ic_gap])
        i = np.nanmean([abs(ic_int[c]) for c in cs if c in ic_int])
        f = np.nanmean([abs(ic_full[c]) for c in cs if c in ic_full])
        out(f"    {gname:10s}{g:11.4f}{i:12.4f}{f:11.4f}")

    # ── 4. 粗糙信号的方向准确率 ──
    out("\n[4] 只用「美股标普 T-1 方向」当信号的准确率")
    sig = df["us_spx_ret1"] if "us_spx_ret1" in df else df["us_dji_ret1"]
    for t in ("gap", "intra", "full"):
        out(f"    {t:6s} {hit_rate(sig, df[t])*100:5.1f}%")

    # ── 5. 交易可行性 ──
    # 日频反手 = 每天卖出+买入各一笔，成本按双边计
    round_trip = 2 * args.cost
    out(f"\n[5] 按上面信号每日反手交易（单边 {args.cost*10000:.0f}bp，双边 {round_trip*10000:.0f}bp）")
    for t in ("gap", "intra", "full"):
        pnl = (np.sign(sig) * df[t] - round_trip).dropna()
        if not len(pnl):
            continue
        out(f"    {t:6s} 日均 {pnl.mean()*100:+.4f}%  年化 {pnl.mean()*252*100:+7.2f}%  "
            f"胜率 {(pnl > 0).mean()*100:5.1f}%  累计 {((1+pnl).prod()-1)*100:+8.2f}%")
    out("    注：gap 一栏即便为正也不可实现——它要求以 T-1 收盘价建仓，而那时美股还没开盘。")

    out("\n" + "-" * 78)
    out("解读要点：")
    out("  * gap 的 IC 高、intra 的 IC 接近 0 → 信息在集合竞价就被定价，开盘后拿不到。")
    out("  * 日股/韩股/港股 T-1 收盘对 A 股当日几乎无增量（同交易时段，已被 A 股消化）。")
    out("  * 真正的增量在「日韩 T 日 8:00-9:00 盘中」，但日线历史里没有，只能实时用。")
    out("=" * 78)

    (DATA_DIR / "_signal_check.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
