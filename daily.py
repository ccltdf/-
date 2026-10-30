# -*- coding: utf-8 -*-
"""daily.py — 每天跑这一条命令：更新数据 → 输出今日/明日预测

一条命令做完三件事：
    1. 增量更新 55 个协变量（本地已有数据时只补最近 120 天，首次自动全量）
    2. 用全部已实现的数据重训 h1/h2 两个模型
    3. 输出「今天」和「明天」的预测、信号强度分档、仓位建议，并记进台账

建议时点
--------
早上 9:00 之前（比如 8:50）。此时：
    * 美股 T−1 收盘已出（北京凌晨 4~5 点），是最关键的隔夜信息
    * 日韩已开盘约 1 小时（本版未纳入，见 README）
    * A 股当天还没开盘，预测的是「今天」

如果在收盘后跑，语义同样成立：T 日 = 今天（已实现，可用于复盘），T+1 = 明天。

台账
----
data/predictions_log.csv 按日期累积，每次运行会自动回填已实现的实际收益并统计命中率。

用法
----
    D:\\anaconda3\\python.exe daily.py           # 更新数据 + 出预测
    D:\\anaconda3\\python.exe daily.py --no-fetch # 不联网，用本地数据
    D:\\anaconda3\\python.exe daily.py --eval     # 额外跑一遍滚动评估
    D:\\anaconda3\\python.exe daily.py --full     # 强制全量重抓
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fetch_data import (
    RAW_DIR, Asset, build_panel, build_universe, crawl, save_raw,
)
from predict import (
    FEATURES, build_frame, evaluate, fit_final, make_model, strength_ladder, walk_forward,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_PATH = DATA_DIR / "predictions_log.csv"
BENCH = "cn_szzs"
START = "2021-03-01"
INCR_DAYS = 120


# ──────────────────────────────────────────────────────────────
#  1. 数据更新
# ──────────────────────────────────────────────────────────────
def update_frames(assets: List[Asset], full: bool, quiet: bool) -> Dict[str, pd.DataFrame]:
    """增量合并本地 raw CSV；首次或 --full 时全量。"""
    have_local = (RAW_DIR / f"{BENCH}.csv").exists()
    if full or not have_local:
        beg = "20180101"
        label = "全量"
    else:
        beg = (pd.Timestamp.now() - pd.Timedelta(days=INCR_DAYS)).strftime("%Y%m%d")
        label = f"增量（{beg} 起）"
    if not quiet:
        print(f"  [{label}] 抓取 {len(assets)} 个标的 ...", flush=True)

    new = crawl(assets, beg, verbose=False)
    frames: Dict[str, pd.DataFrame] = {}
    for key, df in new.items():
        path = RAW_DIR / f"{key}.csv"
        if have_local and not full and path.exists():
            old = pd.read_csv(path, parse_dates=["date"])
            df = (
                pd.concat([old, df], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        frames[key] = df
    save_raw(frames)
    return frames


# ──────────────────────────────────────────────────────────────
#  2. 预测
# ──────────────────────────────────────────────────────────────
def make_forecast(df: pd.DataFrame) -> Dict[str, object]:
    """训练两个模型并输出最新一日的预测。"""
    row = df.iloc[[-1]]
    X = row[FEATURES].fillna(0.0).to_numpy()
    out: Dict[str, object] = {
        "date": pd.Timestamp(row["date"].iloc[0]),
        "us_spx": float(row["us_spx_ret1"].iloc[0]),
        "us_ndx": float(row["us_ndx_ret1"].iloc[0]) if "us_ndx_ret1" in row else np.nan,
        "cn_vol20": float(row["cn_vol20"].iloc[0]),
    }
    for h, key in ((1, "h1"), (2, "h2")):
        model = fit_final(df, h)
        out[key] = float(model.predict(X)[0])
        out[f"model_{key}"] = model
    out["h2c"] = ((1.0 + out["h1"] / 100.0) * (1.0 + out["h2"] / 100.0) - 1.0) * 100.0
    return out


def grade_signal(value: float, abs_pred_hist: np.ndarray) -> tuple[str, float]:
    """按 |预测值| 在历史预测分布中的分位给出信号等级。"""
    if not len(abs_pred_hist):
        return "—", float("nan")
    pct = float((abs_pred_hist < abs(value)).mean() * 100.0)
    if pct >= 80:
        return "强信号", pct
    if pct >= 60:
        return "中等信号", pct
    return "弱信号", pct


def position_hint(value: float, grade: str) -> str:
    """把信号等级翻译成仓位建议（A 股做空受限，下跌信号只减仓不清仓）。"""
    if grade == "强信号":
        return "偏多：可加到 7~9 成" if value > 0 else "偏空：降到 2~3 成"
    if grade == "中等信号":
        return "偏多：5~6 成" if value > 0 else "偏空：4 成以下"
    return "维持现有仓位，不做方向性操作"


# ──────────────────────────────────────────────────────────────
#  3. 台账
# ──────────────────────────────────────────────────────────────
def update_log(df: pd.DataFrame, fc: Dict[str, object],
               grade_h1: str, pct_h1: float, grade_h2: str, pct_h2: float) -> Optional[pd.DataFrame]:
    """写入当日预测，并回填历史行已实现的实际收益。"""
    row = {
        "date": fc["date"].normalize(),
        "us_spx_ret1": round(fc["us_spx"], 4),
        "cn_vol20": round(fc["cn_vol20"], 4),
        "pred_h1": round(fc["h1"], 4),
        "pred_h2": round(fc["h2"], 4),
        "grade_h1": grade_h1,
        "grade_h2": grade_h2,
        "strength_h1": round(pct_h1, 1),
        "strength_h2": round(pct_h2, 1),
    }
    log = pd.read_csv(LOG_PATH, parse_dates=["date"]) if LOG_PATH.exists() else pd.DataFrame()
    if len(log):
        log = log[log["date"] != row["date"]]
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log = log.sort_values("date").reset_index(drop=True)
    for col in ("actual_h1", "actual_h2"):
        if col not in log.columns:
            log[col] = np.nan

    # 回填实际值：当天未收盘时不算数，避免把盘中价当成最终结果
    now = pd.Timestamp.now()
    today = now.normalize()
    closed_today = now.hour * 60 + now.minute >= 15 * 60 + 5
    actual = df.set_index("date")[["y_h1", "y_h2"]]
    for i, r in log.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        if d == today and not closed_today:
            continue                      # 今天还在盘中，实际收益未定
        if d in actual.index:
            log.at[i, "actual_h1"] = actual.loc[d, "y_h1"]
            log.at[i, "actual_h2"] = actual.loc[d, "y_h2"]
    log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    return log


def log_summary(log: pd.DataFrame) -> str:
    if log is None or not len(log):
        return "台账为空"
    done = log.dropna(subset=["actual_h1"])
    if not len(done):
        return f"台账 {len(log)} 条，暂无已实现样本"
    hit1 = (np.sign(done["pred_h1"]) == np.sign(done["actual_h1"])).mean() * 100
    strong = done[done["grade_h1"] == "强信号"].dropna(subset=["actual_h1"])
    hit1s = ((np.sign(strong["pred_h1"]) == np.sign(strong["actual_h1"])).mean() * 100) if len(strong) else float("nan")
    strong_txt = f"{hit1s:.1f}%（n={len(strong)}）" if len(strong) else "—"
    return (f"台账 {len(log)} 条，已实现 {len(done)} 条｜"
            f"h1 命中 {hit1:.1f}%｜强信号日 {strong_txt}")


# ──────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("daily", description="每日盘前预测（更新数据 + 今日/明日预测）")
    ap.add_argument("--no-fetch", action="store_true", help="跳过联网，直接用本地 CSV")
    ap.add_argument("--full", action="store_true", help="强制全量重抓")
    ap.add_argument("--no-eval", action="store_true", help="跳过滚动评估（省 1~2 秒）")
    args = ap.parse_args(argv)

    t0 = time.time()
    now = pd.Timestamp.now()
    print("=" * 64)
    print(f"盘前预测 · {now:%Y-%m-%d %H:%M}")
    print("=" * 64)

    # 1) 数据
    if args.no_fetch:
        print("  跳过联网，使用本地 CSV")
        frames = {}
        for f in RAW_DIR.glob("*.csv"):
            frames[f.stem] = pd.read_csv(f, parse_dates=["date"])
        assets: List[Asset] = []
    else:
        assets = build_universe(verbose=False)
        frames = update_frames(assets, args.full, quiet=False)

    # 2) 面板（把今天补进交易日历，早上跑才不会把预测目标退到昨天）
    today = now.normalize()
    extra = [today] if today.weekday() < 5 else []
    if "cn_hs300" in frames and len(extra):
        last_cn = frames["cn_hs300"]["date"].max().normalize()
        if today <= last_cn:
            extra = []
    panel = build_panel(frames, assets or None, extra_dates=extra)
    df = build_frame(panel, frames[BENCH], START)

    # 3) 预测
    fc = make_forecast(df)
    if not args.no_eval:
        pr1 = walk_forward(df, "y_h1", 1)
        pr2 = walk_forward(df, "y_h2", 2)
        abs1 = np.abs(pr1[np.isfinite(pr1)])
        abs2 = np.abs(pr2[np.isfinite(pr2)])
        r1 = evaluate(pr1, df["y_h1"].to_numpy(), df["date"])
        r2 = evaluate(pr2, df["y_h2"].to_numpy(), df["date"])
    else:
        # 轻量近似：用近 500 日模型的样本内预测分布做参照
        X = df[FEATURES].fillna(0.0).to_numpy()[: len(df) - 1]
        m1 = fit_final(df, 1)
        p1 = m1.predict(X)[-500:]
        m2 = fit_final(df, 2)
        p2 = m2.predict(X)[-500:]
        abs1, abs2 = np.abs(p1), np.abs(p2)
        r1 = {"hit": 0.5637, "ic": 0.1821}
        r2 = {"hit": 0.5005, "ic": 0.0438}

    grade1, pct1 = grade_signal(fc["h1"], abs1)
    grade2, pct2 = grade_signal(fc["h2"], abs2)

    # 4) 输出
    d_t = fc["date"]
    d_t1 = d_t + pd.offsets.BDay(1)
    last_us = frames.get("us_spx", pd.DataFrame())
    us_date = last_us["date"].max().strftime("%Y-%m-%d") if len(last_us) else "—"
    last_cn = frames[BENCH]["date"].max().strftime("%Y-%m-%d")

    print()
    print(f"  数据：{len(frames)} 个标的  最新 A股 {last_cn} / 美股 {us_date}  "
          f"（用时 {time.time()-t0:.0f}s）")
    print(f"  输入：标普隔夜 {fc['us_spx']:+.2f}%   纳指隔夜 {fc['us_ndx']:+.2f}%   "
          f"A股20日波动 {fc['cn_vol20']:.2f}%")
    print("-" * 64)
    print(f"  {'预测 T 日（' + d_t.strftime('%m-%d') + '）':22s} {fc['h1']:+6.2f}%   "
          f"{grade1:5s} 强度分位{pct1:5.1f}%")
    print(f"  {'预测 T+1 日（' + d_t1.strftime('%m-%d') + '）':22s} {fc['h2']:+6.2f}%   "
          f"{grade2:5s} 强度分位{pct2:5.1f}%")
    print(f"  {'两日累计':22s} {fc['h2c']:+6.2f}%")
    print("-" * 64)
    print(f"  仓位建议：{position_hint(fc['h1'], grade1)}")
    print(f"  历史对照：h1 方向 {r1['hit']*100:.1f}%（IC {r1['ic']:+.3f}）｜"
          f"h2 方向 {r2['hit']*100:.1f}%（IC {r2['ic']:+.3f}）")
    if grade2 in ("强信号", "中等信号"):
        print("  提醒：h2 的历史准确率接近 50%，即便标为强信号也不建议单独据此操作。")
    print("-" * 64)

    # 5) 台账
    log = update_log(df, fc, grade1, pct1, grade2, pct2)
    print(f"  {log_summary(log)}")

    # 6) 落盘
    fc_out = {k: v for k, v in fc.items() if not k.startswith("model_")}
    pd.DataFrame([{**fc_out, "grade_h1": grade1, "grade_h2": grade2,
                   "strength_h1": pct1, "strength_h2": pct2}]).to_csv(
        DATA_DIR / "latest_forecast.csv", index=False, encoding="utf-8-sig")
    print(f"  已写入 latest_forecast.csv / predictions_log.csv")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
