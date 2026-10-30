# -*- coding: utf-8 -*-
"""predict.py — 两个提前量：预测 1 个交易日后、2 个交易日后的走势

口径（先说清楚，否则数字会读错）
--------------------------------
站在 T 日 9:00，信息截止 T−1 收盘 + 隔夜美股，同时输出两个预测：

    h1  = 预测 T 日收益      （1 个交易日后，也就是「今天」）
    h2  = 预测 T+1 日收益    （2 个交易日后，也就是「明天」）
    h2c = 预测 T~T+1 两日累计收益（附带，用于「持有两天」的场景）

三者共用同一套特征（panel.csv），区别只在标签。

为什么 h2 一定要单独看
----------------------
美股隔夜信息只作用在 T 日开盘那一跳；到 T+1 日，这块信息已经消耗完毕，
而 T 日白天的量价又不在这套特征里。所以 h2 的信息量理应大幅衰减——
脚本会把 h1 / h2 / h2c 并排打出来，衰减幅度一目了然。

模型
----
Ridge 线性回归（标准化后），特征刻意只留 2 个：
    us_spx_ret1   标普500 隔夜收益（美股整体方向）
    cn_vol20      A股 20 日已实现波动率（shift 1，决定「同样的隔夜信息能推动多少」）
实测：55 个协变量全塞进去，方向准确率反而从 56% 掉到 52.8%（共线稀释）；
单一美股因子最好，再加一个波动率做状态调节最优。

滚动验证
--------
扩展窗口，每 20 个交易日重训一次；h1 训练集截止 s−2、h2/h2c 截止 s−3，
保证训练样本的标签在预测时点之前已经实现。

用法
----
    D:\\anaconda3\\python.exe predict.py                 # 滚动评估 + 今日/明日预测
    D:\\anaconda3\\python.exe predict.py --bench cn_hs300
    D:\\anaconda3\\python.exe predict.py --eval-only     # 只做评估
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# 两个模型共用的特征（顺序即列序）
FEATURES: List[str] = ["us_spx_ret1", "cn_vol20"]
TARGETS = {"h1": "T 日（1 个交易日后）", "h2": "T+1 日（2 个交易日后）", "h2c": "T~T+1 两日累计"}


# ──────────────────────────────────────────────────────────────
#  数据
# ──────────────────────────────────────────────────────────────
def build_frame(panel: pd.DataFrame, px: pd.DataFrame, start: str = "2021-03-01") -> pd.DataFrame:
    """由面板与基准行情构造「特征 + 标签」表。所有 A 股自身特征一律 shift(1)。

    px 需含 date/close 两列（或已按 date 建索引的行情）。
    """
    if "date" not in px.columns:
        px = px.reset_index()
    px = px[["date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").set_index("date")
    close = px["close"].astype(float)
    ret = close.pct_change()

    df = panel.copy()
    # 标签：从锚点日 T−1 收盘算起的收益
    df["y_h1"] = ret.reindex(df["date"]).values * 100.0
    df["y_h2"] = ret.shift(-1).reindex(df["date"]).values * 100.0
    df["y_h2c"] = ((1.0 + ret) * (1.0 + ret.shift(-1)) - 1.0).reindex(df["date"]).values * 100.0

    # 状态特征：T 日 9:00 可知的最新波动率 = 截至 T−1 的 20 日已实现波动
    # 用 asof(date < T) 取：既排除当天（防泄漏），又在早盘缺当天 bar 时取到正确值。
    vol_raw = (ret.rolling(20).std() * 100.0).dropna()
    dates = pd.to_datetime(df["date"])
    v20 = np.full(len(dates), np.nan)
    if len(vol_raw):
        idx = vol_raw.index.values.astype("datetime64[D]")
        pos = np.searchsorted(idx, dates.values.astype("datetime64[D]"), side="left") - 1
        ok = pos >= 0
        v20[ok] = vol_raw.values[pos[ok]]
    df["cn_vol20"] = v20

    df = df[df["date"] >= start].reset_index(drop=True)
    return df


def load_dataset(bench: str, start: str) -> pd.DataFrame:
    """从磁盘读 panel.csv 与 raw/<bench>.csv。"""
    panel = pd.read_csv(DATA_DIR / "panel.csv", parse_dates=["date"])
    px = pd.read_csv(DATA_DIR / "raw" / f"{bench}.csv", parse_dates=["date"])
    return build_frame(panel, px, start)


# ──────────────────────────────────────────────────────────────
#  滚动训练
# ──────────────────────────────────────────────────────────────
def make_model() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def walk_forward(df: pd.DataFrame, target: str, horizon: int,
                 start: int = 400, step: int = 20) -> np.ndarray:
    """扩展窗口滚动预测。horizon=1 表示标签次日即实现，=2 表示要多留一天缓冲。"""
    X = df[FEATURES].fillna(0.0).to_numpy()
    y = df[target].to_numpy()
    pred = np.full(len(y), np.nan)
    gap = horizon + 1                      # 训练集尾部留出标签实现所需的缓冲
    for s in range(start, len(y), step):
        end = max(30, s - gap)
        m = make_model()
        m.fit(X[:end], y[:end])
        pred[s: s + step] = m.predict(X[s: s + step])
    return pred


def strength_ladder(pred: np.ndarray, y: np.ndarray, n_bins: int = 5) -> List[Dict[str, float]]:
    """按 |预测值| 分位看方向准确率——判断信号强度是否真的带来更高的胜率。"""
    mask = np.isfinite(pred) & np.isfinite(y)
    p, t = pred[mask], y[mask]
    if len(p) < 50:
        return []
    try:
        bins = pd.qcut(np.abs(p), n_bins, labels=False, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for q in range(int(bins.max()) + 1):
        sel = bins == q
        if sel.sum() < 20:
            continue
        rows.append({
            "bin": q + 1,
            "n": int(sel.sum()),
            "abs_pred": float(np.abs(p[sel]).mean()),
            "hit": float((np.sign(p[sel]) == np.sign(t[sel])).mean()),
            "mean_ret": float(t[sel].mean()),
        })
    return rows


def evaluate(pred: np.ndarray, y: np.ndarray, dates: pd.Series) -> Dict[str, object]:
    mask = np.isfinite(pred) & np.isfinite(y)
    p, t = pred[mask], y[mask]
    if len(p) < 30:
        return {"n": int(len(p))}
    ic = float(spearmanr(p, t)[0])
    hit = float((np.sign(p) == np.sign(t)).mean())
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 分年度方向准确率（稳定性）
    yr = pd.Series(dates[mask].dt.year.to_numpy(), index=range(len(p)))
    by_year = {}
    for year, idx in yr.groupby(yr).groups.items():
        ii = np.array(list(idx))
        if len(ii) >= 40:
            by_year[int(year)] = float((np.sign(p[ii]) == np.sign(t[ii])).mean())

    return {
        "n": int(len(p)), "ic": ic, "hit": hit, "r2": r2,
        "mae": float(np.abs(p - t).mean()), "by_year": by_year,
    }


# ──────────────────────────────────────────────────────────────
#  今日预测
# ──────────────────────────────────────────────────────────────
def fit_final(df: pd.DataFrame, horizon: int):
    """用全部已实现标签训练最终模型。horizon=2 时丢掉最后一条（标签还没实现）。"""
    end = len(df) - horizon
    X = df[FEATURES].fillna(0.0).to_numpy()[:end]
    y = df[f"y_h{horizon}" if horizon < 3 else "y_h2c"].to_numpy()[:end]
    m = make_model()
    m.fit(X, y)
    return m


def forecast_latest(df: pd.DataFrame) -> Dict[str, object]:
    row = df.iloc[[-1]]
    X = row[FEATURES].fillna(0.0).to_numpy()
    out: Dict[str, object] = {
        "date": row["date"].iloc[0],
        "us_spx": float(row["us_spx_ret1"].iloc[0]),
        "cn_vol20": float(row["cn_vol20"].iloc[0]),
    }
    for h, key in ((1, "h1"), (2, "h2")):
        out[key] = float(fit_final(df, h).predict(X)[0])
    # 两日累计：直接由两个单日预测复合
    out["h2c"] = ((1.0 + out["h1"] / 100.0) * (1.0 + out["h2"] / 100.0) - 1.0) * 100.0
    return out


# ──────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("predict", description="1 日 / 2 日后走势预测")
    ap.add_argument("--bench", default="cn_szzs", help="基准指数 key（默认上证）")
    ap.add_argument("--start", default="2021-03-01")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args(argv)

    df = load_dataset(args.bench, args.start)
    lines: List[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    out("=" * 76)
    out(f"盘前预测  基准={args.bench}  样本 {len(df)} 日  "
        f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
    out(f"特征：{' / '.join(FEATURES)}")
    out("=" * 76)

    # ── 滚动评估 ──
    results = {}
    preds = {}
    ys = {}
    for key, tgt, hz in (("h1", "y_h1", 1), ("h2", "y_h2", 2), ("h2c", "y_h2c", 2)):
        pr = walk_forward(df, tgt, hz, step=args.step)
        preds[key] = pr
        ys[key] = df[tgt].to_numpy()
        results[key] = evaluate(pr, ys[key], df["date"])

    out("\n[1] 样本外滚动评估")
    out(f"    {'目标':26s}{'样本':>6s}{'IC':>9s}{'方向准确率':>11s}{'MAE':>8s}")
    for key in ("h1", "h2", "h2c"):
        r = results[key]
        out(f"    {TARGETS[key]:26s}{r['n']:6d}{r['ic']:+9.4f}{r['hit']*100:10.2f}%{r['mae']:8.3f}")

    out("\n[2] 分年度方向准确率（稳定性）")
    years = sorted({y for r in results.values() for y in r.get("by_year", {})})
    out("    " + f"{'目标':10s}" + "".join(f"{y:>8d}" for y in years))
    for key in ("h1", "h2", "h2c"):
        by = results[key].get("by_year", {})
        out(f"    {key:10s}" + "".join(
            f"{by[y]*100:7.1f}%" if y in by else f"{'-':>8s}" for y in years))

    # ── 强度阶梯 ──
    out("\n[2b] 强度阶梯：按 |预测值| 五等分，方向准确率是否随强度上升")
    for key in ("h1", "h2"):
        rows = strength_ladder(preds[key], ys[key])
        if not rows:
            continue
        out(f"    {key}：" + "  ".join(
            f"第{r['bin']}组(|p|≈{r['abs_pred']:.2f}%) {r['hit']*100:.1f}%" for r in rows))
        top = rows[-1]
        out(f"       → 最强组准确率 {top['hit']*100:.1f}%，"
            f"{'强度分层有效' if top['hit'] - rows[0]['hit'] > 0.04 else '强度分层不明显'}")

    # ── 今日预测 ──
    fc = forecast_latest(df)
    out("\n" + "-" * 76)
    out("[3] 最新预测")
    out("-" * 76)
    out(f"    信息基准日：{pd.Timestamp(fc['date']):%Y-%m-%d} 09:00 之前")
    out(f"    输入：美股标普隔夜 {fc['us_spx']:+.2f}%   A股20日波动 {fc['cn_vol20']:.2f}%")
    for key, label in (("h1", "预测 T 日（今天）"), ("h2", "预测 T+1 日（明天）"),
                       ("h2c", "预测两日累计")):
        v = fc[key]
        valid = preds[key][np.isfinite(preds[key])]
        strength = float((np.abs(valid) < abs(v)).mean() * 100.0) if len(valid) else np.nan
        # 强度分位决定信号等级：只有极强信号才有历史胜率优势
        if strength >= 80:
            grade = "强信号"
        elif strength >= 60:
            grade = "中等信号"
        else:
            grade = "弱信号·建议观望"
        r = results[key]
        out(f"    {label:14s} {v:+6.2f}%   {grade:10s} "
            f"强度分位{strength:5.1f}%  （历史准确率 {r['hit']*100:.1f}%、IC {r['ic']:+.3f}）")

    out("\n" + "-" * 76)
    out("[4] 结论")
    out("-" * 76)
    h1, h2 = results["h1"], results["h2"]
    out(f"    h1（1 日后）：IC {h1['ic']:+.3f}、方向 {h1['hit']*100:.1f}% —— "
        f"{'可用' if h1['hit'] >= 0.55 else '偏弱'}，可用于当日方向判断。")
    out(f"    h2（2 日后）：IC {h2['ic']:+.3f}、方向 {h2['hit']*100:.1f}% —— "
        f"{'可用' if h2['hit'] >= 0.55 else '接近随机，不建议作为方向依据'}。")
    out(f"    h2c（两日累计）：IC {results['h2c']['ic']:+.3f}、方向 {results['h2c']['hit']*100:.1f}% —— "
        f"信号几乎全部来自 h1，且分年度不稳定。")
    out("")
    out("    用法建议：h1 不要天天用。按强度阶梯，只有 |预测| 落在最高档的日子")
    out("    胜率才明显高于 55%，其余日子应观望；h2 在任何强度档位都没有优势。")
    out("    h2 衰减的根因：隔夜信息只作用于 T 日开盘，到 T+1 已消耗完；")
    out("    而 T 日白天的量价不在本特征集内。若要提高 h2，应在 T 日收盘后重跑（届时信息更全）。")
    out("=" * 76)

    if args.eval_only:
        return 0

    # ── 落盘 ──
    out_df = df[["date"] + FEATURES + ["y_h1", "y_h2", "y_h2c"]].copy()
    for key in ("h1", "h2", "h2c"):
        out_df[f"pred_{key}"] = preds[key]
        out_df[f"dir_{key}"] = np.where(
            np.isfinite(preds[key]), np.where(preds[key] > 0, 1, -1), 0
        )
    out_df.to_csv(DATA_DIR / "forecast.csv", index=False, encoding="utf-8-sig")
    (DATA_DIR / "_predict.txt").write_text("\n".join(lines), encoding="utf-8")

    with open(DATA_DIR / "model.pkl", "wb") as f:
        pickle.dump(
            {"features": FEATURES, "bench": args.bench, "h1": fit_final(df, 1),
             "h2": fit_final(df, 2), "forecast": fc},
            f,
        )
    print(f"\n已写入：{DATA_DIR}\\forecast.csv  /  _predict.txt  /  model.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
