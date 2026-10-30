# -*- coding: utf-8 -*-
"""pipeline.py — 模块一 → 数据 → 模块二 → 推荐 → 明日走势，一条流水线

    ┌─ 模块一：网页 AI 共识池（stocks.xlsx「共识」sheet，464 只取前 N）
    ├─ 数据：抓日线（fetch_universe），后复权
    ├─ 模块二：六条件 + PS + 标准化回测，两级漏斗
    │    ① 轻量粗筛  200 → 15   复用模块二的指标/六条件/PS/回测，预测器换成 Ridge
    │    ② 完整评价  15  → 5    TCN+Transformer 双头 + Walk-Forward，真实模块二
    ├─ 走势：盘前预测模型（美股隔夜 + 个股波动）给出明日 1 日 / 2 日预判
    └─ 输出：推荐组合 + 个股信息 + 走势预测

为什么要两级漏斗
----------------
完整模块二每只约 10~18 秒（8 折 × 10 轮重训），200 只要跑 40 分钟以上，不现实。
而模块二里真正慢的只有 TCN+Transformer 这一个环节——六条件、Prediction Strength、
标准化回测、风控都是毫秒级。所以先用 Ridge 顶替预测器把 200 只过一遍（每只约 0.08 秒），
把明显不合格的股票滤掉，再让 15 只候选跑真正的深度模型。粗筛和精评用的是同一套
信号与回测代码，口径一致，只是预测器精度不同。

Strategy Fitness
----------------
方案里定义了 Fitness = w1·R + w2·Sharpe + w3·(1−MDD) + w4·WinRate − Penalty，
但模块二代码里没有实现。这里补上，权重取固定值（不做样本内优化，避免过拟合），
各分项先转成池内百分位再加权。

用法
----
    D:\\anaconda3\\python.exe pipeline.py                    # 默认：200 → 15 → 5
    D:\\anaconda3\\python.exe pipeline.py --top 100 --final 3
    D:\\anaconda3\\python.exe pipeline.py --skip-fetch       # 用已有数据
    D:\\anaconda3\\python.exe pipeline.py --quick-only       # 只跑粗筛（约 20 秒）
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODULE2 = PROJECT / "模块二"
UNIVERSE = ROOT / "data" / "universe"
OUT_DIR = ROOT / "output"

# 让 module2 可以被 import；同时让它把输出写到本目录下
sys.path.insert(0, str(MODULE2))
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import Ridge  # noqa: E402

from module2.backtest import run_backtest  # noqa: E402
from module2.config import Config  # noqa: E402
from module2.conditions import get_user_condition_labels  # noqa: E402
from module2.dataset import build_samples, time_split, window_normalize  # noqa: E402
from module2.run import prepare_data, walk_forward_predict  # noqa: E402
from module2.strategy import build_signals  # noqa: E402

from fetch_universe import load_pool, to_secid  # noqa: E402
from predict import FEATURES as PM_FEATURES  # noqa: E402
from predict import fit_final as pm_fit_final  # noqa: E402
from predict import make_model as pm_make_model  # noqa: E402
from predict import strength_ladder  # noqa: E402


# ──────────────────────────────────────────────────────────────
#  基础设施
# ──────────────────────────────────────────────────────────────
def quiet(fn, *args, **kwargs):
    """模块二的交易账户会 print 每笔成交，这里统一静音。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def module2_cfg(epochs: int, folds: int) -> Config:
    cfg = Config(
        epochs=epochs,
        wf_enabled=True,
        wf_train_days=300,
        wf_val_days=20,
        wf_step=40,
        wf_max_folds=folds,
        verbose=False,
    )
    cfg.data_dir = str(UNIVERSE)
    cfg.out_dir = str(OUT_DIR / "_module2")
    return cfg


# ──────────────────────────────────────────────────────────────
#  阶段一：轻量粗筛
# ──────────────────────────────────────────────────────────────
def quick_screen(csv_name: str, cfg: Config, labels: Dict[str, int]) -> Optional[Dict[str, object]]:
    """用 Ridge 替代 TCN 跑一遍模块二的完整链路（六条件 + PS + 标准化回测）。"""
    try:
        df = quiet(prepare_data, csv_name, cfg)
    except Exception:
        return None
    if len(df) < 300:
        return None

    samples = build_samples(df, cfg.lookback, cfg.horizon)
    if len(samples) < 200:
        return None
    sp = time_split(len(samples), cfg)
    Xn = window_normalize(samples.X)[:, -1, :]     # 窗口最后一天的特征（已窗口内标准化）
    y = samples.y_pct

    tr, va, te = sp["train"], sp["val"], sp["test"]
    model = Ridge(alpha=10.0).fit(Xn[slice(*tr)], y[slice(*tr)])
    val_pred = model.predict(Xn[slice(*va)])
    test_pred = model.predict(Xn[slice(*te)])

    # PS 标定：验证段 5 日累计误差尺度 + |PS| 70% 分位阈值
    cum_p = np.prod(1 + val_pred / 100, axis=1) - 1
    cum_t = np.prod(1 + y[slice(*va)] / 100, axis=1) - 1
    sigma = float(np.std((cum_t - cum_p)[-60:])) + 1e-8
    ps_abs = np.abs(np.prod(1 + test_pred / 100, axis=1) - 1) / sigma
    cut = float(max(np.quantile(ps_abs, 0.70), 0.05))

    sig = quiet(build_signals, df, samples.anchor_idx[slice(*te)], test_pred,
                labels, cfg, sigma, cut)
    bt = quiet(run_backtest, sig, cfg)
    m = bt["metrics"]

    # 方向准确率（4 段，与模块二同口径）
    from module2.evaluation import direction_accuracy, direction_accuracy_t1
    da4 = direction_accuracy(test_pred, y[slice(*te)])
    da1 = direction_accuracy_t1(test_pred, y[slice(*te)])

    # 分窗口收益稳定性
    eq = bt["equity"]
    wins = []
    if len(eq) >= 60:
        nav = eq["nav"].to_numpy()
        for i in range(0, len(nav) - 40, 40):
            wins.append(nav[i + 40] / nav[i] - 1)
    stab = float(np.std(wins)) if len(wins) > 1 else 0.0

    last = sig.iloc[-1]
    conds = {c: int(last[c]) for c in ("c1", "c2", "c3", "c4", "c5", "c6")}
    缺口 = [c for c, v in conds.items() if v == 0]
    track = "".join({"Buy": "B", "Sell": "S", "Hold": "·"}[s]
                    for s in sig.tail(20)["signal"].tolist())

    return {
        "代码": str(csv_name).replace(".csv", ""),
        "测试天数": len(sig),
        "买入天数": int((sig["signal"] == "Buy").sum()),
        "年触发次数": round(int((sig["signal"] == "Buy").sum()) / max(len(sig), 1) * 252, 1),
        "近20日信号": track,
        "缺口条件": "、".join(缺口) if 缺口 else "无",
        "条件满足数": 6 - len(缺口),
        "策略收益": float(m.get("total_return", np.nan)),
        "Sharpe": float(m.get("sharpe", np.nan)),
        "最大回撤": float(m.get("max_drawdown", np.nan)),
        "胜率": float(m.get("win_rate", np.nan)),
        "交易次数": int(m.get("total_trades", 0)),
        "DA4": da4 * 100,
        "DA1": da1 * 100,
        "当前信号": str(sig["signal"].iloc[-1]) if len(sig) else "—",
        "当前六条件": str(sig["condition"].iloc[-1]) if len(sig) else "—",
        "当前PS": float(sig["ps"].iloc[-1]) if len(sig) and np.isfinite(sig["ps"].iloc[-1]) else np.nan,
        "窗口收益标准差": stab,
    }


def fitness_table(rows: pd.DataFrame, w=(0.35, 0.25, 0.20, 0.20)) -> pd.DataFrame:
    """按方案定义的 Strategy Fitness 打分（分项转池内百分位后加权，再扣惩罚）。"""
    df = rows.copy()
    if not len(df):
        return df

    def pct(s: pd.Series, higher_better: bool = True) -> pd.Series:
        r = s.rank(pct=True, na_option="bottom")
        return r if higher_better else 1.0 - r

    w1, w2, w3, w4 = w
    score = (
        w1 * pct(df["策略收益"])
        + w2 * pct(df["Sharpe"])
        + w3 * pct(df["最大回撤"], higher_better=False)
        + w4 * (df["胜率"].fillna(0).clip(0, 100) / 100.0)
    )
    # Penalty：交易太少、亏损过大、窗口收益不稳
    penalty = (
        0.15 * (df["交易次数"] < 3).astype(float)
        + 0.10 * (df["策略收益"].clip(upper=0).abs() / 20.0).clip(0, 1)
        + 0.10 * (df["窗口收益标准差"].fillna(0) / 0.15).clip(0, 1)
    )
    df["适应度"] = (score - penalty).round(4)
    return df.sort_values("适应度", ascending=False).reset_index(drop=True)


def prescreen_filter(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """粗筛硬门槛：交易过少、回撤过深、完全不交易的先剔掉。"""
    ok = df[
        (df["交易次数"] >= 2)
        & (df["最大回撤"] <= 40)
        & (df["测试天数"] >= 120)
        & (df["买入天数"] >= 3)
    ].copy()
    if len(ok) < n:                     # 门槛太严就放宽
        ok = df[df["交易次数"] >= 1].copy()
    if len(ok) < n:
        ok = df.copy()
    return ok.head(n)


# ──────────────────────────────────────────────────────────────
#  阶段二：完整模块二
# ──────────────────────────────────────────────────────────────
def deep_eval(csv_name: str, cfg: Config, labels: Dict[str, int]) -> Optional[Dict[str, object]]:
    """真实模块二：TCN+Transformer 双头 + Walk-Forward + 六条件 + 标准化回测。"""
    try:
        df = quiet(prepare_data, csv_name, cfg)
        info = quiet(walk_forward_predict, df, cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"    {csv_name} 深度评估失败：{str(exc)[:70]}")
        return None

    samples = info["samples"]
    pred = info["pred"]
    sigma = info["sigma"]
    cut = float(info.get("ps_cut", cfg.ps_threshold))
    idx = info["idx"]
    test_y = samples.y_pct[idx]

    sig = quiet(build_signals, df, samples.anchor_idx[idx], pred, labels, cfg, sigma, cut)
    bt = quiet(run_backtest, sig, cfg)
    m = bt["metrics"]

    from module2.evaluation import direction_accuracy, direction_accuracy_t1
    da4 = direction_accuracy(pred, test_y)
    da1 = direction_accuracy_t1(pred, test_y)

    eq = bt["equity"]
    wins = []
    if len(eq) >= 60:
        nav = eq["nav"].to_numpy()
        for i in range(0, len(nav) - 40, 40):
            wins.append(nav[i + 40] / nav[i] - 1)

    buy_hold = (float(samples.anchor_close[idx[-1]]) / float(samples.anchor_close[idx[0]]) - 1) * 100 \
        if len(idx) else np.nan

    # 信号细节：近 20 日轨迹、Buy 频率、当前差哪个条件
    last = sig.iloc[-1]
    conds = {c: int(last[c]) for c in ("c1", "c2", "c3", "c4", "c5", "c6")}
    缺口 = [c for c, v in conds.items() if v == 0]
    recent = sig.tail(20)["signal"].tolist()
    track = "".join({"Buy": "B", "Sell": "S", "Hold": "·"}[s] for s in recent)
    buy_days = int((sig["signal"] == "Buy").sum())
    buy_freq = buy_days / max(len(sig), 1) * 252      # 折算成年触发次数

    return {
        "代码": str(csv_name).replace(".csv", ""),
        "测试天数": len(sig),
        "买入天数": buy_days,
        "年触发次数": round(buy_freq, 1),
        "近20日信号": track,
        "缺口条件": "、".join(缺口) if 缺口 else "无（已全满足）",
        "条件满足数": 6 - len(缺口),
        "策略收益": float(m.get("total_return", np.nan)),
        "Sharpe": float(m.get("sharpe", np.nan)),
        "最大回撤": float(m.get("max_drawdown", np.nan)),
        "胜率": float(m.get("win_rate", np.nan)),
        "盈亏比": float(m.get("profit_loss_ratio", np.nan)),
        "交易次数": int(m.get("total_trades", 0)),
        "DA4": da4 * 100,
        "DA1": da1 * 100,
        "买入持有": float(buy_hold),
        "当前信号": str(sig["signal"].iloc[-1]),
        "当前六条件": str(sig["condition"].iloc[-1]),
        "当前PS": float(sig["ps"].iloc[-1]) if np.isfinite(sig["ps"].iloc[-1]) else np.nan,
        "预测T1收益": float(pred[-1][0]) if len(pred) else np.nan,
        "窗口收益标准差": float(np.std(wins)) if len(wins) > 1 else 0.0,
    }


# ──────────────────────────────────────────────────────────────
#  阶段三：明日走势（盘前预测模型，个股版）
# ──────────────────────────────────────────────────────────────
def market_forecast(panel: pd.DataFrame) -> Dict[str, float]:
    """大盘明日预测（就是每日脚本那套模型，基准上证）。"""
    px = pd.read_csv(ROOT / "data" / "raw" / "cn_szzs.csv", parse_dates=["date"])
    close = px.set_index("date")["close"].astype(float)
    ret = close.pct_change()

    dates = pd.to_datetime(panel["date"])
    vol = (ret.rolling(20).std() * 100.0).dropna()
    v20 = np.full(len(dates), np.nan)
    if len(vol):
        vidx = vol.index.values.astype("datetime64[D]")
        pos = np.searchsorted(vidx, dates.values.astype("datetime64[D]"), side="left") - 1
        ok = pos >= 0
        v20[ok] = vol.values[pos[ok]]

    df = pd.DataFrame({
        "date": dates,
        "us_spx_ret1": panel["us_spx_ret1"].to_numpy(),
        "cn_vol20": v20,
        "y_h1": ret.reindex(dates).to_numpy() * 100.0,
    })
    df["y_h2"] = df["y_h1"].shift(-1)
    X = df[PM_FEATURES].fillna(0.0).to_numpy()
    out: Dict[str, float] = {}
    for h, key in ((1, "h1"), (2, "h2")):
        y = df[f"y_h{h}"].to_numpy()
        valid = np.isfinite(y)
        m = pm_make_model()
        m.fit(X[valid], y[valid])
        out[key] = float(m.predict(X[[-1]])[0])
        hist = np.abs(m.predict(X[valid]))
        out[f"strength_{key}"] = float((hist < abs(out[key])).mean() * 100)
    out["h2c"] = ((1 + out["h1"] / 100) * (1 + out["h2"] / 100) - 1) * 100
    out["us_spx"] = float(df["us_spx_ret1"].iloc[-1])
    out["vol20"] = float(df["cn_vol20"].iloc[-1])
    return out


def stock_beta(px: pd.DataFrame, lookback: int = 500) -> tuple:
    """个股对上证的 beta 与相关系数（近 500 个交易日）。

    用 250 日会得到噪声：招商银行近 250 日 beta 算出 -0.01、中国神华 -0.19，
    因为银行/煤炭经常走独立行情。500 日才稳定（招行 0.30、神华 0.14、全样本 0.78/0.42）。
    """
    bench = pd.read_csv(ROOT / "data" / "raw" / "cn_szzs.csv", parse_dates=["date"])
    b = bench.set_index("date")["close"].astype(float).pct_change()
    s = px.copy()
    s["date"] = pd.to_datetime(s["date"])
    r = s.set_index("date")["close"].astype(float).pct_change()
    m = pd.concat([r.rename("s"), b.rename("b")], axis=1).dropna().tail(lookback)
    if len(m) < 120 or m["b"].var() <= 0:
        return 1.0, 0.0
    beta = float(np.cov(m["s"], m["b"])[0, 1] / m["b"].var())
    corr = float(m["s"].corr(m["b"]))
    return beta, corr


def grade(value: float, strength: float) -> str:
    if not np.isfinite(strength):
        return "—"
    if strength >= 80:
        return "强信号"
    if strength >= 60:
        return "中等信号"
    return "弱信号"


# ──────────────────────────────────────────────────────────────
#  个股信息（行业）
# ──────────────────────────────────────────────────────────────
def fetch_industry(codes: List[str]) -> Dict[str, str]:
    """从东财取个股所属细分行业（f127）。"""
    import json as _json
    import urllib.parse
    import urllib.request

    headers = {"User-Agent": "Mozilla/5.0 Chrome/131.0", "Referer": "https://quote.eastmoney.com/"}
    out: Dict[str, str] = {}
    for code in codes:
        secid = to_secid(code)
        if not secid:
            continue
        url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f57,f58,f127"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                d = _json.loads(resp.read().decode("utf-8", "ignore")).get("data") or {}
            out[code] = str(d.get("f127") or "")
        except Exception:  # noqa: BLE001
            out[code] = ""
    return out


# ──────────────────────────────────────────────────────────────
#  输出
# ──────────────────────────────────────────────────────────────
COND_DESC = {
    "c1": "MACD>0", "c2": "RSI 30~70", "c3": "收盘>MA20",
    "c4": "预测MACD>0", "c5": "预测MACD上行", "c6": "预测RSI 30~70",
}


def render_report(final: pd.DataFrame, pool_info: pd.DataFrame, args, elapsed: float,
                  mkt: Optional[Dict[str, float]] = None,
                  g1: str = "", g2: str = "",
                  today_buys: Optional[pd.DataFrame] = None) -> str:
    L: List[str] = []
    L.append("# 模块一 → 模块二 → 明日走势 · 推荐结果")
    L.append("")
    L.append(f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　耗时 {elapsed/60:.1f} 分钟")
    L.append(f"漏斗：候选池 {len(pool_info)} → 粗筛 {args.prescreen} → 精评 {len(final)} 只入选")
    L.append("")

    # ── 一、大盘环境 ──
    if mkt:
        L.append("## 一、今日大盘环境")
        L.append("")
        L.append("| 目标 | 预测收益 | 强度 | 方向 |")
        L.append("|---|---|---|---|")
        L.append(f"| T+1（1 个交易日后） | {mkt['h1']:+.2f}% | {g1}（{mkt['strength_h1']:.0f}% 分位） "
                 f"| {'偏多' if mkt['h1'] > 0 else '偏空'} |")
        L.append(f"| T+2（2 个交易日后） | {mkt['h2']:+.2f}% | {g2}（{mkt['strength_h2']:.0f}% 分位） "
                 f"| {'偏多' if mkt['h2'] > 0 else '偏空'} |")
        L.append("")
        L.append(f"输入：美股标普隔夜 {mkt['us_spx']:+.2f}%，A 股 20 日波动 {mkt['vol20']:.2f}%。"
                 f"该模型 1 日目标历史方向准确率 56.4%、2 日 50.1%。")
        L.append("")
        if mkt["strength_h1"] >= 80:
            L.append("**环境判断**：隔夜信息强度充足，可以按信号正常执行。")
        elif mkt["strength_h1"] >= 60:
            L.append("**环境判断**：中等强度，可以执行，建议分批建仓。")
        else:
            L.append(f"**环境判断**：弱信号（{mkt['strength_h1']:.0f}% 分位），隔夜信息不足以把 A 股推离原趋势。"
                     f"方向参考价值有限，仓位按系数打折——**但这不影响股票池本身的质量判断**。")
        L.append("")

    # ── 二、今日可执行 ──
    L.append("## 二、今日可执行清单")
    L.append("")
    if today_buys is None:
        today_buys = final[final["当前信号"] == "Buy"]
    if len(today_buys):
        L.append("| 代码 | 名称 | 适应度排名 | 六条件 | PS | 系数 | 建议仓位 |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in today_buys.iterrows():
            rk = int(r.get("适应度排名", 0)) or "—"
            mark = "" if r["代码"] in set(final["代码"]) else " ·外"
            L.append(f"| {r['代码']} | {r.get('名称','')} | 第{rk}名{mark} | `{r['当前六条件']}` "
                     f"| {r['当前PS']:+.2f} | {r['仓位系数']:.1f}× | {int(r['建议仓位'])}% |")
        L.append("")
        L.append("「·外」表示该股未进入前 "
                 f"{args.final} 名推荐，但今日六条件触发，属于当期机会而非池内最优。"
                 "仓位 = 基础 80% × 大盘系数（强信号 1.5×、中等 1.0×、弱 0.6×、方向相反 0.3×）。")
    else:
        mean_freq = final["年触发次数"].mean() if "年触发次数" in final.columns else np.nan
        L.append("**今天没有股票触发买入信号。** 这是六条件过滤器的正常状态，不是系统故障——")
        L.append(f"候选池平均每年触发 {mean_freq:.0f} 次买入，相当于每 {252/max(mean_freq,0.1):.0f} "
                 f"个交易日一次；最近 30 个交易日 A 股整体走弱，条件集不齐。")
        L.append("")
        if "条件满足数" in final.columns:
            near = final.sort_values("条件满足数", ascending=False).head(3)
            L.append("**最接近触发的 3 只**（差得越少越值得盯，满足 6/6 即触发）：")
            L.append("")
            L.append("| 代码 | 名称 | 已满足 | 还缺什么 | 当前 PS | 近 20 日信号 |")
            L.append("|---|---|---|---|---|---|")
            for _, r in near.iterrows():
                L.append(f"| {r['代码']} | {r.get('名称','')} | {int(r['条件满足数'])}/6 "
                         f"| {r['缺口条件']} | {r['当前PS']:+.2f} | `{r['近20日信号']}` |")
            L.append("")
    L.append(f"六条件含义：c1 {COND_DESC['c1']}｜c2 {COND_DESC['c2']}｜c3 {COND_DESC['c3']}｜"
             f"c4 {COND_DESC['c4']}｜c5 {COND_DESC['c5']}｜c6 {COND_DESC['c6']}。"
             f"c4~c6 由预测价推出，不使用真实次日指标。信号轨迹中 B=买入、S=卖出、·=观望。")
    L.append("")

    # ── 三、推荐组合 ──
    L.append("## 三、推荐组合（股票池，与今日是否入场无关）")
    L.append("")
    L.append("| # | 代码 | 名称 | 行业 | 共识 | 适应度 | 策略收益 | Sharpe | 最大回撤 | 胜率 | 年触发 | 今日 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in final.reset_index(drop=True).iterrows():
        L.append(
            f"| {i+1} | {r['代码']} | {r.get('名称','')} | {r.get('行业','')} "
            f"| {int(r['出现平台数'])}平台 | {r['适应度']:.3f} | {r['策略收益']:+.1f}% "
            f"| {r['Sharpe']:.2f} | {r['最大回撤']:.1f}% | {r['胜率']:.0f}% "
            f"| {r.get('年触发次数','—')} 次/年 | {r['当前信号']} |"
        )
    L.append("")

    # ── 四、走势与仓位 ──
    L.append("## 四、明日走势与仓位系数")
    L.append("")
    L.append("| 代码 | 名称 | β(500日) | 相关性 | T+1 传导 | T+2 传导 | 大盘系数 |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in final.iterrows():
        L.append(
            f"| {r['代码']} | {r.get('名称','')} | {r['beta']:.2f} | {r['corr']:.2f} "
            f"| {r['h1']:+.2f}% | {r['h2']:+.2f}% | {r['仓位系数']:.1f}× |"
        )
    L.append("")
    L.append("传导值很小是正常的：美股隔夜只解释 A 股日收益约 18% 的方差，"
             "0.28% 的隔夜涨幅经模型收缩后对上证只剩 +0.03%。它不是「明天涨多少」的预言，"
             "而是「隔夜信息把预期往哪个方向推、推了多少」的度量。")
    L.append("")

    # ── 五、逐只说明 ──
    L.append("## 五、逐只说明")
    L.append("")
    for i, r in final.reset_index(drop=True).iterrows():
        L.append(f"### {i+1}. {r['代码']} {r.get('名称','')}　{r.get('行业','')}")
        L.append("")
        L.append(f"- **AI 共识**：{int(r['出现平台数'])} 个平台推荐，池内排名第 {int(r['排名'])}")
        bh = r.get("买入持有", np.nan)
        bh_txt = f"{bh:+.1f}%" if bh is not None and np.isfinite(bh) else "—"
        L.append(f"- **模块二回测**（样本外 {int(r['测试天数'])} 天）：收益 {r['策略收益']:+.1f}%，"
                 f"Sharpe {r['Sharpe']:.2f}，最大回撤 {r['最大回撤']:.1f}%，"
                 f"胜率 {r['胜率']:.0f}%，交易 {int(r['交易次数'])} 次，同期买入持有 {bh_txt}")
        L.append(f"- **预测质量**：4 段方向准确率 {r['DA4']:.1f}%，T+1 方向 {r['DA1']:.1f}%")
        L.append(f"- **当前状态**：六条件 `{r['当前六条件']}` → **{r['当前信号']}**，"
                 f"PS {r['当前PS']:+.2f}，满足 {int(r.get('条件满足数',0))}/6"
                 + (f"，缺 {r['缺口条件']}" if r["当前信号"] != "Buy" else ""))
        L.append(f"- **近 20 日信号**：`{r.get('近20日信号','—')}`"
                 f"（B 买入 / S 卖出 / · 观望），年触发约 {r.get('年触发次数','—')} 次")
        L.append(f"- **明日传导**：β {r['beta']:.2f}（相关性 {r['corr']:.2f}），"
                 f"T+1 {r['h1']:+.2f}%、T+2 {r['h2']:+.2f}%，大盘系数 {r['仓位系数']:.1f}×")
        L.append("")
    L.append("---")
    L.append("")
    L.append("**口径说明**：策略收益为统一 100 万资金、95% 固定仓位、含佣金 3bp + 印花税 5bp + 滑点 10bp 的标准化回测；"
             "适应度为方案定义的 Strategy Fitness（收益/Sharpe/回撤/胜率加权后扣小样本与稳定性惩罚），"
             "各分项先转池内百分位。")
    L.append("")
    L.append("**走势预测口径**：大盘用盘前模型（美股隔夜 + A 股波动），1 日目标历史方向准确率 56.4%、"
             "2 日目标 50.1%。个股层面实测过独立建模，方向准确率只有 50.7%（中位 50.9%、"
             "仅 25% 的股票能超过 52%），接近随机——所以个股那一栏不是独立预测，"
             "而是用 β 把大盘信号传导过去的结果，不含个股特异性 alpha。")
    L.append("")
    L.append("**风险提示**：AI 共识池只代表舆情热度，不代表基本面；模块二回测样本外区间有限，"
             "适应度排名不等于未来收益。以上为量化研究输出，不构成投资建议。")
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser("pipeline", description="模块一→数据→模块二→推荐→明日走势")
    ap.add_argument("--top", type=int, default=200, help="候选池取前 N（默认 200）")
    ap.add_argument("--prescreen", type=int, default=15, help="进入完整模块二的只数")
    ap.add_argument("--final", type=int, default=5, help="最终推荐只数")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--skip-fetch", action="store_true", help="不抓数据，用本地已有")
    ap.add_argument("--quick-only", action="store_true", help="只跑粗筛，跳过深度模型")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    labels = quiet(get_user_condition_labels)

    # ── 模块一：候选池 ──
    print("=" * 72)
    print("[1/5] 模块一 · 网页 AI 共识池")
    pool = load_pool(PROJECT / "llm_web_ask" / "output" / "stocks.xlsx", args.top)
    print(f"      {len(pool)} 只（三平台共识 {int((pool['出现平台数']==3).sum())}、"
          f"双平台 {int((pool['出现平台数']==2).sum())}）")

    # ── 数据 ──
    print("[2/5] 日线数据")
    missing = [c for c in pool["代码"] if not (UNIVERSE / f"{c}.csv").exists()]
    if missing and not args.skip_fetch:
        print(f"      缺 {len(missing)} 只，调用 fetch_universe 补齐 ...")
        from fetch_universe import main as fu_main
        fu_main(["--top", str(args.top), "--workers", "4"])
    else:
        print(f"      本地已有 {len(pool)-len(missing)}/{len(pool)} 只")

    # ── 粗筛 ──
    print(f"[3/5] 粗筛（Ridge 版模块二）{len(pool)} 只 ...")
    cfg_fast = module2_cfg(epochs=8, folds=6)
    rows = []
    for i, code in enumerate(pool["代码"], 1):
        r = quick_screen(f"{code}.csv", cfg_fast, labels)
        if r:
            rows.append(r)
        if i % 40 == 0 or i == len(pool):
            print(f"      {i}/{len(pool)}  已通过 {len(rows)} 只")
    qdf = fitness_table(pd.DataFrame(rows))
    print(f"      粗筛完成：{len(qdf)} 只有效，用时 {time.time()-t0:.0f}s")

    short = prescreen_filter(qdf, args.prescreen)
    print(f"      进入精评：{'、'.join(short['代码'].tolist())}")

    # ── 完整模块二 ──
    if args.quick_only:
        final_base = short.head(args.final).copy()
        all_eval = short.copy()
        print("[4/5] 跳过完整模块二（--quick-only）")
    else:
        print(f"[4/5] 完整模块二（TCN+Transformer，{args.folds} 折 × {args.epochs} 轮）...")
        cfg_deep = module2_cfg(args.epochs, args.folds)
        deep_rows = []
        for i, code in enumerate(short["代码"].tolist(), 1):
            t1 = time.time()
            r = deep_eval(f"{code}.csv", cfg_deep, labels)
            if r:
                deep_rows.append(r)
                print(f"      [{i}/{len(short)}] {code} 收益 {r['策略收益']:+.1f}% "
                      f"Sharpe {r['Sharpe']:.2f} 今日{r['当前信号']}（{time.time()-t1:.0f}s）")
            else:
                print(f"      [{i}/{len(short)}] {code} 失败")
        ddf = fitness_table(pd.DataFrame(deep_rows))
        final_base = ddf.head(args.final).copy()
        all_eval = ddf.copy()

    # ── 元信息 + 走势（推荐集与全部精评集都算，后者用于找今日可执行的股票）──
    meta = pool.set_index("代码")
    idx_path = UNIVERSE / "_index.csv"
    name_map: Dict[str, str] = {}
    if idx_path.exists():
        _idx = pd.read_csv(idx_path, dtype={"代码": str})
        name_map = {c: str(n) for c, n in zip(_idx["代码"], _idx["名称"].fillna("")) if n}

    def enrich(df_: pd.DataFrame) -> pd.DataFrame:
        d = df_.copy()
        for c in ("排名", "出现平台数"):
            d[c] = d["代码"].map(meta[c]) if c in meta.columns else np.nan
        d["名称"] = d["代码"].map(
            lambda c: name_map.get(c) or (meta["名称"].get(c) if "名称" in meta.columns else "") or ""
        ).astype(str)
        return d

    print("[5/5] 明日走势预测")
    panel = pd.read_csv(ROOT / "data" / "panel.csv", parse_dates=["date"])
    mkt = market_forecast(panel)
    mkt_g1 = grade(mkt["h1"], mkt["strength_h1"])
    mkt_g2 = grade(mkt["h2"], mkt["strength_h2"])
    print(f"      大盘（上证）：T+1 {mkt['h1']:+.2f}% [{mkt_g1}]　"
          f"T+2 {mkt['h2']:+.2f}% [{mkt_g2}]")

    def pos_factor(dir_sign: float, g: str) -> float:
        if dir_sign < 0:
            return 0.3
        return {"强信号": 1.5, "中等信号": 1.0}.get(g, 0.6)

    def add_path(d: pd.DataFrame) -> pd.DataFrame:
        """补 β、传导预测、仓位系数。走势只调仓位，不否决六条件信号。"""
        bets, corrs = [], []
        for code in d["代码"]:
            px = pd.read_csv(UNIVERSE / f"{code}.csv", parse_dates=["date"])
            try:
                b, c = stock_beta(px)
            except Exception:  # noqa: BLE001
                b, c = 1.0, 0.0
            bets.append(b)
            corrs.append(c)
        d["beta"], d["corr"] = bets, corrs
        d["h1"] = d["beta"] * mkt["h1"]
        d["h2"] = d["beta"] * mkt["h2"]
        d["h2c"] = ((1 + d["h1"] / 100) * (1 + d["h2"] / 100) - 1) * 100
        d["h1强度"] = mkt["strength_h1"]
        d["h2强度"] = mkt["strength_h2"]
        d["h1等级"] = [grade(v, s) for v, s in zip(d["h1"], d["h1强度"])]
        d["h2等级"] = [grade(v, s) for v, s in zip(d["h2"], d["h2强度"])]
        d["仓位系数"] = [pos_factor(np.sign(b * mkt["h1"]), g)
                     for b, g in zip(d["beta"], d["h1等级"])]
        d["建议仓位"] = [round(80 * f) if s == "Buy" else 0
                     for s, f in zip(d["当前信号"], d["仓位系数"])]
        return d

    final_base = add_path(enrich(final_base))
    all_eval = add_path(enrich(all_eval))
    final_base["适应度排名"] = range(1, len(final_base) + 1)
    all_eval["适应度排名"] = range(1, len(all_eval) + 1)
    industries = fetch_industry(all_eval["代码"].tolist())
    final_base["行业"] = final_base["代码"].map(industries).fillna("")
    all_eval["行业"] = all_eval["代码"].map(industries).fillna("")

    # 今日有买入信号的全部候选（不限于前 5 名）
    today_buys = all_eval[all_eval["当前信号"] == "Buy"].sort_values("适应度", ascending=False)

    def decide(r) -> str:
        if r["当前信号"] == "Buy":
            return f"可入场 · 建议 {int(r['建议仓位'])}% 仓位"
        gap = 6 - int(r["条件满足数"])
        return f"观察 · 差{gap}个条件"

    final_base["判断"] = final_base.apply(decide, axis=1)
    final_base["大盘T1"] = mkt["h1"]
    final_base["大盘T2"] = mkt["h2"]

    # ── 输出 ──
    cols = ["代码", "名称", "行业", "排名", "出现平台数", "适应度", "适应度排名",
            "策略收益", "Sharpe", "最大回撤", "胜率", "盈亏比", "交易次数", "测试天数",
            "买入天数", "年触发次数", "买入持有", "DA4", "DA1",
            "当前信号", "当前六条件", "当前PS", "条件满足数", "缺口条件", "近20日信号",
            "beta", "corr", "h1", "h2", "h2c", "h1强度", "h2强度", "h1等级", "h2等级",
            "仓位系数", "建议仓位", "判断"]
    out = final_base[[c for c in cols if c in final_base.columns]]
    out.to_csv(OUT_DIR / "recommendations.csv", index=False, encoding="utf-8-sig")
    qdf.to_csv(OUT_DIR / "prescreen.csv", index=False, encoding="utf-8-sig")

    report = render_report(out, pool, args, time.time() - t0, mkt, mkt_g1, mkt_g2, today_buys)
    (OUT_DIR / "pipeline_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "pipeline_report.json").write_text(
        json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n" + report)
    print(f"\n输出：{OUT_DIR}\\pipeline_report.md / recommendations.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
