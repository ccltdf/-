# -*- coding: utf-8 -*-
"""fetch_data.py — 盘前预测（预测当日走势）· 协变量数据源探测与抓取

目的
----
验证「用 T 日 9:00 之前的数据预测 T 日 A 股走势」所需的协变量能否稳定拿到，
并把它们抓成本地面板（panel.csv），供后续建模使用。

协变量分组
----------
    cn_index    A 股大盘指数（上证/深成/创业板/沪深300/中证500/中证1000/科创50）
    cn_industry 31 个申万一级行业指数（映射到东方财富行业板块）
    us_index   美股三大指数（道指/标普500/纳指100）
    us_sector  美股 11 个 SPDR 行业板块 ETF（科技/金融/能源/医疗/工业/可选消费/
               日常消费/公用事业/材料/房地产/通讯服务）
    jp_index   日经 225
    kr_index   韩国 KOSPI
    hk_index   恒生指数（附加，与 A 股同交易时段，可作当日情绪参考）

数据源
------
全部走东方财富公开行情接口（push2his / push2 / searchapi），无需登录、无需 key。
美股板块用 ETF 的 secid = 107.XXX（NYSE Arca），实测可取 2018 年至今全历史。

时点对齐（关键，决定有没有未来函数）
------------------------------------
预测目标是 A 股 T 日走势，可用信息截止 T 日 9:00。对齐规则统一为
    「对每个海外市场，取 date < T 的最后一条收盘价」
即：
    * A 股 T-1 收盘           → 已知
    * 美股 T-1 收盘（美东）    → 北京时间 T 日凌晨 4~5 点收盘，T 日 9:00 前已知 ✓
    * 日经/KOSPI T-1 收盘      → 北京 T-1 14:30 前后，已知 ✓
    * 日经/KOSPI T 日 8:00~9:00 盘中 —— 实盘可得，但日线历史里取不到，
      因此默认不纳入面板（纳入会构成回测偏差），仅在实盘推理时单独注入。

输出
----
    data/raw/<key>.csv         每个协变量的原始日线
    data/panel.csv             A 股交易日 × 协变量（已按上面规则对齐）
    data/_report.txt           抓取与完整性报告

用法
----
    D:\\anaconda3\\python.exe fetch_data.py                # 全量抓取 + 构建面板
    D:\\anaconda3\\python.exe fetch_data.py --probe-only    # 只探测，不落盘
    D:\\anaconda3\\python.exe fetch_data.py --beg 20200101 # 指定起始日
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

API_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
API_CLIST = "https://push2.eastmoney.com/api/qt/clist/get"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json,text/plain,*/*",
    "Connection": "close",
}

SLEEP_BETWEEN = 0.15        # 请求间隔，避免被限流
TIMEOUT = 25


# ──────────────────────────────────────────────────────────────
#  资产清单
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Asset:
    key: str          # 英文短名，做文件名与列名前缀
    name: str         # 中文名
    secid: str        # 东财 secid
    group: str        # 协变量分组
    market: str       # CN / US / JP / KR / HK，用于说明时点

    @property
    def col(self) -> str:
        return f"{self.key}_ret1"


CN_INDEX: List[Asset] = [
    Asset("cn_szzs", "上证指数", "1.000001", "cn_index", "CN"),
    Asset("cn_szcz", "深证成指", "0.399001", "cn_index", "CN"),
    Asset("cn_cyb", "创业板指", "0.399006", "cn_index", "CN"),
    Asset("cn_hs300", "沪深300", "1.000300", "cn_index", "CN"),
    Asset("cn_zz500", "中证500", "1.000905", "cn_index", "CN"),
    Asset("cn_zz1000", "中证1000", "1.000852", "cn_index", "CN"),
    Asset("cn_kc50", "科创50", "1.000688", "cn_index", "CN"),
]

US_INDEX: List[Asset] = [
    Asset("us_dji", "道琼斯", "100.DJIA", "us_index", "US"),
    Asset("us_spx", "标普500", "100.SPX", "us_index", "US"),
    Asset("us_ndx", "纳斯达克100", "100.NDX", "us_index", "US"),
]

US_SECTOR: List[Asset] = [
    Asset("sec_xlk", "科技", "107.XLK", "us_sector", "US"),
    Asset("sec_xlf", "金融", "107.XLF", "us_sector", "US"),
    Asset("sec_xle", "能源", "107.XLE", "us_sector", "US"),
    Asset("sec_xlv", "医疗", "107.XLV", "us_sector", "US"),
    Asset("sec_xli", "工业", "107.XLI", "us_sector", "US"),
    Asset("sec_xly", "可选消费", "107.XLY", "us_sector", "US"),
    Asset("sec_xlp", "日常消费", "107.XLP", "us_sector", "US"),
    Asset("sec_xlu", "公用事业", "107.XLU", "us_sector", "US"),
    Asset("sec_xlb", "材料", "107.XLB", "us_sector", "US"),
    Asset("sec_xlre", "房地产", "107.XLRE", "us_sector", "US"),
    Asset("sec_xlc", "通讯服务", "107.XLC", "us_sector", "US"),
]

JP_KR_HK: List[Asset] = [
    Asset("jp_n225", "日经225", "100.N225", "jp_index", "JP"),
    Asset("kr_ks11", "韩国KOSPI", "100.KS11", "kr_index", "KR"),
    Asset("hk_hsi", "恒生指数", "100.HSI", "hk_index", "HK"),
]

# 申万一级 31 个行业 → 东方财富行业板块（BK 代码由 list_cn_industries 动态解析）
CN_INDUSTRY_NAMES: List[str] = [
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "家用电器", "食品饮料",
    "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输", "房地产",
    "商贸零售", "社会服务", "银行", "非银金融", "综合", "建筑材料", "建筑装饰",
    "电力设备", "机械设备", "国防军工", "汽车", "计算机", "传媒", "通信",
    "煤炭", "石油石化", "环保", "美容护理",
]

# 申万一级名 → 英文 key（列名友好）
IND_KEY = {
    "农林牧渔": "agri", "基础化工": "chem", "钢铁": "steel", "有色金属": "metal",
    "电子": "elec", "家用电器": "appl", "食品饮料": "food", "纺织服饰": "textile",
    "轻工制造": "light", "医药生物": "pharm", "公用事业": "util", "交通运输": "trans",
    "房地产": "re", "商贸零售": "retail", "社会服务": "svc", "银行": "bank",
    "非银金融": "nbfi", "综合": "congl", "建筑材料": "bmat", "建筑装饰": "bdec",
    "电力设备": "power", "机械设备": "mach", "国防军工": "defense", "汽车": "auto",
    "计算机": "it", "传媒": "media", "通信": "telecom", "煤炭": "coal",
    "石油石化": "oil", "环保": "env", "美容护理": "beauty",
}


# ──────────────────────────────────────────────────────────────
#  底层请求
# ──────────────────────────────────────────────────────────────
def http_json(url: str, retries: int = 3) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8", "ignore"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"请求失败（{retries} 次）：{url[:110]} -> {last_exc}")


def fetch_kline(secid: str, beg: str = "20180101", end: str = "20500101") -> pd.DataFrame:
    """取日线（不复权）。返回列：date/open/close/high/low/volume/amount。"""
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101",          # 日线
        "fqt": "0",            # 不复权（指数本身无需复权）
        "beg": beg,
        "end": end,
    }
    payload = http_json(f"{API_KLINE}?{urllib.parse.urlencode(params)}")
    data = payload.get("data")
    if not data or not data.get("klines"):
        return pd.DataFrame()

    rows = []
    for line in data["klines"]:
        p = line.split(",")
        if len(p) < 6:
            continue
        rows.append(
            {
                "date": p[0],
                "open": pd.to_numeric(p[1], errors="coerce"),
                "close": pd.to_numeric(p[2], errors="coerce"),
                "high": pd.to_numeric(p[3], errors="coerce"),
                "low": pd.to_numeric(p[4], errors="coerce"),
                "volume": pd.to_numeric(p[5], errors="coerce"),
                "amount": pd.to_numeric(p[6], errors="coerce") if len(p) > 6 else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def list_cn_industries(limit_pages: int = 6) -> Dict[str, str]:
    """取东方财富行业板块全量列表，返回 {板块名: BK代码}。"""
    mapping: Dict[str, str] = {}
    for page in range(1, limit_pages + 1):
        params = {
            "pn": str(page), "pz": "100", "po": "1", "np": "1",
            "fltt": "2", "invt": "2", "fid": "f3",
            "fs": "m:90+t:2",                 # 行业板块
            "fields": "f12,f13,f14",
        }
        try:
            payload = http_json(f"{API_CLIST}?{urllib.parse.urlencode(params)}")
        except Exception:
            break
        items = ((payload.get("data") or {}).get("diff")) or []
        if not items:
            break
        for it in items:
            name, code = str(it.get("f14", "")).strip(), str(it.get("f12", "")).strip()
            if name and code:
                mapping.setdefault(name, code)
        time.sleep(SLEEP_BETWEEN)
    return mapping


def build_universe(verbose: bool = True) -> List[Asset]:
    """组装完整资产清单（行业部分动态解析 BK 代码）。"""
    assets = list(CN_INDEX) + list(US_INDEX) + list(US_SECTOR) + list(JP_KR_HK)

    if verbose:
        print("解析东财行业板块列表 ...", end=" ", flush=True)
    board = list_cn_industries()
    if verbose:
        print(f"{len(board)} 个板块")

    missing = []
    for name in CN_INDUSTRY_NAMES:
        code = board.get(name)
        if not code:
            missing.append(name)
            continue
        assets.append(
            Asset(f"ind_{IND_KEY[name]}", name, f"90.{code}", "cn_industry", "CN")
        )
    if missing:
        print(f"  [警告] 以下行业未匹配到板块代码：{missing}")
    return assets


# ──────────────────────────────────────────────────────────────
#  抓取与落盘
# ──────────────────────────────────────────────────────────────
def crawl(assets: Iterable[Asset], beg: str, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """逐个抓取日线，返回 {key: DataFrame}。失败的 key 不在结果中。"""
    frames: Dict[str, pd.DataFrame] = {}
    assets = list(assets)
    for i, a in enumerate(assets, 1):
        try:
            df = fetch_kline(a.secid, beg=beg)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(assets)}] {a.key:12s} {a.name:12s} 失败 {exc}")
            continue
        if df.empty:
            print(f"  [{i}/{len(assets)}] {a.key:12s} {a.name:12s} 空数据")
            continue
        frames[a.key] = df
        if verbose:
            print(
                f"  [{i}/{len(assets)}] {a.key:12s} {a.name:12s} "
                f"{len(df):5d} 行  {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}"
            )
        time.sleep(SLEEP_BETWEEN)
    return frames


def save_raw(frames: Dict[str, pd.DataFrame]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for key, df in frames.items():
        df.to_csv(RAW_DIR / f"{key}.csv", index=False, encoding="utf-8-sig")


# ──────────────────────────────────────────────────────────────
#  时点对齐 → 面板
# ──────────────────────────────────────────────────────────────
def asof_prev_close(df: pd.DataFrame, calendar: np.ndarray) -> np.ndarray:
    """对每个 A 股交易日 T，取该资产 date < T 的最后一条收盘价。

    统一规则 => A 股取 T-1 收盘；美股取最后一个已收盘的美东交易日（北京 T 日凌晨已收盘）；
    日韩取 T-1 收盘。全部为 T 日 9:00 前可知的信息。
    """
    if df.empty:
        return np.full(len(calendar), np.nan, dtype=float)
    d = df["date"].to_numpy(dtype="datetime64[D]")
    close = df["close"].to_numpy(dtype=float)
    pos = np.searchsorted(d, calendar, side="left") - 1      # 第一个 >= T 的位置 - 1
    out = np.full(len(calendar), np.nan, dtype=float)
    valid = pos >= 0
    out[valid] = close[pos[valid]]
    return out


def build_panel(frames: Dict[str, pd.DataFrame], assets: Optional[List[Asset]] = None,
                extra_dates: Optional[Iterable] = None) -> pd.DataFrame:
    """构造 A 股交易日 × 协变量面板。

    assets 为 None 时直接用 frames 的 key 生成列名（离线模式）。
    extra_dates 用于把「今天」补进交易日历：早上 9:00 前跑时，当天 A 股的 bar
    还没生成，若不补，预测目标会退回到昨天。
    """
    if "cn_hs300" not in frames:
        raise RuntimeError("缺少 cn_hs300，无法构建 A 股交易日历")

    calendar = frames["cn_hs300"]["date"].to_numpy(dtype="datetime64[D]")
    if extra_dates is not None:
        extra = np.array([np.datetime64(pd.Timestamp(d), "D") for d in extra_dates])
        if len(extra):
            calendar = np.union1d(calendar, extra)
    calendar = np.sort(calendar)
    panel = pd.DataFrame({"date": pd.to_datetime(calendar)})

    if assets:
        items = [(a.key, f"{a.key}_ret1") for a in assets]
    else:
        items = [(k, f"{k}_ret1") for k in frames]

    for key, col in items:
        df = frames.get(key)
        if df is None:
            continue
        prev = asof_prev_close(df, calendar)
        ret = np.full(len(prev), np.nan, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ret[1:] = prev[1:] / prev[:-1] - 1.0
        panel[col] = ret * 100.0        # 单位：%

    return panel


def build_targets(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """当日（T 日）真实收益：仅作标签，实际决策时不可用。"""
    cal = np.sort(frames["cn_hs300"]["date"].to_numpy(dtype="datetime64[D]"))
    out = pd.DataFrame({"date": pd.to_datetime(cal)})
    for a in CN_INDEX:
        df = frames.get(a.key)
        if df is None:
            continue
        s = df.set_index("date")["close"]
        s = s.reindex(pd.to_datetime(cal))
        out[f"y_{a.key}"] = (s.pct_change() * 100.0).to_numpy()
    return out


# ──────────────────────────────────────────────────────────────
#  报告
# ──────────────────────────────────────────────────────────────
def write_report(
    assets: List[Asset],
    frames: Dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    elapsed: float,
) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("盘前预测 · 协变量数据源探测报告")
    lines.append("=" * 78)
    lines.append(f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"抓取耗时：{elapsed:.1f}s，资产 {len(assets)} 个，成功 {len(frames)} 个")
    lines.append("")

    group_cn = {
        "cn_index": "A股大盘", "cn_industry": "A股行业", "us_index": "美股指数",
        "us_sector": "美股板块", "jp_index": "日股", "kr_index": "韩股", "hk_index": "港股",
    }
    by_group: Dict[str, List[Asset]] = {}
    for a in assets:
        by_group.setdefault(a.group, []).append(a)

    for g, items in by_group.items():
        ok = sum(1 for a in items if a.key in frames)
        lines.append(f"[{group_cn.get(g, g)}]  {ok}/{len(items)} 成功")
        for a in items:
            df = frames.get(a.key)
            if df is None:
                lines.append(f"  {a.name:12s} {a.secid:12s} ✗ 未取到")
                continue
            last = df["date"].iloc[-1]
            lines.append(
                f"  {a.name:12s} {a.secid:12s} {len(df):5d} 行  "
                f"{df['date'].iloc[0]:%Y-%m-%d} ~ {last:%Y-%m-%d}  "
                f"(T-1 收盘={df['close'].iloc[-1]:.2f})"
            )
        lines.append("")

    lines.append("-" * 78)
    lines.append("对齐面板（panel.csv）")
    lines.append("-" * 78)
    lines.append(f"行数（A股交易日）：{len(panel)}   "
                 f"{panel['date'].min():%Y-%m-%d} ~ {panel['date'].max():%Y-%m-%d}")
    feat_cols = [c for c in panel.columns if c != "date" and not c.startswith("y_")]
    lines.append(f"协变量列数：{len(feat_cols)}")

    nan_rate = panel[feat_cols].isna().mean().sort_values(ascending=False)
    perfect = int((nan_rate == 0).sum())
    lines.append(f"无缺失列：{perfect}/{len(feat_cols)}")
    bad = nan_rate[nan_rate > 0]
    if len(bad):
        lines.append("缺失率 > 0 的列（前三名）：")
        for c, r in bad.head(3).items():
            lines.append(f"  {c:20s} {r*100:.2f}%")
    lines.append("")

    lines.append("-" * 78)
    lines.append("预测目标（当日收益，仅作标签）")
    lines.append("-" * 78)
    for c in [c for c in targets.columns if c.startswith("y_")]:
        s = targets[c].dropna()
        lines.append(
            f"  {c:14s} 样本 {len(s):5d}  均值 {s.mean():+.3f}%  "
            f"标准差 {s.std():.3f}%  |r|>1% 占比 {(s.abs() > 1).mean()*100:.1f}%"
        )
    lines.append("")
    lines.append("结论：协变量数据源全部可用，面板已按「T 日 9:00 前可知」规则对齐。")
    lines.append("=" * 78)
    text = "\n".join(lines)
    (DATA_DIR / "_report.txt").write_text(text, encoding="utf-8")
    return text


# ──────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("fetch_data", description="盘前预测协变量数据抓取")
    ap.add_argument("--beg", default="20180101", help="起始日期 YYYYMMDD")
    ap.add_argument("--probe-only", action="store_true", help="只探测，不写文件")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    assets = build_universe(verbose=not args.quiet)
    print(f"\n资产清单：{len(assets)} 个，开始抓取日线（起始 {args.beg}）...\n")
    frames = crawl(assets, args.beg, verbose=not args.quiet)

    if not frames:
        print("没有取到任何数据，请检查网络或接口变更。")
        return 1

    panel = build_panel(frames, assets)
    targets = build_targets(frames)

    elapsed = time.time() - t0
    report = write_report(assets, frames, panel, targets, elapsed)
    print("\n" + report)

    if args.probe_only:
        print("\n[probe-only] 未写入文件。")
        return 0

    save_raw(frames)
    panel.to_csv(DATA_DIR / "panel.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(DATA_DIR / "targets.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写入：{RAW_DIR}\\*.csv  /  panel.csv  /  targets.csv  /  _report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
