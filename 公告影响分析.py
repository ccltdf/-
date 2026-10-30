"""读取同花顺公告并分析可能受影响的股票。

用法:
    python 公告影响分析.py --url "https://..." --output output\announcement_analysis.json
    python 公告影响分析.py --file 公告页面.html
    python 公告影响分析.py --text "公告标题和正文"

脚本只做信息抽取和规则分析，不构成投资建议。网页需要登录、验证码或
完全依赖 JavaScript 渲染时，请先在浏览器中保存页面，再使用 --file。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/131 Safari/537.36"
)

POSITIVE_TERMS = {
    "中标": 3, "签订合同": 3, "合同": 1, "获批": 3, "批准": 2, "增持": 3,
    "回购": 3, "分红": 2, "盈利预增": 3, "业绩预增": 3, "扭亏": 2,
    "突破": 2, "投产": 2, "订单": 2, "重大进展": 2, "战略合作": 2,
    "收购": 1, "并购": 1, "融资": 1,
}
NEGATIVE_TERMS = {
    "减持": 3, "终止": 3, "亏损": 3, "业绩预降": 3, "预亏": 3, "下滑": 2,
    "违约": 3, "诉讼": 2, "处罚": 3, "监管": 2, "立案": 3, "退市": 4,
    "质押": 2, "冻结": 2, "商誉减值": 3, "延期": 1, "取消": 2,
    "风险提示": 1, "无法": 2,
}


class VisibleTextParser(HTMLParser):
    """提取 HTML 中的可见文本，跳过脚本、样式和页面控件。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


@dataclass(frozen=True)
class StockImpact:
    code: str
    name: str
    direction: str
    score: int
    reasons: list[str]
    evidence: list[str]


def fetch_url(url: str, timeout: int = 20) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"})
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"读取公告失败，HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"读取公告失败，网络错误: {exc.reason}") from exc


def clean_text(raw: str, is_html: bool) -> str:
    if is_html:
        parser = VisibleTextParser()
        parser.feed(raw)
        raw = "\n".join(parser.parts)
    raw = html.unescape(raw).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_title(text: str) -> str:
    lines = [line for line in text.splitlines() if line]
    return lines[0][:200] if lines else "未识别标题"


def extract_stock_refs(text: str) -> dict[str, str]:
    """识别 6 位 A 股代码及其附近常见的股票名称。"""
    refs: dict[str, str] = {}
    code_pattern = re.compile(r"(?<!\d)(?:沪深|A股)?\s*(\d{6})(?!\d)")
    for match in code_pattern.finditer(text):
        code = match.group(1)
        start = max(0, match.start() - 30)
        context = text[start : match.end() + 30]
        name_match = re.search(
            r"([\u4e00-\u9fff]{2,12})\s*(?:\(|（)\s*" + code + r"\s*(?:\)|）)",
            context,
        )
        if not name_match:
            name_match = re.search(
                r"(?:\(|（)\s*" + code + r"\s*(?:\)|）)\s*([\u4e00-\u9fff]{2,12})",
                context,
            )
        refs.setdefault(code, name_match.group(1) if name_match else "未识别名称")
    return refs


def _term_hits(text: str, terms: dict[str, int]) -> list[tuple[str, int]]:
    return [(term, weight) for term, weight in terms.items() if term in text]


def analyze(text: str) -> dict:
    title = extract_title(text)
    refs = extract_stock_refs(text)
    positive = _term_hits(text, POSITIVE_TERMS)
    negative = _term_hits(text, NEGATIVE_TERMS)
    score = sum(weight for _, weight in positive) - sum(weight for _, weight in negative)
    if score >= 2:
        direction = "利好"
    elif score <= -2:
        direction = "利空"
    else:
        direction = "中性/待确认"
    reasons = [f"{term}（+{weight}）" for term, weight in positive]
    reasons += [f"{term}（-{weight}）" for term, weight in negative]
    evidence = [line[:180] for line in text.splitlines() if any(term in line for term, _ in positive + negative)][:8]
    impacts = [
        asdict(
            StockImpact(
                code=code,
                name=name,
                direction=direction,
                score=score,
                reasons=reasons,
                evidence=evidence,
            )
        )
        for code, name in refs.items()
    ]
    return {
        "title": title,
        "direction": direction,
        "score": score,
        "reasons": reasons,
        "affected_stocks": impacts,
        "stock_count": len(impacts),
        "warning": "规则分析仅供研究参考，不构成投资建议；未识别到代码时不代表公告没有影响。",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取同花顺公告并识别可能受影响的股票")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="同花顺公告页面 URL")
    source.add_argument("--file", type=Path, help="已保存的公告 HTML 或 TXT 文件")
    source.add_argument("--text", help="直接提供公告标题/正文")
    parser.add_argument("--output", type=Path, help="输出 JSON 文件路径")
    return parser


def read_source(args: argparse.Namespace) -> str:
    if args.url:
        return clean_text(fetch_url(args.url), is_html=True)
    if args.file:
        if not args.file.is_file():
            raise RuntimeError(f"文件不存在: {args.file}")
        return clean_text(args.file.read_text(encoding="utf-8", errors="replace"), args.file.suffix.lower() in {".html", ".htm"})
    return clean_text(args.text, is_html=False)


def print_summary(result: dict) -> None:
    print(f"标题: {result['title']}")
    print(f"总体判断: {result['direction']} | 规则分数: {result['score']}")
    print(f"触发词: {', '.join(result['reasons']) or '未识别到明确触发词'}")
    print("可能受影响股票:")
    if not result["affected_stocks"]:
        print("  未识别到公告中的 6 位股票代码")
    for stock in result["affected_stocks"]:
        print(f"  {stock['code']} {stock['name']} - {stock['direction']}")


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        text = read_source(args)
        if not text:
            raise RuntimeError("公告正文为空，请检查 URL、登录状态或保存的页面内容。")
        result = analyze(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"分析结果已保存: {args.output}")
        print_summary(result)
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
