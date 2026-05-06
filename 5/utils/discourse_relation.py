"""显式篇章关系识别与 Arg1/Arg2 提取。"""

from __future__ import annotations

import html
import re

from utils.constants import CONNECTIVE_TO_CATEGORY
from utils.schemas import RelationAnalysisResult, RelationMatch


def _sorted_connectives() -> list[str]:
    """按长度排序，优先匹配多词连接词。"""
    return sorted(CONNECTIVE_TO_CATEGORY.keys(), key=len, reverse=True)


def _guess_since_category(arg1: str, arg2: str) -> tuple[str, str]:
    """对 since 做启发式消歧。"""
    arg1_lower = arg1.lower()
    arg2_lower = arg2.lower()

    if re.search(r"\b(has|have|had)\b[^.?!]{0,30}\bbeen\b", arg1_lower) and re.search(
        r"\b(graduated|left|arrived|started|began|moved|finished|was|were|did)\b", arg2_lower
    ):
        return "TEMPORAL", "根据完成体/过去事件线索，`since` 更像时间起点。"

    if re.search(r"\b(is|are|was|were)\b", arg2_lower) and "located" in arg2_lower:
        return "CONTINGENCY", "根据解释原因的从句线索，`since` 更像因果关系。"

    if any(cue in arg2_lower for cue in ["because", "reason", "therefore", "due to"]):
        return "CONTINGENCY", "根据因果提示词，`since` 更像因果关系。"

    if any(cue in arg2_lower for cue in ["graduated", "left", "arrived", "since then"]):
        return "TEMPORAL", "根据时间起点事件线索，`since` 更像时间关系。"

    return "CONTINGENCY", "未命中强特征时，当前规则默认将 `since` 近似视为因果关系。"


def _trim_argument(text: str) -> str:
    """清理 Arg1 / Arg2 边界噪音。"""
    return text.strip(" \t\r\n,;:-")


def _extract_arguments(text: str, start_char: int, end_char: int) -> tuple[str, str]:
    """根据连接词位置做简化 Arg1 / Arg2 提取。"""
    prefix = _trim_argument(text[:start_char])
    suffix = _trim_argument(text[end_char:])

    if not prefix:
        comma_index = suffix.find(",")
        if comma_index != -1:
            arg2 = _trim_argument(suffix[:comma_index])
            arg1 = _trim_argument(suffix[comma_index + 1 :])
            return arg1 or prefix, arg2 or suffix
        return prefix, suffix
    return prefix, suffix


def _find_connective_matches(text: str) -> list[tuple[str, re.Match[str]]]:
    """扫描文本中的显式连接词。"""
    matches: list[tuple[str, re.Match[str]]] = []
    lowered = text.lower()
    for connective in _sorted_connectives():
        pattern = r"\b" + re.escape(connective) + r"\b"
        for match in re.finditer(pattern, lowered):
            matches.append((connective, match))
    matches.sort(key=lambda item: item[1].start())
    return matches


def _deduplicate_matches(matches: list[tuple[str, re.Match[str]]]) -> list[tuple[str, re.Match[str]]]:
    """去掉重叠匹配，避免多词连接词与子词重复高亮。"""
    deduped: list[tuple[str, re.Match[str]]] = []
    last_end = -1
    for connective, match in matches:
        if match.start() < last_end:
            continue
        deduped.append((connective, match))
        last_end = match.end()
    return deduped


def build_relation_highlight_html(text: str, matches: list[RelationMatch]) -> str:
    """把连接词高亮成可直接渲染的 HTML。"""
    if not matches:
        return f"<div class='result-box'>{html.escape(text)}</div>"

    chunks: list[str] = []
    cursor = 0
    for match in matches:
        chunks.append(html.escape(text[cursor : match.start_char]))
        token = html.escape(text[match.start_char : match.end_char])
        chunks.append(
            f"<span class='connective-tag'>{token} [{html.escape(match.category)}]</span>"
        )
        cursor = match.end_char
    chunks.append(html.escape(text[cursor:]))
    return f"<div class='result-box'>{''.join(chunks)}</div>"


def analyze_explicit_relation(text: str) -> RelationAnalysisResult:
    """执行简化版显式篇章关系分析。"""
    normalized = " ".join(text.split())
    raw_matches = _deduplicate_matches(_find_connective_matches(normalized))

    relation_matches: list[RelationMatch] = []
    for connective, match in raw_matches:
        arg1, arg2 = _extract_arguments(normalized, match.start(), match.end())
        category = CONNECTIVE_TO_CATEGORY[connective]
        explanation = "按显式连接词词表直接映射类别。"
        if connective == "since":
            category, explanation = _guess_since_category(arg1, arg2)

        relation_matches.append(
            RelationMatch(
                connective=normalized[match.start() : match.end()],
                category=category,
                start_char=match.start(),
                end_char=match.end(),
                arg1=arg1,
                arg2=arg2,
                explanation=explanation,
            )
        )

    if relation_matches:
        summary = "已识别显式连接词，并基于 PDTB 顶级类别进行简化标注。"
    else:
        summary = "未在当前句子中识别到词表内的显式连接词。"

    return RelationAnalysisResult(
        input_text=normalized,
        matches=relation_matches,
        summary=summary,
        mode="规则词表 + 启发式 Arg 提取",
    )
