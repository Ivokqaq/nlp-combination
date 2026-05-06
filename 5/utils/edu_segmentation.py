"""规则基线 EDU 切分。"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st

from utils.constants import DISCOURSE_MARKERS
from utils.schemas import RuleSegmentationResult


def _simple_tokenize(text: str) -> list[str]:
    """在 spaCy 不可用时使用的极简分词器。"""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


@st.cache_resource(show_spinner=False)
def load_spacy_pipeline() -> tuple[Any | None, str]:
    """加载 spaCy 管线；若失败则返回降级提示。"""
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm"), "spaCy en_core_web_sm"
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp, "spaCy blank('en') 降级模式"
    except Exception:
        return None, "spaCy 不可用，已回退到正则分词模式"


def _merge_tokens(tokens: list[str]) -> str:
    """把简化 token 列表还原为便于阅读的字符串。"""
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif re.match(r"[,.!?;:%)\]]", token):
            text += token
        elif token in {"'s", "n't"}:
            text += token
        elif text.endswith(("(", "[", '"', "'")):
            text += token
        else:
            text += " " + token
    return text.strip()


def _segment_with_tokens(tokens: list[str], lower_tokens: list[str]) -> RuleSegmentationResult:
    """基于简化 token 信息进行启发式切分。"""
    boundaries = {0}
    for idx, _token in enumerate(tokens):
        if idx == 0:
            continue
        prev = tokens[idx - 1]
        token_lower = lower_tokens[idx]

        if prev in {".", "!", "?", ";"}:
            boundaries.add(idx)
            continue

        if prev in {",", "-", "—", ":"} and token_lower in DISCOURSE_MARKERS:
            boundaries.add(idx)
            continue

        if token_lower in DISCOURSE_MARKERS and prev not in {"(", "["}:
            boundaries.add(idx)

    ordered_boundaries = sorted(boundaries)
    edus: list[str] = []
    boundary_tokens: list[str] = []
    for pos, start_idx in enumerate(ordered_boundaries):
        end_idx = ordered_boundaries[pos + 1] if pos + 1 < len(ordered_boundaries) else len(tokens)
        edu_tokens = tokens[start_idx:end_idx]
        edu_text = _merge_tokens(edu_tokens)
        if edu_text:
            edus.append(edu_text)
            if pos > 0:
                boundary_tokens.append(tokens[start_idx])

    return RuleSegmentationResult(
        edus=edus or [_merge_tokens(tokens)],
        boundary_tokens=boundary_tokens,
        mode="规则启发式",
        notes="基于标点、从属连词和局部上下文的简化规则。",
    )


def segment_rule_based_text(text: str) -> RuleSegmentationResult:
    """对输入文本执行规则基线切分。"""
    nlp, mode = load_spacy_pipeline()

    if nlp is None:
        tokens = _simple_tokenize(text)
        lowered = [token.lower() for token in tokens]
        result = _segment_with_tokens(tokens, lowered)
        result.mode = mode
        result.notes = "spaCy 不可用，已使用正则分词与规则切分。"
        return result

    doc = nlp(text)
    tokens = [token.text for token in doc]
    lowered = [token.text.lower() for token in doc]
    result = _segment_with_tokens(tokens, lowered)
    result.mode = mode
    if "blank" in mode.lower():
        result.notes = "spaCy 模型缺失，当前使用 blank('en') 管线进行降级切分。"
    else:
        result.notes = "已结合 spaCy 分词结果执行规则基线切分。"
    return result
