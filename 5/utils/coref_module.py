"""指代消解后端与降级方案。"""

from __future__ import annotations

import re

import streamlit as st

from utils.constants import PERSON_PRONOUNS
from utils.edu_segmentation import load_spacy_pipeline
from utils.schemas import CorefCluster, CorefResult


@st.cache_resource(show_spinner=False)
def load_fastcoref_backend():
    """加载 fastcoref 模型。"""
    from fastcoref import FCoref

    return FCoref(device="cpu")


def _normalize_cluster_payload(raw_cluster: list, text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """把不同 API 形态的 cluster 结果统一成 mentions + char spans。"""
    mention_texts: list[str] = []
    spans: list[tuple[int, int]] = []

    for item in raw_cluster:
        if isinstance(item, dict):
            start = item.get("start_char")
            end = item.get("end_char")
            mention = item.get("text")
            if isinstance(start, int) and isinstance(end, int):
                spans.append((start, end))
                mention_texts.append(mention or text[start:end])
        elif isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, int) for v in item):
            start, end = item
            if 0 <= start < end <= len(text):
                spans.append((start, end))
                mention_texts.append(text[start:end])
        elif isinstance(item, str):
            mention_texts.append(item)

    if mention_texts and not spans:
        for mention in mention_texts:
            match = re.search(re.escape(mention), text)
            if match:
                spans.append((match.start(), match.end()))

    return mention_texts, spans


def _run_fastcoref(text: str) -> CorefResult:
    """优先使用 fastcoref 执行真实模型推理。"""
    model = load_fastcoref_backend()
    predictions = model.predict(texts=[text])
    prediction = predictions[0] if isinstance(predictions, list) else predictions

    raw_clusters = []
    raw_strings = []
    if hasattr(prediction, "get_clusters"):
        try:
            raw_clusters = prediction.get_clusters(as_strings=False)
        except Exception:
            raw_clusters = []
        try:
            raw_strings = prediction.get_clusters(as_strings=True)
        except Exception:
            raw_strings = []

    clusters: list[CorefCluster] = []
    for idx, cluster in enumerate(raw_clusters or []):
        mention_texts, spans = _normalize_cluster_payload(cluster, text)
        if raw_strings and idx < len(raw_strings) and raw_strings[idx]:
            mention_texts = [str(item) for item in raw_strings[idx]]
        if mention_texts:
            clusters.append(CorefCluster(cluster_id=idx + 1, mentions=mention_texts, spans=spans))

    if not clusters:
        raise ValueError("fastcoref 已加载，但未返回可用 cluster 结果。")

    return CorefResult(
        mode="fastcoref",
        message="当前使用 fastcoref 真实模型输出指代簇结果。",
        clusters=clusters,
    )


def _find_entity_mentions_with_spacy(text: str) -> list[tuple[str, tuple[int, int], str]]:
    """用 spaCy 提取简单命名实体和代词，为降级模式提供基础信息。"""
    nlp, _ = load_spacy_pipeline()
    mentions: list[tuple[str, tuple[int, int], str]] = []
    if nlp is None:
        return mentions

    doc = nlp(text)
    for ent in getattr(doc, "ents", []):
        if ent.label_ in {"PERSON", "ORG", "GPE", "NORP"}:
            entity_type = "PERSON" if ent.label_ == "PERSON" else "THING"
            mentions.append((ent.text, (ent.start_char, ent.end_char), entity_type))

    for token in doc:
        pronoun_type = PERSON_PRONOUNS.get(token.text.lower())
        if pronoun_type:
            mentions.append((token.text, (token.idx, token.idx + len(token.text)), pronoun_type))
    mentions.sort(key=lambda item: item[1][0])
    return mentions


def _heuristic_coref(text: str) -> CorefResult:
    """当 fastcoref 不可用时，使用启发式规则进行演示级聚类。"""
    mentions = _find_entity_mentions_with_spacy(text)
    if not mentions:
        return CorefResult(
            mode="heuristic",
            message="fastcoref 不可用，且未检测到可用于演示的 mention，当前仅保留输入文本展示。",
            clusters=[],
        )

    clusters: list[CorefCluster] = []
    person_cluster: CorefCluster | None = None
    thing_cluster: CorefCluster | None = None
    plural_cluster: CorefCluster | None = None

    for mention_text, span, mention_type in mentions:
        mention_lower = mention_text.lower()

        if mention_type == "PERSON":
            if person_cluster is None:
                person_cluster = CorefCluster(cluster_id=len(clusters) + 1, mentions=[], spans=[])
                clusters.append(person_cluster)
            person_cluster.mentions.append(mention_text)
            person_cluster.spans.append(span)
            continue

        if mention_type == "THING":
            if mention_lower in {"it", "its"}:
                if thing_cluster is None and person_cluster is not None:
                    continue
                if thing_cluster is None:
                    thing_cluster = CorefCluster(cluster_id=len(clusters) + 1, mentions=[], spans=[])
                    clusters.append(thing_cluster)
                thing_cluster.mentions.append(mention_text)
                thing_cluster.spans.append(span)
            else:
                if thing_cluster is None:
                    thing_cluster = CorefCluster(cluster_id=len(clusters) + 1, mentions=[], spans=[])
                    clusters.append(thing_cluster)
                thing_cluster.mentions.append(mention_text)
                thing_cluster.spans.append(span)
            continue

        if mention_type == "PLURAL":
            if plural_cluster is None:
                plural_cluster = CorefCluster(cluster_id=len(clusters) + 1, mentions=[], spans=[])
                clusters.append(plural_cluster)
            plural_cluster.mentions.append(mention_text)
            plural_cluster.spans.append(span)

    return CorefResult(
        mode="heuristic",
        message="fastcoref 当前不可用，页面已切换到启发式演示模式。该模式只用于作业展示和界面联调，不代表真实模型性能。",
        clusters=[cluster for cluster in clusters if cluster.mentions],
    )


def analyze_coreference(text: str) -> CorefResult:
    """统一封装指代消解分析逻辑。"""
    normalized = " ".join(text.split())
    try:
        return _run_fastcoref(normalized)
    except Exception as error:
        fallback_result = _heuristic_coref(normalized)
        fallback_result.message += f" 触发降级原因：{type(error).__name__}。"
        return fallback_result
