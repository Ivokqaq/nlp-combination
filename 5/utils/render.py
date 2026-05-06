"""页面渲染与 HTML 可视化工具。"""

from __future__ import annotations

import html

import streamlit as st

from utils.constants import COLOR_PALETTE
from utils.discourse_relation import build_relation_highlight_html
from utils.schemas import CorefCluster, RelationAnalysisResult


def inject_global_css() -> None:
    """注入作业页面使用的全局样式。"""
    st.markdown(
        """
        <style>
        .info-panel {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            border: 1px solid #d0d7de;
            background: #f8fafc;
            color: #1f2937 !important;
        }
        .info-panel.blue { background: #eef6ff; border-color: #b6d4fe; }
        .info-panel.green { background: #edf7ed; border-color: #b7dfb9; }
        .info-panel.amber { background: #fff8e6; border-color: #f5cf7c; }
        .edu-card {
            border: 1px solid #d0d7de;
            border-left: 6px solid #ec8f3a;
            background: white;
            border-radius: 10px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            color: #1f2937 !important;
        }
        .edu-card.teal { border-left-color: #2a9d8f; }
        .edu-card.orange { border-left-color: #ec8f3a; }
        .boundary-chip {
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.15rem 0.45rem;
            border-radius: 999px;
            font-size: 0.78rem;
            background: #fff2cc;
            color: #6b4f00;
        }
        .result-box {
            border: 1px solid #d0d7de;
            border-radius: 10px;
            background: #ffffff;
            padding: 0.9rem;
            line-height: 1.75;
            margin-bottom: 0.8rem;
            color: #1f2937 !important;
        }
        .connective-tag {
            display: inline-block;
            margin: 0 0.15rem;
            padding: 0.05rem 0.35rem;
            border-radius: 6px;
            background: #1d3557;
            color: #ffffff;
            font-weight: 600;
        }
        .arg-box {
            border-radius: 10px;
            padding: 0.8rem 0.95rem;
            margin-bottom: 0.75rem;
            border: 1px solid #d0d7de;
            color: #1f2937 !important;
        }
        .arg1-box { background: #eef7ff; border-color: #9fc5e8; }
        .arg2-box { background: #fef3e7; border-color: #f6b26b; }
        .cluster-item {
            border: 1px solid #d0d7de;
            border-radius: 10px;
            background: white;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.6rem;
            color: #1f2937 !important;
        }
        .mention-span {
            padding: 0.06rem 0.24rem;
            border-radius: 6px;
            border: 1px solid rgba(0,0,0,0.08);
            color: #1f2937 !important;
        }
        .info-panel strong,
        .edu-card strong,
        .result-box strong,
        .arg-box strong,
        .cluster-item strong,
        .mention-span sup,
        .edu-card,
        .result-box,
        .arg-box,
        .cluster-item {
            color: #1f2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_info_panel(title: str, body: str, tone: str = "blue") -> None:
    """渲染说明信息面板。"""
    st.markdown(
        f"<div class='info-panel {tone}'><strong>{html.escape(title)}</strong><br>{html.escape(body)}</div>",
        unsafe_allow_html=True,
    )


def render_edu_cards(title: str, edus: list[str], boundary_tokens: list[str], accent: str) -> None:
    """把 EDU 切分结果渲染成卡片列表。"""
    st.markdown(f"**{html.escape(title)}**")
    boundary_set = set(boundary_tokens)
    for index, edu in enumerate(edus, start=1):
        first_token = edu.strip().split()[0] if edu.strip() else ""
        chip_html = ""
        if index > 1 and first_token in boundary_set:
            chip_html = f"<div class='boundary-chip'>Boundary Token: {html.escape(first_token)}</div>"
        st.markdown(
            f"<div class='edu-card {accent}'><strong>EDU {index}</strong><br>{html.escape(edu)}{chip_html}</div>",
            unsafe_allow_html=True,
        )


def render_relation_summary(result: RelationAnalysisResult) -> None:
    """渲染显式篇章关系分析结果。"""
    render_info_panel("分析摘要", f"{result.summary} 当前模式：{result.mode}", tone="green")
    st.markdown("#### 连接词高亮")
    st.markdown(build_relation_highlight_html(result.input_text, result.matches), unsafe_allow_html=True)

    if not result.matches:
        st.warning("当前句子未匹配到词表中的显式连接词。")
        return

    for index, match in enumerate(result.matches, start=1):
        st.markdown(f"#### 关系 {index}")
        st.markdown(
            f"<div class='arg-box arg1-box'><strong>Arg1</strong><br>{html.escape(match.arg1 or '未成功提取')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='arg-box arg2-box'><strong>Arg2</strong><br>{html.escape(match.arg2 or '未成功提取')}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Connective: {match.connective} | PDTB 顶级类别: {match.category} | 说明: {match.explanation}"
        )


def _color_for_cluster(cluster_id: int) -> str:
    """根据 cluster id 选择颜色。"""
    return COLOR_PALETTE[(cluster_id - 1) % len(COLOR_PALETTE)]


def render_coref_text(text: str, clusters: list[CorefCluster]) -> None:
    """把指代簇按颜色高亮回原文。"""
    all_mentions: list[tuple[int, int, int]] = []
    for cluster in clusters:
        for start, end in cluster.spans:
            if 0 <= start < end <= len(text):
                all_mentions.append((start, end, cluster.cluster_id))

    all_mentions.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    filtered_mentions: list[tuple[int, int, int]] = []
    current_end = -1
    for start, end, cluster_id in all_mentions:
        if start < current_end:
            continue
        filtered_mentions.append((start, end, cluster_id))
        current_end = end

    rendered_parts: list[str] = []
    cursor = 0
    for start, end, cluster_id in filtered_mentions:
        rendered_parts.append(html.escape(text[cursor:start]))
        color = _color_for_cluster(cluster_id)
        mention_text = html.escape(text[start:end])
        rendered_parts.append(
            f"<span class='mention-span' style='background:{color};'>{mention_text}<sup>#{cluster_id}</sup></span>"
        )
        cursor = end
    rendered_parts.append(html.escape(text[cursor:]))

    st.markdown(
        f"<div class='result-box'>{''.join(rendered_parts) if rendered_parts else html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def render_cluster_list(clusters: list[CorefCluster]) -> None:
    """渲染 cluster 列表。"""
    if not clusters:
        st.info("当前没有可展示的指代簇。")
        return

    for cluster in clusters:
        color = _color_for_cluster(cluster.cluster_id)
        mention_list = ", ".join(cluster.mentions)
        st.markdown(
            f"<div class='cluster-item' style='border-left: 8px solid {color};'>"
            f"<strong>Cluster {cluster.cluster_id}</strong><br>"
            f"{html.escape(mention_list)}</div>",
            unsafe_allow_html=True,
        )
