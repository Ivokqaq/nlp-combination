from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from extractors import ENTITY_COLORS, Entity, SAMPLE_TEXTS, extract_information


st.set_page_config(
    page_title="Week 8 信息抽取与知识图谱系统",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
    <style>
      .entity {
        display: inline-block;
        padding: 0.18rem 0.42rem;
        margin: 0.12rem 0.18rem 0.12rem 0;
        border-radius: 0.5rem;
        line-height: 1.8;
        color: #111827;
        font-weight: 500;
      }
      .entity-tag {
        margin-left: 0.35rem;
        padding: 0.08rem 0.35rem;
        border-radius: 999px;
        font-size: 0.72rem;
        background: rgba(17, 24, 39, 0.1);
      }
      .panel {
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 16px;
        padding: 1rem 1rem 0.75rem;
        background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(255,255,255,1));
      }
      .panel-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
      }
      .mono {
        font-family: "Consolas", "Courier New", monospace;
        white-space: pre-wrap;
        line-height: 1.7;
        font-size: 0.96rem;
      }
      .note {
        border-left: 4px solid #2563eb;
        padding: 0.75rem 1rem;
        background: rgba(219, 234, 254, 0.45);
        border-radius: 0.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.title("Week 8 随堂 Vibe 实验：信息抽取与知识图谱构建系统")
    st.caption(
        "作业版实现：集成 NER、BIO 标注、关系抽取与知识图谱可视化。当前使用规则增强抽取，后续可替换为 spaCy 或大模型 API。"
    )

    with st.sidebar:
        st.subheader("演示控制台")
        sample_name = st.selectbox("选择示例文本", list(SAMPLE_TEXTS.keys()))
        if st.button("加载示例文本", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_TEXTS[sample_name]

        st.markdown("### 作业观察提示")
        st.markdown(
            "- 模块 1：对比实体高亮与 BIO 序列\n"
            "- 模块 2：看 Subject / Predicate / Object 是否合理\n"
            "- 模块 3：观察线性文本如何变成网状结构"
        )

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = SAMPLE_TEXTS["英文商业新闻"]

    text = st.text_area(
        "输入文本（支持中文、英文或混合语料）",
        height=180,
        key="input_text",
        placeholder="例如：Steve Jobs founded Apple in California. Apple is headquartered in Cupertino.",
    )

    extract_clicked = st.button("抽取信息", type="primary", use_container_width=True)
    should_run = extract_clicked or ("result" not in st.session_state and text.strip())

    if should_run and text.strip():
        st.session_state["result"] = extract_information(text)

    result = st.session_state.get("result")
    if not result:
        st.info("输入文本后点击“抽取信息”，系统会生成实体、关系和知识图谱。")
        return

    render_summary(result)

    tab_ner, tab_re, tab_kg, tab_report = st.tabs(
        ["模块 1：NER 与 BIO", "模块 2：关系抽取", "模块 3：知识图谱", "实验报告提示"]
    )

    with tab_ner:
        render_ner_tab(result)

    with tab_re:
        render_relation_tab(result)

    with tab_kg:
        render_graph_tab(result)

    with tab_report:
        render_report_tab(result)


def render_summary(result) -> None:
    entity_count = len(result.entities)
    relation_count = len(result.relations)
    nested_count = len(result.nested_pairs)

    col1, col2, col3 = st.columns(3)
    col1.metric("实体数", entity_count)
    col2.metric("关系数", relation_count)
    col3.metric("嵌套候选", nested_count)


def render_ner_tab(result) -> None:
    show_bio = st.checkbox("查看底层标注（BIO 序列）", value=False)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">文本展示区</div>', unsafe_allow_html=True)
    if show_bio:
        st.markdown(
            f'<div class="mono">{escape(format_bio_text(result))}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(render_highlighted_text(result.text, result.entities), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        st.subheader("实体识别结果")
        entity_rows = [
            {
                "Entity": entity.text,
                "Type": entity.label,
                "Span": f"[{entity.start}, {entity.end})",
            }
            for entity in result.entities
        ]
        if entity_rows:
            st.dataframe(pd.DataFrame(entity_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("当前文本没有识别出实体。")

    with col2:
        st.subheader("理论提醒")
        st.markdown(
            '<div class="note">'
            "BIO 适合展示 flat entity 的边界。<br>"
            "如果一个实体内部还包含另一个实体，单层 BIO 往往就不够用了。"
            "</div>",
            unsafe_allow_html=True,
        )
        if result.nested_pairs:
            nested_rows = [
                {
                    "Outer": f"{outer.text} ({outer.label})",
                    "Inner": f"{inner.text} ({inner.label})",
                }
                for outer, inner in result.nested_pairs
            ]
            st.markdown("**检测到的嵌套候选（用于观察，不参与当前平铺展示）**")
            st.dataframe(pd.DataFrame(nested_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("**检测到的嵌套候选**")
            st.caption("当前文本未发现明显嵌套实体候选。")


def render_relation_tab(result) -> None:
    st.subheader("关系三元组")
    if result.relations:
        rows = [
            {
                "Subject": relation.subject,
                "Predicate": relation.predicate,
                "Object": relation.object,
            }
            for relation in result.relations
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("当前文本没有抽取到关系。你可以尝试输入包含 founded / 位于 / 总部位于 / CEO of 等模式的句子。")

    st.markdown("### 学生观察建议")
    st.markdown(
        "- 输入包含从句或代词的句子，观察规则系统是否会漏掉真正的主语。\n"
        "- 对比实体抽取结果与关系结果，理解 RE 本质上是在实体节点之间预测语义边。"
    )


def render_graph_tab(result) -> None:
    st.subheader("知识图谱交互可视化")
    if not result.entities:
        st.info("先输入一段包含实体的文本，图谱才会出现。")
        return

    net = Network(height="560px", width="100%", directed=True, cdn_resources="in_line")
    net.barnes_hut()

    node_ids: set[str] = set()
    for entity in result.entities:
        node_id = entity_key(entity.text, entity.label)
        node_ids.add(node_id)
        net.add_node(
            node_id,
            label=entity.text,
            title=f"{entity.label}: {entity.text}",
            color=ENTITY_COLORS.get(entity.label, "#d1d5db"),
            size=node_size(entity.label),
            shape="dot",
        )

    for relation in result.relations:
        subject_node = find_node_id(result.entities, relation.subject)
        object_node = find_node_id(result.entities, relation.object)
        if not subject_node:
            subject_node = entity_key(relation.subject, "MISC")
            if subject_node not in node_ids:
                node_ids.add(subject_node)
                net.add_node(subject_node, label=relation.subject, color="#e5e7eb", size=20)
        if not object_node:
            object_node = entity_key(relation.object, "MISC")
            if object_node not in node_ids:
                node_ids.add(object_node)
                net.add_node(object_node, label=relation.object, color="#e5e7eb", size=20)
        net.add_edge(subject_node, object_node, label=relation.predicate, arrows="to")

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 120
            }
          },
          "edges": {
            "smooth": {
              "type": "dynamic"
            },
            "font": {
              "size": 13,
              "strokeWidth": 0
            }
          }
        }
        """
    )
    components.html(net.generate_html(), height=580, scrolling=False)
    st.caption("支持拖拽节点、滚轮缩放和悬停查看信息。")


def render_report_tab(result) -> None:
    st.subheader("可直接写进实验报告的内容")
    st.markdown(
        "1. **模块 1**：说明 NER 的两个核心目标是实体边界识别与实体类型判定；"
        "再解释 BIO 中 `B / I / O` 如何帮助模型表达实体边界。\n"
        "2. **模块 2**：说明 RE 可表示为 `Subject - Predicate - Object` 三元组，"
        "是在已识别实体基础上预测语义关系。\n"
        "3. **模块 3**：说明知识图谱可视化把线性文本转化为图结构，"
        "实体映射为 nodes，关系映射为 edges。"
    )

    st.markdown("### 当前运行结果可写的观察")
    observations = [
        f"本次文本共识别出 {len(result.entities)} 个实体，抽取到 {len(result.relations)} 条关系。",
        "BIO 视图适合展示平铺实体，但对嵌套实体的表达能力有限。",
        "知识图谱比表格更直观，便于观察实体之间的整体连接结构。",
    ]
    if result.nested_pairs:
        observations.append(
            f"当前文本中还发现 {len(result.nested_pairs)} 组嵌套候选，说明真实文本中的实体结构可能比单层序列标注更复杂。"
        )
    for item in observations:
        st.markdown(f"- {item}")

    st.markdown("### 页面截图建议")
    st.markdown(
        "- 一张实体高亮图：突出不同类型实体的颜色差异\n"
        "- 一张 BIO 视图图：展示 token 与 `B/I/O` 标签\n"
        "- 一张关系表截图：展示 `Subject / Predicate / Object`\n"
        "- 一张知识图谱截图：展示节点颜色、边标签、整体结构"
    )


def render_highlighted_text(text: str, entities: list[Entity]) -> str:
    parts: list[str] = ['<div class="mono">']
    cursor = 0
    for entity in sorted(entities, key=lambda item: item.start):
        if entity.start > cursor:
            parts.append(escape(text[cursor:entity.start]))
        color = ENTITY_COLORS.get(entity.label, "#e5e7eb")
        parts.append(
            f'<span class="entity" style="background:{color}">'
            f"{escape(text[entity.start:entity.end])}"
            f'<span class="entity-tag">{escape(entity.label)}</span>'
            "</span>"
        )
        cursor = entity.end
    if cursor < len(text):
        parts.append(escape(text[cursor:]))
    parts.append("</div>")
    return "".join(parts).replace("\n", "<br>")


def format_bio_text(result) -> str:
    return "\n".join(f"{token.tag:>8}  {token.token}" for token in result.token_tags)


def entity_key(text: str, label: str) -> str:
    return f"{text}::{label}"


def find_node_id(entities: list[Entity], target_text: str) -> str | None:
    for entity in entities:
        if entity.text == target_text:
            return entity_key(entity.text, entity.label)
    return None


def node_size(label: str) -> int:
    return {
        "PER": 24,
        "ORG": 30,
        "LOC": 22,
        "MISC": 20,
    }.get(label, 20)


if __name__ == "__main__":
    main()
