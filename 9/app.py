from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
LABEL_ORDER = ["Positive", "Neutral", "Negative"]
LABEL_COLORS = {
    "Positive": "#22c55e",
    "Neutral": "#38bdf8",
    "Negative": "#ef4444",
}


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    source: str


def normalize_label(label: str) -> str:
    value = label.strip().lower()
    if value in {"positive", "pos", "label_2", "5 stars", "4 stars"}:
        return "Positive"
    if value in {"negative", "neg", "label_0", "1 star", "2 stars"}:
        return "Negative"
    if value in {"neutral", "neu", "label_1", "3 stars"}:
        return "Neutral"
    return label.strip().title()


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=MODEL_NAME)


def rule_based_sentiment(text: str) -> SentimentResult:
    positive_words = [
        "好",
        "棒",
        "满意",
        "喜欢",
        "推荐",
        "清晰",
        "流畅",
        "划算",
        "惊喜",
        "舒服",
        "耐用",
        "优秀",
    ]
    negative_words = [
        "差",
        "垃圾",
        "失望",
        "卡",
        "坏",
        "退货",
        "看不清",
        "没电",
        "发热",
        "慢",
        "刺眼",
        "不值",
    ]

    positive_hits = sum(word in text for word in positive_words)
    negative_hits = sum(word in text for word in negative_words)
    if positive_hits > negative_hits:
        return SentimentResult("Positive", min(0.68 + positive_hits * 0.08, 0.96), "Rule-based fallback")
    if negative_hits > positive_hits:
        return SentimentResult("Negative", min(0.68 + negative_hits * 0.08, 0.96), "Rule-based fallback")
    return SentimentResult("Neutral", 0.58, "Rule-based fallback")


def analyze_sentiment(text: str) -> SentimentResult:
    cleaned = text.strip()
    if not cleaned:
        return SentimentResult("Neutral", 0.0, "Empty input")

    try:
        classifier = load_sentiment_pipeline()
        output = classifier(cleaned, truncation=True)[0]
        return SentimentResult(
            label=normalize_label(output["label"]),
            score=float(output["score"]),
            source=MODEL_NAME,
        )
    except Exception:
        return rule_based_sentiment(cleaned)


def make_gauge(result: SentimentResult, title: str = "Confidence Score") -> go.Figure:
    color = LABEL_COLORS.get(result.label, "#64748b")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(result.score * 100, 1),
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": title, "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#0f172a",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#1e293b"},
                    {"range": [50, 75], "color": "#334155"},
                    {"range": [75, 100], "color": "#475569"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=280,
        margin={"l": 24, "r": 24, "t": 52, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
    )
    return fig


def render_result_card(result: SentimentResult, prefix: str = "模型判断") -> None:
    color = LABEL_COLORS.get(result.label, "#64748b")
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-kicker">{prefix}</div>
            <div class="result-label" style="color:{color};">{result.label}</div>
            <div class="result-meta">Confidence Score: {result.score:.3f}</div>
            <div class="result-source">Source: {result.source}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_reviews() -> list[str]:
    reviews = [
        "手机外观很好看，系统也很流畅，拍照效果比预期好。",
        "物流挺快，但包装有点压痕，整体还可以。",
        "用了两天就开始发热，玩游戏半小时电量掉得很快。",
        "屏幕显示清晰，阳光下也能看清，价格很划算。",
        "客服回复慢，问题拖了三天才解决。",
        "音质一般，没有特别惊喜，也没有明显问题。",
        "电池续航很强，出门一天不用带充电宝。",
        "刚收到就发现边框有划痕，体验非常差。",
        "颜色和图片差不多，手感还算舒服。",
        "拍夜景很糊，视频防抖也不稳定。",
        "老人用起来很简单，字体大，声音也够响。",
        "更新系统后偶尔卡顿，希望后续能优化。",
        "这个价位能买到这样的配置，我觉得很满意。",
        "戴久了耳朵疼，降噪效果也不明显。",
        "功能比较基础，适合日常轻度使用。",
    ]
    random.shuffle(reviews)
    return reviews[: random.randint(10, 15)]


def analyze_many(texts: Iterable[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        result = analyze_sentiment(text)
        rows.append(
            {
                "review": text,
                "sentiment": result.label,
                "confidence": round(result.score, 3),
                "source": result.source,
            }
        )
    return pd.DataFrame(rows)


def set_page_style() -> None:
    st.set_page_config(page_title="电商评论情感分析平台", page_icon="SA", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --panel: #111827;
            --panel-soft: #172033;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --line: #2d3748;
        }
        .stApp {
            background: #090d14;
            color: var(--text);
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .hero {
            padding: 18px 0 10px;
            border-bottom: 1px solid var(--line);
            margin-bottom: 16px;
        }
        .hero-title {
            font-size: 32px;
            font-weight: 760;
            margin-bottom: 6px;
        }
        .hero-subtitle {
            color: var(--muted);
            max-width: 880px;
            font-size: 15px;
            line-height: 1.7;
        }
        .result-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            min-height: 172px;
        }
        .result-kicker, .result-source {
            color: var(--muted);
            font-size: 13px;
        }
        .result-label {
            font-size: 36px;
            font-weight: 760;
            margin: 10px 0;
        }
        .result-meta {
            font-size: 16px;
            color: var(--text);
            margin-bottom: 8px;
        }
        .note {
            background: var(--panel-soft);
            border-left: 4px solid #38bdf8;
            padding: 14px 16px;
            border-radius: 6px;
            color: #dbeafe;
            line-height: 1.75;
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">电商评论情感分析与意见挖掘平台</div>
            <div class="hero-subtitle">
                Week 10 NLP Vibe 实验：将自然语言评论转化为 Positive / Negative / Neutral 极性判断，
                再通过 Confidence Score 和批量可视化支持舆情观察。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tab_single_review() -> None:
    st.subheader("模块 1：基础情感分类与置信度量化")
    default_text = "这款手机屏幕很清晰，电池也耐用，整体非常满意。"
    review = st.text_area("输入一段中文商品评论", value=default_text, height=120)

    if st.button("分析单条评论", type="primary", use_container_width=True):
        st.session_state["single_result"] = analyze_sentiment(review)

    result = st.session_state.get("single_result") or analyze_sentiment(default_text)
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        render_result_card(result)
        st.markdown(
            """
            <div class="note">
            观察任务：输入非常明显的好评和差评，比较标签变化与置信度变化。
            工程应用中，置信度可以帮助判断模型输出是否足够可靠。
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(make_gauge(result), use_container_width=True)


def tab_explicit_implicit() -> None:
    st.subheader("模块 2：显式情感 vs. 隐式情感识别")
    st.markdown(
        """
        <div class="note">
        Explicit Sentiment 通常带有明显褒贬词，例如“太棒了”“很垃圾”。
        Implicit Sentiment 表面像事实描述，但暗含态度，例如“玩游戏半小时就没电了”。
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        explicit_text = st.text_area(
            "显式情感评价",
            value="这屏幕画质太垃圾了，完全不值这个价格。",
            height=120,
        )
        explicit_result = analyze_sentiment(explicit_text)
        render_result_card(explicit_result, "显式评价结果")
        st.plotly_chart(make_gauge(explicit_result, "Explicit Confidence"), use_container_width=True)

    with col_b:
        implicit_text = st.text_area(
            "隐式客观描述",
            value="在太阳底下根本看不清屏幕上的字。",
            height=120,
        )
        implicit_result = analyze_sentiment(implicit_text)
        render_result_card(implicit_result, "隐式描述结果")
        st.plotly_chart(make_gauge(implicit_result, "Implicit Confidence"), use_container_width=True)

    st.caption(
        "观察重点：如果隐式描述被判断成 Neutral，说明小型模型可能缺少场景常识或深层语义推理能力。"
    )


def tab_opinion_dashboard() -> None:
    st.subheader("模块 3：舆情挖掘与可视化仪表盘")
    if st.button("生成测试舆情数据并批量分析", type="primary", use_container_width=True):
        st.session_state["batch_df"] = analyze_many(sample_reviews())

    if "batch_df" not in st.session_state:
        st.session_state["batch_df"] = analyze_many(sample_reviews())

    df = st.session_state["batch_df"]
    counts = df["sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0).reset_index()
    counts.columns = ["sentiment", "count"]

    metric_cols = st.columns(3)
    for col, label in zip(metric_cols, LABEL_ORDER):
        value = int(counts.loc[counts["sentiment"] == label, "count"].iloc[0])
        col.metric(label, value)

    chart_col, table_col = st.columns([0.9, 1.1], gap="large")
    with chart_col:
        fig = px.pie(
            counts,
            values="count",
            names="sentiment",
            color="sentiment",
            color_discrete_map=LABEL_COLORS,
            hole=0.42,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e5e7eb"},
            legend_title_text="Sentiment",
            margin={"l": 12, "r": 12, "t": 20, "b": 20},
        )
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "review": "商品评论",
                "sentiment": "情感极性",
                "confidence": st.column_config.NumberColumn("置信度", format="%.3f"),
                "source": "分析来源",
            },
        )

    st.markdown(
        """
        <div class="note">
        观察任务：把单条评论的情感判断聚合为整体口碑比例。
        当 Negative 占比升高时，可以进一步定位产品缺陷、售后问题或潜在舆情风险。
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    set_page_style()
    render_header()
    tabs = st.tabs(
        [
            "单文本分析",
            "显式 / 隐式情感",
            "Opinion Mining Dashboard",
        ]
    )
    with tabs[0]:
        tab_single_review()
    with tabs[1]:
        tab_explicit_implicit()
    with tabs[2]:
        tab_opinion_dashboard()


if __name__ == "__main__":
    main()
