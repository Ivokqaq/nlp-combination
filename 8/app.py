import re
import string
from typing import Iterable

import jieba
import streamlit as st
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"

RULE_DICTIONARY = {
    "i": "我",
    "you": "你",
    "he": "他",
    "she": "她",
    "it": "它",
    "we": "我们",
    "they": "他们",
    "am": "是",
    "is": "是",
    "are": "是",
    "was": "是",
    "were": "是",
    "be": "是",
    "a": "一个",
    "an": "一个",
    "the": "这个",
    "this": "这个",
    "that": "那个",
    "these": "这些",
    "those": "那些",
    "cat": "猫",
    "cats": "猫",
    "dog": "狗",
    "dogs": "狗",
    "rain": "雨",
    "rains": "下雨",
    "raining": "下雨",
    "and": "和",
    "or": "或者",
    "but": "但是",
    "in": "在",
    "on": "在",
    "at": "在",
    "to": "到",
    "from": "从",
    "of": "的",
    "for": "为了",
    "with": "和",
    "without": "没有",
    "like": "喜欢",
    "love": "爱",
    "study": "学习",
    "learn": "学习",
    "machine": "机器",
    "translation": "翻译",
    "language": "语言",
    "natural": "自然",
    "processing": "处理",
    "student": "学生",
    "teacher": "老师",
    "book": "书",
    "school": "学校",
    "computer": "计算机",
    "good": "好",
    "bad": "坏",
    "beautiful": "美丽",
    "important": "重要",
    "difficult": "困难",
    "simple": "简单",
    "because": "因为",
    "if": "如果",
    "when": "当",
    "although": "虽然",
    "who": "谁",
    "which": "哪个",
    "where": "哪里",
    "what": "什么",
    "why": "为什么",
    "how": "如何",
}


st.set_page_config(
    page_title="机器翻译机制与质量评测系统",
    page_icon="MT",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_translator():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def translate_with_nmt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    tokenizer, model = load_translator()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(**inputs, max_length=256, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def split_english_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", text)


def rule_based_translate(text: str) -> str:
    tokens = split_english_tokens(text)
    translated: list[str] = []

    for token in tokens:
        normalized = token.lower().strip(string.punctuation)
        if not normalized:
            translated.append(token)
            continue

        translated.append(RULE_DICTIONARY.get(normalized, token))

    return " ".join(translated).replace(" ,", "，").replace(" .", "。").replace(" ?", "？").replace(" !", "！")


def tokenize_chinese(text: str) -> list[str]:
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


def calc_bleu(reference: str, candidate: str) -> float:
    reference_tokens = tokenize_chinese(reference)
    candidate_tokens = tokenize_chinese(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )


def render_nmt_output(source_text: str, button_label: str) -> str:
    if not st.button(button_label, use_container_width=True):
        return ""

    if not source_text.strip():
        st.warning("请先输入英文原文。")
        return ""

    try:
        with st.spinner("正在调用 NMT 模型生成译文..."):
            translation = translate_with_nmt(source_text)
    except Exception as exc:
        st.error("NMT 模型暂时无法加载或运行。请检查网络、依赖和模型缓存。")
        st.code(str(exc), language="text")
        return ""

    st.session_state["last_nmt_translation"] = translation
    return translation


def score_label(score: float) -> str:
    if score >= 0.7:
        return "较高：候选译文与参考译文在 n-gram 层面非常接近。"
    if score >= 0.4:
        return "中等：候选译文和参考译文有一定重合，但仍存在明显差异。"
    if score > 0:
        return "较低：表面词序列重合较少，可能是翻译质量差，也可能是合理改写没有被 BLEU 充分奖励。"
    return "为 0：候选译文与参考译文几乎没有有效 n-gram 重合，或输入为空。"


st.title("机器翻译机制对比与质量评测系统")
st.caption("Week 9 随堂 Vibe 实验：Machine Translation Mechanisms and BLEU Evaluation")

tab_nmt, tab_compare, tab_bleu = st.tabs(
    ["NMT Engine", "Rule-based vs NMT", "BLEU Score"]
)

with tab_nmt:
    st.subheader("神经机器翻译引擎")
    source = st.text_area(
        "英文原文",
        value="It rains cats and dogs.",
        height=120,
        key="nmt_source",
    )

    nmt_translation = render_nmt_output(source, "生成 NMT 译文")
    if nmt_translation:
        st.markdown("**中文译文**")
        st.success(nmt_translation)

    st.info(
        "观察重点：NMT 是否能根据上下文处理 idiom、多义词和复杂长句，而不是只做词对词替换。"
    )

with tab_compare:
    st.subheader("基于规则的直译 vs 神经网络意译")
    compare_source = st.text_area(
        "英文原文",
        value="It rains cats and dogs.",
        height=120,
        key="compare_source",
    )

    if st.button("对比两种翻译", use_container_width=True):
        if not compare_source.strip():
            st.warning("请先输入英文原文。")
        else:
            literal_translation = rule_based_translate(compare_source)
            try:
                with st.spinner("正在生成 NMT 译文..."):
                    neural_translation = translate_with_nmt(compare_source)
            except Exception as exc:
                neural_translation = ""
                st.error("NMT 模型暂时无法加载或运行。规则直译仍可展示。")
                st.code(str(exc), language="text")

            left, right = st.columns(2)
            with left:
                st.markdown("**Rule-based 逐词直译**")
                st.warning(literal_translation)
            with right:
                st.markdown("**NMT 神经机器翻译**")
                if neural_translation:
                    st.success(neural_translation)
                else:
                    st.info("暂无 NMT 输出。")

    st.info(
        "观察重点：逐词直译在语序、定语从句、一词多义和习语表达上容易失真；NMT 通常更能利用上下文生成自然译文。"
    )

with tab_bleu:
    st.subheader("机器翻译质量自动评测")
    bleu_source = st.text_area(
        "待翻译英文原文",
        value="It rains cats and dogs.",
        height=90,
        key="bleu_source",
    )

    if "candidate_text" not in st.session_state:
        st.session_state["candidate_text"] = st.session_state.get(
            "last_nmt_translation", "下着倾盆大雨。"
        )

    if st.button("用 NMT 自动生成 Candidate", use_container_width=True):
        if not bleu_source.strip():
            st.warning("请先输入英文原文。")
        else:
            try:
                with st.spinner("正在生成 Candidate..."):
                    st.session_state["candidate_text"] = translate_with_nmt(bleu_source)
                st.rerun()
            except Exception as exc:
                st.error("NMT 模型暂时无法加载或运行。你仍然可以手动输入 Candidate。")
                st.code(str(exc), language="text")

    col_a, col_b = st.columns(2)
    with col_a:
        reference = st.text_area(
            "标准中文参考译文 Reference",
            value="下着倾盆大雨。",
            height=130,
            key="reference",
        )
    with col_b:
        candidate = st.text_area(
            "机器生成候选译文 Candidate",
            height=130,
            key="candidate_text",
        )

    evaluate = st.button("计算 BLEU 分数", use_container_width=True)

    if evaluate:
        bleu = calc_bleu(reference, candidate)
        st.metric("BLEU Score", f"{bleu:.4f}")
        st.write(score_label(bleu))

        with st.expander("查看分词结果"):
            st.markdown("**Reference tokens**")
            st.code(" / ".join(tokenize_chinese(reference)), language="text")
            st.markdown("**Candidate tokens**")
            st.code(" / ".join(tokenize_chinese(candidate)), language="text")

    st.info(
        "观察重点：BLEU 主要看 n-gram 重合。语义正确但同义改写可能得分偏低，词汇相同但语序混乱也会影响高阶 n-gram 得分。"
    )
