from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import zipfile

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from nltk.wsd import lesk
from sklearn.metrics.pairwise import cosine_similarity
from spacy import displacy
from transformers import AutoModel, AutoTokenizer


st.set_page_config(page_title="Deep Semantic Analysis Platform", layout="wide")

TREEBANK_TOKENIZER = TreebankWordTokenizer()


@dataclass
class WSDResult:
    target_word: str
    synset_name: Optional[str]
    definition: Optional[str]


@st.cache_resource(show_spinner=False)
def load_nlp_model():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


@st.cache_resource(show_spinner=False)
def load_bert_resources():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def _explain_nltk_error(exc: Exception) -> str:
    if isinstance(exc, zipfile.BadZipFile):
        return (
            "检测到本地 NLTK 资源缓存损坏（BadZipFile）。通常是 `nltk_data` 目录里有一个损坏的 zip "
            "文件或被错误命名为 zip 的文件。请删除损坏的 NLTK 缓存后重试。"
        )
    return f"{type(exc).__name__}: {exc}"


def ensure_nltk_resources() -> tuple[bool, Optional[str]]:
    try:
        wn.ensure_loaded()
        return True, None
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True, force=True)
            wn.ensure_loaded()
            return True, None
        except Exception as exc:
            return False, (
                "WordNet 资源不可用，自动下载也没有成功。"
                f"\n详细信息：{_explain_nltk_error(exc)}"
            )
    except Exception as exc:
        return False, (
            "检测 WordNet 资源时出现异常。"
            f"\n详细信息：{_explain_nltk_error(exc)}"
        )


def normalize_target_word(target_word: str) -> str:
    return target_word.strip().lower()


def run_lesk_wsd(sentence: str, target_word: str) -> WSDResult:
    tokens = TREEBANK_TOKENIZER.tokenize(sentence)
    synset = lesk(tokens, target_word)
    if synset is None:
        return WSDResult(target_word=target_word, synset_name=None, definition=None)
    return WSDResult(
        target_word=target_word,
        synset_name=synset.name(),
        definition=synset.definition(),
    )


def locate_target_span(doc, target_word: str) -> Optional[tuple[int, int]]:
    pieces = [piece for piece in target_word.lower().split() if piece]
    if not pieces:
        return None

    lowered_tokens = [token.text.lower() for token in doc]
    for start in range(len(lowered_tokens) - len(pieces) + 1):
        if lowered_tokens[start : start + len(pieces)] == pieces:
            return start, start + len(pieces)
    return None


def get_contextual_embedding(sentence: str, target_word: str):
    import torch

    tokenizer, model = load_bert_resources()
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()

    target = normalize_target_word(target_word)
    sentence_lower = sentence.lower()
    char_start = sentence_lower.find(target)
    if char_start == -1:
        return None, "目标词未在句子中找到，无法提取 BERT 上下文向量。"

    char_end = char_start + len(target)
    target_positions = []
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if start < char_end and end > char_start:
            target_positions.append(idx)

    if not target_positions:
        return None, "目标词未对齐到 BERT token，无法提取上下文向量。"

    with torch.no_grad():
        outputs = model(**encoded)
    hidden = outputs.last_hidden_state[0]
    vector = hidden[target_positions].mean(dim=0).cpu().numpy()
    return vector, None


def compute_similarity(sentence_1: str, sentence_2: str, target_word: str):
    vector_1, error_1 = get_contextual_embedding(sentence_1, target_word)
    vector_2, error_2 = get_contextual_embedding(sentence_2, target_word)

    if error_1:
        return None, error_1
    if error_2:
        return None, error_2

    score = cosine_similarity([vector_1], [vector_2])[0][0]
    return float(score), None


def get_subtree_text(token) -> str:
    return " ".join(child.text for child in token.subtree)


def looks_like_time(token) -> bool:
    time_entities = {"DATE", "TIME"}
    time_words = {
        "today",
        "tomorrow",
        "yesterday",
        "year",
        "month",
        "week",
        "day",
        "morning",
        "evening",
        "night",
    }
    if token.ent_type_ in time_entities:
        return True
    lemma = token.lemma_.lower()
    return lemma in time_words


def looks_like_location(token) -> bool:
    location_entities = {"GPE", "LOC", "FAC"}
    location_preps = {"in", "at", "on", "from", "to", "into", "inside", "outside"}
    if token.ent_type_ in location_entities:
        return True
    if token.dep_ == "pobj" and token.head.text.lower() in location_preps:
        return True
    return False


def extract_srl(doc) -> list[dict[str, str]]:
    rows = []
    predicates = [token for token in doc if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}]
    if not predicates:
        predicates = [token for token in doc if token.pos_ == "VERB"]

    for predicate in predicates:
        row = {
            "A0 (Agent)": "",
            "Predicate": predicate.lemma_,
            "A1 (Patient)": "",
            "AM-LOC": "",
            "AM-TMP": "",
        }

        for child in predicate.children:
            if child.dep_ in {"nsubj", "nsubjpass", "csubj"} and not row["A0 (Agent)"]:
                row["A0 (Agent)"] = get_subtree_text(child)
            elif child.dep_ in {"dobj", "obj", "attr"} and not row["A1 (Patient)"]:
                row["A1 (Patient)"] = get_subtree_text(child)
            elif child.dep_ in {"prep", "agent"}:
                pobj_children = [grandchild for grandchild in child.children if grandchild.dep_ == "pobj"]
                for pobj in pobj_children:
                    phrase = f"{child.text} {get_subtree_text(pobj)}"
                    if looks_like_time(pobj) and not row["AM-TMP"]:
                        row["AM-TMP"] = phrase
                    elif looks_like_location(pobj) and not row["AM-LOC"]:
                        row["AM-LOC"] = phrase
            elif child.dep_ in {"npadvmod", "advmod"} and looks_like_time(child) and not row["AM-TMP"]:
                row["AM-TMP"] = get_subtree_text(child)

        for ent in doc.ents:
            if ent.label_ in {"DATE", "TIME"} and not row["AM-TMP"] and ent.root.head == predicate:
                row["AM-TMP"] = ent.text
            elif ent.label_ in {"GPE", "LOC", "FAC"} and not row["AM-LOC"] and ent.root.head == predicate:
                row["AM-LOC"] = ent.text

        rows.append(row)

    return rows


def render_dependency_html(doc) -> str:
    return displacy.render(doc, style="dep", options={"compact": True})


def build_wsd_tab():
    st.subheader("WSD: 词义消歧对比测试")
    st.write("对比传统 `Lesk` 方法与 `BERT contextual embeddings` 在多义词语境中的表现。")

    sentence_1 = st.text_input(
        "句子 1",
        value="I went to the bank to deposit my money.",
    )
    target_word = st.text_input("目标词", value="bank")
    sentence_2 = st.text_input(
        "句子 2",
        value="I sat by the river bank.",
    )

    if st.button("运行 WSD 分析", type="primary"):
        normalized_target = normalize_target_word(target_word)
        if not sentence_1.strip() or not sentence_2.strip() or not normalized_target:
            st.error("请填写两个句子和目标词。")
            return

        if normalized_target not in sentence_1.lower() or normalized_target not in sentence_2.lower():
            st.warning("目标词最好同时出现在两个句子中，否则 BERT 对比会失败。")

        resources_ok, resource_error = ensure_nltk_resources()
        if not resources_ok:
            st.error(resource_error)
            st.info(
                "你可以先检查常见的 NLTK 数据目录，例如 `C:/Users/<用户名>/nltk_data`，"
                "看看里面是否有损坏的 `wordnet.zip` 或其他异常文件。"
            )
            return

        result = run_lesk_wsd(sentence_1, normalized_target)
        similarity, similarity_error = compute_similarity(sentence_1, sentence_2, normalized_target)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**传统方法：Lesk + WordNet**")
            if result.synset_name:
                st.write(f"Synset: `{result.synset_name}`")
                st.write(f"Definition: {result.definition}")
                candidate_synsets = wn.synsets(normalized_target)
                if candidate_synsets:
                    st.write("候选词义数量：", len(candidate_synsets))
            else:
                st.info("Lesk 未能给出明确词义。")

        with col2:
            st.markdown("**上下文向量方法：BERT**")
            if similarity_error:
                st.error(similarity_error)
            else:
                st.metric("Cosine Similarity", f"{similarity:.4f}")
                if similarity < 0.6:
                    st.caption("两个语境下的目标词表示差异较明显，符合动态上下文向量的预期。")
                else:
                    st.caption("两个语境下的目标词表示仍较接近，说明上下文差异未必足够强。")

        st.markdown("**实验观察建议**")
        st.write(
            "- 观察 Lesk 是否能把 `bank` 区分成金融机构和河岸。\n"
            "- 观察 BERT 余弦相似度是否因上下文变化而下降。"
        )


def build_srl_tab():
    st.subheader("SRL: 语义角色标注提取与可视化")
    st.write("使用 `spaCy dependency parsing` 和启发式规则近似抽取 `Predicate-Argument Structure`。")

    sentence = st.text_area(
        "输入英文句子",
        value="Apple is manufacturing new smartphones in China this year.",
        height=100,
    )

    if st.button("运行 SRL 分析", type="primary", key="run_srl"):
        if not sentence.strip():
            st.error("请输入句子。")
            return

        try:
            nlp = load_nlp_model()
        except RuntimeError as exc:
            st.error(str(exc))
            return

        doc = nlp(sentence)
        rows = extract_srl(doc)

        if rows:
            st.markdown("**SRL 结构化结果**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("没有识别到明显的谓词-论元结构。")

        st.markdown("**依存关系图**")
        dep_html = render_dependency_html(doc)
        st.components.v1.html(dep_html, height=420, scrolling=True)

        st.markdown("**实验观察建议**")
        st.write(
            "- 检查 `nsubj` 是否被映射到 `A0`。\n"
            "- 检查 `obj/dobj` 是否被映射到 `A1`。\n"
            "- 结合依存图理解谓词与论元之间的语法关系。"
        )


def main():
    st.title("Deep Semantic Analysis Platform")
    st.caption("Week 5 NLP Assignment: WSD and SRL")

    with st.sidebar:
        st.markdown("### 技术栈")
        st.write("- Streamlit")
        st.write("- NLTK / WordNet / Lesk")
        st.write("- transformers / BERT")
        st.write("- spaCy / displaCy")
        st.markdown("### 模块目标")
        st.write("1. 比较传统 WSD 与上下文向量方法")
        st.write("2. 用依存句法近似实现 SRL")

    tab_wsd, tab_srl = st.tabs(["WSD", "SRL"])

    with tab_wsd:
        build_wsd_tab()

    with tab_srl:
        build_srl_tab()


if __name__ == "__main__":
    main()
