import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import nltk
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from nltk.util import ngrams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


st.set_page_config(
    page_title="Language Modeling Lab",
    layout="wide",
)


DEFAULT_CORPUS = """The cat is walking in the garden.
The dog is running in the park.
The cat is sleeping on the sofa.
The language model learns patterns from text.
Natural language processing is fun and practical.
"""

DEFAULT_RNN_TEXT = (
    "hello world hello world hello world hello world hello world "
)

MASK_EXAMPLE = "The man went to the [MASK] to buy some milk."
GPT_EXAMPLE = "Natural language processing helps us"
PPL_EXAMPLE = """The cat is sleeping on the sofa.
This sentence is clear and grammatical.
sofa greenly idea thunder milk because.
"""


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;]", text.lower())


def load_reuters_corpus() -> Tuple[str, str]:
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        try:
            nltk.download("reuters", quiet=True)
        except Exception:
            return DEFAULT_CORPUS, "未能下载 Reuters，已回退到内置示例语料。"

    try:
        from nltk.corpus import reuters

        words = reuters.words()[:500]
        sample = " ".join(words)
        return sample, "已加载 NLTK Reuters 前 500 个词作为示例语料。"
    except Exception:
        return DEFAULT_CORPUS, "读取 Reuters 失败，已回退到内置示例语料。"


@dataclass
class NGramLanguageModel:
    n: int
    tokens: List[str]
    vocab: List[str]
    vocab_size: int
    ngram_counts: Counter
    context_counts: Counter
    word_counts: Counter


def build_ngram_model(text: str, n: int) -> NGramLanguageModel:
    tokens = simple_tokenize(text)
    if not tokens:
        tokens = simple_tokenize(DEFAULT_CORPUS)

    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
    ngram_counts = Counter(ngrams(padded, n))
    context_counts = Counter(ngrams(padded, n - 1)) if n > 1 else Counter()
    word_counts = Counter(tokens)
    vocab = sorted(set(tokens + ["</s>"]))

    return NGramLanguageModel(
        n=n,
        tokens=tokens,
        vocab=vocab,
        vocab_size=len(vocab),
        ngram_counts=ngram_counts,
        context_counts=context_counts,
        word_counts=word_counts,
    )


def sentence_probability(
    sentence: str,
    model: NGramLanguageModel,
    smoothing_delta: float = 0.0,
) -> Tuple[float, List[Dict[str, object]]]:
    sent_tokens = simple_tokenize(sentence)
    if not sent_tokens:
        return 0.0, []

    padded = ["<s>"] * (model.n - 1) + sent_tokens + ["</s>"]
    details: List[Dict[str, object]] = []
    joint_prob = 1.0

    for gram in ngrams(padded, model.n):
        context = gram[:-1]
        target = gram[-1]
        ngram_count = model.ngram_counts.get(gram, 0)

        if model.n == 1:
            context_count = len(model.tokens) + 1
        else:
            context_count = model.context_counts.get(context, 0)

        numerator = ngram_count + smoothing_delta
        denominator = context_count + smoothing_delta * model.vocab_size
        prob = numerator / denominator if denominator > 0 else 0.0
        joint_prob *= prob

        details.append(
            {
                "context": " ".join(context) if context else "<root>",
                "target": target,
                "count(context,target)": ngram_count,
                "count(context)": context_count,
                "probability": prob,
                "seen": "Yes" if ngram_count > 0 else "No",
            }
        )

    return joint_prob, details


class CharRNNLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, cell_type: str):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.cell_type = cell_type

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def build_char_dataset(text: str, seq_len: int = 12):
    text = text or DEFAULT_RNN_TEXT
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}

    encoded = [stoi[ch] for ch in text]
    if len(encoded) <= seq_len:
        encoded = encoded * (seq_len + 2)

    xs = []
    ys = []
    for i in range(len(encoded) - seq_len):
        xs.append(encoded[i : i + seq_len])
        ys.append(encoded[i + 1 : i + seq_len + 1])

    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor, stoi, itos


def generate_text(
    model: CharRNNLM,
    seed: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    length: int = 50,
) -> str:
    if not seed:
        seed = next(iter(stoi))

    model.eval()
    current = seed
    hidden = None

    with torch.no_grad():
        for ch in seed[:-1]:
            if ch in stoi:
                token = torch.tensor([[stoi[ch]]], dtype=torch.long)
                _, hidden = model(token, hidden)

        last_char = seed[-1]
        if last_char not in stoi:
            last_char = next(iter(stoi))

        for _ in range(length):
            token = torch.tensor([[stoi[last_char]]], dtype=torch.long)
            logits, hidden = model(token, hidden)
            next_id = torch.argmax(logits[0, -1]).item()
            next_char = itos[next_id]
            current += next_char
            last_char = next_char

    return current


@st.cache_resource(show_spinner=False)
def load_fill_mask_pipeline():
    return pipeline("fill-mask", model="bert-base-uncased")


@st.cache_resource(show_spinner=False)
def load_text_generation_pipeline():
    return pipeline("text-generation", model="gpt2")


@st.cache_resource(show_spinner=False)
def load_gpt2_for_ppl():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def compute_gpt2_ppl(sentences: Sequence[str]) -> pd.DataFrame:
    tokenizer, model = load_gpt2_for_ppl()
    rows = []

    for sentence in sentences:
        text = sentence.strip()
        if not text:
            continue

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

        with torch.no_grad():
            output = model(**enc, labels=enc["input_ids"])
            loss = output.loss.item()

        ppl = math.exp(loss)
        rows.append(
            {
                "sentence": text,
                "cross_entropy_loss": round(loss, 4),
                "perplexity": round(ppl, 4),
            }
        )

    return pd.DataFrame(rows)


def render_header():
    st.title("语言模型训练与对比分析平台")
    st.caption(
        "基于 Streamlit + nltk + torch + transformers，覆盖统计语言模型、自训练 RNN、BERT/GPT-2 对比与 PPL 评价。"
    )


def render_ngram_tab():
    st.subheader("模块 1：n-gram & Add-one Smoothing")
    st.write("这个模块用于演示统计语言模型的联合概率计算，以及未见事件导致的零概率问题。")

    corpus_source = st.radio(
        "语料来源",
        ["内置示例语料", "NLTK Reuters 示例", "手动输入"],
        horizontal=True,
    )

    helper_text = ""
    if corpus_source == "NLTK Reuters 示例":
        corpus_text, helper_text = load_reuters_corpus()
    elif corpus_source == "手动输入":
        corpus_text = st.session_state.get("manual_corpus", DEFAULT_CORPUS)
    else:
        corpus_text = DEFAULT_CORPUS

    n = st.selectbox("n-gram 阶数", [2, 3], index=1)
    if helper_text:
        st.info(helper_text)

    corpus_text = st.text_area(
        "基础英文语料",
        value=corpus_text,
        height=180,
        key="manual_corpus" if corpus_source == "手动输入" else None,
    )

    model = build_ngram_model(corpus_text, n=n)
    st.write(
        f"当前语料共有 `{len(model.tokens)}` 个 token，词表大小 `{model.vocab_size}`。"
    )

    freq_df = (
        pd.DataFrame(model.word_counts.most_common(15), columns=["token", "count"])
        if model.word_counts
        else pd.DataFrame(columns=["token", "count"])
    )
    st.dataframe(freq_df, use_container_width=True, hide_index=True)

    sentence = st.text_input(
        "输入待计算句子",
        value="the cat is running in the garden",
    )
    use_smoothing = st.checkbox("开启 Add-one / Laplace smoothing", value=False)

    raw_prob, raw_details = sentence_probability(sentence, model, smoothing_delta=0.0)
    smooth_prob, smooth_details = sentence_probability(sentence, model, smoothing_delta=1.0)
    selected_prob = smooth_prob if use_smoothing else raw_prob
    selected_details = smooth_details if use_smoothing else raw_details

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("当前模式句子联合概率", f"{selected_prob:.8e}")
    metric_col2.metric("未平滑概率", f"{raw_prob:.8e}")
    metric_col3.metric("Add-one 后概率", f"{smooth_prob:.8e}")

    if raw_prob == 0.0 and smooth_prob > 0.0:
        st.warning("检测到未见 n-gram：未平滑时联合概率为 0，开启 Add-one 后概率被重新分配。")

    st.dataframe(pd.DataFrame(selected_details), use_container_width=True, hide_index=True)


def render_rnn_tab():
    st.subheader("模块 2：Train Your Own RNN Language Model")
    st.write("建议先用重复模式很强的短文本做实验，方便观察隐藏状态如何学习序列规律。")

    training_text = st.text_area(
        "自定义训练语料",
        value=DEFAULT_RNN_TEXT,
        height=160,
    )

    col1, col2, col3, col4 = st.columns(4)
    hidden_size = col1.slider("Hidden Size", 16, 128, 64, step=16)
    epochs = col2.slider("Epochs", 10, 200, 60, step=10)
    learning_rate = col3.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    cell_type = col4.selectbox("RNN Cell", ["RNN", "LSTM"], index=1)

    if st.button("开始训练 RNN-LM", type="primary"):
        x_tensor, y_tensor, stoi, itos = build_char_dataset(training_text)
        model = CharRNNLM(len(stoi), hidden_size, cell_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        chart_placeholder = st.empty()
        progress = st.progress(0)
        losses = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x_tensor)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_tensor.reshape(-1))
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            chart_placeholder.line_chart(pd.DataFrame({"loss": losses}))
            progress.progress((epoch + 1) / epochs)

        st.session_state["rnn_model"] = model
        st.session_state["rnn_stoi"] = stoi
        st.session_state["rnn_itos"] = itos
        st.session_state["rnn_losses"] = losses
        st.success("训练完成，可以用 Seed 生成文本了。")

    if "rnn_losses" in st.session_state:
        st.write(f"最终训练损失：`{st.session_state['rnn_losses'][-1]:.4f}`")
        seed = st.text_input("起始字符 / Seed", value="h", key="rnn_seed")
        gen_len = st.slider("生成长度", 20, 100, 50, step=10)

        if st.button("生成文本"):
            generated = generate_text(
                st.session_state["rnn_model"],
                seed,
                st.session_state["rnn_stoi"],
                st.session_state["rnn_itos"],
                length=gen_len,
            )
            st.code(generated)


def render_pretrained_tab():
    st.subheader("模块 3：Masked LM vs. Causal LM")
    st.write("这个模块用于直观比较 BERT 的掩码预测和 GPT-2 的从左到右续写。")

    left, right = st.columns(2)

    with left:
        st.markdown("#### BERT (`bert-base-uncased`)")
        mask_sentence = st.text_input("输入包含 [MASK] 的句子", value=MASK_EXAMPLE)
        if st.button("运行 BERT 填空"):
            if "[MASK]" not in mask_sentence:
                st.error("请输入包含 [MASK] 的句子。")
            else:
                try:
                    fill_mask = load_fill_mask_pipeline()
                    predictions = fill_mask(mask_sentence, top_k=5)
                    bert_df = pd.DataFrame(
                        [
                            {
                                "token": item["token_str"].strip(),
                                "score": round(float(item["score"]), 4),
                                "sequence": item["sequence"],
                            }
                            for item in predictions
                        ]
                    )
                    st.dataframe(bert_df, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"BERT 加载或推理失败：{exc}")

    with right:
        st.markdown("#### GPT-2 (`gpt2`)")
        prompt = st.text_area("输入续写 Prompt", value=GPT_EXAMPLE, height=120)
        if st.button("运行 GPT-2 续写"):
            try:
                text_gen = load_text_generation_pipeline()
                outputs = text_gen(
                    prompt,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                    num_return_sequences=1,
                    pad_token_id=text_gen.model.config.eos_token_id,
                )
                st.code(outputs[0]["generated_text"])
            except Exception as exc:
                st.error(f"GPT-2 加载或推理失败：{exc}")


def render_ppl_tab():
    st.subheader("模块 4：Cross-Entropy & Perplexity")
    st.write("使用 GPT-2 计算句子的交叉熵损失，并根据 `PPL = exp(loss)` 输出困惑度。")

    ppl_input = st.text_area(
        "每行输入一个待评估句子",
        value=PPL_EXAMPLE,
        height=180,
    )

    if st.button("计算 PPL"):
        sentences = [line for line in ppl_input.splitlines() if line.strip()]
        if not sentences:
            st.error("请至少输入一条句子。")
            return

        try:
            result_df = compute_gpt2_ppl(sentences)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            if len(result_df) >= 2:
                best_row = result_df.loc[result_df["perplexity"].idxmin()]
                worst_row = result_df.loc[result_df["perplexity"].idxmax()]
                st.info(
                    "一般来说，PPL 越小，说明模型越不“困惑”。"
                    f" 当前最低 PPL 句子是：`{best_row['sentence']}`；"
                    f" 最高 PPL 句子是：`{worst_row['sentence']}`。"
                )
        except Exception as exc:
            st.error(f"PPL 计算失败：{exc}")


def main():
    render_header()
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. n-gram & Smoothing",
            "2. Train RNN-LM",
            "3. BERT vs GPT-2",
            "4. Perplexity",
        ]
    )

    with tab1:
        render_ngram_tab()
    with tab2:
        render_rnn_tab()
    with tab3:
        render_pretrained_tab()
    with tab4:
        render_ppl_tab()


if __name__ == "__main__":
    main()
