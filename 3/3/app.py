from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from gensim import downloader as api
from gensim.models import FastText, Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_CORPUS_PATH = BASE_DIR / "data" / "sample_corpus.txt"


def ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass


def load_default_corpus() -> str:
    return SAMPLE_CORPUS_PATH.read_text(encoding="utf-8")


def sentence_split(text: str) -> list[str]:
    ensure_nltk_resources()
    sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    return sentences


def tokenize_for_embeddings(text: str) -> list[list[str]]:
    ensure_nltk_resources()
    sentences = sentence_split(text)
    tokenized_sentences = []
    for sentence in sentences:
        tokens = [
            token.lower()
            for token in nltk.word_tokenize(sentence)
            if any(char.isalpha() for char in token)
        ]
        if tokens:
            tokenized_sentences.append(tokens)
    return tokenized_sentences


def build_tfidf_dataframe(docs: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    df = pd.DataFrame(
        matrix.toarray(),
        index=[f"Sentence {i + 1}" for i in range(len(docs))],
        columns=vectorizer.get_feature_names_out(),
    )
    return df, matrix


def extract_top_keywords(tfidf_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    scores = tfidf_df.sum(axis=0).sort_values(ascending=False).head(top_n)
    return scores.reset_index().rename(columns={"index": "keyword", 0: "score"})


def compute_lsa_coordinates(docs: list[str], matrix_type: str) -> pd.DataFrame:
    if matrix_type == "TF-IDF":
        vectorizer = TfidfVectorizer(stop_words="english")
    else:
        vectorizer = CountVectorizer(stop_words="english")

    doc_term = vectorizer.fit_transform(docs)
    term_doc = doc_term.T
    n_components = 2
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    coords = svd.fit_transform(term_doc)

    terms = vectorizer.get_feature_names_out()
    if matrix_type == "TF-IDF":
        strength = np.asarray(doc_term.sum(axis=0)).ravel()
    else:
        strength = np.asarray(doc_term.sum(axis=0)).ravel()

    lsa_df = pd.DataFrame(coords, columns=["x", "y"])
    lsa_df["term"] = terms
    lsa_df["strength"] = strength
    lsa_df = lsa_df.sort_values("strength", ascending=False).reset_index(drop=True)
    return lsa_df


def train_word2vec_model(
    tokenized_sentences: list[list[str]],
    architecture: str,
    window: int,
    vector_size: int,
    epochs: int,
) -> Word2Vec:
    sg = 0 if architecture == "CBOW" else 1
    return Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=1,
        sg=sg,
        epochs=epochs,
        seed=42,
    )


def train_fasttext_model(
    tokenized_sentences: list[list[str]],
    window: int,
    vector_size: int,
    epochs: int,
) -> FastText:
    return FastText(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=1,
        epochs=epochs,
        seed=42,
    )


@st.cache_resource(show_spinner=False)
def load_glove_model():
    return api.load("glove-twitter-25")


def average_sentence_vector(sentence: str, model: FastText) -> np.ndarray | None:
    ensure_nltk_resources()
    tokens = [
        token.lower()
        for token in nltk.word_tokenize(sentence)
        if any(char.isalpha() for char in token)
    ]
    if not tokens:
        return None
    vectors = [model.wv[token] for token in tokens]
    return np.mean(vectors, axis=0)


def cosine_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(cosine_similarity([vec_a], [vec_b])[0][0])


def plot_lsa(lsa_df: pd.DataFrame, max_terms: int = 30) -> plt.Figure:
    plot_df = lsa_df.head(max_terms)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot_df["x"], plot_df["y"], s=60, alpha=0.75)
    for _, row in plot_df.iterrows():
        ax.text(row["x"] + 0.01, row["y"] + 0.01, row["term"], fontsize=9)
    ax.set_title("LSA 2D Projection of Terms")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


st.set_page_config(
    page_title="Semantic Analysis Playground",
    layout="wide",
)

st.title("Week 4 Semantic Analysis Playground")
st.caption(
    "Integrating TF-IDF, LSA, Word2Vec, GloVe, FastText, and simple Sent2Vec-style averaging."
)

default_corpus = load_default_corpus()
corpus_text = st.text_area(
    "English corpus for training / analysis",
    value=default_corpus,
    height=260,
    help="Recommended length: around 500-1000 English words.",
)

docs = sentence_split(corpus_text) if corpus_text.strip() else []
tokenized_sentences = tokenize_for_embeddings(corpus_text) if corpus_text.strip() else []

if not docs or len(docs) < 3:
    st.warning("Please provide at least 3 English sentences so the four modules can work reliably.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Module 1: TF-IDF + LSA",
        "Module 2: Word2Vec",
        "Module 3: GloVe",
        "Module 4: FastText + Sent2Vec",
    ]
)


with tab1:
    st.subheader("Traditional Statistical Representation")
    st.write(
        "The corpus is split into sentences as documents. We first compute TF-IDF, "
        "then reduce a term-document matrix to 2 dimensions with TruncatedSVD to simulate LSA."
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        tfidf_df, _ = build_tfidf_dataframe(docs)
        st.markdown("**Sentence-level TF-IDF matrix (preview)**")
        st.dataframe(tfidf_df.round(3).iloc[:, :20], use_container_width=True)

        top_keywords = extract_top_keywords(tfidf_df, top_n=5)
        st.markdown("**Top 5 keywords by aggregated TF-IDF weight**")
        st.dataframe(top_keywords, use_container_width=True, hide_index=True)

    with right_col:
        matrix_type = st.radio(
            "Matrix for LSA projection",
            options=["TF-IDF", "CountVectorizer"],
            horizontal=True,
        )
        lsa_df = compute_lsa_coordinates(docs, matrix_type=matrix_type)
        st.markdown("**LSA 2D visualization of terms**")
        st.pyplot(plot_lsa(lsa_df), clear_figure=True)
        st.info(
            "If words often appear in similar sentence contexts, LSA may place them closer "
            "in the low-dimensional semantic space."
        )


with tab2:
    st.subheader("Word2Vec: CBOW vs Skip-Gram")
    st.write(
        "CBOW predicts the center word from context, while Skip-Gram predicts context words "
        "from a center word. On small corpora, their nearest neighbors can differ noticeably."
    )

    control_col, result_col = st.columns([1, 1])

    with control_col:
        architecture = st.radio("Training architecture", ["CBOW", "Skip-Gram"], horizontal=True)
        window = st.slider("Context window", min_value=2, max_value=10, value=4)
        vector_size = st.slider("Vector size", min_value=20, max_value=200, value=60, step=10)
        epochs = st.slider("Training epochs", min_value=20, max_value=300, value=120, step=20)
        query_word = st.text_input("Target word", value="city").strip().lower()

        w2v_model = train_word2vec_model(
            tokenized_sentences=tokenized_sentences,
            architecture=architecture,
            window=window,
            vector_size=vector_size,
            epochs=epochs,
        )

    with result_col:
        vocab_size = len(w2v_model.wv)
        st.metric("Vocabulary size", vocab_size)
        if query_word:
            try:
                similar_words = w2v_model.wv.most_similar(query_word, topn=5)
                similar_df = pd.DataFrame(similar_words, columns=["word", "cosine_similarity"])
                st.markdown(f"**Top 5 similar words for `{query_word}`**")
                st.dataframe(similar_df, use_container_width=True, hide_index=True)
            except KeyError:
                st.error(f"`{query_word}` is not in the current Word2Vec vocabulary.")
        else:
            st.info("Enter a word to inspect its nearest neighbors.")


with tab3:
    st.subheader("Pretrained GloVe and Word Analogies")
    st.write(
        "This module uses `glove-twitter-25`. The first load may take time because the model "
        "may need to be downloaded."
    )

    try:
        glove_model = load_glove_model()
        analogy_col, similarity_col = st.columns([1, 1])

        with analogy_col:
            word_a = st.text_input("A", value="king", key="glove_a").strip().lower()
            word_b = st.text_input("B", value="man", key="glove_b").strip().lower()
            word_c = st.text_input("C", value="woman", key="glove_c").strip().lower()

            if word_a and word_b and word_c:
                try:
                    analogy_results = glove_model.most_similar(
                        positive=[word_a, word_c],
                        negative=[word_b],
                        topn=5,
                    )
                    analogy_df = pd.DataFrame(
                        analogy_results,
                        columns=["candidate", "score"],
                    )
                    st.markdown(
                        f"**Analogy result for `{word_a} - {word_b} + {word_c}`**"
                    )
                    st.dataframe(analogy_df, use_container_width=True, hide_index=True)
                    st.success(f"Top prediction: `{analogy_df.iloc[0]['candidate']}`")
                except KeyError as exc:
                    st.error(f"GloVe vocabulary does not contain: {exc}")

        with similarity_col:
            sim_word_1 = st.text_input("Similarity word 1", value="doctor", key="sim1").strip().lower()
            sim_word_2 = st.text_input("Similarity word 2", value="nurse", key="sim2").strip().lower()

            if sim_word_1 and sim_word_2:
                try:
                    score = glove_model.similarity(sim_word_1, sim_word_2)
                    st.metric(
                        label=f"Cosine similarity: {sim_word_1} vs {sim_word_2}",
                        value=f"{score:.4f}",
                    )
                except KeyError as exc:
                    st.error(f"GloVe vocabulary does not contain: {exc}")
    except Exception as exc:
        st.error(
            "Unable to load `glove-twitter-25`. Please check your network or gensim data cache."
        )
        st.code(str(exc))


with tab4:
    st.subheader("FastText OOV Robustness and Sentence Vectors")
    st.write(
        "FastText uses subword n-grams, so it can often produce useful vectors for misspelled "
        "or unseen words. We also build sentence vectors with simple average pooling."
    )

    train_col, sentence_col = st.columns([1, 1])

    with train_col:
        ft_window = st.slider("FastText window", min_value=2, max_value=10, value=4)
        ft_vector_size = st.slider("FastText vector size", min_value=20, max_value=200, value=60, step=10)
        ft_epochs = st.slider("FastText epochs", min_value=20, max_value=300, value=120, step=20)
        oov_word = st.text_input("OOV / typo test word", value="computeer").strip().lower()

        baseline_w2v = train_word2vec_model(
            tokenized_sentences=tokenized_sentences,
            architecture="Skip-Gram",
            window=ft_window,
            vector_size=ft_vector_size,
            epochs=ft_epochs,
        )
        fasttext_model = train_fasttext_model(
            tokenized_sentences=tokenized_sentences,
            window=ft_window,
            vector_size=ft_vector_size,
            epochs=ft_epochs,
        )

        if oov_word:
            st.markdown(f"**OOV comparison for `{oov_word}`**")
            try:
                _ = baseline_w2v.wv[oov_word]
                st.success("Word2Vec found this word in the vocabulary.")
            except KeyError:
                st.warning("Word2Vec: 未登录词 (out-of-vocabulary).")

            try:
                fasttext_neighbors = fasttext_model.wv.most_similar(oov_word, topn=5)
                ft_df = pd.DataFrame(fasttext_neighbors, columns=["word", "cosine_similarity"])
                st.markdown("**FastText nearest neighbors**")
                st.dataframe(ft_df, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"FastText failed on this input: {exc}")

    with sentence_col:
        sentence_1 = st.text_area(
            "Sentence 1",
            value="The programmer writes code and improves the model after each experiment.",
            height=120,
        )
        sentence_2 = st.text_area(
            "Sentence 2",
            value="The engineer tests the system and refines the algorithm after every trial.",
            height=120,
        )

        vec_1 = average_sentence_vector(sentence_1, fasttext_model)
        vec_2 = average_sentence_vector(sentence_2, fasttext_model)

        if vec_1 is not None and vec_2 is not None:
            score = cosine_score(vec_1, vec_2)
            st.metric("Sentence cosine similarity", f"{score:.4f}")
            st.info(
                "This is a simple Sent2Vec-style approximation: tokenize the sentence, "
                "retrieve word vectors, and average them into one global vector."
            )
        else:
            st.warning("Please enter two non-empty English sentences.")
