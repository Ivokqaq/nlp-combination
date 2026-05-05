import html
import re
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import streamlit as st

try:
    import spacy
except Exception:
    spacy = None


@dataclass
class TreeNode:
    label: str
    children: List["TreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


def _tree_depth(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    return 1 + max(_tree_depth(child) for child in node.children)


def _leaf_count(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    return sum(_leaf_count(child) for child in node.children)


def _assign_tree_positions(
    node: TreeNode,
    depth: int,
    leaf_x: List[int],
    positions: dict,
) -> float:
    if node.is_leaf:
        x = leaf_x[0]
        leaf_x[0] += 1
        positions[id(node)] = (x, depth)
        return x
    child_xs = [_assign_tree_positions(child, depth + 1, leaf_x, positions) for child in node.children]
    x = sum(child_xs) / len(child_xs)
    positions[id(node)] = (x, depth)
    return x


def render_constituency_svg(root: TreeNode) -> str:
    positions = {}
    _assign_tree_positions(root, depth=0, leaf_x=[0], positions=positions)
    depth = _tree_depth(root)
    leaves = max(1, _leaf_count(root))

    x_step = 90
    y_step = 90
    margin_x = 40
    margin_y = 40
    width = leaves * x_step + margin_x * 2
    height = depth * y_step + margin_y * 2

    lines = []
    labels = []

    def walk(node: TreeNode):
        x, y = positions[id(node)]
        px = margin_x + x * x_step
        py = margin_y + y * y_step
        labels.append(
            f"<text x='{px}' y='{py}' text-anchor='middle' font-size='15' font-family='Arial'>{html.escape(node.label)}</text>"
        )
        for child in node.children:
            cx, cy = positions[id(child)]
            cpx = margin_x + cx * x_step
            cpy = margin_y + cy * y_step
            lines.append(f"<line x1='{px}' y1='{py + 8}' x2='{cpx}' y2='{cpy - 18}' stroke='#6b7280' stroke-width='1.5'/>")
            walk(child)

    walk(root)

    return (
        f"<svg width='100%' height='{height}' viewBox='0 0 {width} {height}' preserveAspectRatio='xMinYMin meet' "
        "xmlns='http://www.w3.org/2000/svg' style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px'>"
        + "".join(lines)
        + "".join(labels)
        + "</svg>"
    )


def render_dependency_svg(tokens: List[str], edges: List[Tuple[int, int, str]]) -> str:
    n = len(tokens)
    x_step = 95
    baseline_y = 260
    margin_x = 40
    width = margin_x * 2 + max(1, n - 1) * x_step + 80
    height = 320

    token_x = {i + 1: margin_x + i * x_step + 40 for i in range(n)}
    marker = (
        "<defs><marker id='arrow' markerWidth='10' markerHeight='8' refX='9' refY='4' orient='auto' markerUnits='strokeWidth'>"
        "<path d='M0,0 L10,4 L0,8 z' fill='#2563eb'/></marker></defs>"
    )
    lines = [marker]
    labels = []

    for idx, tok in enumerate(tokens, start=1):
        x = token_x[idx]
        labels.append(f"<text x='{x}' y='{baseline_y}' text-anchor='middle' font-size='16' font-family='Arial'>{html.escape(tok)}</text>")
        labels.append(f"<text x='{x}' y='{baseline_y + 22}' text-anchor='middle' font-size='12' fill='#64748b'>{idx}</text>")

    for head, dep, rel in edges:
        dx = token_x[dep]
        if head == 0:
            lines.append(
                f"<line x1='{dx}' y1='55' x2='{dx}' y2='{baseline_y - 28}' stroke='#2563eb' stroke-width='1.7' marker-end='url(#arrow)'/>"
            )
            labels.append(
                f"<text x='{dx + 6}' y='48' font-size='12' fill='#1d4ed8' font-family='Arial'>{html.escape(rel)}</text>"
            )
            continue

        hx = token_x[head]
        left = min(hx, dx)
        right = max(hx, dx)
        span = right - left
        arc_h = max(45, min(150, 30 + span * 0.6))
        y0 = baseline_y - 35
        ctrl_y = y0 - arc_h
        path = f"M {hx} {y0} Q {(hx + dx) / 2:.2f} {ctrl_y:.2f} {dx} {y0}"
        lines.append(f"<path d='{path}' fill='none' stroke='#2563eb' stroke-width='1.7' marker-end='url(#arrow)'/>")
        labels.append(
            f"<text x='{(hx + dx) / 2:.2f}' y='{ctrl_y - 6:.2f}' text-anchor='middle' font-size='12' fill='#1d4ed8' font-family='Arial'>{html.escape(rel)}</text>"
        )

    return (
        f"<svg width='100%' height='{height}' viewBox='0 0 {width} {height}' preserveAspectRatio='xMinYMin meet' "
        "xmlns='http://www.w3.org/2000/svg' style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px'>"
        + "".join(lines)
        + "".join(labels)
        + "</svg>"
    )


def make_word(tag: str, word: str) -> TreeNode:
    return TreeNode(tag, [TreeNode(word)])


def build_demo_structures(attach_to: str):
    tokens = ["The", "boy", "saw", "the", "man", "with", "the", "telescope"]

    np_subject = TreeNode("NP", [make_word("Det", "The"), make_word("N", "boy")])
    np_object = TreeNode("NP", [make_word("Det", "the"), make_word("N", "man")])
    pp = TreeNode("PP", [make_word("P", "with"), TreeNode("NP", [make_word("Det", "the"), make_word("N", "telescope")])])

    if attach_to == "NP":
        np_object.children.append(pp)
        vp = TreeNode("VP", [make_word("V", "saw"), np_object])
        edges = [
            (0, 3, "root"),
            (3, 2, "nsubj"),
            (2, 1, "det"),
            (3, 5, "obj"),
            (5, 4, "det"),
            (5, 8, "nmod"),
            (8, 7, "det"),
            (8, 6, "case"),
        ]
    else:
        vp = TreeNode("VP", [make_word("V", "saw"), np_object, pp])
        edges = [
            (0, 3, "root"),
            (3, 2, "nsubj"),
            (2, 1, "det"),
            (3, 5, "obj"),
            (5, 4, "det"),
            (3, 8, "obl"),
            (8, 7, "det"),
            (8, 6, "case"),
        ]

    root = TreeNode("S", [np_subject, vp])
    return tokens, root, edges



def build_fruit_flies_structures(reading: str):
    tokens = ["Fruit", "flies", "like", "a", "banana"]

    if reading == "NOUN":
        # [Fruit flies] like [a banana]
        root = TreeNode(
            "S",
            [
                TreeNode("NP", [make_word("N", "Fruit"), make_word("N", "flies")]),
                TreeNode("VP", [make_word("V", "like"), TreeNode("NP", [make_word("Det", "a"), make_word("N", "banana")])]),
            ],
        )
        edges = [
            (0, 3, "root"),
            (3, 2, "nsubj"),
            (2, 1, "compound"),
            (3, 5, "obj"),
            (5, 4, "det"),
        ]
    else:
        # Fruit flies [like a banana]
        root = TreeNode(
            "S",
            [
                TreeNode("NP", [make_word("N", "Fruit")]),
                TreeNode("VP", [make_word("V", "flies"), TreeNode("PP", [make_word("P", "like"), TreeNode("NP", [make_word("Det", "a"), make_word("N", "banana")])])]),
            ],
        )
        edges = [
            (0, 2, "root"),
            (2, 1, "nsubj"),
            (2, 5, "obl"),
            (5, 3, "case"),
            (5, 4, "det"),
        ]

    return tokens, root, edges


def build_chinese_ambiguity_structures(sentence: str, reading: str):
    nj_bridge = "南京市长江大桥"
    dog_sent = "咬死了猎人的狗"

    if sentence == nj_bridge:
        tokens = ["南京市", "长江", "大桥"]
        if reading == "LOC":
            # [Bite-kill [hunter DE dog]]
            root = TreeNode("NP", [make_word("N", tokens[0]), TreeNode("NP", [make_word("N", tokens[1]), make_word("N", tokens[2])])])
            edges = [
                (0, 3, "root"),
                (3, 2, "compound"),
                (3, 1, "nmod"),
            ]
            gloss = "Reading A: Nanjing city's Yangtze River Bridge."
        else:
            # [[City+Yangtze] Bridge]
            root = TreeNode("NP", [TreeNode("NP", [make_word("N", tokens[0]), make_word("N", tokens[1])]), make_word("N", tokens[2])])
            edges = [
                (0, 3, "root"),
                (3, 2, "compound"),
                (2, 1, "compound"),
            ]
            gloss = "Reading B: a bridge over the Nanjing-city section of the Yangtze."
        return tokens, root, edges, gloss

    if sentence == dog_sent:
        tokens = ["咬死了", "猎人", "的", "狗"]
        if reading == "DOG_BITE":
            # [[Bite-kill hunter] DE dog]
            root = TreeNode(
                "NP",
                [
                    TreeNode("CP", [TreeNode("VP", [make_word("V", tokens[0]), make_word("N", tokens[1])]), make_word("DEG", tokens[2])]),
                    make_word("N", tokens[3]),
                ],
            )
            edges = [
                (0, 4, "root"),
                (4, 1, "acl"),
                (1, 2, "obj"),
                (4, 3, "case"),
            ]
            gloss = "Reading A: the dog that bit the hunter to death."
        else:
            # [Bite-kill [hunter DE dog]]
            root = TreeNode(
                "VP",
                [
                    make_word("V", tokens[0]),
                    TreeNode("NP", [TreeNode("NP", [make_word("N", tokens[1])]), make_word("DEG", tokens[2]), make_word("N", tokens[3])]),
                ],
            )
            edges = [
                (0, 1, "root"),
                (1, 4, "obj"),
                (4, 2, "nmod"),
                (4, 3, "case"),
            ]
            gloss = "Reading B: (someone) bit the hunter's dog to death."
        return tokens, root, edges, gloss

    return None, None, None, "Unsupported sentence. Try one of the built-in examples."


def simple_parse(sentence: str, attach_to: str):
    words = re.findall(r"[A-Za-z']+", sentence)
    if len(words) < 3:
        return None
    lw = [w.lower() for w in words]
    det = {"the", "a", "an", "this", "that"}
    prep = {"with", "on", "in", "at", "to", "from", "by"}
    verbs = {"saw", "eat", "eats", "ate", "sleep", "sleeps", "kicked", "kicks", "liked", "likes", "chased", "chases"}

    subj_start = 0
    subj_end = 2 if len(words) >= 2 and lw[0] in det else 1
    if len(words) <= subj_end:
        return None
    verb_idx = subj_end
    if lw[verb_idx] not in verbs and len(words) > verb_idx + 1 and lw[verb_idx + 1] in verbs:
        verb_idx += 1

    obj_start = verb_idx + 1
    if obj_start >= len(words):
        return None
    obj_end = obj_start + 2 if len(words) > obj_start + 1 and lw[obj_start] in det else obj_start + 1
    obj_end = min(obj_end, len(words))
    pp_idx = next((i for i in range(obj_end, len(words)) if lw[i] in prep), -1)

    def build_np(span_words: List[str]) -> Optional[TreeNode]:
        if not span_words:
            return None
        if len(span_words) >= 2 and span_words[0].lower() in det:
            return TreeNode("NP", [make_word("Det", span_words[0]), make_word("N", " ".join(span_words[1:]))])
        return TreeNode("NP", [make_word("N", " ".join(span_words))])

    np_subj = build_np(words[subj_start:subj_end])
    np_obj = build_np(words[obj_start:obj_end]) if obj_start < len(words) else None
    if np_subj is None or np_obj is None:
        return None

    edges = []
    verb_token_idx = verb_idx + 1
    edges.append((0, verb_token_idx, "root"))
    subj_head = subj_end
    obj_head = obj_end
    edges.append((verb_token_idx, subj_head, "nsubj"))
    if subj_end - subj_start == 2 and lw[subj_start] in det:
        edges.append((subj_head, subj_start + 1, "det"))
    edges.append((verb_token_idx, obj_head, "obj"))
    if obj_end - obj_start == 2 and lw[obj_start] in det:
        edges.append((obj_head, obj_start + 1, "det"))

    vp_children = [make_word("V", words[verb_idx]), np_obj]

    if pp_idx != -1 and pp_idx + 1 < len(words):
        pp_np = build_np(words[pp_idx + 1 :])
        if pp_np:
            pp = TreeNode("PP", [make_word("P", words[pp_idx]), pp_np])
            if attach_to == "NP":
                np_obj.children.append(pp)
            else:
                vp_children.append(pp)

            pp_head = len(words)
            edges.append(((obj_head if attach_to == "NP" else verb_token_idx), pp_head, "nmod" if attach_to == "NP" else "obl"))
            edges.append((pp_head, pp_idx + 1, "case"))
            if len(words) - (pp_idx + 1) >= 2 and lw[pp_idx + 1] in det:
                edges.append((pp_head, pp_idx + 2, "det"))

    root = TreeNode("S", [np_subj, TreeNode("VP", vp_children)])
    return words, root, edges


@st.cache_resource
def get_spacy_nlp():
    if spacy is None:
        return None, "spaCy is not installed."

    model_names = ["en_core_web_sm", "en_core_web_md"]
    for name in model_names:
        try:
            return spacy.load(name), None
        except Exception:
            continue
    return None, "spaCy is installed, but no English model was found (e.g., en_core_web_sm)."


def extract_core_arguments_spacy(sentence: str):
    nlp, err = get_spacy_nlp()
    if nlp is None:
        return [], err

    doc = nlp(sentence)
    target_labels = {"nsubj", "dobj", "pobj", "root"}
    rows = []
    for tok in doc:
        dep_label = tok.dep_.lower()
        if dep_label in target_labels:
            rows.append(
                {
                    "dep": tok.dep_,
                    "token": tok.text,
                    "head": tok.head.text if tok.dep_.upper() != "ROOT" else "ROOT",
                    "lemma": tok.lemma_,
                    "index": tok.i + 1,
                }
            )
    return rows, None


def extract_core_arguments_fallback(tokens: List[str], dep_edges: List[Tuple[int, int, str]]):
    dep_to_head = {dep: head for head, dep, _ in dep_edges}
    dep_to_rel = {dep: rel for _, dep, rel in dep_edges}
    rows = []

    for dep_idx in sorted(dep_to_rel.keys()):
        rel = dep_to_rel[dep_idx]
        rel_norm = rel.lower()
        keep = rel_norm in {"nsubj", "dobj", "pobj", "root"} or rel_norm == "obj"
        if not keep:
            continue

        display_rel = "dobj" if rel_norm == "obj" else rel
        head_idx = dep_to_head.get(dep_idx, 0)
        head_tok = "ROOT" if head_idx == 0 else tokens[head_idx - 1]
        rows.append(
            {
                "dep": display_rel,
                "token": tokens[dep_idx - 1],
                "head": head_tok,
                "lemma": tokens[dep_idx - 1].lower(),
                "index": dep_idx,
            }
        )

    return rows


st.set_page_config(page_title="Syntactic Parsing Studio", layout="wide")
st.title("Syntactic Parsing Studio: Constituency + Dependency")

with st.expander("PDF Summary (based on 1415136.pdf)", expanded=True):
    st.markdown(
        """
This lecture is organized around **Syntactic Parsing**:

1. Why parsing matters  
Natural language has hierarchical and recursive structure. Surface word order alone is not enough.

2. Two mainstream representations  
- **Constituency Parsing** focuses on phrase nesting (NP / VP / PP).
- **Dependency Parsing** focuses on head-dependent relations between words.

3. Attachment ambiguity  
A classic example is `The boy saw the man with the telescope.`
- VP attachment: `with the telescope` modifies `saw`.
- NP attachment: `with the telescope` modifies `man`.

The lecture also covers CFG/PCFG, graph-based vs transition-based parsing, UAS/LAS, and treebanks such as PTB/CTB.
"""
    )

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Input & Ambiguity")
    sentence = st.text_input(
        "Sentence (English demo)",
        value="The boy saw the man with the telescope.",
        help="The lightweight parser works best for simple SVO + PP sentences.",
    )
    parse_mode = st.radio("Parse mode", ["Demo ambiguity examples (recommended)", "Simple auto parse"], horizontal=True)
    demo_choice = st.selectbox(
        "Demo sentence",
        ["The boy saw the man with the telescope.", "Fruit flies like a banana."],
        disabled=(parse_mode != "Demo ambiguity examples (recommended)"),
    )
    if demo_choice == "The boy saw the man with the telescope.":
        attach_choice = st.radio("PP attachment", ["VP", "NP"], horizontal=True)
        fruit_reading = None
    else:
        fruit_reading = st.radio(
            "Reading",
            ["NOUN", "VERB"],
            horizontal=True,
            format_func=lambda x: "NOUN: [Fruit flies] + like (verb)" if x == "NOUN" else "VERB: flies (verb) + like a banana (PP)",
        )
        attach_choice = "VP"

with col_right:
    st.subheader("Interpretation")
    if parse_mode == "Demo ambiguity examples (recommended)" and demo_choice == "Fruit flies like a banana.":
        if fruit_reading == "NOUN":
            st.info("Current reading: `fruit flies` is an NP (insects), and `like` is a verb.")
        else:
            st.info("Current reading: `flies` is a verb, and `like a banana` is a PP.")
    elif attach_choice == "VP":
        st.info("Current reading: `with the telescope` modifies the action `saw`.")
    else:
        st.info("Current reading: `with the telescope` modifies the noun `man`.")

if parse_mode == "Demo ambiguity examples (recommended)":
    if demo_choice == "The boy saw the man with the telescope.":
        tokens, tree_root, dep_edges = build_demo_structures(attach_choice)
    else:
        tokens, tree_root, dep_edges = build_fruit_flies_structures(fruit_reading or "NOUN")
else:
    parsed = simple_parse(sentence, attach_choice)
    if parsed is None:
        st.warning("This sentence is outside the simple parser coverage. Try the demo sentence or a simpler SVO+PP sentence.")
        tokens, tree_root, dep_edges = build_demo_structures(attach_choice)
    else:
        tokens, tree_root, dep_edges = parsed

st.markdown("### Dual View")
left, right = st.columns(2)
with left:
    st.caption("Constituency Tree")
    st.markdown(render_constituency_svg(tree_root), unsafe_allow_html=True)
with right:
    st.caption("Dependency Graph")
    st.markdown(render_dependency_svg(tokens, dep_edges), unsafe_allow_html=True)

st.markdown("### Dependency Relations")
rows = [{"head": h, "dep": d, "label": r} for h, d, r in dep_edges]
st.dataframe(rows, use_container_width=True, hide_index=True)

st.markdown("### Core Argument Extractor")
current_sentence_for_core = " ".join(tokens)
spacy_rows, spacy_err = extract_core_arguments_spacy(current_sentence_for_core)

if spacy_err is not None:
    st.info(
        "spaCy parser unavailable in this runtime; showing fallback results from the current dependency graph. "
        f"Reason: {spacy_err}"
    )
    core_rows = extract_core_arguments_fallback(tokens, dep_edges)
    core_source = "Fallback (current dependency graph)"
else:
    core_rows = spacy_rows
    core_source = "spaCy dependency parser"

st.caption(f"Source: {core_source} | Sentence: {current_sentence_for_core}")
if core_rows:
    st.dataframe(core_rows, use_container_width=True, hide_index=True)
else:
    st.warning("No nsubj/dobj/pobj/ROOT items were found for this sentence.")

st.markdown("### Chinese Ambiguity Playground")
cn_input = st.text_input(
    "Try a Chinese ambiguity sentence",
    value="南京市长江大桥",
    help="Built-in examples: 南京市长江大桥, 咬死了猎人的狗",
)
cn_sentence = re.sub(r"\s+", "", cn_input)

supported_cn = {"南京市长江大桥", "咬死了猎人的狗"}
if cn_sentence not in supported_cn:
    st.info("This prototype currently supports two built-in Chinese ambiguity examples.")
else:
    if cn_sentence == "南京市长江大桥":
        cn_reading = st.radio(
            "Choose reading",
            ["LOC", "RIVER_SECTION"],
            horizontal=True,
            format_func=lambda x: "A: [City][Yangtze Bridge]" if x == "LOC" else "B: [City+Yangtze][Bridge]",
        )
    else:
        cn_reading = st.radio(
            "Choose reading",
            ["DOG_BITE", "DOG_IS_BITTEN"],
            horizontal=True,
            format_func=lambda x: "A: dog bites hunter" if x == "DOG_BITE" else "B: someone bites hunter's dog",
        )

    cn_tokens, cn_tree, cn_edges, cn_gloss = build_chinese_ambiguity_structures(cn_sentence, cn_reading)
    st.info(cn_gloss)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Constituency Tree (Chinese)")
        st.markdown(render_constituency_svg(cn_tree), unsafe_allow_html=True)
    with c2:
        st.caption("Dependency Graph (Chinese)")
        st.markdown(render_dependency_svg(cn_tokens, cn_edges), unsafe_allow_html=True)

    st.caption("Dependency relations (Chinese)")
    cn_rows = [{"head": h, "dep": d, "label": r} for h, d, r in cn_edges]
    st.dataframe(cn_rows, use_container_width=True, hide_index=True)
