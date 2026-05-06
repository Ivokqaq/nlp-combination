from __future__ import annotations

import contextlib
import os
import re
import runpy
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
NLP_ROOT = ROOT.parent
COMMON_DEPS = ROOT / ".deps"
NLTK_DATA = ROOT / "nltk_data"
ASSIGNMENT_1_ENHANCEMENTS = ROOT / "assets" / "assignment1-enhance.css"

if NLTK_DATA.exists():
    os.environ["NLTK_DATA"] = str(NLTK_DATA)


ASSIGNMENTS = [
    {
        "label": "作业 1",
        "title": "作业 1 · 静态网页",
        "kind": "html",
        "path": NLP_ROOT / "1" / "index.html",
    },
    {
        "label": "作业 2",
        "title": "作业 2 · Syntactic Parsing Studio",
        "kind": "streamlit",
        "path": NLP_ROOT / "2" / "app.py",
    },
    {
        "label": "作业 3",
        "title": "作业 3 · Vector Space & Embeddings",
        "kind": "streamlit",
        "path": NLP_ROOT / "3" / "3" / "app.py",
    },
    {
        "label": "作业 4",
        "title": "作业 4 · Deep Semantic Analysis",
        "kind": "streamlit",
        "path": NLP_ROOT / "4" / "app.py",
    },
    {
        "label": "作业 5",
        "title": "作业 5 · Discourse Analysis",
        "kind": "streamlit",
        "path": NLP_ROOT / "5" / "app.py",
    },
    {
        "label": "作业 6",
        "title": "作业 6 · Language Modeling Lab",
        "kind": "streamlit",
        "path": NLP_ROOT / "6" / "app.py",
    },
    {
        "label": "作业 7",
        "title": "作业 7 · Information Extraction & KG",
        "kind": "streamlit",
        "path": NLP_ROOT / "7" / "app.py",
    },
    {
        "label": "作业 8",
        "title": "作业 8 · Machine Translation",
        "kind": "streamlit",
        "path": NLP_ROOT / "8" / "app.py",
    },
    {
        "label": "作业 9",
        "title": "作业 9 · Sentiment Analysis",
        "kind": "streamlit",
        "path": NLP_ROOT / "9" / "app.py",
    },
]


st.set_page_config(
    page_title="NLP 作业整合平台",
    page_icon="NLP",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.5rem;
        max-width: 1280px;
      }
      div[role="radiogroup"] {
        gap: .35rem;
        flex-wrap: wrap;
      }
      div[role="radiogroup"] label {
        border: 1px solid rgba(148, 163, 184, .55);
        border-radius: 8px;
        padding: .35rem .75rem;
        background: #ffffff;
      }
      div[role="radiogroup"] label:has(input:checked) {
        border-color: #2563eb;
        background: #eff6ff;
      }
      div[role="radiogroup"] label p {
        font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@contextlib.contextmanager
def patched_streamlit_page_config():
    original = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        yield
    finally:
        st.set_page_config = original


@contextlib.contextmanager
def project_runtime(project_file: Path):
    project_dir = project_file.parent
    deps_dir = project_dir / ".deps"
    old_cwd = Path.cwd()
    old_path = list(sys.path)

    extra_paths = [str(project_dir)]
    if COMMON_DEPS.exists():
        extra_paths.append(str(COMMON_DEPS))
    if deps_dir.exists():
        extra_paths.append(str(deps_dir))

    for extra_path in reversed(extra_paths):
        if extra_path in sys.path:
            sys.path.remove(extra_path)
        sys.path.insert(0, extra_path)

    os.chdir(project_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def render_html_project(index_path: Path) -> None:
    if not index_path.exists():
        st.error(f"找不到入口文件：{index_path}")
        return

    html = index_path.read_text(encoding="utf-8-sig", errors="replace")
    html = inline_local_assets(html, index_path.parent)
    components.html(html, height=2100, scrolling=False)


def inline_local_assets(html: str, base_dir: Path) -> str:
    def read_asset(asset_name: str) -> str | None:
        asset_path = (base_dir / asset_name).resolve()
        try:
            asset_path.relative_to(base_dir.resolve())
        except ValueError:
            return None
        if not asset_path.exists() or not asset_path.is_file():
            return None
        return asset_path.read_text(encoding="utf-8-sig", errors="replace")

    def replace_stylesheet(match: re.Match[str]) -> str:
        href = match.group("href").lstrip("./")
        if href.startswith(("http://", "https://", "//")):
            return match.group(0)
        content = read_asset(href)
        if content is None:
            return match.group(0)
        return f"<style>\n{content}\n</style>"

    def replace_script(match: re.Match[str]) -> str:
        src = match.group("src").lstrip("./")
        if src.startswith(("http://", "https://", "//")):
            return match.group(0)
        content = read_asset(src)
        if content is None:
            return match.group(0)
        return f"<script>\n{content}\n</script>"

    html = re.sub(
        r"<link\b(?=[^>]*rel=[\"']stylesheet[\"'])(?=[^>]*href=[\"'](?P<href>[^\"']+)[\"'])[^>]*>",
        replace_stylesheet,
        html,
        flags=re.IGNORECASE,
    )
    html = re.sub(
        r"<script\b(?=[^>]*src=[\"'](?P<src>[^\"']+)[\"'])[^>]*>\s*</script>",
        replace_script,
        html,
        flags=re.IGNORECASE,
    )

    if ASSIGNMENT_1_ENHANCEMENTS.exists():
        enhancements = ASSIGNMENT_1_ENHANCEMENTS.read_text(encoding="utf-8")
        html = html.replace("</head>", f"<style>\n{enhancements}\n</style>\n</head>")
    return html


def render_streamlit_project(app_path: Path) -> None:
    if not app_path.exists():
        st.error(f"找不到入口文件：{app_path}")
        return

    with project_runtime(app_path), patched_streamlit_page_config():
        runpy.run_path(str(app_path), run_name="__main__")


def render_assignment(assignment: dict[str, object]) -> None:
    path = assignment["path"]
    assert isinstance(path, Path)

    try:
        if assignment["kind"] == "html":
            render_html_project(path)
        else:
            render_streamlit_project(path)
    except Exception as exc:
        st.error("当前作业加载失败。")
        st.exception(exc)
        st.info(
            "如果这里提示缺少模型或依赖，可以先在对应作业目录安装 requirements，"
            "或者把依赖统一安装到 combination 的运行环境。"
        )


st.title("NLP 作业整合平台")

selected_label = st.radio(
    "选择作业",
    [assignment["label"] for assignment in ASSIGNMENTS],
    horizontal=True,
    label_visibility="collapsed",
)
selected = next(item for item in ASSIGNMENTS if item["label"] == selected_label)

st.caption(str(selected["path"]))
st.divider()
render_assignment(selected)
