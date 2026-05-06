"""NeuralEDUSeg 样本抓取与解析工具。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
import streamlit as st

from utils.constants import DEMO_EDU_SAMPLE, NEURALEDUSEG_API_URL
from utils.schemas import EDUSample

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "neuraleduseg_sample.json"


def _safe_request_json(url: str, timeout: int = 10) -> list[dict] | dict:
    """发送请求并尝试解析 JSON。"""
    response = requests.get(url, timeout=timeout, headers={"Accept": "application/vnd.github+json"})
    response.raise_for_status()
    return response.json()


def _safe_request_text(url: str, timeout: int = 10) -> str:
    """发送请求并返回文本内容。"""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _extract_boundary_tokens(edus: list[str]) -> list[str]:
    """从 EDU 列表中抽取每个 EDU 起始位置的边界词。"""
    boundary_tokens: list[str] = []
    for edu in edus[1:]:
        stripped = edu.strip()
        if not stripped:
            continue
        boundary_tokens.append(stripped.split()[0])
    return boundary_tokens


def _build_sample(
    source_name: str,
    source_mode: str,
    raw_text: str,
    edus: list[str],
    notes: str,
    preview: str,
) -> EDUSample:
    """构造统一的数据结构。"""
    clean_edus = [edu.strip() for edu in edus if edu and edu.strip()]
    if not clean_edus:
        clean_edus = [raw_text.strip()]
    return EDUSample(
        source_name=source_name,
        source_mode=source_mode,
        raw_text=" ".join(raw_text.split()),
        edus=clean_edus,
        boundary_tokens=_extract_boundary_tokens(clean_edus),
        notes=notes,
        original_preview=preview[:400],
    )


def _parse_tagged_markup(text: str) -> list[str]:
    """解析带显式标签的 EDU 文本。"""
    edu_matches = re.findall(r"<EDU[^>]*>(.*?)</EDU>", text, flags=re.IGNORECASE | re.DOTALL)
    if edu_matches:
        return [" ".join(match.split()) for match in edu_matches]

    split_matches = re.split(r"</?s>|</?p>|<edu>|</edu>|\[EDU\]|\[/EDU\]|\|\|\|", text, flags=re.IGNORECASE)
    split_matches = [segment.strip() for segment in split_matches if segment and segment.strip()]
    if len(split_matches) > 1:
        return split_matches
    return []


def _parse_line_based_edus(text: str) -> list[str]:
    """解析“一行一个 EDU”的简单格式。"""
    lines = [line.strip() for line in text.splitlines()]
    candidate_lines = [line for line in lines if line and not line.startswith("#")]
    if len(candidate_lines) >= 2 and all(len(line.split()) >= 2 for line in candidate_lines[: min(5, len(candidate_lines))]):
        return candidate_lines
    return []


def _parse_conll_like_edus(text: str) -> list[str]:
    """解析近似 CoNLL 的 token/label 格式。"""
    lines = [line.rstrip("\n") for line in text.splitlines()]
    token_rows: list[list[str]] = []
    for line in lines:
        if not line.strip():
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 2:
            token_rows.append(parts)

    if len(token_rows) < 3:
        return []

    edus: list[str] = []
    current_tokens: list[str] = []

    for row in token_rows:
        token = row[0]
        label_candidates = [cell.upper() for cell in row[1:]]
        is_boundary = any(
            label in {"1", "B", "B-EDU", "EDU_START", "START", "BEGIN"}
            or label.endswith("-B")
            for label in label_candidates
        )
        if is_boundary and current_tokens:
            edus.append(" ".join(current_tokens))
            current_tokens = []
        current_tokens.append(token)

    if current_tokens:
        edus.append(" ".join(current_tokens))
    return edus if len(edus) >= 2 else []


def parse_neuraleduseg_sample(text: str, source_name: str, source_mode: str) -> EDUSample:
    """根据真实文本内容尽量自适应解析 NeuralEDUSeg 样本。"""
    preview = text[:500]

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            edus = payload.get("edus") or payload.get("segments") or []
            raw_text = payload.get("text") or " ".join(edus)
            if isinstance(edus, list) and edus:
                return _build_sample(
                    source_name=source_name,
                    source_mode=source_mode,
                    raw_text=raw_text,
                    edus=[str(item) for item in edus],
                    notes="已按 JSON 结构解析真实样本。",
                    preview=preview,
                )
    except json.JSONDecodeError:
        pass

    parsers = [
        ("显式 EDU 标签", _parse_tagged_markup),
        ("逐行 EDU 样式", _parse_line_based_edus),
        ("CoNLL / 序列标注样式", _parse_conll_like_edus),
    ]

    for parser_name, parser in parsers:
        edus = parser(text)
        if len(edus) >= 2:
            raw_text = " ".join(segment.strip() for segment in edus)
            return _build_sample(
                source_name=source_name,
                source_mode=source_mode,
                raw_text=raw_text,
                edus=edus,
                notes=f"已按“{parser_name}”成功解析样本格式。",
                preview=preview,
            )

    normalized_text = " ".join(text.split())
    return _build_sample(
        source_name=source_name,
        source_mode=source_mode,
        raw_text=normalized_text,
        edus=[normalized_text],
        notes="无法可靠确认真实标注格式，已按单段文本回退展示。",
        preview=preview,
    )


def _save_cache(sample: EDUSample) -> None:
    """把成功解析的样本写入本地缓存。"""
    payload = {
        "source_name": sample.source_name,
        "source_mode": sample.source_mode,
        "raw_text": sample.raw_text,
        "edus": sample.edus,
        "boundary_tokens": sample.boundary_tokens,
        "notes": sample.notes,
        "original_preview": sample.original_preview,
    }
    CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cached_sample() -> EDUSample | None:
    """读取本地缓存样本。"""
    if not CACHE_FILE.exists():
        return None
    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return EDUSample(**payload)
    except Exception:
        return None


def _build_demo_sample() -> EDUSample:
    """构造内置演示样本。"""
    return _build_sample(
        source_name=DEMO_EDU_SAMPLE["source_name"],
        source_mode=DEMO_EDU_SAMPLE["source_mode"],
        raw_text=DEMO_EDU_SAMPLE["raw_text"],
        edus=DEMO_EDU_SAMPLE["edus"],
        notes=DEMO_EDU_SAMPLE["notes"],
        preview=DEMO_EDU_SAMPLE["raw_text"],
    )


def _pick_remote_sample_file(files: list[dict]) -> dict | None:
    """从 GitHub API 返回的文件列表中挑选最像样本文件的条目。"""
    if not files:
        return None

    def score(item: dict) -> tuple[int, int]:
        name = item.get("name", "").lower()
        size = int(item.get("size") or 0)
        priority = 0
        if item.get("type") != "file":
            priority -= 100
        if any(name.endswith(ext) for ext in (".edus", ".txt", ".out", ".sample", ".seg")):
            priority += 30
        if "sample" in name or "demo" in name:
            priority += 20
        if "preprocess" in name:
            priority += 10
        if 30 <= size <= 50_000:
            priority += 10
        return priority, -size

    sorted_files = sorted(files, key=score, reverse=True)
    return sorted_files[0] if sorted_files else None


@st.cache_data(show_spinner=False)
def get_neuraleduseg_sample() -> EDUSample:
    """获取 NeuralEDUSeg 样本，并在失败时自动降级。"""
    try:
        files = _safe_request_json(NEURALEDUSEG_API_URL, timeout=10)
        if not isinstance(files, list):
            raise ValueError("GitHub API 返回的内容不是文件列表。")
        sample_file = _pick_remote_sample_file(files)
        if not sample_file or not sample_file.get("download_url"):
            raise ValueError("未找到可下载的样本文件。")

        source_name = sample_file.get("name", "unknown_sample")
        raw_text = _safe_request_text(sample_file["download_url"], timeout=10)
        sample = parse_neuraleduseg_sample(raw_text, source_name=source_name, source_mode="online")
        _save_cache(sample)
        return sample
    except Exception as online_error:
        cached = _load_cached_sample()
        if cached is not None:
            cached.source_mode = "cached"
            cached.notes = f"在线抓取失败（{type(online_error).__name__}），已回退到本地缓存。"
            return cached

        demo = _build_demo_sample()
        demo.notes = f"在线抓取失败（{type(online_error).__name__}），且本地无缓存，已回退到内置演示样本。"
        return demo
