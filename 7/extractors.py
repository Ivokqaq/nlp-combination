from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class TokenTag:
    token: str
    start: int
    end: int
    tag: str


@dataclass
class ExtractionResult:
    text: str
    entities: list[Entity]
    all_candidates: list[Entity]
    nested_pairs: list[tuple[Entity, Entity]]
    relations: list[Relation]
    token_tags: list[TokenTag]


SAMPLE_TEXTS = {
    "英文商业新闻": (
        "Steve Jobs founded Apple in California. "
        "Apple is headquartered in Cupertino. "
        "Tim Cook is the CEO of Apple."
    ),
    "中文商业新闻": (
        "雷军创办了小米集团，小米集团总部位于北京。"
        "张一鸣创建了字节跳动，字节跳动位于北京。"
    ),
    "嵌套实体示例": (
        "University of California, Los Angeles is located in Los Angeles. "
        "该机构与OpenAI保持合作。"
    ),
    "中英混合示例": (
        "OpenAI位于San Francisco。Sam Altman leads OpenAI. "
        "Steve Jobs founded Apple, and Apple is headquartered in Cupertino."
    ),
}


ENTITY_COLORS = {
    "PER": "#fde68a",
    "ORG": "#bfdbfe",
    "LOC": "#bbf7d0",
    "MISC": "#e9d5ff",
}


ENTITY_PRIORITY = {
    "ORG": 4,
    "PER": 3,
    "LOC": 2,
    "MISC": 1,
}


EN_PERSONS = [
    "Steve Jobs",
    "Tim Cook",
    "Bill Gates",
    "Elon Musk",
    "Sam Altman",
    "Satya Nadella",
]

EN_ORGS = [
    "Apple",
    "Microsoft",
    "OpenAI",
    "Google",
    "Meta",
    "Tesla",
    "ByteDance",
    "University of California, Los Angeles",
]

EN_LOCS = [
    "California",
    "Cupertino",
    "Seattle",
    "San Francisco",
    "Los Angeles",
    "Beijing",
    "Shanghai",
]

ZH_PERSONS = [
    "雷军",
    "张一鸣",
    "马化腾",
    "任正非",
]

ZH_ORGS = [
    "小米集团",
    "字节跳动",
    "腾讯",
    "清华大学",
    "北京大学",
    "上海财经大学",
]

ZH_LOCS = [
    "北京",
    "上海",
    "深圳",
    "杭州",
    "广州",
    "中国",
]


def extract_information(text: str) -> ExtractionResult:
    all_candidates = collect_entity_candidates(text)
    nested_pairs = build_nested_pairs(all_candidates)
    entities = flatten_entities(all_candidates)
    relations = extract_relations(text, entities)
    token_tags = build_bio_tags(text, entities)
    return ExtractionResult(
        text=text,
        entities=entities,
        all_candidates=all_candidates,
        nested_pairs=nested_pairs,
        relations=relations,
        token_tags=token_tags,
    )


def collect_entity_candidates(text: str) -> list[Entity]:
    candidates: list[Entity] = []

    candidates.extend(scan_lexicon(text, EN_PERSONS, "PER"))
    candidates.extend(scan_lexicon(text, EN_ORGS, "ORG"))
    candidates.extend(scan_lexicon(text, EN_LOCS, "LOC"))
    candidates.extend(scan_lexicon(text, ZH_PERSONS, "PER"))
    candidates.extend(scan_lexicon(text, ZH_ORGS, "ORG"))
    candidates.extend(scan_lexicon(text, ZH_LOCS, "LOC"))

    org_patterns = [
        r"\b(?:[A-Z][A-Za-z&.\-]+(?:\s+[A-Z][A-Za-z&.\-]+){0,4})\s+(?:University|College|Institute|Bank|Group|Company|Corporation|Corp|Inc|Ltd)\b",
    ]
    loc_patterns = [
        r"[\u4e00-\u9fff]{2,}(?:市|省|县|区|国)",
        r"\b(?:New York|Los Angeles|San Francisco|California|Beijing|Shanghai|Seattle|Cupertino)\b",
    ]
    person_patterns = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b",
    ]

    candidates.extend(scan_patterns(text, org_patterns, "ORG"))
    candidates.extend(scan_patterns(text, loc_patterns, "LOC"))
    candidates.extend(scan_patterns(text, person_patterns, "PER"))

    return filter_entities(dedupe_entities(candidates))


def scan_lexicon(text: str, lexicon: Iterable[str], label: str) -> list[Entity]:
    entities: list[Entity] = []
    for term in lexicon:
        for match in re.finditer(re.escape(term), text):
            entities.append(Entity(term, label, match.start(), match.end()))
    return entities


def scan_patterns(text: str, patterns: Iterable[str], label: str) -> list[Entity]:
    entities: list[Entity] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = normalize_span(match.group(0))
            start = match.start()
            end = start + len(value)
            if value:
                entities.append(Entity(value, label, start, end))
    return entities


def normalize_span(value: str) -> str:
    return value.strip(" ,.;:，。；：()[]{}\"'")


def dedupe_entities(entities: list[Entity]) -> list[Entity]:
    seen = set()
    unique: list[Entity] = []
    for entity in sorted(
        entities,
        key=lambda item: (item.start, item.end, -ENTITY_PRIORITY.get(item.label, 0), item.text),
    ):
        key = (entity.start, entity.end, entity.label, entity.text)
        if key not in seen and entity.text:
            seen.add(key)
            unique.append(entity)
    return unique


def filter_entities(entities: list[Entity]) -> list[Entity]:
    blocked_person_terms = set(EN_ORGS + EN_LOCS + ZH_ORGS + ZH_LOCS)
    org_suffixes = ("University", "College", "Institute", "Bank", "Group", "Company", "Corporation")
    filtered: list[Entity] = []
    for entity in entities:
        if entity.label == "PER":
            if entity.text in blocked_person_terms:
                continue
            if any(suffix in entity.text for suffix in org_suffixes):
                continue
            if " of " in entity.text:
                continue
        filtered.append(entity)
    return filtered


def build_nested_pairs(entities: list[Entity]) -> list[tuple[Entity, Entity]]:
    nested_pairs: list[tuple[Entity, Entity]] = []
    for outer in entities:
        for inner in entities:
            if outer == inner:
                continue
            if outer.start <= inner.start and inner.end <= outer.end:
                if outer.start < inner.start or inner.end < outer.end:
                    nested_pairs.append((outer, inner))
    seen = set()
    unique_pairs: list[tuple[Entity, Entity]] = []
    for outer, inner in nested_pairs:
        key = (
            outer.start,
            outer.end,
            outer.label,
            inner.start,
            inner.end,
            inner.label,
        )
        if key not in seen:
            seen.add(key)
            unique_pairs.append((outer, inner))
    return unique_pairs


def flatten_entities(entities: list[Entity]) -> list[Entity]:
    ranked = sorted(
        entities,
        key=lambda item: (
            -(item.end - item.start),
            -ENTITY_PRIORITY.get(item.label, 0),
            item.start,
        ),
    )
    occupied: list[tuple[int, int]] = []
    selected: list[Entity] = []
    for entity in ranked:
        if any(overlaps(entity.start, entity.end, left, right) for left, right in occupied):
            continue
        occupied.append((entity.start, entity.end))
        selected.append(entity)
    return sorted(selected, key=lambda item: (item.start, item.end))


def overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def build_bio_tags(text: str, entities: list[Entity]) -> list[TokenTag]:
    tokens = tokenize(text)
    token_tags: list[TokenTag] = []
    for token, start, end in tokens:
        tag = "O"
        for entity in entities:
            if overlaps(start, end, entity.start, entity.end):
                prefix = "B" if start == entity.start else "I"
                tag = f"{prefix}-{entity.label}"
                break
        token_tags.append(TokenTag(token=token, start=start, end=end, tag=tag))
    return token_tags


def tokenize(text: str) -> list[tuple[str, int, int]]:
    pattern = re.compile(
        r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]",
        re.UNICODE,
    )
    tokens: list[tuple[str, int, int]] = []
    for match in pattern.finditer(text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens


def extract_relations(text: str, entities: list[Entity]) -> list[Relation]:
    sentences = split_sentences(text)
    relations: list[Relation] = []

    relation_specs = [
        {
            "predicate": "FOUNDER_OF",
            "keywords": ["founded", "co-founded", "创办", "创立", "创建"],
            "subject_types": ["PER"],
            "object_types": ["ORG"],
        },
        {
            "predicate": "HEADQUARTERED_IN",
            "keywords": ["headquartered in", "总部位于"],
            "subject_types": ["ORG"],
            "object_types": ["LOC"],
        },
        {
            "predicate": "LOCATED_IN",
            "keywords": ["located in", "位于", "坐落于"],
            "subject_types": ["ORG"],
            "object_types": ["LOC"],
        },
        {
            "predicate": "CEO_OF",
            "keywords": ["ceo of", "chief executive of", "首席执行官", "担任"],
            "subject_types": ["PER"],
            "object_types": ["ORG"],
        },
        {
            "predicate": "EMPLOYED_BY",
            "keywords": ["works at", "worked at", "joined", "任职于", "就职于", "加入"],
            "subject_types": ["PER"],
            "object_types": ["ORG"],
        },
        {
            "predicate": "BORN_IN",
            "keywords": ["born in", "出生于"],
            "subject_types": ["PER"],
            "object_types": ["LOC"],
        },
    ]

    for sent_text, sent_start, sent_end in sentences:
        sent_entities = [
            entity
            for entity in entities
            if entity.start >= sent_start and entity.end <= sent_end
        ]
        sent_lower = sent_text.lower()

        for spec in relation_specs:
            for keyword in spec["keywords"]:
                keyword_lower = keyword.lower()
                if keyword_lower not in sent_lower:
                    continue
                if spec["predicate"] == "LOCATED_IN" and "总部位于" in sent_text and keyword == "位于":
                    continue
                keyword_pos = sent_lower.find(keyword_lower)
                absolute_keyword_start = sent_start + keyword_pos
                absolute_keyword_end = absolute_keyword_start + len(keyword)
                subject = choose_entity(
                    sent_entities,
                    spec["subject_types"],
                    before=absolute_keyword_start,
                    after=None,
                )
                obj = choose_entity(
                    sent_entities,
                    spec["object_types"],
                    before=None,
                    after=absolute_keyword_end,
                )
                if subject and obj and subject.text != obj.text:
                    relations.append(Relation(subject.text, spec["predicate"], obj.text))

    seen = set()
    unique: list[Relation] = []
    for relation in relations:
        key = (relation.subject, relation.predicate, relation.object)
        if key not in seen:
            seen.add(key)
            unique.append(relation)
    return unique


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    boundaries = []
    start = 0
    for match in re.finditer(r"[。！？!?;\n]+", text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            offset = text[start:end].find(sentence)
            sent_start = start + offset
            boundaries.append((sentence, sent_start, sent_start + len(sentence)))
        start = end
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            offset = text[start:].find(sentence)
            sent_start = start + offset
            boundaries.append((sentence, sent_start, sent_start + len(sentence)))
    if not boundaries and text.strip():
        sentence = text.strip()
        sent_start = text.find(sentence)
        boundaries.append((sentence, sent_start, sent_start + len(sentence)))
    return boundaries


def choose_entity(
    entities: list[Entity],
    allowed_types: list[str],
    before: int | None,
    after: int | None,
) -> Entity | None:
    candidates = [entity for entity in entities if entity.label in allowed_types]
    if before is not None:
        candidates = [entity for entity in candidates if entity.end <= before]
    if after is not None:
        candidates = [entity for entity in candidates if entity.start >= after]
    if not candidates:
        return None
    if before is not None:
        return sorted(candidates, key=lambda item: abs(item.end - before))[0]
    if after is not None:
        return sorted(candidates, key=lambda item: abs(item.start - after))[0]
    return candidates[0]
