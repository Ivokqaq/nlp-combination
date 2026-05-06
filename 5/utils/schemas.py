"""项目中共享的数据结构定义。"""

from dataclasses import dataclass, field


@dataclass
class EDUSample:
    """保存 NeuralEDUSeg 样本解析结果。"""

    source_name: str
    source_mode: str
    raw_text: str
    edus: list[str]
    boundary_tokens: list[str] = field(default_factory=list)
    notes: str = ""
    original_preview: str = ""


@dataclass
class RuleSegmentationResult:
    """保存规则基线切分结果。"""

    edus: list[str]
    boundary_tokens: list[str]
    mode: str
    notes: str = ""


@dataclass
class RelationMatch:
    """保存单个显式连接词匹配结果。"""

    connective: str
    category: str
    start_char: int
    end_char: int
    arg1: str
    arg2: str
    explanation: str


@dataclass
class RelationAnalysisResult:
    """保存浅层篇章关系分析结果。"""

    input_text: str
    matches: list[RelationMatch]
    summary: str
    mode: str


@dataclass
class CorefCluster:
    """保存单个指代簇。"""

    cluster_id: int
    mentions: list[str]
    spans: list[tuple[int, int]]


@dataclass
class CorefResult:
    """保存指代消解结果。"""

    mode: str
    message: str
    clusters: list[CorefCluster]
