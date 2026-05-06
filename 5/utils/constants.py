"""项目常量与内置演示样本。"""

APP_TITLE = "Week 6 篇章分析综合平台"

APP_DESCRIPTION = """
这是一个围绕 **Discourse Analysis** 与 **Coreference Resolution** 的教学型 Web 系统。
页面实现严格对应课程作业的三个模块：

- 模块 1：EDU 切分，展示 **规则基线 vs NeuralEDUSeg 真实数据**
- 模块 2：显式连接词识别、**PDTB 顶级类别** 标注与 **Arg1/Arg2** 提取
- 模块 3：**coreference clusters** 高亮与等价类输出
"""

NEURALEDUSEG_API_URL = "https://api.github.com/repos/PKU-TANGENT/NeuralEDUSeg/contents/data/rst"
NEURALEDUSEG_RAW_BASE = "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/"

DEMO_EDU_SAMPLE = {
    "source_name": "built_in_demo_sample.txt",
    "source_mode": "built-in demo",
    "raw_text": (
        "The bank also says it will use its network to channel the investments, "
        "although the market remains volatile, and investors are still cautious."
    ),
    "edus": [
        "The bank also says it will use its network to channel the investments,",
        "although the market remains volatile,",
        "and investors are still cautious.",
    ],
    "notes": "未能成功获取远程样本时，系统自动回退到内置演示样本。",
}

DEFAULT_RELATION_TEXT = (
    "Third-quarter sales in Europe were exceptionally strong, boosted by promotional "
    "programs and new products - although weaker foreign currencies reduced the "
    "company's earnings."
)

SINCE_CAUSAL_EXAMPLE = (
    "Guangzhou has a wide water area with many rivers and water systems since it is "
    "located in the water-rich area of southern China."
)

SINCE_TEMPORAL_EXAMPLE = (
    "She has been living in Shanghai since she graduated from Shanghai University of "
    "Finance and Economics."
)

DEFAULT_COREF_TEXT = (
    "Barack Obama was born in Hawaii. He became the 44th President of the United States. "
    "Obama served two terms, and his policies influenced many people. Michelle Obama "
    "supported him during that period, and they remained a prominent public couple."
)

CONNECTIVE_TO_CATEGORY = {
    "although": "COMPARISON",
    "though": "COMPARISON",
    "but": "COMPARISON",
    "however": "COMPARISON",
    "whereas": "COMPARISON",
    "while": "COMPARISON",
    "because": "CONTINGENCY",
    "therefore": "CONTINGENCY",
    "thus": "CONTINGENCY",
    "so": "CONTINGENCY",
    "since": "AMBIGUOUS",
    "if": "CONTINGENCY",
    "when": "TEMPORAL",
    "after": "TEMPORAL",
    "before": "TEMPORAL",
    "meanwhile": "TEMPORAL",
    "and": "EXPANSION",
    "or": "EXPANSION",
    "for example": "EXPANSION",
    "in addition": "EXPANSION",
    "also": "EXPANSION",
}

DISCOURSE_MARKERS = {
    "because",
    "although",
    "though",
    "when",
    "while",
    "since",
    "if",
    "that",
    "which",
    "who",
    "whereas",
}

PERSON_PRONOUNS = {
    "he": "PERSON",
    "him": "PERSON",
    "his": "PERSON",
    "she": "PERSON",
    "her": "PERSON",
    "hers": "PERSON",
    "they": "PLURAL",
    "them": "PLURAL",
    "their": "PLURAL",
    "theirs": "PLURAL",
    "it": "THING",
    "its": "THING",
}

COLOR_PALETTE = [
    "#fff2cc",
    "#d9ead3",
    "#d0e0e3",
    "#ead1dc",
    "#f4cccc",
    "#cfe2f3",
    "#d9d2e9",
    "#fce5cd",
]
