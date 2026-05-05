"""Week 6 篇章分析与指代消解系统主入口。"""

from pathlib import Path
import sys

import streamlit as st

# 在 Streamlit Community Cloud 上，个别情况下项目根目录不会稳定出现在导入路径中。
# 这里显式把当前文件所在目录加入 sys.path，避免 `from utils...` 导入失败。
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from utils.constants import (
    APP_DESCRIPTION,
    APP_TITLE,
    DEFAULT_COREF_TEXT,
    DEFAULT_RELATION_TEXT,
    SINCE_CAUSAL_EXAMPLE,
    SINCE_TEMPORAL_EXAMPLE,
)
from utils.coref_module import analyze_coreference
from utils.data_loader import get_neuraleduseg_sample
from utils.discourse_relation import analyze_explicit_relation
from utils.edu_segmentation import segment_rule_based_text
from utils.render import (
    inject_global_css,
    render_cluster_list,
    render_coref_text,
    render_edu_cards,
    render_info_panel,
    render_relation_summary,
)


def init_page() -> None:
    """初始化页面配置与全局样式。"""
    st.set_page_config(
        page_title="Week 6 篇章分析综合平台",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()


def render_header() -> None:
    """渲染顶部标题区域。"""
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)

    with st.sidebar:
        st.header("运行说明")
        st.markdown(
            """
            - 本项目优先展示课程理论对应的三个模块。
            - 如果网络、模型或依赖不可用，页面会自动进入降级模式。
            - 降级模式不会伪装成真实模型结果，界面会明确提示当前状态。
            """
        )
        st.caption(f"当前工作目录：`{Path.cwd()}`")


def module_edu_segmentation() -> None:
    """渲染模块 1：EDU 切分。"""
    st.subheader("模块 1：话语分割（EDU 切分）")
    render_info_panel(
        "理论对应",
        "P28-30 介绍 EDU 与话语分割任务，P32-37 介绍基于 BiLSTM-CRF 与 Restricted Self-attention 的神经话语分割。",
        tone="blue",
    )
    render_info_panel(
        "模块目标",
        "对比规则基线与 NeuralEDUSeg 真实样本标注，观察简单启发式规则与人工/神经网络标注之间的差异。",
        tone="green",
    )

    sample = get_neuraleduseg_sample()
    baseline = segment_rule_based_text(sample.raw_text)

    status_text = f"数据来源：{sample.source_mode} | 样本文件：{sample.source_name}"
    if sample.notes:
        status_text += f" | 备注：{sample.notes}"
    st.caption(status_text)

    with st.expander("查看当前样本文本", expanded=False):
        st.text_area("纯文本输入", sample.raw_text, height=180, key="edu_raw_text", disabled=True)

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("#### 规则基线切分结果")
        st.caption(
            f"使用方式：{baseline.mode}；共切分出 {len(baseline.edus)} 个 EDU；"
            f"边界词：{', '.join(baseline.boundary_tokens) if baseline.boundary_tokens else '无'}"
        )
        render_edu_cards(
            title="Rule Baseline",
            edus=baseline.edus,
            boundary_tokens=baseline.boundary_tokens,
            accent="orange",
        )

    with right_col:
        st.markdown("#### NeuralEDUSeg 真实数据结果")
        st.caption(
            f"数据模式：{sample.source_mode}；共解析出 {len(sample.edus)} 个 EDU；"
            f"边界词：{', '.join(sample.boundary_tokens) if sample.boundary_tokens else '无'}"
        )
        render_edu_cards(
            title="NeuralEDUSeg Ground Truth",
            edus=sample.edus,
            boundary_tokens=sample.boundary_tokens,
            accent="teal",
        )

    st.markdown("#### 观察建议")
    st.markdown(
        """
        - 观察规则基线是否过度依赖标点或从属连词，从而出现“切得太多”或“切得不够”的情况。
        - 对照 PPT P35-36：受限自注意力（Restricted Self-attention）强调局部窗口内的重要上下文，这有助于在长句中更稳定地判断边界。
        - 如果当前页面显示的是缓存或内置演示样本，这表示程序已进入降级模式，但界面仍保留了“规则 vs 标注”的对比结构。
        """
    )


def module_discourse_relation() -> None:
    """渲染模块 2：显式篇章关系提取。"""
    st.subheader("模块 2：浅层篇章分析与显式关系提取")
    render_info_panel(
        "理论对应",
        "P46-47 介绍 PDTB 与显式篇章关系，P51 说明显式连接词消歧，尤其是 since 的时间/因果歧义。",
        tone="blue",
    )

    example_choice = st.radio(
        "快速载入示例",
        [
            "课件默认例句（although）",
            "since 因果示例",
            "since 时间示例",
            "手动输入",
        ],
        horizontal=True,
    )

    if example_choice == "课件默认例句（although）":
        default_text = DEFAULT_RELATION_TEXT
    elif example_choice == "since 因果示例":
        default_text = SINCE_CAUSAL_EXAMPLE
    elif example_choice == "since 时间示例":
        default_text = SINCE_TEMPORAL_EXAMPLE
    else:
        default_text = DEFAULT_RELATION_TEXT

    relation_text = st.text_area(
        "输入英文句子",
        value=default_text,
        height=150,
        help="建议先尝试 although 和 since 两类例句，便于观察显式连接词类别与 Arg1/Arg2 的提取效果。",
    )

    relation_result = analyze_explicit_relation(relation_text)
    render_relation_summary(relation_result)

    st.markdown("#### 观察建议")
    st.markdown(
        """
        - 重点比较 `since` 在不同语境下的类别变化：`TEMPORAL` vs `CONTINGENCY`。
        - 当前实现是“课程展示友好的简化版 PDTB 显式关系提取”，Arg1/Arg2 提取属于启发式近似，不等同于完整标注器。
        - 如果连接词能被找到但类别不够准确，这恰好可以作为实验报告中“连接词消歧难点”的案例。
        """
    )


def module_coreference() -> None:
    """渲染模块 3：指代消解。"""
    st.subheader("模块 3：指代消解（Coreference Resolution）可视化")
    render_info_panel(
        "理论对应",
        "P62-64 介绍 mention / entity / cluster 的基本概念，P65-75 介绍 mention detection、mention ranking 以及端到端指代消解。",
        tone="blue",
    )

    coref_text = st.text_area(
        "输入英文段落",
        value=DEFAULT_COREF_TEXT,
        height=220,
        help="建议输入带有 he / she / it / they 的多句文本，便于观察跨句回指效果。",
    )

    coref_result = analyze_coreference(coref_text)
    render_info_panel(
        "当前后端",
        coref_result.message,
        tone="green" if coref_result.mode == "fastcoref" else "amber",
    )

    st.markdown("#### 原文高亮")
    render_coref_text(coref_text, coref_result.clusters)

    st.markdown("#### 等价类输出")
    render_cluster_list(coref_result.clusters)

    st.markdown("#### 观察建议")
    st.markdown(
        """
        - 观察代词是否能正确链接到先行词，尤其是跨句的 `he / she / it / they`。
        - 如果页面提示当前使用的是“启发式降级模式”，请在报告中说明这是由于本地环境缺少 `fastcoref` 运行条件，并把该模式作为工程兜底方案。
        - 可以结合 PPT P72-P75 讨论：为什么 mention ranking / end-to-end coref 的核心都在于“为当前 mention 选择最合适的先行词或聚类”。
        """
    )


def main() -> None:
    """主函数。"""
    init_page()
    render_header()
    tab1, tab2, tab3 = st.tabs(
        ["话语分割 EDU", "浅层篇章关系", "指代消解"]
    )

    with tab1:
        module_edu_segmentation()
    with tab2:
        module_discourse_relation()
    with tab3:
        module_coreference()


if __name__ == "__main__":
    main()
