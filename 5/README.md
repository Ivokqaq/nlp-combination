# Week 6 篇章分析综合平台

这是一个面向课程作业的 Streamlit Web 应用，覆盖以下 3 个模块：

1. EDU 切分：规则基线 vs NeuralEDUSeg 真实样本
2. 浅层篇章关系提取：显式连接词、PDTB 顶级类别、Arg1 / Arg2
3. 指代消解：coreference cluster 高亮与等价类输出

## 运行步骤

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## 降级方案

- `spaCy` 模型缺失：
  页面会自动退化到 `spacy.blank("en")` 或正则分词模式。

- `NeuralEDUSeg` 网络样本抓取失败：
  页面会优先读取本地缓存；若缓存不存在，则使用内置演示样本。

- `fastcoref` 安装或加载失败：
  模块 3 会切换到启发式演示模式，并在页面中明确说明当前不是模型真实输出。

## 说明

- 本项目优先保证“课程展示可运行”和“理论点可观察”。
- 如果某些本地环境无法顺利安装 `fastcoref`，建议先完成模块 1 和模块 2，并在报告中说明模块 3 的工程降级策略。
