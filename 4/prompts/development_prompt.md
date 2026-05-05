# NLP Assignment Development Prompt

请帮我开发一个基于 Streamlit 的 NLP 课程作业 Web 系统，包含两个标签页：WSD 和 SRL。

## 目标

实现一个“深层语义分析平台”，要求代码清晰、适合教学展示，并能支持实验报告撰写。

## 模块 1：WSD

- 输入一个包含多义词的英文句子
- 输入目标词
- 输入第二个包含同一目标词的英文句子
- 使用 `nltk.wsd.lesk` 做传统词义消歧
- 输出预测的 `Synset` 和 `Definition`
- 使用 `bert-base-uncased` 提取目标词在两个句子中的上下文向量
- 计算两个向量的余弦相似度
- 页面上同时展示传统方法与上下文向量方法的结果

## 模块 2：SRL

- 输入一个英文句子，默认值为：`Apple is manufacturing new smartphones in China this year.`
- 使用 `spaCy` 的 `en_core_web_sm` 做依存句法分析
- 采用启发式规则近似 SRL：
  - `nsubj -> A0`
  - `dobj/obj -> A1`
  - 地点相关成分 -> `AM-LOC`
  - 时间相关成分 -> `AM-TMP`
- 以表格方式展示 `[A0, Predicate, A1, AM-LOC, AM-TMP]`
- 页面底部渲染依存句法图，帮助解释抽取结果

## 工程要求

- 使用 Python
- 使用 Streamlit 作为 Web 框架
- 代码拆分清晰，函数命名明确
- 出现模型或资源缺失时，要给出友好的报错提示
- 适合课程作业展示，不需要训练大型模型

## 希望输出

- 完整可运行的 `app.py`
- `requirements.txt`
- 必要的说明文档
