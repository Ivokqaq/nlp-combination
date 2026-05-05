# NLP Week 4 Streamlit Assignment Prompt

请帮我用 Python 开发一个可运行的 Streamlit Web 应用，主题为“语义分析综合测试平台”。要求如下：

1. 整体要求
- 使用 `streamlit` 作为前端框架。
- 使用 `gensim`、`scikit-learn`、`nltk` 作为主要 NLP / 向量建模工具。
- 页面结构使用 4 个标签页（tabs）。
- 默认载入一段 500-1000 词的英文语料，但也允许用户在文本框中替换为自己的语料。
- 所有输入与输出都要尽量可解释，适合课堂演示。

2. 模块 1：TF-IDF 与 LSA
- 将英文文本按句子切分为文档集合。
- 使用 `TfidfVectorizer` 计算 TF-IDF 矩阵，并在界面中展示。
- 提取权重最高的 5 个关键词。
- 在 `TF-IDF` 或 `CountVectorizer` 矩阵基础上，使用 `TruncatedSVD` 做 LSA 降维到 2D。
- 用散点图可视化词汇在二维空间中的分布，并在图中标注词。
- 页面中补一句解释，说明为什么共现频繁的词可能在低维空间中更接近。

3. 模块 2：Word2Vec（CBOW vs Skip-Gram）
- 使用 `gensim.models.Word2Vec` 在当前语料上实时训练。
- 提供单选按钮切换 `CBOW (sg=0)` 和 `Skip-Gram (sg=1)`。
- 提供 `window`、`vector_size`、`epochs` 等可调参数。
- 提供一个单词输入框，输出最相似的 5 个词和相似度分数。
- 若单词不在词表中，要给出友好提示。
- 最好附带一个简短提示，说明两种架构的直观区别。

4. 模块 3：GloVe 与词类比
- 使用 `gensim.downloader` 加载 `glove-twitter-25` 预训练模型。
- 提供三个输入框 `A, B, C`，完成 `A - B + C` 的词类比检索。
- 输出 Top 5 候选词，并高亮最可能结果。
- 再提供两个单词输入框，计算词义相似度分数。
- 如果输入词不在预训练词表中，要明确提示。
- 在页面上提醒用户：首次下载模型可能需要一点时间和网络。

5. 模块 4：FastText 与句向量
- 使用 `gensim.models.FastText` 在当前语料上实时训练。
- 做 OOV 测试：输入一个拼写错误的词，比如 `computeer`。
- 对 `Word2Vec` 取向量时用 `try/except` 捕获 `KeyError`，提示“未登录词”。
- 对 `FastText` 则继续给出最相似词结果，展示子词特征的优势。
- 再提供两个较长句子输入框，用训练好的 FastText 词向量做平均池化（average pooling）生成句向量。
- 计算并显示两个句子的余弦相似度。

6. 工程要求
- 代码尽量集中在一个 `app.py` 中，必要时可拆出少量辅助函数。
- 添加清晰的标题、说明文字、参数侧边栏或模块内控件。
- 代码风格清晰，便于课程作业提交。
- 对异常情况做好处理，例如空文本、过短语料、缺失模型下载等。
- 最终请同时输出：
  - 项目目录结构
  - `requirements.txt`
  - 完整代码
  - 运行方式
