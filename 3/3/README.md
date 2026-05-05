# Semantic Analysis Playground

这是一个面向 NLP 课程作业的 `Streamlit` 小项目，主题是“语义分析综合测试平台”。

## 项目结构

```text
3/
├─ app.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ sample_corpus.txt
└─ prompts/
   └─ development_prompt.md
```

## 运行方式

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 使用说明

- 默认会读取 `data/sample_corpus.txt` 作为演示语料。
- 你也可以直接在页面文本框里粘贴自己的英文语料。
- `GloVe` 模块首次运行可能需要联网下载 `glove-twitter-25`。

## 模块说明

- `Tab 1`：TF-IDF 与 LSA
- `Tab 2`：Word2Vec（CBOW / Skip-Gram）
- `Tab 3`：GloVe 词类比与词相似度
- `Tab 4`：FastText OOV 测试与句向量相似度
