# 机器翻译机制对比与质量评测系统

这是 Week 9 随堂 Vibe 实验项目，用 Streamlit 构建三类机器翻译体验：

1. NMT Engine：调用 Hugging Face `Helsinki-NLP/opus-mt-en-zh` 做英译中。
2. Rule-based vs NMT：用词典逐词直译模拟早期规则翻译，并与 NMT 对比。
3. BLEU Score：用 `nltk.translate.bleu_score` 计算 Candidate 与 Reference 的 BLEU 分数。

## 运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

首次运行会下载 Hugging Face 模型，可能需要较长时间和可用网络。代码使用 `AutoTokenizer` 和 `AutoModelForSeq2SeqLM` 生成译文，避免不同 `transformers` 版本中 `pipeline` 任务名变化的问题。若模型无法加载，规则翻译和手动 BLEU 评测仍可使用。

## 建议展示样例

- Idiom: `It rains cats and dogs.`
- 长句: `The student who studied natural language processing carefully completed the machine translation experiment.`
- BLEU 对比:
  - Reference: `下着倾盆大雨。`
  - Candidate 1: `下着倾盆大雨。`
  - Candidate 2: `雨 下着 倾盆 大。`
  - Candidate 3: `雨下得很大。`
