# 语言模型训练与对比分析平台

这是一个基于 `Streamlit` 的 NLP 课程作业项目，覆盖四个模块：

1. `n-gram & Add-one smoothing`
2. `RNN language model` 从零训练
3. `BERT` 与 `GPT-2` 生成机制对比
4. `Cross-Entropy` 与 `Perplexity (PPL)` 计算

## 运行方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 模块说明

- 模块 1：用统计计数构建 `bigram/trigram`，展示未平滑与 `Add-one smoothing` 的联合概率差异。
- 模块 2：用 `torch.nn.RNN` 或 `torch.nn.LSTM` 训练字符级语言模型，动态显示 `loss` 曲线并生成文本。
- 模块 3：用 `transformers.pipeline` 对比 `bert-base-uncased` 的 `fill-mask` 和 `gpt2` 的自回归续写。
- 模块 4：用 `GPT-2` 计算句子损失，并根据 `PPL = exp(loss)` 输出困惑度。

## 注意事项

- 首次运行模块 3 和模块 4 时，`transformers` 可能需要下载模型文件。
- 若本机未缓存 `NLTK Reuters`，模块 1 会自动回退到内置示例语料。
- 建议课堂演示时优先使用较短文本，以保证加载和训练速度更稳定。
