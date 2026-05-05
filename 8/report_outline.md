# 实验报告写作提纲

## 一、实验目的

本实验围绕 Machine Translation 的机制对比与质量评测展开，通过 Streamlit 实现一个交互式 Web 系统，比较 Rule-based MT 和 Neural Machine Translation，并使用 BLEU 进行自动评测。

## 二、理论背景

机器翻译经历了 Rule-based MT、Example-based MT、Statistical MT 和 Neural MT 等阶段。Rule-based MT 依赖词典和人工规则，机制直观但扩展困难。NMT 使用神经网络建模源语言到目标语言的条件概率，常见结构包括 Encoder-Decoder、Attention 和 Transformer。

BLEU 是经典机器翻译自动评测指标。它通过比较 Candidate 与 Reference 的 n-gram 重合程度，并加入 Brevity Penalty，衡量机器译文与参考译文的表面相似度。

## 三、系统设计

系统包含三个标签页：

1. NMT Engine：输入英文句子，调用 Hugging Face 模型输出中文译文。
2. Rule-based vs NMT：使用英汉词典逐词替换，并与 NMT 结果并排展示。
3. BLEU Score：输入 Reference 和 Candidate，计算 BLEU 分数并解释结果。

## 四、实验观察

建议观察 `It rains cats and dogs.`、包含定语从句的长句、一词多义句子等案例。

Rule-based MT 通常会在 idiom、语序调整和上下文理解上失败。NMT 更可能生成符合中文习惯的译文，但也可能出现事实偏差或不稳定输出。

BLEU 对字面 n-gram 重合敏感。语义合理但表达不同的译文可能分数偏低；词汇相似但语序错误的译文也会在高阶 n-gram 上受到惩罚。

## 五、总结

本实验展示了不同机器翻译机制的核心差异：规则方法可解释但僵硬，NMT 更灵活但依赖模型和数据，BLEU 可自动评测但不能完全代表真实语义质量。
