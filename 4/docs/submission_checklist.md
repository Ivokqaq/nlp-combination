# 提交前检查清单

## 必交内容

- 实验报告
- Web 页面展示结果
- 核心代码
- 使用工具与 LLM 说明

## 建议整理的文件

- `app.py`
- `requirements.txt`
- 实验报告 PDF 或 Word
- 页面运行截图
- 如老师允许，可补充录屏或导出的页面说明

## 运行前检查

- 已安装 `requirements.txt` 中依赖
- 已下载 `en_core_web_sm`
- 网络可访问 Hugging Face 模型，或已提前缓存 `bert-base-uncased`
- NLTK 的 `punkt`、`wordnet`、`omw-1.4` 可正常下载

## 报告中建议明确写出的信息

- 为什么 WSD 选用 `Lesk` 和 `BERT`
- 为什么 SRL 使用“依存句法 + 启发式规则”的近似实现
- WSD 和 SRL 模块分别对应哪些课件页
- 实验观察到的现象与局限性
