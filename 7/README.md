# Week 8 信息抽取与知识图谱构建系统

这是一个适合作业演示的 Streamlit Web 应用，覆盖三项核心模块：

- 命名实体识别（NER）与实体高亮
- BIO 标注序列展示
- 关系抽取与知识图谱可视化

## 运行方式

```powershell
pip install -r requirements.txt
streamlit run app.py
```

如果你使用的是本地 Python，也可以写成：

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 当前实现说明

- 默认采用规则增强抽取，适合课堂演示和页面展示
- 支持中文、英文和中英混合示例
- 知识图谱使用 `pyvis` 生成交互式网络图
- 后续可将 `extractors.py` 中的抽取逻辑替换为 `spaCy`、`HanLP` 或大模型 API

## 推荐演示文本

1. `Steve Jobs founded Apple in California. Apple is headquartered in Cupertino.`
2. `雷军创办了小米集团，小米集团总部位于北京。`
3. `University of California, Los Angeles is located in Los Angeles.`

## 适合截图的页面

- 实体高亮展示
- BIO 序列展示
- 关系三元组表格
- 知识图谱网络图
