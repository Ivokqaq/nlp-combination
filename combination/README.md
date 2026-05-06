# NLP 作业整合平台

这个目录提供一个统一的 Streamlit 入口，把 `D:\study\nlp\1` 到 `D:\study\nlp\9` 的作业按标签式导航整合到一起。

运行方式：

```powershell
.\run.ps1
```

也可以在已经安装好依赖的环境里直接运行 `streamlit run app.py`。

说明：

- 作业 1 是静态 HTML，会在页面中内嵌显示。
- 作业 2-9 会按当前选择懒加载，避免一次性执行所有模型和控件。
- 子作业自己的 `st.set_page_config` 会被整合入口屏蔽，防止 Streamlit 重复配置页面时报错。
- 依赖会优先使用 `combination\.deps`、各作业目录本身和其中的 `.deps` 目录。
- NLTK 数据会从 `combination\nltk_data` 读取。
