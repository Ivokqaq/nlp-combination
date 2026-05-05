# Week 10 Vibe Lab: Sentiment Analysis and Opinion Mining

This Streamlit app implements a small e-commerce sentiment analysis platform for the NLP Week 10 lab.

## Features

- Single-review sentiment classification with confidence gauge.
- Explicit vs. implicit sentiment comparison.
- Batch opinion mining dashboard with simulated product reviews and a pie chart.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app uses `lxyuan/distilbert-base-multilingual-cased-sentiments-student` through Hugging Face `pipeline`.
If the model cannot be loaded, the app falls back to a small rule-based demo classifier so the interface can still be reviewed.
