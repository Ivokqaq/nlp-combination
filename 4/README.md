# Week 5 NLP Assignment

This project implements a Streamlit-based semantic analysis platform with:

- Word Sense Disambiguation (WSD)
- Semantic Role Labeling (SRL)

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the app:

```bash
streamlit run app.py
```

## Deliverables

- `app.py`: main Streamlit app
- `requirements.txt`: dependency list
- `prompts/development_prompt.md`: reusable AI coding prompt

## Notes

- The WSD module compares a traditional Lesk method with BERT contextual embeddings.
- The SRL module is a lightweight heuristic approximation built on spaCy dependency parsing.
