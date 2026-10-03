# Lexical Book Search

A Goodreads search engine with field-specific inverted indexes, tiered ranking, vector-space scoring, BM25, and unigram language models. Supporting components implement MinHash LSH, spelling correction, snippets, and ranking evaluation. A Streamlit interface provides interactive search.

## Run

From this directory, using a dedicated Python environment:

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python build_indexes.py
streamlit run UI/main.py
```

`prepare_data.py` converts the shared book CSV to `crawled.json` and `preprocessed.json`; `build_indexes.py` writes `indexes/`. The BERT notebook uses the preprocessed JSON.

## Implementation

`Logic/` contains retrieval and evaluation components; `UI/` contains the interface. `Logic/LSHFakeData.json` is a synthetic near-duplicate fixture. Run the existing smoke checks with `python manual_tests.py`.
