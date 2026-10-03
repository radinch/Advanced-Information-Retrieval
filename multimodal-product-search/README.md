# Multimodal Product Search

Hybrid retrieval over an ABO-Home product catalog using combined image-text embeddings and SPLADE sparse representations. Reciprocal rank fusion combines candidates, a cross-encoder reranks text-bearing queries, and local language models generate grounded structured answers. Optional QLoRA adapts answer formatting.

## Run

Use Python 3.10–3.12 in a dedicated environment. Restore the inputs listed in [`data/README.md`](data/README.md), then run from this directory:

```bash
python -m pip install ".[retrieval]"
python src/build_index.py
python src/retrieve.py --prefusion rrf
python src/rerank.py
python src/build_judgment_pool.py --finalize data/qrels_pool.csv
python src/evaluate.py
```

The supplied judgments apply to the original catalog and query set. For a new experiment, create a fresh pool with `python src/build_judgment_pool.py`, annotate its relevance grades and reasons, then finalize it.

For local answer generation:

```bash
python -m pip install ".[generation]"
python src/generate_answer.py --llm_backend local --load_in_4bit
```

A CUDA GPU is recommended for neural indexing and generation. [`hybrid_multimodal_retrieval.ipynb`](hybrid_multimodal_retrieval.ipynb) provides the interactive workflow; `src/tests.py` retains the original artifact validation checks.

## Results

[`reports/evaluation_report.md`](reports/evaluation_report.md) summarizes the recorded ranking metrics, and [`reports/error_analysis.md`](reports/error_analysis.md) examines representative queries. Model comparisons and answer summaries are also retained. These are supplied experimental records, not new runs. Pooled judgments are incomplete, and some methods are evaluated on different query subsets; compare scores with their coverage and query counts.
