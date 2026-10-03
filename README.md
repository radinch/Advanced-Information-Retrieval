# Advanced Information Retrieval

Implementations and experimental studies developed for the Advanced Information Retrieval course at Sharif University of Technology. The repository connects classical search methods with neural representation learning and multimodal retrieval through projects on book discovery, text analysis, and product search.

## Projects

### Lexical Book Search

A Goodreads search engine built around text preprocessing, field-specific inverted indexes, and tiered retrieval. Ranking methods include vector-space scoring, BM25, and unigram language models with smoothing. MinHash LSH supports near-duplicate detection, while spelling correction and query-focused snippets improve the search experience. A Streamlit interface exposes the retrieval pipeline for interactive exploration.

### Text Representation Learning

Three notebooks investigate how text representations affect prediction and semantic structure:

- **Word embeddings:** skip-gram Word2Vec with negative sampling, Gensim FastText, and optional native FastText, compared through semantic consistency tests and vector visualizations.
- **Classification and clustering:** TF-IDF versus sentence embeddings for book-genre prediction, including class imbalance experiments and K-Means, spherical K-Means, DBSCAN, and hierarchical clustering.
- **BERT adaptation:** full fine-tuning, LoRA, and a frozen-encoder classification head for book-title prediction, followed by layer-wise analysis of pretrained and adapted representations.

### Multimodal Product Search

A hybrid search pipeline over an ABO-Home product catalog, supporting text, image, and combined queries. Qwen3-VL embeddings provide dense multimodal representations, and SPLADE supplies sparse lexical signals. Reciprocal rank fusion combines candidates before BGE cross-encoder reranking. Local Qwen2.5 models generate grounded structured answers from retrieved evidence; an optional QLoRA experiment adapts answer formatting.

## Repository Structure

| Directory | Contents |
| --- | --- |
| [`lexical-book-search/`](lexical-book-search/) | Search implementation, Streamlit interface, data preparation, and smoke checks. |
| [`text-representation-learning/`](text-representation-learning/) | Embedding, classification, clustering, and BERT notebooks. |
| [`multimodal-product-search/`](multimodal-product-search/) | Retrieval and generation scripts, interactive notebook, relevance judgments, and reports. |
| [`data/`](data/) | Shared Goodreads metadata corpus. |

## Evaluation

The projects examine ranking effectiveness, classification performance, and semantic cluster structure. Product-search reports include MAP, NDCG, precision, recall, and qualitative error analysis. Their pooled judgments are incomplete, and some methods use different query subsets; scores should be interpreted alongside judged coverage and query counts. Recorded plots, tables, and reports are retained with the implementations.

## Getting Started

Follow each project's README for dependencies and execution commands, using separate Python environments. The book-search module prepares the preprocessed records required by the BERT notebook. GPU execution is recommended for BERT and the multimodal models.

The embedding-enriched book dataset and product catalog, queries, and images must be supplied separately; expected locations are documented in the corresponding data folders. Model checkpoints, generated indexes, and verbose logs are excluded from version control.