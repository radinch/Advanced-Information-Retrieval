# Advanced Information Retrieval

Implementations and experiments developed for the Advanced Information Retrieval course at Sharif University of Technology. The repository covers lexical retrieval, text representation learning, and multimodal search, with an emphasis on ranking, semantic representations, and empirical evaluation.

## Projects

| Project | Focus |
| --- | --- |
| [Lexical Book Search](lexical-book-search/) | Inverted and tiered indexes; vector-space, BM25, and language-model ranking; near-duplicate detection, query correction, and snippets. |
| [Text Representation Learning](text-representation-learning/) | Word2Vec and FastText; genre classification and semantic clustering; BERT adaptation and layer-wise analysis. |
| [Multimodal Product Search](multimodal-product-search/) | Dense and sparse hybrid retrieval, cross-encoder reranking, relevance evaluation, and grounded answer generation. |

## Usage

Each project has its own environment and execution instructions. Start with its README; the book-search module also prepares the preprocessed records used by the BERT notebook. The shared Goodreads corpus is in [`data/`](data/), and recorded product-search results are in [`multimodal-product-search/reports/`](multimodal-product-search/reports/).

The embedding-enriched book dataset and product catalog, queries, and images must be supplied separately. Their expected locations are documented in the corresponding data folders. Model checkpoints, generated indexes, and verbose logs are excluded from version control.
