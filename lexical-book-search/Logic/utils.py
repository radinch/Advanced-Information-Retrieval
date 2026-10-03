from typing import Dict, List
import json
import os

try:
    from .Search import SearchEngine
    from .spell_correction import SpellCorrection
    from .preprocess import preprocess_docs
    from .indexer.index import Index
    from .indexer.document_lengths_index import DocumentLengthsIndex
    from .indexer.metadata_index import Metadata_index
    from .indexer.tiered_index import Tiered_index
    from .indexer.indexes_enum import Indexes
except ImportError:
    from Logic.Search import SearchEngine
    from Logic.spell_correction import SpellCorrection
    from Logic.preprocess import preprocess_docs
    from Logic.indexer.index import Index
    from Logic.indexer.document_lengths_index import DocumentLengthsIndex
    from Logic.indexer.metadata_index import Metadata_index
    from Logic.indexer.tiered_index import Tiered_index
    from Logic.indexer.indexes_enum import Indexes

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INDEX_DIR = os.path.join(PROJECT_ROOT, 'indexes')
RAW_DATA_CANDIDATES = [
    os.path.join(PROJECT_ROOT, 'crawled.json'),
    os.path.join(PROJECT_ROOT, 'preprocessed.json'),
    'crawled.json',
    'preprocessed.json',
]


def _first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def _load_json(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def _index_files_exist(index_dir=INDEX_DIR):
    required = [
        'documents_index.json',
        'characters_index.json',
        'genres_index.json',
        'description_index.json',
        'characters_document_length_index.json',
        'genres_document_length_index.json',
        'description_document_length_index.json',
        'documents_metadata_index.json',
    ]
    return all(os.path.exists(os.path.join(index_dir, name)) for name in required)


def build_indexes(raw_data_path=None, index_dir=INDEX_DIR, preprocessed_output_path=None):
    """
    Build all indexes from crawled/raw JSON data.

    This helper is safe to call from notebooks or scripts:
        from Logic import utils
        utils.build_indexes('crawled.json')
    """
    raw_data_path = raw_data_path or _first_existing(RAW_DATA_CANDIDATES)
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f'Could not find crawled/preprocessed data. Expected one of: {RAW_DATA_CANDIDATES}'
        )
    docs = _load_json(raw_data_path, [])
    # If the file appears raw, preprocess it. If it is already tokenized, this is idempotent enough.
    preprocess_docs(docs)
    os.makedirs(index_dir, exist_ok=True)
    if preprocessed_output_path:
        with open(preprocessed_output_path, 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False, indent=4)
    index = Index(docs)
    index.store_index(index_dir)
    DocumentLengthsIndex(index_dir)
    Metadata_index(index_dir)
    Tiered_index(index_dir)
    return index


# Load books for UI display. Prefer raw crawled data if available because it is readable.
data_path = _first_existing(RAW_DATA_CANDIDATES)
data = _load_json(data_path, []) if os.path.exists(data_path) else []
books_dataset = {str(book.get('id')): book for book in data if isinstance(book, dict) and book.get('id') is not None}

# Build corpus for spell correction once.
corpus = []
for book in data:
    if not isinstance(book, dict):
        continue
    description = book.get('description', book.get('descriptions', ''))
    if isinstance(description, list):
        description = ' '.join(str(x) for x in description)
    genres = book.get('genres', [])
    characters = book.get('characters', [])
    corpus.append(
        str(description or '') + ' ' +
        ' '.join(str(x) for x in genres if x is not None) + ' ' +
        ' '.join(str(x) for x in characters if x is not None)
    )

os.makedirs(INDEX_DIR, exist_ok=True)
spell_pkl_path = os.path.join(INDEX_DIR, 'spell_correction.pkl')
spell_correction_obj = SpellCorrection(
    all_documents=corpus,
    load_path=spell_pkl_path,
    save_path=spell_pkl_path if corpus else None,
)

search_engine = None
try:
    if not _index_files_exist(INDEX_DIR) and data:
        build_indexes(data_path, INDEX_DIR)
    if _index_files_exist(INDEX_DIR):
        search_engine = SearchEngine(INDEX_DIR)
except Exception:
    search_engine = None


def search(
    query: str,
    max_result_count: int,
    method: str = 'ltn.lnn',
    weights: list = None,
    should_print=False,
    preferred_genre: str = None,
):
    """
    Find relevant documents for a query.
    """
    global search_engine
    if weights is None:
        weights = [0.3, 0.3, 0.4]
    weight_dict = {
        Indexes.CHARACTERS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.DESCRIPTIONS: weights[2],
    }
    if search_engine is None:
        if data:
            build_indexes(data_path, INDEX_DIR)
            search_engine = SearchEngine(INDEX_DIR)
        else:
            raise FileNotFoundError('No dataset/indexes found. Create crawled.json and run utils.build_indexes().')
    return search_engine.search(
        query,
        method,
        weight_dict,
        max_results=max_result_count,
        safe_ranking=True,
    )


def get_book_by_id(id: str, books_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get a book by id from either a dict-based or list-based dataset.
    """
    if isinstance(books_dataset, dict):
        return books_dataset.get(str(id), {})
    for book in books_dataset:
        if str(book.get('id')) == str(id):
            return book
    return {}
