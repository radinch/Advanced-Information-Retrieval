import ast
import csv
import json
import os
import re
import string
from typing import Any, Iterable, List

try:
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover - fallback for minimal environments
    PorterStemmer = None


class Preprocessor:
    def __init__(self, custom_stopwords_path: str = './Logic/stopwords.txt'):
        """
        Initialize the text preprocessor.

        The class intentionally uses only lightweight NLTK components that do not
        require downloading corpora. It accepts strings, lists, and missing values.
        """
        self.link_pattern = re.compile(
            r"\S*https?://\S*|\S*www\.\S*|\S+\.(?:ir|com|org|net|edu|gov)\S*|\S+@\S+",
            flags=re.IGNORECASE,
        )
        self.token_pattern = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")
        self.stemmer = PorterStemmer() if PorterStemmer is not None else None
        self.stopwords = self._load_stopwords(custom_stopwords_path)

    def _load_stopwords(self, custom_stopwords_path: str) -> set:
        candidates = [
            custom_stopwords_path,
            './stopwords.txt',
            os.path.join(os.path.dirname(__file__), 'stopwords.txt'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stopwords.txt'),
        ]
        stopwords = set()
        for path in candidates:
            if path and os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    stopwords.update(w.strip().lower() for w in f if w.strip())
        # Add a compact default list because the provided file is intentionally tiny.
        stopwords.update({
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'for',
            'from', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'i', 'in',
            'into', 'is', 'it', 'its', 'of', 'on', 'or', 'our', 'ours', 'she',
            'so', 'such', 'the', 'their', 'theirs', 'them', 'they', 'to', 'was',
            'we', 'were', 'with', 'you', 'your', 'yours', 'not', 'no', 'than',
            'then', 'there', 'these', 'those', 'will', 'would', 'can', 'could',
            'may', 'might', 'must', 'do', 'does', 'did', 'done', 'if', 'because',
            'while', 'during', 'over', 'under', 'again', 'further', 'once', 'here',
            'all', 'any', 'both', 'few', 'more', 'most', 'other', 'some', 'same',
            'own', 'too', 'very', 's', 't', 'm', 'll', 're', 've', 'd'
        })
        return stopwords

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, list):
            return ' '.join(self._stringify(v) for v in value)
        if isinstance(value, tuple) or isinstance(value, set):
            return ' '.join(self._stringify(v) for v in value)
        return str(value)

    def _basic_tokens(self, text: Any) -> List[str]:
        text = self._stringify(text)
        text = self.link_pattern.sub(' ', text)
        text = text.lower()
        return [match.group(0).strip("'") for match in self.token_pattern.finditer(text)]

    def preprocess_text(self, text: str) -> str:
        """
        Apply the preprocessing pipeline to a single text and return a whitespace
        separated normalized string.
        """
        return ' '.join(self.remove_stopwords(text))

    def remove_stopwords(self, text: str) -> list:
        """
        Tokenize, normalize, and remove stopwords from text.
        """
        result = []
        for token in self._basic_tokens(text):
            if not token or token in self.stopwords:
                continue
            normalized = self.normalize(token)
            if normalized and normalized not in self.stopwords:
                result.append(normalized)
        return result

    def normalize(self, word: str) -> str:
        """
        Normalize one token by lowercasing, stripping punctuation, and stemming.
        """
        if word is None:
            return ''
        word = self.link_pattern.sub(' ', str(word).lower()).strip()
        word = word.strip(string.punctuation + '“”’‘—–…')
        match = self.token_pattern.search(word)
        if not match:
            return ''
        word = match.group(0).strip("'")
        if not word:
            return ''
        if self.stemmer is not None:
            return self.stemmer.stem(word)
        # Tiny fallback stemmer for environments without nltk.
        for suffix in ('ingly', 'edly', 'ing', 'edly', 'ed', 'ies', 's'):
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                if suffix == 'ies':
                    return word[:-3] + 'y'
                return word[:-len(suffix)]
        return word

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing to a list of text documents.
        """
        return [self.preprocess_text(document) for document in documents]


def _split_list_field(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    value = str(value).strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v).strip().strip('"\'') for v in parsed if str(v).strip()]
    except Exception:
        pass
    # Goodreads CSVs often use comma-separated, pipe-separated, or bracketed text.
    cleaned = value.strip('[]')
    parts = re.split(r'\s*[,|;]\s*', cleaned)
    return [p.strip().strip('"\'') for p in parts if p.strip().strip('"\'')]


def _field_tokens(preprocessor: Preprocessor, value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        tokens = []
        for item in value:
            tokens.extend(preprocessor.remove_stopwords(item))
        return tokens
    return preprocessor.remove_stopwords(value)


def preprocess_docs(docs: list):
    """
    Apply preprocessing to searchable fields in-place.

    The searchable fields are converted into token lists because the indexer stores
    term frequencies. Original raw metadata can be preserved in a separate file for
    UI display.
    """
    preprocessor = Preprocessor()
    searchable_fields = ['title', 'description', 'descriptions', 'author', 'genres', 'characters']
    for doc in docs:
        for field in searchable_fields:
            if field in doc:
                tokens = _field_tokens(preprocessor, doc.get(field))
                if field == 'descriptions' and 'description' not in doc:
                    doc['description'] = tokens
                else:
                    doc[field] = tokens
        if 'description' not in doc and 'descriptions' in doc:
            doc['description'] = doc['descriptions']
    return docs


def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a Goodreads CSV file into the JSON structure expected by the project.
    """
    def first(row, names, default=''):
        lower_map = {k.lower(): k for k in row.keys()}
        for name in names:
            key = lower_map.get(name.lower())
            if key is not None and row.get(key) not in (None, ''):
                return row.get(key)
        return default

    docs = []
    with open(csv_file_path, 'r', encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader):
            doc = {
                'id': str(first(row, ['bookId', 'book_id', 'id', 'bookID'], idx)),
                'title': first(row, ['title', 'book_title', 'name']),
                'author': first(row, ['author', 'authors', 'writer']),
                'description': first(row, ['description', 'descriptions', 'summary', 'text']),
                'genres': _split_list_field(first(row, ['genres', 'genre'])),
                'characters': _split_list_field(first(row, ['characters', 'character'])),
                'languages': _split_list_field(first(row, ['languages', 'language'])),
                'publish_date': first(row, ['publish_date', 'publication_date', 'publishedDate', 'published']),
                'num_pages': first(row, ['num_pages', 'pages', 'number_of_pages']),
                'avg_rating': first(row, ['avg_rating', 'average_rating', 'rating']),
            }
            docs.append(doc)

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(docs, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    csv_to_json('top_3000_rated_books.csv', 'crawled.json')
    with open('crawled.json', 'r', encoding='utf-8') as file:
        docs = json.load(file)
    preprocess_docs(docs)
    with open('preprocessed.json', 'w', encoding='utf-8') as file:
        json.dump(docs, file, ensure_ascii=False, indent=4)
