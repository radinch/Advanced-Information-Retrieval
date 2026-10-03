try:
    from .preprocess import Preprocessor
    from .Scorer import Scorer
    from .indexer import Indexes, Index_types, Index_reader
except ImportError:  # Allows running this file directly from Logic/
    from preprocess import Preprocessor
    from Scorer import Scorer
    from indexer import Indexes, Index_types, Index_reader


class SearchEngine:
    def __init__(self, path='indexes/'):
        """
        Initialize the search engine from stored indexes.
        """
        if path and not path.endswith('/'):
            path += '/'
        self.path = path
        self.fields = [Indexes.CHARACTERS, Indexes.GENRES, Indexes.DESCRIPTIONS]
        self.preprocessor = Preprocessor()

        self.document_indexes = {
            Indexes.CHARACTERS: Index_reader(path, Indexes.CHARACTERS).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES).index,
            Indexes.DESCRIPTIONS: Index_reader(path, Indexes.DESCRIPTIONS).index,
        }

        self.tiered_index = {}
        for field in self.fields:
            try:
                self.tiered_index[field] = Index_reader(path, field, Index_types.TIERED).index
            except Exception:
                self.tiered_index[field] = {
                    'first_tier': self.document_indexes[field],
                    'second_tier': {},
                    'third_tier': {},
                }

        self.document_lengths_index = {}
        for field in self.fields:
            try:
                self.document_lengths_index[field] = Index_reader(path, field, Index_types.DOCUMENT_LENGTH).index
            except Exception:
                self.document_lengths_index[field] = self._infer_document_lengths(self.document_indexes[field])

        self.documents_index = Index_reader(path, Indexes.DOCUMENTS).index

        try:
            self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA).index
        except Exception:
            self.metadata_index = self._build_fallback_metadata()
        # Backward compatibility with the original typo in metadata_index.py.
        if 'average_document_length' not in self.metadata_index and 'averge_document_length' in self.metadata_index:
            self.metadata_index['average_document_length'] = self.metadata_index['averge_document_length']

    def _infer_document_lengths(self, index):
        lengths = {}
        for posting in index.values():
            for doc_id, tf in posting.items():
                lengths[doc_id] = lengths.get(doc_id, 0) + int(tf)
        return lengths

    def _build_fallback_metadata(self):
        return {
            'document_count': len(self.documents_index),
            'average_document_length': {
                field.value: (
                    sum(self.document_lengths_index[field].values()) / len(self.document_lengths_index[field])
                    if self.document_lengths_index[field] else 0.0
                )
                for field in self.fields
            },
        }

    def _normalize_weights(self, weights):
        if isinstance(weights, dict):
            normalized = {}
            for key, value in weights.items():
                if isinstance(key, Indexes):
                    normalized[key] = float(value)
                else:
                    for field in self.fields:
                        if str(key) in (field.value, field.name):
                            normalized[field] = float(value)
            return {field: normalized.get(field, 0.0) for field in self.fields}
        if isinstance(weights, (list, tuple)):
            values = list(weights) + [0.0, 0.0, 0.0]
            return {
                Indexes.CHARACTERS: float(values[0]),
                Indexes.GENRES: float(values[1]),
                Indexes.DESCRIPTIONS: float(values[2]),
            }
        return {field: 1.0 for field in self.fields}

    def _preprocess_query(self, query):
        if isinstance(query, str):
            return self.preprocessor.remove_stopwords(query)
        tokens = []
        for token in query or []:
            normalized = self.preprocessor.normalize(token)
            if normalized and normalized not in self.preprocessor.stopwords:
                tokens.append(normalized)
        return tokens

    def search(
        self,
        query,
        method,
        weights,
        safe_ranking=True,
        max_results=10,
        smoothing_method=None,
        alpha=0.5,
        lamda=0.5,
    ):
        """
        Search for documents relevant to the query.
        """
        query_tokens = self._preprocess_query(query)
        if not query_tokens:
            return []
        weights = self._normalize_weights(weights)
        scores = {}
        final_scores = {}
        method_normalized = (method or '').lower().replace('-', '.')

        if smoothing_method or method_normalized in {'unigram', 'language_model', 'language model'}:
            self.find_scores_with_unigram_model(
                query_tokens, smoothing_method or 'mixture', weights, scores, alpha=alpha, lamda=lamda
            )
        elif method_normalized in {'okapibm25', 'okapi bm25', 'bm25'}:
            self.find_scores_with_safe_ranking(query_tokens, 'OkapiBM25', weights, scores)
        elif safe_ranking:
            self.find_scores_with_safe_ranking(query_tokens, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(query_tokens, method, weights, max_results, scores)

        self.aggregate_scores(weights, scores, final_scores)
        ranked = sorted(final_scores.items(), key=lambda item: (-item[1], str(item[0])))
        if max_results is not None and max_results != -1:
            ranked = ranked[:int(max_results)]
        return ranked

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregate scores from different fields.
        """
        for field, weight in weights.items():
            if weight == 0 or field not in scores:
                continue
            for doc_id, score in scores[field].items():
                final_scores[doc_id] = final_scores.get(doc_id, 0.0) + weight * score

    def _score_field(self, field, index, query, method):
        scorer = Scorer(index, len(self.documents_index))
        method_normalized = (method or '').lower().replace('-', '.')
        if method_normalized in {'okapibm25', 'okapi bm25', 'bm25'}:
            return scorer.compute_socres_with_okapi_bm25(
                query,
                self._get_average_length(field),
                self.document_lengths_index.get(field, {}),
            )
        return scorer.compute_scores_with_vector_space_model(query, method)

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Compute scores using tiered indexes. It stops after a tier when enough
        aggregate candidates have been found.
        """
        temporary_final = {}
        for tier_name in ('first_tier', 'second_tier', 'third_tier'):
            tier_scores = {}
            for field, weight in weights.items():
                if weight == 0:
                    continue
                index = self.tiered_index.get(field, {}).get(tier_name, {})
                tier_scores[field] = self._score_field(field, index, query, method)
                scores[field] = self.merge_scores(scores.get(field, {}), tier_scores[field])
            self.aggregate_scores(weights, tier_scores, temporary_final)
            if max_results not in (None, -1) and len(temporary_final) >= int(max_results):
                break

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Compute scores using the complete field indexes.
        """
        for field, weight in weights.items():
            if weight == 0:
                continue
            scores[field] = self._score_field(field, self.document_indexes[field], query, method)

    def find_scores_with_unigram_model(
        self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5
    ):
        """
        Compute scores with a unigram language model for each field.
        """
        for field, weight in weights.items():
            if weight == 0:
                continue
            scorer = Scorer(self.document_indexes[field], len(self.documents_index))
            scores[field] = scorer.compute_scores_with_unigram_model(
                query,
                smoothing_method,
                document_lengths=self.document_lengths_index.get(field, {}),
                alpha=alpha,
                lamda=lamda,
            )

    def merge_scores(self, scores1, scores2):
        """
        Merge two score dictionaries by summing scores for repeated documents.
        """
        merged = dict(scores1)
        for doc_id, score in scores2.items():
            merged[doc_id] = merged.get(doc_id, 0.0) + score
        return merged

    def _get_average_length(self, field):
        avg_lengths = self.metadata_index.get('average_document_length', {})
        keys = [field.value, field.name, field]
        # Historical project code sometimes used plural descriptions.
        if field == Indexes.DESCRIPTIONS:
            keys.extend(['descriptions', 'description'])
        for key in keys:
            if key in avg_lengths:
                return avg_lengths[key]
        lengths = self.document_lengths_index[field]
        return sum(lengths.values()) / len(lengths) if lengths else 0.0


if __name__ == '__main__':
    search_engine = SearchEngine()
    result = search_engine.search(
        'magic adventure',
        'lnc.ltc',
        {Indexes.CHARACTERS: 1, Indexes.GENRES: 1, Indexes.DESCRIPTIONS: 1},
    )
    print(result)
