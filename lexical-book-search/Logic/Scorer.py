import math
from collections import Counter
from typing import Dict, Iterable, List


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initialize a scorer over an inverted index: {term: {document_id: tf}}.
        """
        self.index = index or {}
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None

    def _normalize_query(self, query):
        if query is None:
            return []
        if isinstance(query, str):
            return [token for token in query.split() if token]
        return [str(token) for token in query if str(token)]

    def get_list_of_documents(self, query):
        """
        Return documents that contain at least one query term.
        """
        docs = set()
        for term in self._normalize_query(query):
            docs.update(self.index.get(term, {}).keys())
        return list(docs)

    def get_idf(self, term):
        """
        Return smoothed inverse document frequency.
        """
        if term not in self.idf:
            df = len(self.index.get(term, {}))
            self.idf[term] = math.log((self.N + 1) / (df + 1)) + 1
        return self.idf[term]

    def get_query_tfs(self, query):
        """
        Return query term frequencies.
        """
        return dict(Counter(self._normalize_query(query)))

    def compute_scores_with_vector_space_model(self, query, method):
        """
        Compute scores using a SMART-style VSM method such as lnc.ltc.
        """
        if not method:
            method = 'lnc.ltc'
        method = method.replace('-', '.')
        if '.' in method:
            document_method, query_method = method.split('.', 1)
        else:
            document_method, query_method = method[:3], method[3:6] or 'lnn'
        document_method = (document_method + 'nnn')[:3]
        query_method = (query_method + 'nnn')[:3]

        query = self._normalize_query(query)
        query_tfs = self.get_query_tfs(query)
        scores = {}
        for document_id in self.get_list_of_documents(query):
            score = self.get_vector_space_model_score(
                query, query_tfs, document_id, document_method, query_method
            )
            if score != 0:
                scores[document_id] = score
        return scores

    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Return a VSM score for one document and one query.
        """
        query_terms = list(query_tfs.keys())
        doc_weights = {}
        query_weights = {}

        for term in query_terms:
            doc_tf = self.index.get(term, {}).get(document_id, 0)
            q_tf = query_tfs.get(term, 0)

            d_weight = self._apply_tf(doc_tf, document_method[0])
            q_weight = self._apply_tf(q_tf, query_method[0])

            if document_method[1] == 't':
                d_weight *= self.get_idf(term)
            if query_method[1] == 't':
                q_weight *= self.get_idf(term)

            doc_weights[term] = d_weight
            query_weights[term] = q_weight

        if document_method[2] == 'c':
            doc_weights = self._cosine_normalize(doc_weights)
        if query_method[2] == 'c':
            query_weights = self._cosine_normalize(query_weights)

        return sum(doc_weights.get(term, 0.0) * query_weights.get(term, 0.0) for term in query_terms)

    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25. Kept with the original typo for compatibility.
        """
        query = self._normalize_query(query)
        scores = {}
        for document_id in self.get_list_of_documents(query):
            score = self.get_okapi_bm25_score(
                query, document_id, average_document_field_length, document_lengths
            )
            if score != 0:
                scores[document_id] = score
        return scores

    def compute_scores_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """Compatibility alias with corrected spelling."""
        return self.compute_socres_with_okapi_bm25(query, average_document_field_length, document_lengths)

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Return the Okapi BM25 score of a document for a query.
        """
        k1 = 1.5
        b = 0.75
        avgdl = float(average_document_field_length or 0.0)
        dl = float(document_lengths.get(str(document_id), document_lengths.get(document_id, 0)) or 0.0)
        if avgdl <= 0:
            avgdl = max(dl, 1.0)
        score = 0.0
        for term in query:
            posting = self.index.get(term, {})
            tf = float(posting.get(document_id, posting.get(str(document_id), 0)) or 0.0)
            if tf <= 0:
                continue
            df = len(posting)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * (dl / avgdl))
            score += idf * ((tf * (k1 + 1)) / denominator)
        return score

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Compute query likelihood scores for documents using a unigram language model.
        """
        query = self._normalize_query(query)
        document_lengths = document_lengths or self._infer_document_lengths()
        candidate_docs = set(document_lengths.keys()) | set(self.get_list_of_documents(query))
        scores = {}
        for document_id in candidate_docs:
            scores[document_id] = self.compute_score_with_unigram_model(
                query, document_id, smoothing_method, document_lengths, alpha, lamda
            )
        return scores

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Return the log query-likelihood score of a document.
        """
        self._prepare_collection_stats()
        smoothing_method = (smoothing_method or 'mixture').lower()
        dl = float(document_lengths.get(document_id, document_lengths.get(str(document_id), 0)) or 0.0)
        if dl <= 0:
            dl = 1.0
        score = 0.0
        epsilon = 1e-12

        for term in query:
            tf = float(self.index.get(term, {}).get(document_id, self.index.get(term, {}).get(str(document_id), 0)) or 0.0)
            cf = float(self._collection_frequencies.get(term, 0.0))
            p_collection = cf / self._collection_length if self._collection_length else epsilon

            if smoothing_method in ('naive', 'none'):
                probability = tf / dl
            elif smoothing_method in ('bayes', 'dirichlet', 'bayesian'):
                mu = max(float(alpha), 0.0)
                # Treat alpha <= 1 as a ratio and scale it to a useful Dirichlet prior.
                if mu <= 1:
                    mu = mu * max(self._collection_length / self.N, 1.0)
                probability = (tf + mu * p_collection) / (dl + mu)
            elif smoothing_method in ('mixture', 'jm', 'jelinek-mercer'):
                lam = min(max(float(lamda), 0.0), 1.0)
                probability = (1 - lam) * (tf / dl) + lam * p_collection
            else:
                raise ValueError(f'Unknown smoothing method: {smoothing_method}')

            score += math.log(max(probability, epsilon))
        return score

    def _apply_tf(self, tf, mode):
        """
        Apply SMART tf weighting.
        """
        tf = float(tf or 0.0)
        if tf <= 0:
            return 0.0
        mode = (mode or 'n').lower()
        if mode == 'n':
            return tf
        if mode == 'l':
            return 1.0 + math.log(tf)
        if mode == 'b':
            return 1.0
        if mode == 'a':
            return 0.5 + 0.5 * tf
        return tf

    def _cosine_normalize(self, weights):
        """
        Cosine-normalize a sparse vector represented as a dict.
        """
        norm = math.sqrt(sum(float(w) ** 2 for w in weights.values()))
        if norm == 0:
            return dict(weights)
        return {term: float(weight) / norm for term, weight in weights.items()}

    def _infer_document_lengths(self):
        lengths = Counter()
        for posting in self.index.values():
            for doc_id, tf in posting.items():
                lengths[doc_id] += int(tf)
        return dict(lengths)

    def _prepare_collection_stats(self):
        """
        Cache collection term frequencies and total collection length.
        """
        if self._collection_frequencies is not None:
            return
        frequencies = Counter()
        for term, posting in self.index.items():
            frequencies[term] = sum(int(tf) for tf in posting.values())
        self._collection_frequencies = dict(frequencies)
        self._collection_length = float(sum(frequencies.values())) or 1.0
