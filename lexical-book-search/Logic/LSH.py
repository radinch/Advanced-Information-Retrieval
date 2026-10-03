import itertools
import json
import os
import random
import re
import zlib
from collections import defaultdict

import numpy as np


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize MinHashLSH.
        """
        self.documents = [self._document_to_text(document) for document in documents]
        self.num_hashes = max(int(num_hashes), 1)
        self._all_shingles = None
        self._doc_shingles = None
        self._shingle_to_row = None

    def _document_to_text(self, document):
        if isinstance(document, dict):
            for key in ('description', 'descriptions', 'summary', 'text'):
                value = document.get(key)
                if isinstance(value, list):
                    return ' '.join(str(v) for v in value)
                if value:
                    return str(value)
            return ' '.join(str(v) for v in document.values())
        if isinstance(document, list):
            return ' '.join(str(v) for v in document)
        return '' if document is None else str(document)

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of k-word shingles.
        """
        text = self._document_to_text(document).lower()
        words = re.findall(r"[a-zA-Z0-9]+", text)
        if not words:
            return set()
        if len(words) < k:
            return {' '.join(words)}
        return {' '.join(words[i:i + k]) for i in range(len(words) - k + 1)}

    def build_characteristic_matrix(self):
        """
        Build a binary shingle-document characteristic matrix.
        """
        self._doc_shingles = [self.shingle_document(doc, 2) for doc in self.documents]
        all_shingles = sorted(set().union(*self._doc_shingles)) if self._doc_shingles else []
        self._all_shingles = all_shingles
        self._shingle_to_row = {shingle: idx for idx, shingle in enumerate(all_shingles)}
        matrix = np.zeros((len(all_shingles), len(self.documents)), dtype=np.uint8)
        for doc_id, shingles in enumerate(self._doc_shingles):
            for shingle in shingles:
                row = self._shingle_to_row[shingle]
                matrix[row, doc_id] = 1
        return matrix

    def _hash_params(self, n_rows):
        prime = 4294967311  # > 2**32
        params = []
        for i in range(self.num_hashes):
            a = (1103515245 + 2 * i * 12345) % prime
            b = (12345 + i * 2654435761) % prime
            if a == 0:
                a = 1
            params.append((a, b, prime))
        return params

    def min_hash_signature(self):
        """
        Generate a MinHash signature matrix of shape (num_hashes, num_documents).
        """
        matrix = self.build_characteristic_matrix()
        n_rows, n_docs = matrix.shape
        signature = np.full((self.num_hashes, n_docs), np.inf)
        if n_rows == 0 or n_docs == 0:
            return np.zeros((self.num_hashes, n_docs), dtype=np.int64)

        row_indices = np.arange(n_rows, dtype=np.int64)
        for hash_idx, (a, b, prime) in enumerate(self._hash_params(n_rows)):
            hashed_rows = (a * row_indices + b) % prime
            for doc_id in range(n_docs):
                present_rows = np.where(matrix[:, doc_id] == 1)[0]
                if present_rows.size:
                    signature[hash_idx, doc_id] = int(np.min(hashed_rows[present_rows]))
        signature[np.isinf(signature)] = 0
        return signature.astype(np.int64)

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into LSH buckets. Bucket keys include the band id to avoid
        accidental collisions across bands.
        """
        if signature.size == 0:
            return {}
        n_hashes, n_docs = signature.shape
        bands = max(1, min(int(bands), n_hashes))
        rows_per_band = max(1, int(rows_per_band))
        buckets = defaultdict(list)

        for band in range(bands):
            start = band * rows_per_band
            end = min(start + rows_per_band, n_hashes)
            if start >= n_hashes:
                break
            for doc_id in range(n_docs):
                band_tuple = tuple(int(x) for x in signature[start:end, doc_id])
                bucket_hash = zlib.crc32(repr((band, band_tuple)).encode('utf-8'))
                key = f'{band}:{bucket_hash}'
                buckets[key].append(doc_id)
        return dict(buckets)

    def perform_lsh(self):
        """
        Perform the entire LSH process.
        """
        signature = self.min_hash_signature()
        if self.num_hashes >= 25:
            num_bands = 25
        else:
            num_bands = max(1, self.num_hashes)
        rows = max(1, self.num_hashes // num_bands)
        return self.lsh_buckets(signature, num_bands, rows)

    def jaccard_score(self, first_set, second_set):
        """
        Calculate Jaccard similarity between two sets.
        """
        first_set = set(first_set)
        second_set = set(second_set)
        union = first_set | second_set
        if not union:
            return 1.0
        return len(first_set & second_set) / len(union)

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test near duplicate detection based on Jaccard similarity.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1
                    first_doc_id, second_doc_id = comb
                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)
                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        if len(all_documents) <= 2:
                            current_score += 1
                            continue
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)
                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)
                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1
                    if current_score == 5:
                        correct_near_duplicates += 1

        score = correct_near_duplicates / all_near_duplicates if all_near_duplicates else 0.0
        print('your final score in near duplicate detection:', score)
        return score


def main():
    fake_path = os.path.join(os.path.dirname(__file__), 'LSHFakeData.json')
    if os.path.exists(fake_path):
        with open(fake_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        descriptions = [d.get('descriptions', d.get('description', '')) for d in docs]
        lsh = MinHashLSH(descriptions, 100)
        buckets = lsh.perform_lsh()
        lsh.jaccard_similarity_test(buckets, descriptions)


if __name__ == '__main__':
    main()
