import os
import pickle
import re
from collections import Counter


class SpellCorrection:
    def __init__(self, all_documents=None, load_path=None, save_path=None):
        """
        Build or load a k-gram vocabulary for spell correction.
        """
        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = {}

    def _tokenize(self, text):
        if text is None:
            return []
        if isinstance(text, list):
            text = ' '.join(str(x) for x in text)
        return re.findall(r"[a-zA-Z0-9]+", str(text).lower())

    def k_gram_word(self, word, k=2):
        """
        Convert a word into a set of k-grams with boundary markers.
        """
        word = str(word).lower().strip()
        if not word:
            return set()
        padded = f'${word}$'
        if len(padded) <= k:
            return {padded}
        return {padded[i:i + k] for i in range(len(padded) - k + 1)}

    def jaccard_score(self, first_set, second_set):
        """
        Calculate Jaccard similarity.
        """
        first_set = set(first_set)
        second_set = set(second_set)
        union = first_set | second_set
        if not union:
            return 1.0
        return len(first_set & second_set) / len(union)

    def k_gramming_and_counting(self, all_documents):
        """
        Create k-grams and corpus frequency counts for every corpus word.
        """
        counter = Counter()
        for document in all_documents:
            counter.update(self._tokenize(document))
        all_k_gram_words = {word: self.k_gram_word(word) for word in counter}
        return all_k_gram_words, dict(counter)

    def save(self, path):
        """
        Save the k-grams data and word counter to a file.
        """
        data = {'all_k_gram_words': self.all_k_gram_words, 'word_counter': self.word_counter}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        """
        Load the k-gram data and word counter from a file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.all_k_gram_words = data.get('all_k_gram_words', {})
        self.word_counter = data.get('word_counter', {})

    def find_nearest_words(self, word):
        """
        Return up to 5 candidate corrections ranked by Jaccard and frequency.
        """
        word = str(word).lower().strip()
        if not word:
            return []
        if word in self.word_counter:
            return [word]
        query_kgrams = self.k_gram_word(word)
        if not query_kgrams or not self.all_k_gram_words:
            return [word]

        max_freq = max(self.word_counter.values(), default=1)
        candidates = []
        for vocab_word, vocab_kgrams in self.all_k_gram_words.items():
            jac = self.jaccard_score(query_kgrams, vocab_kgrams)
            if jac <= 0:
                continue
            # Edit-distance-free reranking: similarity dominates, frequency breaks ties.
            normalized_tf = self.word_counter.get(vocab_word, 0) / max_freq
            length_penalty = 1.0 / (1.0 + abs(len(vocab_word) - len(word)))
            score = 0.75 * jac + 0.20 * normalized_tf + 0.05 * length_penalty
            candidates.append((score, jac, self.word_counter.get(vocab_word, 0), vocab_word))
        candidates.sort(reverse=True)
        return [candidate[-1] for candidate in candidates[:5]] or [word]

    def spell_check(self, query):
        """
        Correct each token in a query. Known words are preserved.
        """
        if not query:
            return ''
        corrected = []
        for token in str(query).split():
            prefix = re.match(r'^\W*', token).group(0)
            suffix = re.search(r'\W*$', token).group(0)
            core = token[len(prefix):len(token) - len(suffix) if suffix else len(token)]
            if not core:
                corrected.append(token)
                continue
            lower_core = core.lower()
            if lower_core in self.word_counter:
                corrected_core = lower_core
            else:
                corrected_core = self.find_nearest_words(lower_core)[0]
            if core[:1].isupper():
                corrected_core = corrected_core.capitalize()
            corrected.append(f'{prefix}{corrected_core}{suffix}')
        return ' '.join(corrected)
