import re
import string
from typing import Callable, List, Tuple


class Snippet:
    """
    Generate query-focused snippets while preserving original document words.
    """

    def __init__(self, normalize_function: Callable, remove_stopword_function: Callable, number_of_words_on_each_side: int = 5):
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.normalize = normalize_function
        self.remove_stopword = remove_stopword_function
        self.win_size = (2 * number_of_words_on_each_side) + 1

    def _tokenize_with_punctuation(self, text: str):
        return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text or '', flags=re.UNICODE)

    def _is_word(self, token: str):
        return bool(re.match(r"\w+(?:'\w+)?$", token, flags=re.UNICODE))

    def find_snippet(self, raw_doc: str, query: str) -> Tuple[str, List[str]]:
        """
        Return a highlighted snippet and normalized query words absent from the document.
        """
        if isinstance(raw_doc, list):
            raw_doc = ' '.join(str(x) for x in raw_doc)
        raw_doc = '' if raw_doc is None else str(raw_doc)
        query_tokens = self.remove_stopword(query)
        query_set = {self.normalize(token) for token in query_tokens if self.normalize(token)}
        if not raw_doc:
            return '', sorted(query_set)
        if not query_set:
            return raw_doc[:300], []

        doc_tokens = self._tokenize_with_punctuation(raw_doc)
        normalized_cache = [self.normalize(token) if self._is_word(token) else '' for token in doc_tokens]
        normalized_doc_set = {token for token in normalized_cache if token}
        not_exist_words = sorted(query_set - normalized_doc_set)

        windows = self._identify_best_windows(doc_tokens, normalized_cache, query_set)
        if not windows:
            # Fallback: start of document when there is no match.
            word_positions = [i for i, token in enumerate(doc_tokens) if self._is_word(token)]
            end = word_positions[min(len(word_positions), self.win_size) - 1] + 1 if word_positions else min(len(doc_tokens), self.win_size)
            windows = [(0, end)]
        merged_windows = self._merge_windows(windows)
        final_snippet = self._create_snippet_text(doc_tokens, normalized_cache, merged_windows, query_set)
        return final_snippet, not_exist_words

    def _word_window_to_token_window(self, word_positions, center_word_idx):
        start_word = max(0, center_word_idx - self.number_of_words_on_each_side)
        end_word = min(len(word_positions) - 1, center_word_idx + self.number_of_words_on_each_side)
        start_token = word_positions[start_word]
        end_token = word_positions[end_word] + 1
        return start_token, end_token

    def _identify_best_windows(self, doc_tokens: list, normalized_cache: list, query_set: set) -> List[Tuple[int, int]]:
        """
        Find windows centered on query matches. Highest-density windows are returned.
        """
        word_positions = [idx for idx, token in enumerate(doc_tokens) if self._is_word(token)]
        if not word_positions:
            return []
        windows = []
        seen_terms = set()
        for word_idx, token_idx in enumerate(word_positions):
            normalized = normalized_cache[token_idx]
            if normalized in query_set:
                seen_terms.add(normalized)
                start, end = self._word_window_to_token_window(word_positions, word_idx)
                score = sum(1 for j in range(start, end) if normalized_cache[j] in query_set)
                windows.append((score, start, end))

        if not windows:
            return []
        # Keep enough windows to cover different query terms but avoid very long snippets.
        windows.sort(key=lambda item: (-item[0], item[1]))
        selected = []
        covered = set()
        for score, start, end in windows:
            terms = {normalized_cache[j] for j in range(start, end) if normalized_cache[j] in query_set}
            if not terms - covered and len(selected) >= max(1, len(query_set)):
                continue
            selected.append((start, end))
            covered.update(terms)
            if len(selected) >= max(3, min(5, len(query_set))):
                break
            if covered >= query_set:
                break
        return sorted(selected)

    def _merge_windows(self, windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping or adjacent token windows.
        """
        if not windows:
            return []
        windows = sorted(windows)
        merged = [list(windows[0])]
        for start, end in windows[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def _create_snippet_text(self, doc_tokens: list, normalized_cache: list,
                             merged_windows: List[Tuple[int, int]], query_set: set) -> str:
        """
        Construct final snippet with *** markers around matched original words.
        """
        pieces = []
        for start, end in merged_windows:
            tokens = []
            for idx in range(start, min(end, len(doc_tokens))):
                token = doc_tokens[idx]
                if normalized_cache[idx] in query_set and self._is_word(token):
                    token = f'***{token}***'
                tokens.append(token)
            text = ''
            for token in tokens:
                if not text:
                    text = token
                elif re.match(r'^[,.;:!?%\)\]\}]$', token):
                    text += token
                elif re.match(r'^[\(\[\{]$', token):
                    text += ' ' + token
                else:
                    text += ' ' + token
            pieces.append(text.strip())
        return ' ... '.join(piece for piece in pieces if piece)
