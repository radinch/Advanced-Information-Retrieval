import copy
import json
import os
import time
from collections import Counter

try:
    from .indexes_enum import Indexes
except ImportError:
    from indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create all project indexes from preprocessed documents.
        """
        self.preprocessed_documents = preprocessed_documents or []
        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.CHARACTERS.value: self.index_characters(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.DESCRIPTIONS.value: self.index_descriptions(),
        }

    def _doc_id(self, document):
        return str(document.get('id'))

    def _tokens(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            tokens = []
            for item in value:
                if isinstance(item, str):
                    tokens.extend(item.split())
                else:
                    tokens.append(str(item))
            return [t for t in tokens if t]
        if isinstance(value, str):
            return [t for t in value.split() if t]
        return [str(value)]

    def _build_inverted_index(self, field_name):
        index = {}
        for document in self.preprocessed_documents:
            doc_id = self._doc_id(document)
            for term, tf in Counter(self._tokens(document.get(field_name, []))).items():
                index.setdefault(term, {})[doc_id] = int(tf)
        return index

    def index_documents(self):
        """
        Index documents by document ID.
        """
        current_index = {}
        for document in self.preprocessed_documents:
            if document is None or document.get('id') is None:
                continue
            current_index[self._doc_id(document)] = document
        return current_index

    def index_characters(self):
        """Build the character-field inverted index."""
        return self._build_inverted_index(Indexes.CHARACTERS.value)

    def index_genres(self):
        """Build the genre-field inverted index."""
        return self._build_inverted_index(Indexes.GENRES.value)

    def index_descriptions(self):
        """Build the description-field inverted index."""
        return self._build_inverted_index(Indexes.DESCRIPTIONS.value)

    def get_posting_list(self, word: str, index_type: str):
        """
        Return document IDs containing word in the selected index.
        """
        try:
            return list(self.index[index_type].get(word, {}).keys())
        except Exception:
            return []

    def _add_terms(self, index_name, doc_id, tokens):
        for term, tf in Counter(tokens).items():
            self.index[index_name].setdefault(term, {})[doc_id] = int(tf)

    def add_document_to_index(self, document: dict):
        """
        Add one document to all indexes.
        """
        doc_id = self._doc_id(document)
        self.index[Indexes.DOCUMENTS.value][doc_id] = document
        self._add_terms(Indexes.CHARACTERS.value, doc_id, self._tokens(document.get(Indexes.CHARACTERS.value, [])))
        self._add_terms(Indexes.GENRES.value, doc_id, self._tokens(document.get(Indexes.GENRES.value, [])))
        self._add_terms(Indexes.DESCRIPTIONS.value, doc_id, self._tokens(document.get(Indexes.DESCRIPTIONS.value, [])))

    def remove_document_from_index(self, document_id: str):
        """
        Remove one document from all indexes.
        """
        document_id = str(document_id)
        self.index[Indexes.DOCUMENTS.value].pop(document_id, None)
        for index_name in (Indexes.CHARACTERS.value, Indexes.GENRES.value, Indexes.DESCRIPTIONS.value):
            empty_terms = []
            for term, posting in self.index[index_name].items():
                posting.pop(document_id, None)
                if not posting:
                    empty_terms.append(term)
            for term in empty_terms:
                del self.index[index_name][term]

    def delete_dummy_keys(self, index_before_add, index, key):
        if key in index_before_add[index] and len(index_before_add[index][key]) == 0:
            del index_before_add[index][key]

    def check_if_key_exists(self, index_before_add, index, key):
        if key not in index_before_add[index]:
            index_before_add[index].setdefault(key, {})

    def check_add_remove_is_correct(self):
        """
        Check if add/remove operations are correct.
        """
        dummy_document = {
            'id': '100',
            'characters': ['sandman', 'robin'],
            'genres': ['mystery', 'crime'],
            'description': ['good'],
        }
        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return
        for index_name, key in [
            (Indexes.CHARACTERS.value, 'sandman'),
            (Indexes.CHARACTERS.value, 'robin'),
            (Indexes.GENRES.value, 'mystery'),
            (Indexes.GENRES.value, 'crime'),
            (Indexes.DESCRIPTIONS.value, 'good'),
        ]:
            self.check_if_key_exists(index_before_add, index_name, key)
            if set(index_after_add[index_name][key]).difference(set(index_before_add[index_name][key])) != {dummy_document['id']}:
                print(f'Add is incorrect, {key}')
                return
            self.delete_dummy_keys(index_before_add, index_name, key)
        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)
        print('Remove is correct' if index_after_remove == index_before_add else 'Remove is incorrect')

    def store_index(self, path: str = 'indexes/', index_name: str = None):
        """
        Store one index or all indexes as JSON files.
        """
        os.makedirs(path, exist_ok=True)
        if index_name is None:
            for name in self.index:
                self.store_index(path, name)
            return
        if isinstance(index_name, Indexes):
            index_name = index_name.value
        if index_name not in self.index:
            raise ValueError('Invalid index name')
        file_path = os.path.join(path, f'{index_name}_index.json')
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.index[index_name], file, ensure_ascii=False, indent=4)

    def load_index(self, path: str):
        """
        Load all index JSON files from path and replace self.index.
        """
        loaded = {}
        for name in (Indexes.DOCUMENTS.value, Indexes.CHARACTERS.value, Indexes.GENRES.value, Indexes.DESCRIPTIONS.value):
            file_path = os.path.join(path, f'{name}_index.json')
            with open(file_path, 'r', encoding='utf-8') as file:
                loaded[name] = json.load(file)
        self.index = loaded
        return loaded

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        print('comparing indexes')
        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Brute-force sanity check for an inverted index.
        """
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue
            if check_word in self._tokens(document[index_type]):
                docs.append(self._doc_id(document))
            if len(docs) == 3:
                break
        brute_force_time = time.time() - start

        start = time.time()
        posting_list = self.get_posting_list(check_word, index_type)
        implemented_time = time.time() - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)
        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')
            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            print('Indexing is bad')
            return False
        print('Indexing is wrong')
        return False


def main():
    pass


if __name__ == '__main__':
    main()
