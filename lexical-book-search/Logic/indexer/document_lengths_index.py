import json
import os

try:
    from .indexes_enum import Indexes, Index_types
    from .index_reader import Index_reader
except ImportError:
    from indexes_enum import Indexes, Index_types
    from index_reader import Index_reader


class DocumentLengthsIndex:
    def __init__(self, path='indexes/'):
        """
        Create document-length indexes for characters, genres, and descriptions.
        """
        self.path = path if path.endswith(os.sep) or path.endswith('/') else path + os.sep
        self.documents_index = Index_reader(self.path, index_name=Indexes.DOCUMENTS).index
        self.document_length_index = {
            Indexes.CHARACTERS: self.get_documents_length(Indexes.CHARACTERS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.DESCRIPTIONS: self.get_documents_length(Indexes.DESCRIPTIONS.value),
        }
        self.store_document_lengths_index(self.path, Indexes.CHARACTERS)
        self.store_document_lengths_index(self.path, Indexes.GENRES)
        self.store_document_lengths_index(self.path, Indexes.DESCRIPTIONS)

    def _tokens(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            tokens = []
            for item in value:
                tokens.extend(str(item).split())
            return [t for t in tokens if t]
        if isinstance(value, str):
            return [t for t in value.split() if t]
        return [str(value)]

    def get_documents_length(self, where):
        """
        Return {document_id: field_token_count} for the requested field.
        """
        lengths = {}
        for doc_id, document in self.documents_index.items():
            if where == 'description' and where not in document and 'descriptions' in document:
                value = document.get('descriptions')
            else:
                value = document.get(where, [])
            lengths[str(doc_id)] = len(self._tokens(value))
        return lengths

    def store_document_lengths_index(self, path, index_name):
        """
        Store one document-length index to disk.
        """
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f'{index_name.value}_{Index_types.DOCUMENT_LENGTH.value}_index.json')
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.document_length_index[index_name], file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    document_lengths_index = DocumentLengthsIndex('../../indexes/')
    print('Document lengths index stored successfully.')
