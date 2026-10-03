import json
import os

try:
    from .index_reader import Index_reader
    from .indexes_enum import Indexes, Index_types
except ImportError:
    from index_reader import Index_reader
    from indexes_enum import Indexes, Index_types


class Metadata_index:
    def __init__(self, path='indexes/'):
        """
        Create metadata needed by ranking models.
        """
        self.path = path if path.endswith(os.sep) or path.endswith('/') else path + os.sep
        self.documents = self.read_documents(self.path)
        self.metadata_index = self.create_metadata_index()
        self.store_metadata_index(self.path)

    def read_documents(self, path):
        """
        Read the documents index.
        """
        return Index_reader(path, index_name=Indexes.DOCUMENTS).index

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

    def create_metadata_index(self):
        """
        Create the metadata index.
        """
        average_document_length = {
            Indexes.CHARACTERS.value: self.get_average_document_field_length(Indexes.CHARACTERS.value),
            Indexes.GENRES.value: self.get_average_document_field_length(Indexes.GENRES.value),
            Indexes.DESCRIPTIONS.value: self.get_average_document_field_length(Indexes.DESCRIPTIONS.value),
            # Backward-compatible alias used in some skeleton comments.
            'descriptions': self.get_average_document_field_length(Indexes.DESCRIPTIONS.value),
        }
        return {
            'average_document_length': average_document_length,
            # Historical typo retained so old code does not break.
            'averge_document_length': average_document_length,
            'document_count': len(self.documents),
        }

    def get_average_document_field_length(self, where):
        """
        Return average token length for a document field.
        """
        if not self.documents:
            return 0.0
        total = 0
        for document in self.documents.values():
            if where == 'description' and where not in document and 'descriptions' in document:
                value = document.get('descriptions')
            else:
                value = document.get(where, [])
            total += len(self._tokens(value))
        return total / len(self.documents)

    def store_metadata_index(self, path):
        """
        Store metadata index to disk.
        """
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f'{Indexes.DOCUMENTS.value}_{Index_types.METADATA.value}_index.json')
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.metadata_index, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    meta_index = Metadata_index('../../indexes/')
