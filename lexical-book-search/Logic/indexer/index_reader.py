try:
    from .indexes_enum import Indexes, Index_types
except ImportError:
    from indexes_enum import Indexes, Index_types
import json
import os


class Index_reader:
    def __init__(self, path: str, index_name: Indexes, index_type: Index_types = None):
        self.path = path if path.endswith(os.sep) or path.endswith('/') else path + os.sep
        self.index_name = index_name
        self.index_type = index_type
        self.index = self.get_index()

    def get_index(self):
        absolute_path = os.path.join(self.path, self.index_name.value)
        if self.index_type is not None:
            absolute_path = absolute_path + '_' + self.index_type.value
        absolute_path = absolute_path + '_index.json'
        with open(absolute_path, 'r', encoding='utf-8') as file:
            return json.load(file)
