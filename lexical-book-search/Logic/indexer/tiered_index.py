import json
import os

try:
    from .indexes_enum import Indexes, Index_types
    from .index_reader import Index_reader
except ImportError:
    from indexes_enum import Indexes, Index_types
    from index_reader import Index_reader


class Tiered_index:
    def __init__(self, path='indexes/'):
        """
        Build tiered indexes from the regular field indexes.
        """
        self.path = path if path.endswith(os.sep) or path.endswith('/') else path + os.sep
        self.index = {
            Indexes.CHARACTERS: Index_reader(self.path, index_name=Indexes.CHARACTERS).index,
            Indexes.GENRES: Index_reader(self.path, index_name=Indexes.GENRES).index,
            Indexes.DESCRIPTIONS: Index_reader(self.path, index_name=Indexes.DESCRIPTIONS).index,
        }
        self.tiered_index = {
            Indexes.CHARACTERS: self.convert_to_tiered_index(3, 2, Indexes.CHARACTERS),
            Indexes.DESCRIPTIONS: self.convert_to_tiered_index(10, 5, Indexes.DESCRIPTIONS),
            Indexes.GENRES: self.convert_to_tiered_index(1, 0, Indexes.GENRES),
        }
        self.store_tiered_index(self.path, Indexes.CHARACTERS)
        self.store_tiered_index(self.path, Indexes.DESCRIPTIONS)
        self.store_tiered_index(self.path, Indexes.GENRES)

    def convert_to_tiered_index(self, first_tier_threshold: int, second_tier_threshold: int, index_name):
        """
        Split posting lists into first, second, and third tiers by term frequency.
        """
        if index_name not in self.index:
            raise ValueError('Invalid index type')

        current_index = self.index[index_name]
        first_tier = {}
        second_tier = {}
        third_tier = {}
        for term, posting in current_index.items():
            for doc_id, tf in posting.items():
                tf = int(tf)
                if tf >= first_tier_threshold:
                    first_tier.setdefault(term, {})[doc_id] = tf
                elif tf >= second_tier_threshold:
                    second_tier.setdefault(term, {})[doc_id] = tf
                else:
                    third_tier.setdefault(term, {})[doc_id] = tf
        return {
            'first_tier': first_tier,
            'second_tier': second_tier,
            'third_tier': third_tier,
        }

    def store_tiered_index(self, path, index_name):
        """
        Store the tiered index to disk.
        """
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f'{index_name.value}_{Index_types.TIERED.value}_index.json')
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.tiered_index[index_name], file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    tiered = Tiered_index(path='../../indexes/')
