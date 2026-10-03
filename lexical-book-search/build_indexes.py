"""Build lexical book-search indexes from crawled.json or another Goodreads JSON file."""
import argparse
import os
from Logic import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='crawled.json', help='Path to raw Goodreads JSON data')
    parser.add_argument('--index-dir', default='indexes', help='Directory where indexes will be written')
    parser.add_argument('--preprocessed-output', default='preprocessed.json', help='Where to store preprocessed JSON')
    args = parser.parse_args()

    index_dir = os.path.abspath(args.index_dir)
    preprocessed_output = os.path.abspath(args.preprocessed_output) if args.preprocessed_output else None
    utils.build_indexes(args.data, index_dir, preprocessed_output)
    print(f'Indexes written to {index_dir}')
