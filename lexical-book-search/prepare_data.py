"""Convert the shared Goodreads CSV to raw and preprocessed book records."""
import argparse
import json
from pathlib import Path

from Logic.preprocess import csv_to_json, preprocess_docs


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=root.parent / 'data/top_3000_rated_books.csv')
    parser.add_argument('--raw', type=Path, default=root / 'crawled.json')
    parser.add_argument('--preprocessed', type=Path, default=root / 'preprocessed.json')
    args = parser.parse_args()
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.preprocessed.parent.mkdir(parents=True, exist_ok=True)
    csv_to_json(args.csv, args.raw)
    records = json.loads(args.raw.read_text(encoding='utf-8'))
    preprocess_docs(records)
    args.preprocessed.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared {len(records)} books: {args.raw} and {args.preprocessed}')


if __name__ == '__main__':
    main()
