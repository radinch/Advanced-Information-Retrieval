import json
import os
import shutil

from Logic.Search import SearchEngine
from Logic.indexer.indexes_enum import Indexes
from Logic.spell_correction import SpellCorrection
from Logic.snippet import Snippet
from Logic.preprocess import Preprocessor
from Logic.Evaluation import Evaluation
from Logic.LSH import MinHashLSH
from Logic import utils


SAMPLE_DATA = [
    {
        "id": "1",
        "title": "The Wizard School",
        "author": "A. Mage",
        "description": "A young wizard studies magic at a secret school and fights a dark lord.",
        "genres": ["Fantasy", "Adventure"],
        "characters": ["Harry", "Hermione", "Dumbledore"],
        "languages": ["English"],
        "publish_date": "2001",
        "num_pages": "320",
        "avg_rating": "4.8",
    },
    {
        "id": "2",
        "title": "Space Voyage",
        "author": "B. Star",
        "description": "A spaceship crew explores distant planets and survives an alien encounter.",
        "genres": ["Science Fiction", "Adventure"],
        "characters": ["Nova", "Orion"],
        "languages": ["English"],
        "publish_date": "2005",
        "num_pages": "280",
        "avg_rating": "4.5",
    },
    {
        "id": "3",
        "title": "Love in Paris",
        "author": "C. Heart",
        "description": "Two artists find love in Paris while painting beautiful streets and cafes.",
        "genres": ["Romance"],
        "characters": ["Claire", "Julien"],
        "languages": ["English"],
        "publish_date": "2010",
        "num_pages": "250",
        "avg_rating": "4.3",
    },
    {
        "id": "4",
        "title": "Dark Magic Academy",
        "author": "D. Spell",
        "description": "Students at an academy learn forbidden magic and battle a dark curse.",
        "genres": ["Fantasy", "Dark Academia"],
        "characters": ["Mara", "Elias"],
        "languages": ["English"],
        "publish_date": "2015",
        "num_pages": "410",
        "avg_rating": "4.6",
    },
]


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    sample_json = "sample_crawled.json"
    sample_preprocessed = "sample_preprocessed.json"
    sample_index_dir = "sample_indexes"

    if os.path.exists(sample_index_dir):
        shutil.rmtree(sample_index_dir)

    with open(sample_json, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_DATA, f, ensure_ascii=False, indent=4)

    print("1. Building sample indexes...")
    utils.build_indexes(sample_json, sample_index_dir, sample_preprocessed)

    required_files = [
        "documents_index.json",
        "characters_index.json",
        "genres_index.json",
        "description_index.json",
        "characters_document_length_index.json",
        "genres_document_length_index.json",
        "description_document_length_index.json",
        "documents_metadata_index.json",
        "characters_tiered_index.json",
        "genres_tiered_index.json",
        "description_tiered_index.json",
    ]

    for filename in required_files:
        path = os.path.join(sample_index_dir, filename)
        assert_true(os.path.exists(path), f"Missing index file: {filename}")

    print("   OK")

    print("2. Testing VSM search...")
    engine = SearchEngine(sample_index_dir)
    weights = {
        Indexes.CHARACTERS: 0.2,
        Indexes.GENRES: 0.2,
        Indexes.DESCRIPTIONS: 0.6,
    }

    result = engine.search("magic dark school", "ltn.lnn", weights, max_results=3)
    print("   ltn.lnn:", result)
    assert_true(len(result) > 0, "Search returned no results")
    assert_true(result[0][0] in {"1", "4"}, "Expected book 1 or 4 to rank first for magic/dark/school")

    result = engine.search("magic dark school", "ltc.lnc", weights, max_results=3)
    print("   ltc.lnc:", result)
    assert_true(len(result) > 0, "Search returned no results")

    print("   OK")

    print("3. Testing BM25 search...")
    result = engine.search("magic dark school", "OkapiBM25", weights, max_results=3)
    print("   BM25:", result)
    assert_true(len(result) > 0, "BM25 returned no results")
    assert_true(result[0][0] in {"1", "4"}, "Expected book 1 or 4 to rank first for BM25 query")

    print("   OK")

    print("4. Testing unigram language model...")
    result = engine.search(
        "magic dark school",
        "unigram",
        weights,
        max_results=3,
        smoothing_method="mixture",
        lamda=0.5,
    )
    print("   Unigram:", result)
    assert_true(len(result) > 0, "Unigram model returned no results")

    print("   OK")

    print("5. Testing spell correction...")
    spell = SpellCorrection([
        "magic wizard school adventure fantasy",
        "space voyage alien planet science fiction",
    ])
    corrected = spell.spell_check("magik wizrd schol")
    print("   Corrected:", corrected)
    assert_true(corrected == "magic wizard school", "Spell correction failed")

    print("   OK")

    print("6. Testing snippet generation...")
    preprocessor = Preprocessor()
    snippet = Snippet(
        normalize_function=preprocessor.normalize,
        remove_stopword_function=preprocessor.remove_stopwords,
        number_of_words_on_each_side=3,
    )
    text, missing = snippet.find_snippet(
        "A young wizard studies magic at a secret school and fights a dark lord.",
        "wizard magic lord",
    )
    print("   Snippet:", text)
    print("   Missing:", missing)
    assert_true("***wizard***" in text, "Snippet did not highlight wizard")
    assert_true("***magic***" in text, "Snippet did not highlight magic")
    assert_true("***lord***" in text, "Snippet did not highlight lord")
    assert_true(missing == [], "No query word should be missing")

    print("   OK")

    print("7. Testing evaluation metrics...")
    actual = [["1", "4"], ["2"]]
    predicted = [["1", "3", "4"], ["2", "1"]]
    evaluation = Evaluation("manual-test")
    metrics = evaluation.calculate_evaluation(actual, predicted)
    print("   Metrics:", metrics)

    assert_true(round(metrics["precision"], 6) == 0.583333, "Precision mismatch")
    assert_true(round(metrics["recall"], 6) == 1.000000, "Recall mismatch")
    assert_true(round(metrics["f1"], 6) == 0.736842, "F1 mismatch")
    assert_true(round(metrics["map"], 6) == 0.916667, "MAP mismatch")
    assert_true(round(metrics["mrr"], 6) == 1.000000, "MRR mismatch")

    print("   OK")

    print("8. Testing LSH near-duplicate detection...")
    docs = [
        "A young wizard studies magic at a secret school.",
        "A young wizard studies magic at a secret school.",
        "A spaceship crew explores distant planets.",
        "Two artists find love in Paris.",
    ]
    lsh = MinHashLSH(docs, 100)
    buckets = lsh.perform_lsh()

    duplicate_pair_found = False
    for bucket_docs in buckets.values():
        if 0 in bucket_docs and 1 in bucket_docs:
            duplicate_pair_found = True
            break

    print("   Duplicate pair found:", duplicate_pair_found)
    assert_true(duplicate_pair_found, "LSH did not place identical documents in the same bucket")

    print("   OK")
    print()
    print("All manual tests passed.")


if __name__ == "__main__":
    main()