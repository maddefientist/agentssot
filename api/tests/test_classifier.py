"""Classifier accuracy on the golden corpus.

Goal: ≥ 90% accuracy across 50 hand-labelled items. Drops below this
fail the build and force a prompt revision.
"""
import json
import os
from pathlib import Path

import pytest


CORPUS_PATH = Path(__file__).parent / "golden" / "classifier_corpus.jsonl"
TARGET_ACCURACY = 0.90


def _load_corpus() -> list[dict]:
    with CORPUS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.integration
def test_classifier_corpus_accuracy():
    """Run the classifier against every line in the corpus, assert accuracy."""
    if not os.environ.get("CLASSIFIER_TEST_LIVE"):
        pytest.skip("Set CLASSIFIER_TEST_LIVE=1 to hit live Ollama.")
    from app.llm.classifier import classify

    corpus = _load_corpus()
    assert len(corpus) >= 10, "corpus must have at least 10 items"

    correct = 0
    misses: list[tuple[str, str, str]] = []
    for entry in corpus:
        result = classify(entry["content"])
        actual = result["memory_type"]
        if actual == entry["expected_type"]:
            correct += 1
        else:
            misses.append((entry["content"][:60], entry["expected_type"], actual))

    accuracy = correct / len(corpus)
    if accuracy < TARGET_ACCURACY:
        for content, expected, actual in misses:
            print(f"MISS [{expected} → {actual}] {content}")
    assert accuracy >= TARGET_ACCURACY, (
        f"classifier accuracy {accuracy:.1%} below target {TARGET_ACCURACY:.0%}; "
        f"{len(misses)} miss(es) — see stdout"
    )


def test_classifier_list_field_normalization():
    """List-typed fields must always be lists even if the model returns null/string/scalar."""
    from app.llm.classifier import _normalize_list_fields

    # None → []
    assert _normalize_list_fields(None) == []
    # str → [str]
    assert _normalize_list_fields("/home/user") == ["/home/user"]
    # list passes through
    assert _normalize_list_fields(["a", "b"]) == ["a", "b"]
    # scalar (int) → []
    assert _normalize_list_fields(42) == []
