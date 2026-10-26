"""Keeps the poster-benchmark fixture in step with the README that quotes it.

The model table in README.md is measured against one specific prompt. Recording it
only as prose left room for the run and the write-up to diverge, so the prompt now
also lives in tests/fixtures/poster_benchmark.toml and these tests fail on drift.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "poster_benchmark.toml"
README_PATH = REPO_ROOT / "README.md"

#: Number words the README uses when it states how many strings are scored.
NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def load_fixture() -> dict[str, Any]:
    """The benchmark recipe, parsed from TOML."""
    return tomllib.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def flatten(chunk: str) -> str:
    """Collapse a Markdown fragment to comparable text: no quotes, no reflow."""
    body = " ".join(re.sub(r"^>\s?", "", line) for line in chunk.splitlines())
    return " ".join(body.replace("`", "").replace("'", "").split())


def read_blockquotes() -> list[str]:
    """Every blockquote in the README, in order."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in README_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return ["\n".join(block) for block in blocks]


def blockquote_containing(needle: str) -> str:
    """The README blockquote that carries the prompt."""
    for block in read_blockquotes():
        if needle in block:
            return block
    raise AssertionError(f"no README blockquote mentions {needle!r}")


def test_fixture_prompt_matches_readme() -> None:
    """The pinned prompt and the README's verbatim quote agree."""
    assert flatten(blockquote_containing("BREW BETTER")) == flatten(load_fixture()["prompt"])


def test_every_required_string_is_actually_requested() -> None:
    """Scoring a poster on text the prompt never asked for would be unfair."""
    fixture = load_fixture()
    prompt = flatten(fixture["prompt"])
    missing = [s for s in fixture["required_strings"] if flatten(s) not in prompt]
    assert not missing


def test_readme_counts_the_strings_it_scores() -> None:
    """The README spells out how many strings must survive; that count is the fixture's."""
    readme = README_PATH.read_text(encoding="utf-8")
    stated = re.search(r"\b(\w+)\s+required strings\b", readme)
    assert stated, "README no longer states how many required strings are scored"
    word = stated.group(1)
    assert word in NUMBER_WORDS, f"unrecognised count word {word!r}"
    assert NUMBER_WORDS[word] == len(load_fixture()["required_strings"])


def test_fixture_records_the_table_generation_size() -> None:
    """The recipe's dimensions match the size the README says the rows used."""
    fixture = load_fixture()
    assert fixture["width"] == fixture["height"] == 1024
