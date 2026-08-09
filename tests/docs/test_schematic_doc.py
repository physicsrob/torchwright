"""docs/schematic.md stays true to the code it documents."""

import json
import re
from pathlib import Path

from torchwright.schematic.format import (
    SCHEMATIC_FILENAME,
    SCHEMATIC_FORMAT,
    SCHEMATIC_SCHEMA_FILENAME,
    SCHEMATIC_SCHEMA_SOURCE,
    SCHEMATIC_SUPPORT_FILENAME,
    decode_cols,
)

_DOC = Path(__file__).parents[2] / "docs" / "schematic.md"


def test_doc_names_the_format_and_files():
    text = _DOC.read_text(encoding="utf-8")
    assert SCHEMATIC_FORMAT in text
    assert SCHEMATIC_FILENAME in text
    assert SCHEMATIC_SCHEMA_FILENAME in text
    assert SCHEMATIC_SUPPORT_FILENAME in text


def test_doc_run_encoding_example_decodes_as_documented():
    text = _DOC.read_text(encoding="utf-8")
    match = re.search(r"`(\[\[.*?\]\])` decodes to\s+`(\[.*?\])`", text)
    assert match is not None, "run-encoding example missing from the doc"
    runs = json.loads(match.group(1))
    expected = json.loads(match.group(2))
    assert decode_cols(runs) == expected


def test_doc_covers_every_required_section():
    text = _DOC.read_text(encoding="utf-8")
    schema = json.loads(SCHEMATIC_SCHEMA_SOURCE.read_text(encoding="utf-8"))
    documented_words = set(re.findall(r"[a-z_.]+", text))
    for section in schema["required"]:
        assert section in documented_words, f"doc never mentions section {section!r}"
