"""Consistency checks for the generated noise artefacts.

These tests do NOT re-measure. They verify that the committed artefacts
(`docs/op_noise_data.json`, `docs/numerical_noise.md`, and per-op docstring
footers) agree with each other and cover every op declared in
`scripts/measure_op_noise.py`.

Number drift (code edited, numbers not regenerated) is a separate check
— see ``tests/docs/test_numerical_noise_drift.py``.

If any of these tests fail, the fix is::

    make measure-noise

which regenerates all three artefacts from a fresh measurement run.
"""

from __future__ import annotations

import json

from scripts.measure_op_noise import (
    DOCS_JSON,
    DOCS_MD,
    REPO_ROOT,
    _footer_summary,
    _target_ops,
    footer_source_files,
    render_markdown,
)
from torchwright.debug.noise import NOISE_FOOTER_MARKER


def _load_json() -> dict:
    assert DOCS_JSON.exists(), (
        f"{DOCS_JSON} is missing. Run `make measure-noise` to generate it."
    )
    return json.loads(DOCS_JSON.read_text())


def test_every_target_op_has_data() -> None:
    data = _load_json()
    # Ops are identified by (machine, name) — the relu and swiglu
    # libraries share op names (both have a `square`).
    json_ops = {(op["machine"], op["name"]) for op in data["ops"]}
    declared = {(t.machine, t.name) for t in _target_ops()}
    missing = declared - json_ops
    extra = json_ops - declared
    assert not missing, (
        f"ops declared in _target_ops() but missing from {DOCS_JSON.name}: {sorted(missing)}. "
        f"Run `make measure-noise`."
    )
    assert not extra, (
        f"ops in {DOCS_JSON.name} but not declared in _target_ops(): {sorted(extra)}. "
        f"Run `make measure-noise`."
    )


def test_footer_source_files_cover_both_libraries() -> None:
    """The sync-back file set (modal_measure_noise.py) covers every
    target op's module in both library subpackages.  Pins the B0-split
    regression where a non-recursive ``ops/*.py`` glob returned zero
    footer-bearing files once the ops moved into ``ops/relu/`` and
    ``ops/swiglu/``.
    """
    files = footer_source_files()
    declared = {REPO_ROOT / t.source_file for t in _target_ops()}
    assert declared <= set(files), (
        f"footer_source_files() misses op modules: "
        f"{sorted(str(p) for p in declared - set(files))}"
    )
    rels = {f.relative_to(REPO_ROOT).as_posix() for f in files}
    assert any(r.startswith("torchwright/ops/relu/") for r in rels), rels
    assert any(r.startswith("torchwright/ops/swiglu/") for r in rels), rels
    for f in files:
        assert f.exists(), f"footer_source_files() names a missing file: {f}"


def test_markdown_matches_json() -> None:
    data = _load_json()
    expected = render_markdown(data)
    actual = DOCS_MD.read_text()
    assert expected == actual, (
        f"{DOCS_MD.name} does not match the rendered form of {DOCS_JSON.name}. "
        f"Run `make measure-noise`."
    )


def test_docstring_footers_match_json() -> None:
    data = _load_json()
    commit = data["commit"]
    targets = {(t.machine, t.name): t for t in _target_ops()}
    for op in data["ops"]:
        target = targets[(op["machine"], op["name"])]
        source = (REPO_ROOT / target.source_file).read_text()
        summary = _footer_summary(op)
        rel_txt = (
            "n/a"
            if summary["max_rel_error"] != summary["max_rel_error"]
            else f"{summary['max_rel_error']:.4g}"
        )
        expected_body = (
            f"Max error: {summary['max_abs_error']:.4g} abs, {rel_txt} rel "
            f"over {summary['total_samples']} samples;"
        )
        expected_commit_line = (
            f"measured at commit {commit}. See docs/numerical_noise.md."
        )
        assert NOISE_FOOTER_MARKER in source, (
            f"docstring for {op['name']!r} has no {NOISE_FOOTER_MARKER} block. "
            f"Run `make measure-noise`."
        )
        assert expected_body in source, (
            f"docstring footer for {op['name']!r} does not contain "
            f"{expected_body!r}. Run `make measure-noise`."
        )
        assert expected_commit_line in source, (
            f"docstring footer for {op['name']!r} does not contain "
            f"{expected_commit_line!r}. Run `make measure-noise`."
        )
