"""Regenerate docs/calculator_layer_table.json with the sweep run on Modal.

The calculator layer/step/parameter sweep is pure CPU but hungry: every
cell now runs a real ``optimize=2`` compile at the family's canonical
geometry (the witnessed ``layers`` number), and the scratchpad graphs at
n=9,10 build for minutes before the compile even starts — it belongs on
Modal workers, not the local machine.  ``modal-run`` cannot sync
artifacts back, and this sweep's whole point is the committed JSON, so
this is one of the purpose-built entrypoints CLAUDE.md carves out for
artifact sync-back: :func:`scripts.calculator_layer_table.collect_cell`
runs remotely — one worker per (impl, n) cell, fanned out with
``.map`` — and only the JSON write and the table print happen locally.

Modal workers start with ``TW_SCHEDULE_CACHE_DIR`` unset (and
``collect_cell`` drops it defensively), so every witnessed compile is a
fresh schedule solve, never a cache replay.

Usage::

    uv run modal run modal_layer_table.py
    uv run modal run modal_layer_table.py --ns 2,3 --out /tmp/t.json
    uv run modal run modal_layer_table.py --impls calculator_scratchpad --ns 6,10
"""

import json
from pathlib import Path

import modal

from modal_image import IMAGE

app = modal.App("torchwright-layer-table", image=IMAGE)


@app.function(cpu=8, memory=32768, timeout=3600)
def collect_cell_remote(cell: tuple) -> dict:
    from scripts.calculator_layer_table import collect_cell

    impl_name, n = cell
    return collect_cell(impl_name, n)


@app.local_entrypoint()
def main(
    ns: str = "2,3,4,5,6,7,8,9,10",
    out: str = "docs/calculator_layer_table.json",
    impls: str = "",
) -> None:
    from scripts.calculator_layer_table import IMPLS, render, table_config

    ns_list = [int(x) for x in ns.split(",")]
    impl_list = [x for x in impls.split(",") if x] or IMPLS
    cells = [(impl, n) for impl in impl_list for n in ns_list]
    results: dict = {impl: [] for impl in impl_list}
    for (impl, _n), row in zip(cells, collect_cell_remote.map(cells), strict=False):
        results[impl].append(row)
    payload = {"config": table_config(ns_list), "results": results}
    if len(impl_list) == len(IMPLS):
        print(render(payload))
    else:
        print(json.dumps(results, indent=2))
    with Path(out).open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
