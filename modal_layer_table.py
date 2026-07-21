"""Regenerate docs/calculator_layer_table.json with the sweep run on Modal.

The calculator layer/step/parameter sweep is pure CPU but hungry (the
scratchpad graphs at n=9,10 build for minutes) — it belongs on a Modal
worker, not the local machine.  ``modal-run`` cannot sync artifacts back,
and this sweep's whole point is the committed JSON, so this is one of the
purpose-built entrypoints CLAUDE.md carves out for artifact sync-back:
:func:`scripts.calculator_layer_table.collect` runs remotely and returns
the payload; only the JSON write and the table print happen locally.

Usage::

    uv run modal run modal_layer_table.py
    uv run modal run modal_layer_table.py --ns 2,3 --out /tmp/t.json
"""

import json

import modal

from modal_image import IMAGE

app = modal.App("torchwright-layer-table", image=IMAGE)


@app.function(cpu=4, memory=16384, timeout=3600)
def collect_remote(ns: list[int]) -> dict:
    from scripts.calculator_layer_table import collect

    return collect(ns)


@app.local_entrypoint()
def main(ns: str = "2,3,4,5,6,7,8,9,10", out: str = "docs/calculator_layer_table.json"):
    from scripts.calculator_layer_table import render

    payload = collect_remote.remote([int(x) for x in ns.split(",")])
    print(render(payload))
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
