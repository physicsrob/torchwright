"""Regenerate the op-noise artefacts on Modal's CPU and sync them back locally.

`make measure-noise` runs on the local CPU, but the canonical drift check
(`tests/docs/test_numerical_noise_drift.py` under `make test`) runs on Modal's
CPU, and a few measurements sit on precision boundaries where local-CPU and
Modal-CPU round differently. This entrypoint runs `scripts.measure_op_noise` on a
Modal CPU worker (the same environment as the drift check) and returns the
regenerated `docs/op_noise_data.json`, `docs/numerical_noise.md`, and every
`torchwright/ops/*.py` (for the in-place noise-footer edits) so the local tree
matches what `make test` will measure.

This is the artefact-sync-back case the CLAUDE.md "Running scripts on GPU" rules
carve out as the only acceptable reason for a purpose-built `modal_*.py` — and it
imports `IMAGE` from `modal_image` rather than duplicating it.

    uv run modal run modal_measure_noise.py
"""

import pathlib
import subprocess
import sys

import modal

from modal_image import IMAGE

app = modal.App("torchwright-measure-noise", image=IMAGE)


# Match the test shard's container exactly (modal_test.py: gpu a100-80gb, cpu 8,
# memory 32768). `_measure_all` forces CPU compute, but reproducing the same
# container is what makes a few precision-boundary measurements (e.g. reciprocal
# p99) match what the drift check under `make test` measures.
@app.function(gpu="a100-80gb", cpu=8, memory=32768, timeout=1800)
def measure() -> dict:
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.measure_op_noise"]
    ).returncode
    if rc != 0:
        raise RuntimeError(f"scripts.measure_op_noise exited {rc}")

    from scripts.measure_op_noise import DOCS_JSON, DOCS_MD, REPO_ROOT

    out: dict[str, str] = {}
    for p in (DOCS_JSON, DOCS_MD):
        out[str(p.relative_to(REPO_ROOT))] = p.read_text()
    for f in sorted((REPO_ROOT / "torchwright" / "ops").glob("*.py")):
        out[str(f.relative_to(REPO_ROOT))] = f.read_text()
    return out


@app.local_entrypoint()
def main():
    files = measure.remote()
    root = pathlib.Path(__file__).resolve().parent
    for rel, content in sorted(files.items()):
        (root / rel).write_text(content)
        print(f"wrote {rel} ({len(content)} bytes)")
