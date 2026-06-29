"""Run pytest on Modal with GPU access.

In sharded mode (default for full suite), tests are distributed across
independent Modal containers, each with its own A100 GPU.

Usage (via Makefile):
    make test                    # sharded across GPUs
    make test FILE=tests/foo.py  # single container, no sharding
    make test ARGS="-k test_foo" # filter applied to all shards
"""

import shlex
import subprocess
import sys
import time

import modal

from modal_image import IMAGE

app = modal.App("torchwright-test", image=IMAGE)

# ── Shard definitions ─────────────────────────────────────────────
# Simple file-level sharding.  Heavy compiled-test files get their
# own container; everything else is batched together.
# New test files are caught by the catch-all shard automatically.

_HEAVY_FILES: list[str] = [
    # The deepest token model (d=3072, 96 heads, chain-of-thought transcript) —
    # its own container.  RoPE recency graphs compile far deeper than the old
    # counter scheme, so the example/calculator tests are now the wall-time
    # drivers and must be spread across containers (they piled into the
    # catch-all and timed it out).
    "tests/compile/forward/test_forward_calculator_scratchpad.py",
]

# Each inner list becomes one container.  Splitting medium files across
# multiple groups keeps any single shard from dominating wall time.  The
# RoPE-era example/calculator compiles are the heavy items, grouped by the
# example family they exercise so the load spreads evenly.
_MEDIUM_FILE_GROUPS: list[list[str]] = [
    [
        "tests/debug/test_probe.py",
        "tests/compile/forward/test_sort_digits.py",
        "tests/ops/test_resampling_primitives.py",
    ],
    # Calculator family (d=1024–3072, ~90 layers each).
    [
        "tests/compile/forward/test_forward_calculator.py",
    ],
    [
        "tests/compile/forward/test_forward_calculator_v2.py",
        "tests/compile/forward/test_kv_cache.py",
        "tests/compile/forward/test_batched_decode_with_past.py",
    ],
    # Token-stream examples (d=1024, ~44 layers each).
    [
        "tests/compile/forward/test_forward_adder.py",
        "tests/compile/forward/test_caesar_cipher.py",
        "tests/compile/forward/test_fibonacci.py",
        "tests/compile/forward/test_binary_increment.py",
    ],
    # Tests that compile the adder via its create_network_parts.
    [
        "tests/compile/forward/test_module.py",
        "tests/compile/forward/test_matrix_occupancy.py",
        "tests/compile/forward/test_graph_analysis.py",
        "tests/debug/test_onnx_debug_session.py",
    ],
    # HF / ONNX parity exports.
    [
        "tests/hf/test_rope_token.py",
        "tests/hf/test_rope_self_match_token.py",
        "tests/hf/test_calculator_parity.py",
    ],
    # RoPE capability compiles (local recency lobe, content slow-planes, offset).
    [
        "tests/compile/forward/test_rope_local_recency.py",
        "tests/compile/forward/test_rope_content_slow_planes.py",
        "tests/compile/forward/test_rope_offset.py",
        "tests/compile/forward/test_rope_offset_all_deltas.py",
        "tests/compile/forward/test_bucketed_argmin.py",
    ],
    # Example-level reference-eval + range-printer compile.
    [
        "tests/examples/test_calculator_arithmetic.py",
        "tests/examples/test_calculator_scratchpad.py",
        "tests/examples/test_range_printer.py",
        "tests/examples/test_adder_v2.py",
    ],
]

_ALL_NAMED_FILES = _HEAVY_FILES + [f for g in _MEDIUM_FILE_GROUPS for f in g]

SHARDS = [
    *_HEAVY_FILES,
    *(" ".join(group) for group in _MEDIUM_FILE_GROUPS),
    "tests " + " ".join(f"--ignore={f}" for f in _ALL_NAMED_FILES),
]


# ── Remote function ───────────────────────────────────────────────


@app.function(gpu="a100-80gb", cpu=8, memory=32768, timeout=2400)
def run_pytest(pytest_args: str, shard_id: int = 0, extra_args: str = "") -> int:
    tag = f"[shard {shard_id}]"
    t0 = time.time()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *shlex.split(pytest_args),
        "-v",
        "--tb=short",
        "--no-header",
        "--durations=0",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    print(f"{tag} {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"{tag} {line}", end="")
    proc.wait()
    elapsed = time.time() - t0
    print(f"\n{tag} finished in {elapsed:.0f}s (exit {proc.returncode})")
    return proc.returncode


# ── Entrypoint ────────────────────────────────────────────────────


@app.local_entrypoint()
def main(file: str = "tests", args: str = ""):
    if file != "tests":
        rc = run_pytest.remote(pytest_args=file, shard_id=0, extra_args=args)
        sys.exit(rc)

    shards = SHARDS
    print(f"Running {len(shards)} shards in parallel:")
    for i, s in enumerate(shards):
        label = s[:90] + "…" if len(s) > 90 else s
        print(f"  shard {i}: {label}")

    t0 = time.time()
    results = list(
        run_pytest.map(shards, range(len(shards)), kwargs={"extra_args": args})
    )
    elapsed = time.time() - t0

    failed = sum(1 for rc in results if rc != 0)
    print(f"\n{'=' * 60}")
    print(f"All shards finished in {elapsed:.0f}s")
    for i, rc in enumerate(results):
        status = "PASS" if rc == 0 else "FAIL"
        print(f"  shard {i}: {status} (rc={rc})")
    if failed:
        print(f"{failed}/{len(results)} shards failed")
    print(f"{'=' * 60}")

    sys.exit(1 if failed else 0)
