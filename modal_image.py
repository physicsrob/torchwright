"""Shared Modal image used by modal_run / modal_test."""

import modal

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    # test-onnx adds CPU onnxruntime so the ONNX-runtime and HF-parity tests
    # actually execute here instead of import-skipping (see the test-onnx group
    # in pyproject.toml).  Modal builds from torchwright/pyproject.toml
    # standalone (it rejects the umbrella's workspace table) and syncs
    # torchwright/uv.lock --frozen, so that lock must be regenerated
    # out-of-workspace whenever these groups change (run `make modal-lock`);
    # `make test` runs `check-modal-lock` first to catch drift.
    .uv_sync(groups=["dev", "test-onnx"], extra_options="--no-install-project")
    .add_local_file("E8.8.1024.txt", "/root/E8.8.1024.txt")
    .add_local_dir("docs", "/root/docs")
    .add_local_python_source(
        "torchwright", "examples", "tests", "scripts", "modal_image"
    )
)
