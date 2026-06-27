"""Guard: torchwright/uv.lock must carry every package the Modal test image installs.

The Modal test image (``modal_image.py``) and any other ``--frozen`` consumer
install from ``torchwright/uv.lock`` — a STANDALONE lock that ``uv`` cannot
refresh from inside the umbrella workspace (``uv`` resolves to the umbrella root,
whose ``[tool.uv.workspace]`` table Modal rejects).  So the lock can silently
drift from ``pyproject.toml``: a dependency added to a synced group but missing
from the lock gets installed as *nothing*, and the tests that need it
import-skip in CI without anyone noticing — exactly the failure this guard
exists to prevent (it is what bit the ONNX/HF test surface before).

This is a fast, offline presence check: does every requirement in the
Modal-synced groups have a corresponding locked package?  It catches the
dominant drift (a new dependency entirely absent from the lock).  It does NOT
verify version constraints — for a full consistency check and to fix any drift,
regenerate the lock out-of-workspace with ``make modal-lock``.

It is wired in as a prerequisite of ``make test`` (the gate this project
actually uses) — replacing the old GitHub Actions ``uv ... --locked`` step, now
that that workflow is gone.

Keep ``MODAL_SYNC_GROUPS`` in step with ``modal_image.py``'s ``groups=[...]``.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # torchwright/

# Mirror modal_image.py's uv_sync(groups=...).
MODAL_SYNC_GROUPS = ["dev", "test-onnx"]

_LEADING_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _canon(name: str) -> str:
    """PEP 503 canonical form so e.g. 'onnxruntime_gpu' == 'onnxruntime-gpu'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())

    requirements = list(pyproject.get("project", {}).get("dependencies", []))
    groups = pyproject.get("dependency-groups", {})
    for group in MODAL_SYNC_GROUPS:
        requirements += groups.get(group, [])

    # canonical package name -> the raw requirement string (for messages)
    required: dict[str, str] = {}
    for req in requirements:
        match = _LEADING_NAME.match(req.strip())
        if match:
            required[_canon(match.group(0))] = req

    locked = {_canon(pkg["name"]) for pkg in lock.get("package", [])}

    missing = sorted(name for name in required if name not in locked)
    if missing:
        sys.stderr.write(
            "torchwright/uv.lock is missing packages required by the "
            f"Modal-synced groups {MODAL_SYNC_GROUPS}:\n"
        )
        for name in missing:
            sys.stderr.write(f"  - {required[name]}\n")
        sys.stderr.write(
            "\nThe Modal test image installs from this standalone lock with "
            "--frozen, so these deps would be silently absent and their tests "
            "would import-skip.\nRegenerate the lock out-of-workspace:\n\n"
            "    make modal-lock\n"
        )
        return 1

    print(
        f"torchwright/uv.lock carries all {len(required)} packages required by "
        f"groups {MODAL_SYNC_GROUPS}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
