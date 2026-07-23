"""Run an arbitrary Python module or script on Modal.

Usage (via Makefile):
    make modal-run MODULE=scripts.investigate_phase_e
    make modal-run MODULE=scripts.foo ARGS="--flag x"
    make modal-run SCRIPT=path/to/one_shot.py ARGS="..."
    make modal-run MODULE=scripts.cpu_only_thing CPU_ONLY=1

Direct:
    uv run modal run modal_run.py --module scripts.investigate_phase_e
"""

import shlex
import subprocess
import sys
import time

import modal

from modal_image import IMAGE

app = modal.App("torchwright-run", image=IMAGE)


def _build_cmd(module: str, script: str, args: str) -> list[str]:
    cmd = [sys.executable]
    if module:
        cmd += ["-m", module]
    else:
        cmd.append(script)
    if args:
        cmd += shlex.split(args)
    return cmd


@app.function(gpu="a100-80gb", cpu=8, memory=32768, timeout=1800)
def run_gpu(module: str, script: str, args: str) -> int:
    cmd = _build_cmd(module, script, args)
    print(f"[remote/gpu] {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    print(f"[remote/gpu] exit {rc} in {time.time() - t0:.0f}s")
    return rc


@app.function(cpu=4, memory=8192, timeout=1800)
def run_cpu(module: str, script: str, args: str) -> int:
    cmd = _build_cmd(module, script, args)
    print(f"[remote/cpu] {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    print(f"[remote/cpu] exit {rc} in {time.time() - t0:.0f}s")
    return rc


@app.local_entrypoint()
def main(
    module: str = "",
    script: str = "",
    args: str = "",
    *,
    cpu_only: bool = False,
) -> None:
    if not module and not script:
        print(
            "error: pass --module <dotted.name> or --script <path>",
            file=sys.stderr,
        )
        sys.exit(2)
    if module and script:
        print("error: pass --module OR --script, not both", file=sys.stderr)
        sys.exit(2)
    fn = run_cpu if cpu_only else run_gpu
    sys.exit(fn.remote(module=module, script=script, args=args))
