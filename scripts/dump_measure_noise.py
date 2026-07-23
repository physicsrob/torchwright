"""Run measure-noise and dump the resulting JSON to stdout.

Used to capture Modal-measured noise values for commit when the local
venv is broken or we need A100-specific measurements.  Usage:

    make modal-run MODULE=scripts.dump_measure_noise > docs/op_noise_data.json
"""

import sys

from scripts.measure_op_noise import _measure_all, render_json


def main() -> None:
    measurements = _measure_all()
    # Get commit SHA from env or fallback
    import subprocess

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    import datetime

    now = datetime.datetime.utcnow().isoformat() + "Z"
    json_text = render_json(measurements, commit=sha, measured_at=now)
    # Print a separator so we can grep out the real JSON block
    sys.stderr.write("=== NOISE JSON BEGIN ===\n")
    sys.stdout.write(json_text)
    sys.stdout.flush()
    sys.stderr.write("\n=== NOISE JSON END ===\n")


if __name__ == "__main__":
    main()
