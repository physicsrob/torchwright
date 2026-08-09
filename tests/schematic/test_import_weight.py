"""The schematic package must import without torch or numpy.

Reading a schematic is the pip-install-and-look use case; paying a
multi-second torch import (or requiring numpy before a support archive
is opened) would defeat it.  Run in a subprocess so this test's own
environment cannot mask a regression.
"""

import subprocess
import sys

_PROBE = """
import sys

import torchwright
import torchwright.schematic
import torchwright.schematic.format
import torchwright.schematic.support
import torchwright.schematic.validate

heavy = [name for name in ("torch", "numpy") if name in sys.modules]
assert not heavy, f"schematic import pulled heavy deps: {heavy}"
"""


def test_schematic_imports_stay_torch_and_numpy_free():
    subprocess.run(
        [sys.executable, "-c", _PROBE],
        check=True,
        capture_output=True,
        text=True,
    )
