import pytest
import torch

import torchwright.graph.node as _node_module
from torchwright.compiler.device import set_device


@pytest.fixture(autouse=True)
def reset_node_id_counter():
    """Reset the global node ID counter to 0 before each test.

    Node.__hash__ returns node_id, so the counter's value at test time
    determines set/dict iteration order in the compiler.  Earlier tests
    that allocate nodes shift the counter, which shifts column assignments
    in the residual-stream scheduler, which drifts compiled pixel values
    by a few percent (Mode-B flakiness in test_game_graph.py) and also
    perturbs fuse_consecutive_linears candidate ordering (Mode-A bug in
    test_optimize.py).  Resetting here makes every test see node_id=0 for
    its first node, independent of test-suite execution order.
    """
    _node_module.global_node_id = 0


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        default="auto",
        help="Torch device for tests: 'cpu', 'cuda', or 'auto' (default: auto-detect)",
    )


@pytest.fixture(scope="session", autouse=True)
def device(request):
    """Auto-detect the best available torch device, or use --device override."""
    choice = request.config.getoption("--device")
    if choice == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(choice)

    set_device(dev)
    return dev


def pytest_sessionstart(session):
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n[gpu] {props.name}, {props.total_memory / 2**30:.1f} GiB VRAM")


def pytest_sessionfinish(session, exitstatus):
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(
            f"\n[gpu] peak allocated: {peak:.2f} GiB, peak reserved: {reserved:.2f} GiB"
        )
