"""``annotate`` / ``annotated`` — hierarchical provenance labels with dedup.

The dedup behavior matters in practice: sibling functions within one
subsystem call each other, each re-pushing its label, which without dedup
accumulates ``a/b/a/b`` paths. ``annotate`` collapses a component that is
already active in the path while still nesting *distinct* labels.
"""

from torchwright.graph import annotate, annotated
from torchwright.graph import node as _node


def _cur():
    return _node._current_annotation.get()


def test_basic_nesting():
    assert _cur() is None
    with annotate("render"):
        assert _cur() == "render"
        with annotate("texture"):
            assert _cur() == "render/texture"
        assert _cur() == "render"
    assert _cur() is None


def test_distinct_labels_still_nest():
    with annotate("proj"), annotate("paint"):
        assert _cur() == "proj/paint"


def test_dedup_consecutive_same_label():
    with annotate("paint"), annotate("paint"), annotate("paint"):
        assert _cur() == "paint"


def test_dedup_compound_label():
    with annotate("pmrk/R_CheckPlane"), annotate("pmrk/R_CheckPlane"):
        assert _cur() == "pmrk/R_CheckPlane"


def test_dedup_interleaved_components():
    # the A/B/A/B shape produced by sibling functions calling each other
    with annotate("stor/R_StoreWallRange"), annotate("pix/R_DrawColumn"):
        with annotate("pix/R_DrawColumn"):
            with annotate("tex"):
                assert _cur() == "stor/R_StoreWallRange/pix/R_DrawColumn/tex"


def test_reset_on_exit_even_on_error():
    try:
        with annotate("x"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert _cur() is None


def test_annotated_decorator_dedup():
    @annotated("bsp")
    def outer():
        @annotated("bsp")
        def inner():
            return _cur()

        return inner()

    assert outer() == "bsp"
    assert _cur() is None
