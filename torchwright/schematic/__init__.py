"""The schematic: the hash-bound record shipped with every compiled artifact.

Format home and consumer surface for ``torchwright_schematic.json``.
``load_schematic`` reads and validates a manifest file alone;
``load_schematic_bundle`` binds it to the artifact directory.
Submodules: ``format`` (constants, hashing, the run encoding),
``validate`` (the checks both builder and readers run), ``support``
(the nonzero-coordinate npz sidecar), ``reader`` (the typed views).
Importing this package never imports torch; numpy loads only when a
support archive is opened.
"""

__all__ = [
    "Schematic",
    "SchematicBundle",
    "SchematicValidationError",
    "SupportArchive",
    "load_schematic",
    "load_schematic_bundle",
]


def __getattr__(name: str) -> object:
    if name == "SchematicValidationError":
        from torchwright.schematic.validate import SchematicValidationError

        return SchematicValidationError
    if name == "SupportArchive":
        from torchwright.schematic.support import SupportArchive

        return SupportArchive
    if name in __all__:
        from torchwright.schematic import reader

        return getattr(reader, name)
    raise AttributeError(name)
