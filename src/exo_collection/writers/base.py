"""Stable Writer lifecycle and built-in Writer type registry."""

from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class Writer(Protocol):
    """Common lifecycle implemented by every append-only modality Writer."""

    @property
    def closed(self) -> bool: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


WriterClass: TypeAlias = type[Writer]


def built_in_writer_registry() -> dict[str, WriterClass]:
    """Return a fresh registry so callers cannot mutate global core state."""

    from .binary_block import BlockBinaryWriter
    from .hdf5_signal import Hdf5SignalWriter

    return {
        "block_binary": BlockBinaryWriter,
        "hdf5_signal": Hdf5SignalWriter,
    }


def resolve_writer_type(name: str) -> WriterClass:
    try:
        return built_in_writer_registry()[name]
    except KeyError as exc:
        raise KeyError(f"unknown Writer type: {name}") from exc


__all__ = ["Writer", "WriterClass", "built_in_writer_registry", "resolve_writer_type"]
