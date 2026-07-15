"""Small lock-free shared-memory buffer for lossy, downsampled UI previews."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from time import perf_counter_ns

import numpy as np
from numpy.typing import ArrayLike, NDArray


HEADER_WORDS = 3  # generation, valid length, host monotonic timestamp
HEADER_NBYTES = HEADER_WORDS * np.dtype("<u8").itemsize


@dataclass(frozen=True, slots=True)
class PreviewBufferDescriptor:
    name: str
    capacity: int
    dtype: str = "<f4"


class SharedPreviewBuffer:
    """Single-writer/multi-reader preview exchange using a sequence lock.

    Raw acquisition never enters this buffer. A producer may overwrite old
    preview data at any time; readers retry if they observe an in-progress copy.
    """

    def __init__(self, memory: shared_memory.SharedMemory, capacity: int, *, owner: bool) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._memory = memory
        self.capacity = int(capacity)
        self._owner = owner
        self._closed = False
        self._header = np.ndarray((HEADER_WORDS,), dtype="<u8", buffer=memory.buf[:HEADER_NBYTES])
        self._values = np.ndarray(
            (self.capacity,),
            dtype="<f4",
            buffer=memory.buf[HEADER_NBYTES : HEADER_NBYTES + self.capacity * 4],
        )

    @classmethod
    def create(cls, capacity: int) -> SharedPreviewBuffer:
        size = HEADER_NBYTES + int(capacity) * np.dtype("<f4").itemsize
        memory = shared_memory.SharedMemory(create=True, size=size)
        result = cls(memory, int(capacity), owner=True)
        result._header[:] = 0
        result._values[:] = 0
        return result

    @classmethod
    def attach(cls, descriptor: PreviewBufferDescriptor) -> SharedPreviewBuffer:
        if descriptor.dtype != "<f4":
            raise ValueError(f"unsupported preview dtype: {descriptor.dtype}")
        memory = shared_memory.SharedMemory(name=descriptor.name, create=False)
        expected = HEADER_NBYTES + descriptor.capacity * 4
        if memory.size < expected:
            memory.close()
            raise ValueError("shared preview segment is smaller than its descriptor")
        return cls(memory, descriptor.capacity, owner=False)

    @property
    def descriptor(self) -> PreviewBufferDescriptor:
        self._ensure_open()
        return PreviewBufferDescriptor(self._memory.name, self.capacity)

    def write(self, values: ArrayLike, host_monotonic_ns: int | None = None) -> int:
        self._ensure_open()
        array = np.asarray(values, dtype="<f4").reshape(-1)
        if array.size > self.capacity:
            # Preview is deliberately lossy; preserve a uniform view of the newest frame.
            positions = np.linspace(0, array.size - 1, self.capacity, dtype=np.int64)
            array = array[positions]
        generation = int(self._header[0])
        if generation % 2:
            generation += 1
        self._header[0] = generation + 1  # odd: copy in progress
        self._values[: array.size] = array
        self._header[1] = array.size
        self._header[2] = perf_counter_ns() if host_monotonic_ns is None else host_monotonic_ns
        self._header[0] = generation + 2  # even: stable snapshot
        return generation + 2

    def read(self, retries: int = 20) -> tuple[NDArray[np.float32], int, int]:
        self._ensure_open()
        for _ in range(retries):
            before = int(self._header[0])
            if before % 2:
                continue
            length = min(int(self._header[1]), self.capacity)
            timestamp = int(self._header[2])
            values = self._values[:length].copy()
            after = int(self._header[0])
            if before == after and after % 2 == 0:
                return values, timestamp, after
        raise RuntimeError("preview buffer changed continuously while reading")

    def close(self) -> None:
        if self._closed:
            return
        # Drop exported ndarray views before closing the underlying mapping.
        del self._header
        del self._values
        self._memory.close()
        self._closed = True

    def unlink(self) -> None:
        if not self._owner:
            raise PermissionError("only the shared-memory owner may unlink it")
        self._memory.unlink()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("shared preview buffer is closed")

    def __enter__(self) -> SharedPreviewBuffer:
        self._ensure_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

