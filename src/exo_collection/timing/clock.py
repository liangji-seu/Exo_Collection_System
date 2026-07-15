"""Host audit and non-jumping acquisition clock primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClockReading:
    host_monotonic_ns: int
    host_utc_ns: int


class HostClock:
    """Clock source used by orchestration and injectable in tests."""

    def monotonic_ns(self) -> int:
        return time.perf_counter_ns()

    def utc_ns(self) -> int:
        return time.time_ns()

    def read(self) -> ClockReading:
        # The acquisition timestamp is sampled first because it is the alignment
        # authority; UTC is audit metadata and may jump due to system time sync.
        return ClockReading(self.monotonic_ns(), self.utc_ns())


HOST_CLOCK = HostClock()

