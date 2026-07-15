from __future__ import annotations

import numpy as np

from exo_collection.acquisition.buffers import SharedPreviewBuffer


def test_shared_preview_buffer_round_trip_and_downsampling() -> None:
    owner = SharedPreviewBuffer.create(5)
    attached = SharedPreviewBuffer.attach(owner.descriptor)
    try:
        generation = attached.write(np.arange(10), host_monotonic_ns=123)
        values, timestamp, observed_generation = owner.read()
        assert values.tolist() == [0.0, 2.0, 4.0, 6.0, 9.0]
        assert timestamp == 123
        assert observed_generation == generation
        assert generation % 2 == 0
    finally:
        attached.close()
        owner.close()
        owner.unlink()

