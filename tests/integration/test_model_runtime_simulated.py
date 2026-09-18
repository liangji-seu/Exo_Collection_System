"""End-to-end online inference against spawned simulated preview workers.

Exercises the full boundary the desktop UI uses: ``ModalityPreviewProcessHandle``
(spawn) → ``RecordingStreamEndpoint`` → ``SubscriptionHub`` → feature window →
``DemoModelRuntime`` → ``MockTorqueOutput``.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from exo_collection.apps.collector.device_preview import (
    ModalityPreviewProcessHandle,
    ProfileModalityAdapterFactory,
)
from exo_collection.apps.model_runtime.backends import build_model_runtime
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.spec import ModelSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput

MODALITIES = ("imu", "encoder")


def _spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": "1.0.0",
            "model": {"backend": "demo", "demo": {"amplitude": 1.0, "scale": 10.0}},
            "features": {
                "control_rate_hz": 50.0,
                "window_s": 0.2,
                "inputs": [
                    {"modality": "imu", "channels": None},
                    {"modality": "encoder", "channels": None},
                ],
            },
            "output": {
                "kind": "torque_left_right",
                "single_output_maps_to": "right_torque",
                "torque_limit_nm": 5.0,
            },
            "safety": {"enable_required": True, "zero_on_error": True, "watchdog_s": 0.25},
        }
    )


@pytest.mark.integration
@pytest.mark.skipif(os.name != "nt", reason="Windows spawn contract")
def test_simulated_online_inference_end_to_end() -> None:
    handles: dict[str, ModalityPreviewProcessHandle] = {}
    recording_started: set[str] = set()
    trial_uuid: str | None = None
    controller: InferenceController | None = None
    try:
        for modality in MODALITIES:
            handle = ModalityPreviewProcessHandle(
                ProfileModalityAdapterFactory(profile_key="simulated", modality=modality),
                device_id=f"rt_{modality}",
                modality=modality,
                simulated=True,
                health_poll_interval_s=0.05,
                recording_queue_size=512,
            )
            handles[modality] = handle
            handle.start()

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            for handle in handles.values():
                handle.poll_events(limit=200)
            if all(handle.recording_endpoint is not None for handle in handles.values()):
                break
            assert all(handle.is_alive for handle in handles.values())
            time.sleep(0.01)
        assert all(handle.recording_endpoint is not None for handle in handles.values())

        trial_uuid = str(uuid.uuid4())
        hub = SubscriptionHub()
        for modality, handle in handles.items():
            endpoint = handle.recording_endpoint
            assert endpoint is not None
            hub.register_endpoint(modality, endpoint)
            handle.begin_recording(trial_uuid)
            recording_started.add(modality)

        spec = _spec()
        output = MockTorqueOutput(5.0, enable_required=True)
        controller = InferenceController(hub, spec, build_model_runtime(spec), output)
        controller.set_enabled(True)
        controller.start()

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and len(output.history) < 2:
            time.sleep(0.01)
        controller.stop_and_wait(timeout_ms=3000)

        assert controller.last_error is None
        assert hub.count("imu") > 0
        assert hub.count("encoder") > 0
        assert len(output.history) >= 1
        for command in output.history:
            assert abs(command.left_nm) <= 5.0 + 1e-6
            assert abs(command.right_nm) <= 5.0 + 1e-6
    finally:
        if controller is not None and controller.isRunning():
            controller.request_stop()
            controller.wait(2000)
        if trial_uuid is not None:
            for modality in tuple(recording_started):
                try:
                    handles[modality].end_recording(trial_uuid)
                except Exception:
                    pass
        for handle in handles.values():
            if handle.is_alive:
                handle.request_stop()
                handle.join(timeout=2.0)
            if handle.is_alive:
                handle.terminate(timeout=2.0)
            handle.close()
