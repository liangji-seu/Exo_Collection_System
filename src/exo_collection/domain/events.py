"""Structured device, sample, metric, and health messages."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from time import perf_counter_ns, time_ns
from typing import Annotated, Any, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from .models import NonEmptyStr, UTCDateTime, utc_now


class EventModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )


class DeviceStatus(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    PREPARING = "PREPARING"
    READY = "READY"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"
    FAULT = "FAULT"
    CLOSED = "CLOSED"


class HealthStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class EdgeType(StrEnum):
    RISING = "rising"
    FALLING = "falling"


class BaseEvent(EventModel):
    event_uuid: UUID = Field(default_factory=uuid4)
    event_type: str
    session_uuid: UUID | None = None
    trial_uuid: UUID | None = None
    device_id: NonEmptyStr
    modality: NonEmptyStr
    clock_domain: NonEmptyStr
    host_monotonic_ns: int = Field(default_factory=perf_counter_ns, ge=0)
    host_utc_ns: int = Field(default_factory=time_ns, ge=0)
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)


class SampleBatch(BaseEvent):
    event_type: Literal["sample_batch"] = "sample_batch"
    first_sample_index: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    sequence_number: int = Field(ge=0)
    device_timestamp: int | float | None = None
    sample_rate_hz: float | None = Field(default=None, gt=0)
    data: Any = Field(repr=False)


class FrameBatch(BaseEvent):
    event_type: Literal["frame_batch"] = "frame_batch"
    first_frame_index: int = Field(ge=0)
    frame_count: int = Field(gt=0)
    sequence_number: int = Field(ge=0)
    device_timestamp: int | float | None = None
    frame_rate_hz: float | None = Field(default=None, gt=0)
    data: Any = Field(repr=False)
    channel: int | None = Field(default=None, ge=0, le=255)
    tail_flags: int = Field(default=0, ge=0, le=255)


class SyncPulseEvent(BaseEvent):
    event_type: Literal["sync_pulse"] = "sync_pulse"
    pulse_id: NonEmptyStr
    source_device: NonEmptyStr
    edge_type: EdgeType
    sample_index: int = Field(ge=0)
    amplitude: float
    pulse_width_ns: int | None = Field(default=None, ge=0)
    detection_threshold: float
    confidence: float = Field(ge=0.0, le=1.0)
    detector_version: NonEmptyStr


class DeviceStatusEvent(BaseEvent):
    event_type: Literal["device_status"] = "device_status"
    status: DeviceStatus
    previous_status: DeviceStatus | None = None
    message: str | None = None
    error_code: str | None = None


class MetricEvent(BaseEvent):
    event_type: Literal["metric"] = "metric"
    metric_name: NonEmptyStr
    value: int | float
    unit: str | None = None
    tags: dict[str, JsonValue] = Field(default_factory=dict)


DomainEvent = Annotated[
    Union[SampleBatch, FrameBatch, SyncPulseEvent, DeviceStatusEvent, MetricEvent],
    Field(discriminator="event_type"),
]


class HealthSnapshot(EventModel):
    """Point-in-time health returned by ``ModalityAdapter.health``."""

    device_id: NonEmptyStr
    modality: NonEmptyStr
    status: HealthStatus = HealthStatus.UNKNOWN
    device_status: DeviceStatus = DeviceStatus.DISCONNECTED
    connected: bool = False
    ready: bool = False
    sampling: bool = False
    sampled_at_utc: UTCDateTime = Field(default_factory=utc_now)
    host_monotonic_ns: int = Field(default_factory=perf_counter_ns, ge=0)
    last_data_host_monotonic_ns: int | None = Field(default=None, ge=0)
    actual_sample_rate_hz: float | None = Field(default=None, ge=0)
    nominal_sample_rate_hz: float | None = Field(default=None, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    queue_capacity: int | None = Field(default=None, gt=0)
    dropped_packets: int = Field(default=0, ge=0)
    sequence_gaps: int = Field(default=0, ge=0)
    temperature_c: float | None = None
    message: str | None = None
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)

    @field_validator("last_data_host_monotonic_ns")
    @classmethod
    def validate_last_data_time(cls, value: int | None) -> int | None:
        return value

    @property
    def queue_utilization(self) -> float | None:
        if self.queue_capacity is None:
            return None
        return self.queue_depth / self.queue_capacity


# Clear semantic alias for callers that use the architecture's wording.
DeviceHealthSnapshot = HealthSnapshot

