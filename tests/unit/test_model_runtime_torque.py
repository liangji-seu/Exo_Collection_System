"""Unit tests for the online-testing torque protocol and Mock output."""

from __future__ import annotations

import math
import struct

import pytest

from exo_collection.adapters.encoder.teensy_serial import calc_crc8 as teensy_crc8
from exo_collection.apps.model_runtime.torque import (
    HEAD_CONTROL,
    HEAD_ENABLE,
    FRAME_TAIL,
    ENABLE_CMD,
    DISABLE_CMD,
    MockTorqueOutput,
    build_control_frame,
    build_enable_frame,
    calc_crc8,
    clamp_torque,
    disable_frame,
    enable_frame,
)


def test_crc8_matches_teensy_serial() -> None:
    samples = [b"", b"\x00", b"\x01\x02\x03", bytes(range(256))]
    for data in samples:
        assert calc_crc8(data) == teensy_crc8(data)


def test_control_frame_layout() -> None:
    frame = build_control_frame(0x1234, 1.5, -2.25, 5.0)
    assert len(frame) == 17
    assert frame[0] == HEAD_CONTROL
    assert frame[-1] == FRAME_TAIL
    seq, left, right, limit = struct.unpack("<H3f", frame[1:15])
    assert seq == 0x1234
    assert left == pytest.approx(1.5)
    assert right == pytest.approx(-2.25)
    assert limit == pytest.approx(5.0)
    assert calc_crc8(frame[1:15]) == frame[-2]


def test_enable_and_disable_frames() -> None:
    enable = enable_frame()
    assert enable == build_enable_frame(ENABLE_CMD)
    assert len(enable) == 4
    assert enable[0] == HEAD_ENABLE
    assert enable[1] == ENABLE_CMD
    assert enable[-1] == FRAME_TAIL
    assert calc_crc8(bytes([ENABLE_CMD])) == enable[2]

    disable = disable_frame()
    assert disable[1] == DISABLE_CMD


def test_clamp_torque() -> None:
    assert clamp_torque(3.0, 5.0) == 3.0
    assert clamp_torque(7.0, 5.0) == 5.0
    assert clamp_torque(-7.0, 5.0) == -5.0
    assert clamp_torque(float("nan"), 5.0) == 0.0
    assert clamp_torque(float("inf"), 5.0) == 0.0
    assert clamp_torque(None, 5.0) == 0.0
    assert clamp_torque("not-a-number", 5.0) == 0.0


def test_mock_disabled_by_default_when_enable_required() -> None:
    output = MockTorqueOutput(5.0, enable_required=True)
    assert output.enabled is False
    output.send(3.0, 4.0)
    last = output.last
    assert last is not None
    assert last.left_nm == 0.0
    assert last.right_nm == 0.0
    assert last.enabled is False


def test_mock_enabled_by_default_when_not_required() -> None:
    output = MockTorqueOutput(5.0, enable_required=False)
    assert output.enabled is True


def test_mock_enable_then_send_and_clamp() -> None:
    output = MockTorqueOutput(5.0, enable_required=True)
    output.set_enabled(True)
    output.send(6.0, -6.0)
    last = output.last
    assert last is not None
    assert last.left_nm == 5.0
    assert last.right_nm == -5.0
    assert last.torque_limit_nm == 5.0


def test_mock_disarm_zeroes_and_blocks() -> None:
    output = MockTorqueOutput(5.0, enable_required=True)
    output.set_enabled(True)
    output.send(2.0, 2.0)
    output.set_enabled(False)
    # Disarm must record a zero command and then block subsequent sends.
    assert output.last.left_nm == 0.0
    assert output.last.right_nm == 0.0
    output.send(2.0, 2.0)
    assert output.last.left_nm == 0.0


def test_mock_zero_records_zero() -> None:
    output = MockTorqueOutput(5.0, enable_required=False)
    output.send(2.0, 2.0)
    output.zero()
    assert output.last.left_nm == 0.0
    assert output.last.right_nm == 0.0


def test_mock_history_is_bounded_and_ordered() -> None:
    output = MockTorqueOutput(20.0, enable_required=False, history_capacity=4)
    for i in range(10):
        output.send(float(i), 0.0)
    history = output.history
    assert len(history) == 4
    assert [cmd.seq for cmd in history] == [6, 7, 8, 9]
    assert history[0].left_nm == 6.0


def test_mock_frames_match_builder() -> None:
    output = MockTorqueOutput(5.0, enable_required=False, record_frames=True)
    output.send(1.25, -3.5)
    last = output.last
    assert last is not None
    assert last.frame == build_control_frame(last.seq, 1.25, -3.5, 5.0)


def test_mock_no_frames_when_disabled() -> None:
    output = MockTorqueOutput(5.0, enable_required=False, record_frames=False)
    output.send(1.0, 1.0)
    assert output.last.frame == b""
