"""Byte-preserving gaitway-3D recording writer.

The gaitway adapter publishes every TCP packet twice: the continuous Type-I
total GRF/COP also flows through the normal HDF5 ``SampleBatch`` path, while a
:class:`~exo_collection.domain.events.GaitwayPacketEvent` preserves the exact
on-wire bytes.  This writer consumes those packet events and records, under a
``gaitway/`` subdirectory of the Trial::

    gaitway_raw.bin     verbatim TCP packets (each self-framed by its U16 size)
    gaitway_type1.csv   parsed Type-I total GRF/COP/treadmill-status samples
    gaitway_type2.csv   parsed Type-II left/right decomposition samples
    gaitway_meta.json   connection/parser provenance and packet counters
    gaitway_log.txt     human-readable acquisition log

Keeping the raw binary makes the parser fixable offline — a later parser bug
never requires re-acquiring a subject.  Type II is recorded as gaitway's
internal left/right decomposition result (``grf_source_type`` in the meta),
never as two independent physical plates.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from exo_collection.adapters.force_plate.gaitway_tcp import (
    FORCE_PLATE_CHANNELS,
    GRF_SOURCE_TYPE_DECOMPOSED,
    PACKET_TYPE_I,
    PACKET_TYPE_II,
    TYPE_II_CHANNELS,
    parse_type_i_packet,
    parse_type_ii_packet,
)
from exo_collection.domain.events import GaitwayPacketEvent
from exo_collection.storage.layout import TrialLayout

_log = logging.getLogger(__name__)

GAITWAY_DIR = "gaitway"
RAW_BIN_RELATIVE = f"{GAITWAY_DIR}/gaitway_raw.bin"
TYPE1_CSV_RELATIVE = f"{GAITWAY_DIR}/gaitway_type1.csv"
TYPE2_CSV_RELATIVE = f"{GAITWAY_DIR}/gaitway_type2.csv"
META_JSON_RELATIVE = f"{GAITWAY_DIR}/gaitway_meta.json"
LOG_TXT_RELATIVE = f"{GAITWAY_DIR}/gaitway_log.txt"

TYPE1_COLUMNS = (
    "host_rx_time_ns",
    "packet_id",
    "sample_index",
    "fx_total",
    "fy_total",
    "fz_total",
    "cop_x_total",
    "cop_y_total",
    "tz_total",
    "treadmill_speed",
    "treadmill_elevation",
    "heart_rate",
    "digital_inputs",
)

TYPE2_COLUMNS = (
    "host_rx_time_ns",
    "packet_id",
    "step_index",
    "sample_index_in_step",
    "foot_contact",
    "digital_inputs",
    "fx_l",
    "fy_l",
    "fz_l",
    "cop_x_l",
    "cop_y_l",
    "fx_r",
    "fy_r",
    "fz_r",
    "cop_x_r",
    "cop_y_r",
)


class GaitwayPacketWriter:
    """Incremental raw + CSV recorder for gaitway packet events.

    Files are written under ``*.partial`` names and left for the Trial
    publication step to rename, matching every other modality Writer.  A
    malformed packet is logged and counted but never aborts the stream; the
    raw bytes are always preserved first.
    """

    def __init__(
        self,
        layout: TrialLayout,
        *,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        self._layout = layout
        self._config = dict(config)
        self._metadata = dict(metadata)
        self._sample_rate_hz = float(self._config.get("sample_rate_hz", 1000.0))
        self._type_i_mode = int(self._config.get("type_i_mode", 2))
        self._type_ii_mode = int(self._config.get("type_ii_mode", 2))
        self._save_raw = bool(self._config.get("save_raw_packets", True))
        self._save_csv = bool(self._config.get("save_parsed_csv", True))

        self._raw_path = layout.partial_path(RAW_BIN_RELATIVE)
        self._type1_path = layout.partial_path(TYPE1_CSV_RELATIVE)
        self._type2_path = layout.partial_path(TYPE2_CSV_RELATIVE)
        self._meta_path = layout.partial_path(META_JSON_RELATIVE)
        self._log_path = layout.partial_path(LOG_TXT_RELATIVE)

        self._raw_file: BinaryIO | None = None
        self._type1_file: TextIO | None = None
        self._type2_file: TextIO | None = None
        self._type1_writer: csv.writer | None = None
        self._type2_writer: csv.writer | None = None

        self._type1_packets = 0
        self._type1_samples = 0
        self._type2_packets = 0
        self._type2_samples = 0
        self._malformed_packets = 0
        self._first_packet_ns: int | None = None
        self._last_packet_ns: int | None = None
        self._connected_at_utc = datetime.now(timezone.utc).isoformat()
        self._closed = False
        self._log_lines: list[str] = []
        self._last_error: str | None = None

        self._open_files()

    # -- file lifecycle ----------------------------------------------------

    def _open_files(self) -> None:
        self._log_lines.append(
            f"{_iso_now()} opened gaitway writer in {self._layout.recording_directory}"
        )
        if self._save_raw:
            self._raw_file = self._raw_path.open("wb")
        if self._save_csv:
            self._type1_file = self._type1_path.open("w", encoding="utf-8", newline="")
            self._type1_writer = csv.writer(self._type1_file)
            self._type1_writer.writerow(TYPE1_COLUMNS)
            self._type2_file = self._type2_path.open("w", encoding="utf-8", newline="")
            self._type2_writer = csv.writer(self._type2_file)
            self._type2_writer.writerow(TYPE2_COLUMNS)

    def append_event(self, event: GaitwayPacketEvent) -> None:
        """Record one packet: raw bytes first, then a best-effort CSV parse."""
        received_ns = int(event.host_monotonic_ns)
        self._first_packet_ns = (
            received_ns if self._first_packet_ns is None else self._first_packet_ns
        )
        self._last_packet_ns = received_ns

        if self._raw_file is not None:
            self._raw_file.write(bytes(event.raw_bytes))

        if event.packet_type == PACKET_TYPE_I:
            self._type1_packets += 1
            if self._save_csv:
                self._write_type1(event, received_ns)
        elif event.packet_type == PACKET_TYPE_II:
            self._type2_packets += 1
            if self._save_csv:
                self._write_type2(event, received_ns)
        else:
            self._malformed_packets += 1
            self._log_lines.append(
                f"{_iso_now()} unexpected packet_type {event.packet_type} ignored"
            )

    def _write_type1(self, event: GaitwayPacketEvent, received_ns: int) -> None:
        try:
            _header, data = parse_type_i_packet(bytes(event.raw_bytes))
        except Exception as exc:  # noqa: BLE001 - a bad packet must never abort recording
            self._malformed_packets += 1
            self._last_error = f"type1 parse: {type(exc).__name__}: {exc}"
            self._log_lines.append(f"{_iso_now()} {self._last_error}")
            return
        count = data.shape[0]
        period_ns = round(1_000_000_000 / self._sample_rate_hz)
        first_host_ns = received_ns - (count - 1) * period_ns
        for index in range(count):
            row = data[index]
            self._type1_writer.writerow(
                [
                    first_host_ns + index * period_ns,
                    int(event.packet_id),
                    int(event.sample_index) + index,
                    *[float(item) for item in row],
                ]
            )
        self._type1_samples += count

    def _write_type2(self, event: GaitwayPacketEvent, received_ns: int) -> None:
        try:
            _header, data = parse_type_ii_packet(bytes(event.raw_bytes))
        except Exception as exc:  # noqa: BLE001 - see above
            self._malformed_packets += 1
            self._last_error = f"type2 parse: {type(exc).__name__}: {exc}"
            self._log_lines.append(f"{_iso_now()} {self._last_error}")
            return
        count = data.shape[0]
        step_index = int(event.sample_index)
        for index in range(count):
            row = data[index]
            foot_contact = int(row[0])
            digital = int(row[1])
            forces = [float(item) for item in row[2:]]
            self._type2_writer.writerow(
                [
                    received_ns,
                    int(event.packet_id),
                    step_index + index,
                    index,
                    foot_contact,
                    digital,
                    *forces,
                ]
            )
        self._type2_samples += count

    def close(self) -> None:
        """Flush data files and write ``gaitway_meta.json`` + ``gaitway_log.txt``."""
        if self._closed:
            return
        self._closed = True

        try:
            for stream in (self._raw_file, self._type1_file, self._type2_file):
                if stream is not None:
                    stream.flush()
        finally:
            for stream, attr in (
                (self._raw_file, "_raw_file"),
                (self._type1_file, "_type1_file"),
                (self._type2_file, "_type2_file"),
            ):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
                setattr(self, attr, None)

        meta = {
            "grf_source_type": GRF_SOURCE_TYPE_DECOMPOSED,
            "protocol": self._metadata.get("protocol", "TM-ICD-0004-ARS A5"),
            "server_host": self._config.get("server_host"),
            "server_port": self._config.get("server_port"),
            "sample_rate_hz": self._sample_rate_hz,
            "type_i_mode": self._type_i_mode,
            "type_ii_mode": self._type_ii_mode,
            "type_i_enabled": self._type_i_mode > 0,
            "type_ii_enabled": self._type_ii_mode > 0,
            "packet_count_type_i": self._type1_packets,
            "packet_count_type_ii": self._type2_packets,
            "sample_count_type_i": self._type1_samples,
            "sample_count_type_ii": self._type2_samples,
            "malformed_packet_count": self._malformed_packets,
            "first_packet_host_monotonic_ns": self._first_packet_ns,
            "last_packet_host_monotonic_ns": self._last_packet_ns,
            "force_unit": "N",
            "cop_unit": "m",
            "connected_at_utc": self._connected_at_utc,
            "closed_at_utc": _iso_now(),
            "type1_channels": list(FORCE_PLATE_CHANNELS),
            "type2_channels": list(TYPE_II_CHANNELS),
            "type1_csv_columns": list(TYPE1_COLUMNS),
            "type2_csv_columns": list(TYPE2_COLUMNS),
            "last_error": self._last_error,
        }
        self._log_lines.append(f"{_iso_now()} closed gaitway writer")
        self._meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._log_path.write_text(
            "\n".join(self._log_lines) + "\n", encoding="utf-8"
        )

    @property
    def packet_count_type_i(self) -> int:
        return self._type1_packets

    @property
    def packet_count_type_ii(self) -> int:
        return self._type2_packets

    @property
    def malformed_packet_count(self) -> int:
        return self._malformed_packets


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "GAITWAY_DIR",
    "RAW_BIN_RELATIVE",
    "TYPE1_CSV_RELATIVE",
    "TYPE2_CSV_RELATIVE",
    "META_JSON_RELATIVE",
    "LOG_TXT_RELATIVE",
    "TYPE1_COLUMNS",
    "TYPE2_COLUMNS",
    "GaitwayPacketWriter",
]
