"""Field self-check for the h/p/cosmos gaitway-3D instrumented treadmill.

This is a standalone diagnostic — it does **not** start a formal experiment and
does not touch the recording pipeline.  It opens one TCP connection to the
gaitway streaming server (default port 49500), requests both Type I (total
GRF/COP/Tz) and Type II (left/right decomposition), collects for ~10-20 s, then
writes ``gaitway_test_report.json`` and ``gaitway_test_plot.png``.

The verdicts answer the questions a field operator cares about:

* did we receive Type I / Type II at all;
* how many packets / samples of each;
* the Fz total / Fz-left / Fz-right magnitude ranges;
* whether ``FzL + FzR ≈ FzTotal`` (magnitude agreement);
* whether right/left single-stance and double-stance phases are visible.

Type II is gaitway's *internal* left/right decomposition of a single platform,
not a second physical force plate — the report records that provenance.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from struct import unpack_from
from typing import Any

import numpy as np

from exo_collection import __version__
from exo_collection.adapters.force_plate.gaitway_tcp import (
    GAITWAY_DEFAULT_PORT,
    PACKET_ACK,
    PACKET_NAK,
    PACKET_SETTINGS,
    PACKET_TYPE_I,
    PACKET_TYPE_II,
    GaitwayPacketError,
    GaitwayPacketFramer,
    build_start_ds_command,
    parse_type_i_packet,
    parse_type_ii_packet,
)

# Canonical channel indices produced by the parsers (see gaitway_tcp.py).
_FZ_TOTAL = 2  # FORCE_PLATE_CHANNELS -> "fz"
_FZ_LEFT = 4  # TYPE_II_CHANNELS     -> "fz_l"
_FZ_RIGHT = 9  # TYPE_II_CHANNELS    -> "fz_r"

# Field-check thresholds (Newtons).  A walking adult loads each limb with
# hundreds of newtons; these separations are deliberately loose.
_SINGLE_STANCE_MIN_N = 300.0  # the stance limb must carry a clear load
_SINGLE_STANCE_RATIO = 3.0  # stance limb must dominate the swing limb by 3x
_DOUBLE_STANCE_MIN_N = 100.0  # both limbs must carry a non-trivial load
_MAGNITUDE_AGREEMENT_TOL = 0.15  # |mean(FzL+FzR)-mean(FzTotal)|/mean(FzTotal)

DEFAULT_REPORT_NAME = "gaitway_test_report.json"
DEFAULT_PLOT_NAME = "gaitway_test_plot.png"


@dataclass
class GaitwayTestReport:
    """Machine-readable result of one gaitway self-check run."""

    ok: bool
    host: str
    port: int
    sample_rate_hz: int
    duration_requested_s: float
    duration_actual_s: float
    started_at_utc: str
    software_version: str
    grf_source_type: str
    sent_command: str
    server_response_summary: str
    gaitway_settings_version: int | None
    settings_packet_hex: str | None
    type_i_received: bool
    type_ii_received: bool
    type_i_packet_count: int
    type_ii_packet_count: int
    type_i_sample_count: int
    type_ii_sample_count: int
    fz_total_min: float | None
    fz_total_max: float | None
    fz_left_min: float | None
    fz_left_max: float | None
    fz_right_min: float | None
    fz_right_max: float | None
    checks: dict[str, dict[str, Any]]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def _range_of(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    return float(np.min(values)), float(np.max(values))


def compute_checks(
    *,
    fz_total: np.ndarray,
    fz_left: np.ndarray,
    fz_right: np.ndarray,
    type_i_received: bool,
    type_ii_received: bool,
) -> dict[str, dict[str, Any]]:
    """Compute the field self-check verdicts from the collected force traces.

    Pure and side-effect free so it can be unit-tested without a treadmill.
    """
    checks: dict[str, dict[str, Any]] = {}
    checks["type_i_received"] = {
        "ok": bool(type_i_received),
        "detail": (
            f"{int(fz_total.size)} samples"
            if type_i_received
            else "no Type I packets received"
        ),
    }
    checks["type_ii_received"] = {
        "ok": bool(type_ii_received),
        "detail": (
            f"{int(fz_left.size)} samples"
            if type_ii_received
            else "no Type II packets received"
        ),
    }

    if type_i_received and type_ii_received and fz_total.size and fz_left.size:
        mean_total = float(np.mean(fz_total))
        mean_left_right = float(np.mean(fz_left) + np.mean(fz_right))
        relative_error = (
            abs(mean_left_right - mean_total) / abs(mean_total)
            if mean_total
            else float("inf")
        )
        checks["fz_left_plus_right_matches_total"] = {
            "ok": relative_error < _MAGNITUDE_AGREEMENT_TOL,
            "detail": (
                f"mean(FzL+FzR)={mean_left_right:.1f} N vs "
                f"mean(FzTotal)={mean_total:.1f} N "
                f"(relative error {relative_error * 100:.1f}%)"
            ),
            "mean_fz_total": mean_total,
            "mean_fz_left_plus_right": mean_left_right,
            "relative_error": relative_error,
        }
    else:
        checks["fz_left_plus_right_matches_total"] = {
            "ok": False,
            "detail": "skipped — needs both Type I and Type II",
        }

    if type_ii_received and fz_left.size:
        right_single = (fz_right > _SINGLE_STANCE_MIN_N) & (
            fz_right > _SINGLE_STANCE_RATIO * fz_left
        )
        left_single = (fz_left > _SINGLE_STANCE_MIN_N) & (
            fz_left > _SINGLE_STANCE_RATIO * fz_right
        )
        double = (fz_left > _DOUBLE_STANCE_MIN_N) & (fz_right > _DOUBLE_STANCE_MIN_N)
        total = int(fz_left.size)
        checks["right_single_stance_detected"] = {
            "ok": bool(right_single.sum()),
            "detail": f"{int(right_single.sum())}/{total} samples (FzR dominant)",
        }
        checks["left_single_stance_detected"] = {
            "ok": bool(left_single.sum()),
            "detail": f"{int(left_single.sum())}/{total} samples (FzL dominant)",
        }
        checks["double_stance_detected"] = {
            "ok": bool(double.sum()),
            "detail": f"{int(double.sum())}/{total} samples (FzL>0 and FzR>0)",
        }
    else:
        for name in (
            "right_single_stance_detected",
            "left_single_stance_detected",
            "double_stance_detected",
        ):
            checks[name] = {"ok": False, "detail": "skipped — no Type II samples"}

    return checks


def run_gaitway_test(
    *,
    host: str = "127.0.0.1",
    port: int = GAITWAY_DEFAULT_PORT,
    sample_rate_hz: int = 1000,
    trigger_mode: int = 0,
    sync_out_enabled: bool = False,
    type_i_mode: int = 2,
    type_ii_mode: int = 2,
    duration_s: float = 15.0,
    connect_timeout_s: float = 5.0,
    socket_timeout_s: float = 0.2,
    out_dir: Path | str = ".",
) -> GaitwayTestReport:
    """Connect to the gaitway server, collect Type I + Type II, and report.

    Never raises for a *protocol* problem (missing Type II, NAK, dropped
    connection) — those become structured ``errors``/``checks`` in the returned
    report so a field operator sees the diagnosis.  Only programming errors
    (e.g. bad port) propagate.
    """
    out_path = Path(out_dir)
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    command = build_start_ds_command(
        sample_rate_hz=sample_rate_hz,
        trigger_mode=trigger_mode,
        sync_out_enabled=sync_out_enabled,
        type_i_mode=type_i_mode,
        type_ii_mode=type_ii_mode,
        seconds=0,
    )

    errors: list[str] = []
    server_summary = "not connected"
    settings_version: int | None = None
    settings_packet_hex: str | None = None

    fz_total: list[float] = []
    fz_left: list[float] = []
    fz_right: list[float] = []
    fz_total_t_ns: list[int] = []
    fz_lr_t_ns: list[int] = []
    type_i_packets = 0
    type_ii_packets = 0

    sock: socket.socket | None = None
    duration_actual_s = 0.0
    deadline = time.perf_counter_ns() + round(duration_s * 1_000_000_000)
    try:
        sock = socket.create_connection((host, port), timeout=connect_timeout_s)
        sock.settimeout(socket_timeout_s)
        framer = GaitwayPacketFramer()
        pending: list[bytes] = []

        def _send(text: str) -> None:
            assert sock is not None
            sock.sendall(text.encode("ascii") + b"\r\n")

        def _read_packet() -> bytes | None:
            nonlocal pending
            assert sock is not None
            while True:
                if pending:
                    return pending.pop(0)
                try:
                    chunk = sock.recv(65_535)
                except socket.timeout:
                    return None
                if not chunk:
                    raise ConnectionError("gaitway server closed the connection")
                packets = framer.feed(chunk)
                if packets:
                    pending.extend(packets[1:])
                    return packets[0]

        _send("getDSsettings")
        # Drain until the settings packet arrives (ACK may precede it).
        got_settings = False
        settings_deadline = time.perf_counter_ns() + round(
            connect_timeout_s * 1_000_000_000
        )
        while time.perf_counter_ns() < settings_deadline and not got_settings:
            packet = _read_packet()
            if packet is None:
                continue
            ptype = int(unpack_from("<H", packet, 2)[0])
            if ptype == PACKET_SETTINGS:
                settings_packet_hex = packet.hex()
                if len(packet) >= 6:
                    settings_version = int(unpack_from("<H", packet, 4)[0])
                got_settings = True
                server_summary = "getDSsettings ACK + settings received"
            elif ptype == PACKET_NAK:
                server_summary = "getDSsettings rejected (NAK)"
                raise GaitwayPacketError("gaitway rejected getDSsettings")
        if not got_settings:
            server_summary = "getDSsettings settings packet not received (continuing)"

        _send(command)
        start_ns = time.perf_counter_ns()

        while time.perf_counter_ns() < deadline:
            packet = _read_packet()
            if packet is None:
                continue
            ptype = int(unpack_from("<H", packet, 2)[0])
            received_ns = time.perf_counter_ns()
            if ptype == PACKET_TYPE_I:
                try:
                    header, data = parse_type_i_packet(packet)
                except GaitwayPacketError:
                    errors.append("malformed Type I packet skipped")
                    continue
                type_i_packets += 1
                count = data.shape[0]
                first_ns = max(
                    0,
                    received_ns
                    - round((count - 1) * 1_000_000_000 / sample_rate_hz),
                )
                for index in range(count):
                    fz_total.append(float(data[index, _FZ_TOTAL]))
                    fz_total_t_ns.append(
                        first_ns + round(index * 1_000_000_000 / sample_rate_hz)
                    )
            elif ptype == PACKET_TYPE_II:
                try:
                    _header, data = parse_type_ii_packet(packet)
                except GaitwayPacketError:
                    errors.append("malformed Type II packet skipped")
                    continue
                type_ii_packets += 1
                for index in range(data.shape[0]):
                    fz_left.append(float(data[index, _FZ_LEFT]))
                    fz_right.append(float(data[index, _FZ_RIGHT]))
                    fz_lr_t_ns.append(received_ns)
            elif ptype == PACKET_NAK:
                errors.append(f"gaitway rejected command (NAK) during collection")
            # ACK/settings packets during streaming are ignored for the report.

        duration_actual_s = (time.perf_counter_ns() - start_ns) / 1_000_000_000
        server_summary = f"collected {duration_actual_s:.1f}s"

        try:
            _send("stopDS")
            stop_deadline = time.perf_counter_ns() + round(5.0 * 1_000_000_000)
            while time.perf_counter_ns() < stop_deadline:
                packet = _read_packet()
                if packet is not None and int(unpack_from("<H", packet, 2)[0]) == PACKET_ACK:
                    server_summary += " + stopDS ACK"
                    break
        except (OSError, ConnectionError):
            errors.append("stopDS not acknowledged")

    except (OSError, ConnectionError) as exc:
        errors.append(f"connection error: {type(exc).__name__}: {exc}")
    except GaitwayPacketError as exc:
        errors.append(f"protocol error: {exc}")
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    fz_total_arr = np.asarray(fz_total, dtype=np.float64)
    fz_left_arr = np.asarray(fz_left, dtype=np.float64)
    fz_right_arr = np.asarray(fz_right, dtype=np.float64)
    type_i_received = type_i_packets > 0
    type_ii_received = type_ii_packets > 0

    checks = compute_checks(
        fz_total=fz_total_arr,
        fz_left=fz_left_arr,
        fz_right=fz_right_arr,
        type_i_received=type_i_received,
        type_ii_received=type_ii_received,
    )

    fz_total_min, fz_total_max = _range_of(fz_total_arr)
    fz_left_min, fz_left_max = _range_of(fz_left_arr)
    fz_right_min, fz_right_max = _range_of(fz_right_arr)

    report = GaitwayTestReport(
        ok=bool(not errors and type_i_received and type_ii_received),
        host=host,
        port=port,
        sample_rate_hz=sample_rate_hz,
        duration_requested_s=duration_s,
        duration_actual_s=round(duration_actual_s, 3),
        started_at_utc=started_at_utc,
        software_version=__version__,
        grf_source_type="gaitway_single_platform_decomposed_left_right",
        sent_command=command,
        server_response_summary=server_summary,
        gaitway_settings_version=settings_version,
        settings_packet_hex=settings_packet_hex,
        type_i_received=type_i_received,
        type_ii_received=type_ii_received,
        type_i_packet_count=type_i_packets,
        type_ii_packet_count=type_ii_packets,
        type_i_sample_count=int(fz_total_arr.size),
        type_ii_sample_count=int(fz_left_arr.size),
        fz_total_min=fz_total_min,
        fz_total_max=fz_total_max,
        fz_left_min=fz_left_min,
        fz_left_max=fz_left_max,
        fz_right_min=fz_right_min,
        fz_right_max=fz_right_max,
        checks=checks,
        errors=errors,
    )

    # Best-effort plot; a plotting failure must not hide the JSON report.
    try:
        _write_plot(
            out_path / DEFAULT_PLOT_NAME,
            report=report,
            fz_total=fz_total_arr,
            fz_total_t_ns=np.asarray(fz_total_t_ns, dtype=np.int64),
            fz_left=fz_left_arr,
            fz_right=fz_right_arr,
            fz_lr_t_ns=np.asarray(fz_lr_t_ns, dtype=np.int64),
        )
    except Exception as exc:  # pragma: no cover - matplotlib/env specific
        errors.append(f"plot write failed: {type(exc).__name__}: {exc}")

    report.write(out_path / DEFAULT_REPORT_NAME)
    return report


def _write_plot(
    path: Path,
    *,
    report: GaitwayTestReport,
    fz_total: np.ndarray,
    fz_total_t_ns: np.ndarray,
    fz_left: np.ndarray,
    fz_right: np.ndarray,
    fz_lr_t_ns: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _seconds(t_ns: np.ndarray) -> np.ndarray:
        if t_ns.size == 0:
            return np.asarray([], dtype=np.float64)
        return (t_ns - t_ns.min()).astype(np.float64) / 1_000_000_000

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.suptitle(
        f"gaitway-3D self-check — {report.host}:{report.port} "
        f"({report.sample_rate_hz} Hz, {'PASS' if report.ok else 'CHECK'})"
    )

    ax0 = axes[0]
    ax0.plot(_seconds(fz_total_t_ns), fz_total, color="#0f766e", lw=1.0, label="Fz total (Type I)")
    ax0.set_ylabel("Fz total [N]")
    ax0.grid(alpha=0.3)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    ax1.plot(_seconds(fz_lr_t_ns), fz_left, color="#2563eb", lw=1.0, label="Fz left (Type II)")
    ax1.plot(_seconds(fz_lr_t_ns), fz_right, color="#dc2626", lw=1.0, label="Fz right (Type II)")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("Fz [N]")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: ``python -m exo_collection.adapters.force_plate.gaitway_test``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="gaitway-3D field self-check (Type I + Type II, ~10-20 s)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=GAITWAY_DEFAULT_PORT)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--trigger", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--sync-out", action="store_true")
    parser.add_argument("--type-i", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--type-ii", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args(argv)

    report = run_gaitway_test(
        host=args.host,
        port=args.port,
        sample_rate_hz=args.sample_rate,
        trigger_mode=args.trigger,
        sync_out_enabled=args.sync_out,
        type_i_mode=args.type_i,
        type_ii_mode=args.type_ii,
        duration_s=args.seconds,
        out_dir=args.out_dir,
    )

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nreport: {Path(args.out_dir) / DEFAULT_REPORT_NAME}")
    print(f"plot:   {Path(args.out_dir) / DEFAULT_PLOT_NAME}")
    return 0 if report.ok else 1


__all__ = [
    "GaitwayTestReport",
    "compute_checks",
    "run_gaitway_test",
    "main",
    "DEFAULT_REPORT_NAME",
    "DEFAULT_PLOT_NAME",
]
