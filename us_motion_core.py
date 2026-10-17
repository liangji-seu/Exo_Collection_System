"""Read-only ultrasound loading and fixed traditional NCC motion diagnostics.

Coordinates are native ADC points; positive direction means deeper/right.
No learning, index creation, interpolation in time, or input modification occurs.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import struct
import time
import zlib

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass
class MotionSession:
    path: Path
    times: np.ndarray
    profiles: np.ndarray
    valid: np.ndarray
    depth: np.ndarray
    channels: tuple[int, ...]
    metadata: dict
    raw_frames: np.ndarray | None = None
    raw_depth: np.ndarray | None = None


class LoadCancelled(RuntimeError):
    """The caller cancelled a read-only load."""


HEADER = struct.Struct('<8sHHQQQQqQQII')
DEPTH = np.arange(20, 977)
RAW_DEPTH = np.arange(997)


def load_session(path, progress=None, cancel=None):
    """Scan every block, CRC-check, then align each physical channel causally.

    Supported format: EXOUSBLK v1, one captured uint8[1000] packet per
    block. Unsupported layouts fail explicitly; existing indexes are not used.
    """
    started = time.perf_counter()
    path = Path(path).resolve()
    binary = path / 'ultrasound.bin' if path.is_dir() else path
    folder = binary.parent
    def report(p, text):
        if cancel and cancel():
            raise LoadCancelled('已取消加载')
        if progress:
            progress(int(p), text)
    report(0, '读取并校验超声数据')
    meta_path = binary.with_suffix('.meta.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
        if meta.get('dtype') != 'uint8' or meta.get('sample_shape') != [1000]:
            raise ValueError('不支持的超声格式：需要 uint8[1000] 完整采集包')
    records = {}
    size = binary.stat().st_size
    previous_sequence = -1
    minimum_sample = 0
    count = 0
    with binary.open('rb') as stream:
        while stream.tell() < size:
            offset = stream.tell()
            raw = stream.read(HEADER.size)
            if len(raw) != HEADER.size:
                raise ValueError(f'超声块头截断，偏移 {offset}')
            h = HEADER.unpack(raw)
            if h[:3] != (b'EXOUSBLK', 1, HEADER.size) or h[5:7] != (1, 1000):
                raise ValueError(f'不支持的超声块格式，偏移 {offset}')
            if h[3] <= previous_sequence or h[4] < minimum_sample:
                raise ValueError(f'超声块序号或采样索引倒退，偏移 {offset}')
            previous_sequence, minimum_sample = h[3], h[4] + h[5]
            packet = stream.read(1000)
            if len(packet) != 1000:
                raise ValueError(f'超声包截断，偏移 {offset}')
            if zlib.crc32(packet) & 0xffffffff != h[11]:
                raise ValueError(f'超声 CRC 校验失败，偏移 {offset}')
            if packet[0] != 0 or packet[-1] != 255 or not 1 <= packet[1] <= 4:
                raise ValueError(f'无法识别完整超声采集包，偏移 {offset}')
            channel = packet[1]
            rows = records.setdefault(channel, [])
            if rows and h[8] < rows[-1][0]:
                raise ValueError(f'通道 {channel} 主机时间倒退')
            rows.append((h[8], packet[2:-1]))
            count += 1
            if count % 256 == 0:
                report(65 * stream.tell() / max(size, 1), '读取并校验超声数据')
    if not records:
        raise ValueError('超声文件为空')
    manifest_path = next((p for p in (folder / '.exo/manifest.json', folder / 'manifest.json') if p.exists()), None)
    timing = json.loads(manifest_path.read_text(encoding='utf-8-sig')).get('timing', {}) if manifest_path else {}
    start = timing.get('start_host_monotonic_ns')
    fallback = start is None
    start = min(rows[0][0] for rows in records.values()) if fallback else int(start)
    stop = int(timing.get('stop_host_monotonic_ns') or max(rows[-1][0] for rows in records.values()))
    if stop < start:
        raise ValueError('会话结束时间早于起点')
    # Integer nanoseconds avoid accidentally selecting future packets at grid boundaries.
    query_ns = np.arange(start, stop + 1, 50_000_000, dtype=np.int64)
    times = (query_ns - start).astype(np.float64) / 1e9
    channels = tuple(sorted(records))
    profiles = np.full((len(times), len(channels), 957), np.nan, dtype=np.float32)
    # Keep the captured ADC values for display, separate from matching features.
    # The other three bytes of each 1000-byte packet are framing/channel metadata.
    raw_frames = np.full((len(times), len(channels), 997), np.nan, dtype=np.float32)
    valid = np.zeros((len(times), len(channels)), dtype=bool)
    source_times = np.full(valid.shape, np.nan)
    gap_segment = np.full(valid.shape, -1, dtype=np.int64)
    for c, channel in enumerate(channels):
        rows = records[channel]
        stamps = np.array([r[0] for r in rows], dtype=np.int64)
        indices = np.searchsorted(stamps, query_ns, side='right') - 1
        safe = np.maximum(indices, 0)
        good = (indices >= 0) & ((query_ns - stamps[safe]) <= 75_000_000)
        segments = np.cumsum(np.r_[False, np.diff(stamps) > 75_000_000])
        valid[:, c] = good
        source_times[good, c] = (stamps[indices[good]] - start) / 1e9
        gap_segment[good, c] = segments[indices[good]]
        selected = np.flatnonzero(good)
        for first in range(0, len(selected), 256):
            ii = selected[first:first + 256]
            adc = np.stack([np.frombuffer(rows[indices[i]][1], dtype=np.uint8) for i in ii]).astype(np.float32)
            raw_frames[ii, c] = adc
            env = uniform_filter1d(np.abs(adc - np.median(adc, axis=1, keepdims=True)), 20, axis=1, mode='nearest')
            profiles[ii, c] = np.log1p(env)[:, 20:977]
            report(65 + 34 * (c + (first + len(ii)) / max(len(selected), 1)) / len(channels), '按主机时间对齐和计算包络')
    metadata = dict(start_host_monotonic_ns=start, stop_host_monotonic_ns=stop,
                    origin_source='首条 host 时间（缺少 manifest 起点）' if fallback else str(manifest_path),
                    origin_fallback=fallback, source_times=source_times, gap_segment=gap_segment,
                    packet_count=count, binary_path=str(binary), load_seconds=time.perf_counter() - started,
                    alignment='20 Hz previous-only; age <= 75 ms', warnings=['缺少 manifest 起点，使用首条 host 时间'] if fallback else [])
    report(100, '加载完成')
    return MotionSession(folder, times, profiles, valid, DEPTH.copy(), channels, metadata,
                         raw_frames=raw_frames, raw_depth=RAW_DEPTH.copy())


def _standardize(a):
    q = np.quantile(a, [.25, .5, .75]).astype(np.float32)
    return (a - q[1]) / max(q[2] - q[0], 1e-6)


def _mean(a):
    # scipy mirror equals torch reflect padding (edge sample is not repeated).
    return uniform_filter1d(a, 49, axis=-1, mode='mirror')


def _moments(a):
    mean = _mean(a)
    return mean, np.maximum(_mean(a * a) - mean * mean, 0)


def _ncc(a, b):
    ma, va = _moments(a)
    mb, vb = _moments(b)
    return (_mean(a * b) - ma * mb) / np.sqrt(va * vb + 1e-5)


def _displacement(a, b):
    shifted = np.lib.stride_tricks.sliding_window_view(np.pad(b, (24, 24), mode='reflect'), len(b))
    cost = np.clip(_ncc(a, shifted), -1, 1)
    return uniform_filter1d((np.argmax(cost, axis=0) - 24).astype(np.float32), 5, mode='nearest')


def _warp(a, displacement):
    return np.interp(np.arange(957) + displacement, np.arange(957), a).astype(np.float32)


def _blocks(direction, screen):
    blocks = []
    for start in range(86, 870, 49):
        stop = start + 49
        values = direction[start:stop][screen[start:stop]]
        label = 'unknown'
        if len(values) / 49 >= .70:
            if np.mean(values <= -1) >= .80:
                label = 'left'
            elif np.mean(values >= 1) >= .80:
                label = 'right'
            elif np.mean(np.abs(values) < 1) > .50:
                label = 'near_zero'
        blocks.append(dict(start_index=start, stop_index=stop, depth_start=int(DEPTH[start]),
                           depth_stop=int(DEPTH[stop - 1]), label=label,
                           median_direction=float(np.median(values)) if len(values) else float('nan')))
    return blocks


def estimate_pair(current, past):
    started = time.perf_counter()
    current, past = np.asarray(current, dtype=np.float32), np.asarray(past, dtype=np.float32)
    if current.shape != (957,) or past.shape != (957,) or not np.isfinite(current).all() or not np.isfinite(past).all():
        raise ValueError('NCC 输入必须是有限的 957 点轮廓')
    a, b = _standardize(current), _standardize(past)
    v, u = _displacement(a, b), _displacement(b, a)
    match = _ncc(a, _warp(b, v))
    reverse = _warp(_ncc(b, _warp(a, u)), v)
    cycle = v + _warp(u, v)
    _, variance = _moments(a)
    global_var = np.var(a)
    texture = np.sqrt(variance + 1e-8) / np.sqrt(global_var + 1e-8)
    support = np.zeros(957, dtype=bool)
    support[72:885] = True
    screen = support & (global_var >= 1e-6) & (texture >= .15) & (match >= .70) & (reverse >= .70) & (np.abs(cycle) <= 2)
    direction = -v
    return dict(direction=direction, screen_pass=screen, support=support,
                blocks=_blocks(direction, screen), compute_ms=(time.perf_counter() - started) * 1000,
                match_ncc=match, reverse_match_ncc=reverse, cycle_residual=cycle, texture_ratio=texture)


def estimate_frame(session, index, channel, lag_seconds=.20):
    if channel not in session.channels:
        raise ValueError(f'不存在物理通道 {channel}')
    if not 0 <= index < len(session.times):
        raise IndexError('帧索引越界')
    if not np.isfinite(lag_seconds) or lag_seconds <= 0:
        raise ValueError('历史间隔必须为正数')
    c = session.channels.index(channel)
    current_time = float(session.times[index])
    target = current_time - lag_seconds
    j = int(np.searchsorted(session.times, target + 1e-10, side='right')) - 1
    reason = ''
    if j < 0:
        reason = '没有足够历史帧'
    elif target - session.times[j] > .075 + 1e-10:
        reason = '历史帧已过期'
    elif not session.valid[j:index + 1, c].all():
        reason = '当前帧、历史帧或期间存在缺失/过期数据'
    elif np.any(np.diff(session.times[j:index + 1]) > .075 + 1e-10):
        reason = '历史区间跨越时间缺口'
    elif 'gap_segment' in session.metadata and session.metadata['gap_segment'][j, c] != session.metadata['gap_segment'][index, c]:
        reason = '历史区间跨越原始通道采样缺口'
    current = session.profiles[index, c]
    past = session.profiles[j, c] if j >= 0 else None
    if not reason:
        result = estimate_pair(current, past)
    else:
        direction = np.full(957, np.nan, dtype=np.float32)
        support = np.zeros(957, dtype=bool)
        support[72:885] = True
        screen = np.zeros(957, dtype=bool)
        result = dict(direction=direction, screen_pass=screen, support=support, blocks=_blocks(direction, screen), compute_ms=0.)
    result.update(current=current, past=past, depth=session.depth, current_time=current_time,
                  past_time=float(session.times[j]) if j >= 0 else None, valid_pair=not bool(reason), reason=reason or '有效历史对')
    result.update(raw_current=session.raw_frames[index, c] if session.raw_frames is not None else None,
                  raw_depth=session.raw_depth)
    return result
