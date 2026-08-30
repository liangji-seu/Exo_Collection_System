"""Global low-level keyboard hook that turns the LinTx button into marker events.

The LinTx button is a USB HID keyboard that types the comma key (vk=0xBC,
``VK_OEM_COMMA``).  This module listens for that key globally via a
``WH_KEYBOARD_LL`` hook so presses are captured even when the Collector window
does not have keyboard focus.

The hook callback fires on the hook's dedicated thread and therefore must never
touch Qt.  It only enqueues ``(host_monotonic_ns, host_utc_ns)`` timestamps into
a thread-safe queue; the Collector's main thread drains them with a ``QTimer``.
"""

from __future__ import annotations

import ctypes
import queue
import time
import threading
from ctypes import wintypes

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_KEYUP = 0x0101
WM_SYSKEYUP = 0x0105
BUTTON_VK = 0xBC  # VK_OEM_COMMA — the LinTx button's comma key
START_STOP_VK = 0xBE  # VK_OEM_PERIOD — the start/stop toggle button's period key
VK_SHIFT = 0x10


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


_HookProc = ctypes.WINFUNCTYPE(
    ctypes.c_longlong, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
)

_user32 = ctypes.windll.user32
_user32.SetWindowsHookExW.restype = ctypes.c_void_p
_user32.SetWindowsHookExW.argtypes = (
    ctypes.c_int,
    _HookProc,
    wintypes.HINSTANCE,
    wintypes.DWORD,
)
_user32.CallNextHookEx.restype = ctypes.c_longlong
_user32.CallNextHookEx.argtypes = (
    ctypes.c_void_p,
    ctypes.c_int,
    wintypes.WPARAM,
    wintypes.LPARAM,
)
_user32.UnhookWindowsHookEx.restype = wintypes.BOOL
_user32.UnhookWindowsHookEx.argtypes = (ctypes.c_void_p,)
_user32.GetAsyncKeyState.restype = ctypes.c_short
_user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)


class ButtonMarkerListener:
    """Thread-safe listener delivering one timestamp pair per comma key-down."""

    def __init__(
        self,
        *,
        vk: int = BUTTON_VK,
        queue_size: int = 256,
        ignore_shift: bool = False,
    ) -> None:
        self._vk = vk
        self._ignore_shift = ignore_shift
        self._queue: queue.Queue[tuple[int, int]] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._hook: int | None = None
        self._proc: _HookProc | None = None
        self._down = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="button-marker-hook",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def drain(self) -> list[tuple[int, int]]:
        """Pop every pending ``(host_monotonic_ns, host_utc_ns)`` press."""
        events: list[tuple[int, int]] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _run(self) -> None:
        self._proc = _HookProc(self._on_event)
        self._hook = _user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, None, 0)
        if not self._hook:
            return
        msg = wintypes.MSG()
        try:
            while not self._stop.is_set():
                while _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE
                    _user32.TranslateMessage(ctypes.byref(msg))
                    _user32.DispatchMessageW(ctypes.byref(msg))
                time.sleep(0.01)
        finally:
            if self._hook:
                _user32.UnhookWindowsHookEx(self._hook)
                self._hook = None

    def _on_event(self, nCode, wParam, lParam):
        if nCode >= 0:
            if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                kb = ctypes.cast(lParam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                if kb.vkCode == self._vk and not self._down:
                    if self._ignore_shift and (
                        _user32.GetAsyncKeyState(VK_SHIFT) & 0x8000
                    ):
                        # Shift held → this is the ">" operator label, not the
                        # bare "." toggle key.  Skip so it stays available to
                        # the Collector's prompt-label event filter.
                        return _user32.CallNextHookEx(None, nCode, wParam, lParam)
                    self._down = True
                    try:
                        self._queue.put_nowait((time.perf_counter_ns(), time.time_ns()))
                    except queue.Full:
                        pass
            elif wParam in (WM_KEYUP, WM_SYSKEYUP):
                kb = ctypes.cast(lParam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                if kb.vkCode == self._vk:
                    self._down = False
        return _user32.CallNextHookEx(None, nCode, wParam, lParam)


__all__ = ["ButtonMarkerListener", "BUTTON_VK", "START_STOP_VK"]
