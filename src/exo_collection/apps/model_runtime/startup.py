"""Cancellable explicit Run intent while loading/connecting/warming inputs."""
from enum import Enum
from time import monotonic


class StartupPhase(Enum):
    IDLE = "idle"
    LOADING = "loading"
    CONNECTING = "connecting"
    WARMING = "warming"
    STARTED = "started"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StartupCoordinator:
    def __init__(self, *, clock=monotonic):
        self._clock = clock
        self.generation = 0
        self.phase = StartupPhase.IDLE
        self.active = False
        self.deadline = None
        self.required = ()
        self.attempted = set()
        self.reason = ""

    def begin(self):
        if not self.active:
            self.generation += 1
            self.active = True
            self.phase = StartupPhase.IDLE
            self.deadline = None
            self.required = ()
            self.attempted = set()
            self.reason = ""
        return self.generation

    def matches(self, generation):
        return self.active and generation == self.generation

    def wait(self, generation, phase, *, timeout_s):
        if not self.matches(generation):
            return False
        if phase is not self.phase:
            self.phase = phase
            self.deadline = self._clock() + timeout_s
        return True

    @property
    def expired(self):
        return self.active and self.deadline is not None and self._clock() >= self.deadline

    def cancel(self, reason):
        if self.active:
            self.generation += 1
            self.active = False
            self.phase = StartupPhase.CANCELLED
            self.reason = reason
            self.deadline = None

    def fail(self, reason):
        self.cancel(reason)
        self.phase = StartupPhase.FAILED
        self.reason = reason

    def finish(self, generation):
        if not self.matches(generation):
            return False
        self.active = False
        self.phase = StartupPhase.STARTED
        self.deadline = None
        return True
