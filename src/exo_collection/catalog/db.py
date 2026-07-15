"""Catalog engine setup and migration entry point."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic, sleep
from typing import BinaryIO

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, event
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session, sessionmaker


_MIGRATION_LOCK_TIMEOUT_SECONDS = 30.0
_MIGRATION_LOCK_POLL_SECONDS = 0.05


class _MigrationFileLock:
    """Cross-process advisory lock stored next to one SQLite catalog.

    The lock file is intentionally persistent: the operating system releases
    its byte-range lock when a process exits, so a crashed migrator cannot
    leave a stale sentinel that blocks future application starts.
    """

    def __init__(self, path: Path, *, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._handle: BinaryIO | None = None

    @staticmethod
    def _try_lock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def __enter__(self) -> _MigrationFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()

        deadline = monotonic() + self.timeout_seconds
        while True:
            try:
                self._try_lock(handle)
                self._handle = handle
                return self
            except OSError as exc:
                if monotonic() >= deadline:
                    handle.close()
                    raise TimeoutError(
                        f"Timed out waiting for Catalog migration lock: {self.path}"
                    ) from exc
                sleep(_MIGRATION_LOCK_POLL_SECONDS)

    def __exit__(self, *_exc: object) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            self._unlock(handle)
        finally:
            handle.close()


class Catalog:
    """Owns a SQLite catalog configured for short concurrent desktop transactions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        url = URL.create("sqlite", database=str(self.path))
        self.engine = create_engine(url, future=True, pool_pre_ping=True)
        self._configure_sqlite(self.engine)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, class_=Session)

    @staticmethod
    def _configure_sqlite(engine: Engine) -> None:
        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(dbapi_connection: object, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    def migrate(self, revision: str = "head") -> None:
        lock_path = self.path.with_name(f"{self.path.name}.migrate.lock")
        with _MigrationFileLock(lock_path, timeout_seconds=_MIGRATION_LOCK_TIMEOUT_SECONDS):
            migrations = Path(__file__).with_name("migrations")
            config = Config()
            config.set_main_option("script_location", str(migrations))
            config.set_main_option(
                "sqlalchemy.url", self.engine.url.render_as_string(hide_password=False)
            )
            command.upgrade(config, revision)

    def session(self) -> Session:
        return self.session_factory()

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> Catalog:
        self.migrate()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
