"""Catalog engine setup and migration entry point."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, event
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session, sessionmaker


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
        migrations = Path(__file__).with_name("migrations")
        config = Config()
        config.set_main_option("script_location", str(migrations))
        config.set_main_option("sqlalchemy.url", self.engine.url.render_as_string(hide_password=False))
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

