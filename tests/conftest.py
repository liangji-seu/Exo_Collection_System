"""Keep online-window test sessions out of the operator's retained logs."""

import pytest


@pytest.fixture(autouse=True)
def isolated_online_window_logs(request, tmp_path, monkeypatch):
    if not request.node.path.name.startswith("test_model_runtime_"):
        return
    from exo_collection.apps.model_runtime import window

    diagnostics_type = window.Diagnostics
    session_type = window.SessionLog
    recording_type = window.OnlineRecording
    monkeypatch.setattr(window, "OnlineRecording",
        lambda _directory, **kwargs: recording_type(tmp_path / "online_recordings", **kwargs))
    monkeypatch.setattr(
        window, "Diagnostics",
        lambda _directory, **kwargs: diagnostics_type(tmp_path / "diagnostics", **kwargs),
    )
    monkeypatch.setattr(
        window, "SessionLog",
        lambda _directory, metadata: session_type(tmp_path / "sessions", metadata),
    )
