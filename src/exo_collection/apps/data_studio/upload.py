"""Manual, offline SSH/SCP upload for finalized Trial packages.

Passwords and private-key passphrases exist only in the UI process and the
spawned upload worker. They are sent over a multiprocessing pipe *after* the
process starts; they are never put in a command line, configuration file, log,
Catalog, or Manifest. Remote shell commands are deliberately not used. SCP transfers
files while SFTP creates directories, reads remote files for SHA-256
verification, and atomically publishes the verified staging directory.
"""

from __future__ import annotations

import hashlib
import base64
import json
import multiprocessing
import os
import re
import stat
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from multiprocessing.connection import Connection
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import UUID, uuid4

from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import read_activity
from exo_collection.storage.layout import (
    name_has_storage_suffix,
    path_has_unpublished_component,
)
from exo_collection.storage.manifest import TrialManifest, load_manifest


_COPY_BUFFER_SIZE = 1024 * 1024
_SAFE_REMOTE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
UPLOAD_AUDIT_DIRECTORY = ".upload-audit"


def _is_safe_remote_segment(value: str) -> bool:
    """Return whether one path component is safe and non-navigational."""

    return (
        value not in {"", ".", ".."}
        and _SAFE_REMOTE_SEGMENT.fullmatch(value) is not None
    )


class UploadError(RuntimeError):
    """Expected upload failure whose message is safe to show in the UI."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class UploadCancelled(UploadError):
    def __init__(self, message: str = "上传已取消。") -> None:
        super().__init__("CANCELLED", message)


@dataclass(frozen=True, slots=True)
class HostKeyInfo:
    """Public SSH server identity presented for first-use confirmation."""

    lookup_hostname: str
    algorithm: str
    key_base64: str
    sha256_fingerprint: str


class UnknownHostKeyError(UploadError):
    def __init__(self, host_key: HostKeyInfo) -> None:
        super().__init__(
            "UNKNOWN_HOST_KEY",
            "这是首次连接该主机，必须由操作者核对并确认 SSH 主机指纹。",
        )
        self.host_key = host_key


class UploadPhase(StrEnum):
    VALIDATING = "VALIDATING"
    CONNECTING = "CONNECTING"
    UPLOADING = "UPLOADING"
    VERIFYING = "VERIFYING"
    PUBLISHING = "PUBLISHING"
    COMPLETED = "COMPLETED"


class AuthenticationMethod(StrEnum):
    PASSWORD = "PASSWORD"
    PRIVATE_KEY = "PRIVATE_KEY"


@dataclass(frozen=True, slots=True)
class OfflineUploadRequest:
    """One ephemeral upload request.

    ``password`` is excluded from repr/equality so accidental diagnostics do
    not reveal it.  Callers must not persist this object.
    """

    dataset_root: Path
    manifest_path: Path
    host: str
    port: int
    username: str
    remote_workdir: str
    password: str | None = field(default=None, repr=False, compare=False)
    private_key_path: Path | None = field(
        default=None, repr=False, compare=False
    )
    private_key_passphrase: str | None = field(
        default=None, repr=False, compare=False
    )
    transfer_batch_uuid: UUID = field(default_factory=uuid4)
    accepted_host_key: HostKeyInfo | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        root = Path(self.dataset_root).expanduser().resolve()
        manifest = Path(self.manifest_path).expanduser().resolve()
        if not manifest.is_relative_to(root):
            raise ValueError("Manifest 必须位于当前数据根目录内。")
        host = _require_text("host", self.host)
        username = _require_text("username", self.username)
        remote_workdir = validate_remote_directory(self.remote_workdir)
        password = str(self.password) if self.password is not None else None
        private_key_path = (
            Path(self.private_key_path).expanduser().resolve()
            if self.private_key_path is not None
            else None
        )
        passphrase = (
            str(self.private_key_passphrase)
            if self.private_key_passphrase is not None
            else None
        )
        if password and private_key_path is not None:
            raise ValueError("密码认证和 SSH 私钥认证只能选择一种。")
        if not password and private_key_path is None:
            raise ValueError("请提供密码或 SSH 私钥。")
        for label, secret in (("密码", password), ("私钥口令", passphrase)):
            if secret is not None and any(ord(character) < 32 for character in secret):
                raise ValueError(f"{label}不能包含控制字符。")
        if private_key_path is not None and not private_key_path.is_file():
            raise ValueError(f"SSH 私钥文件不存在：{private_key_path}")
        if private_key_path is None and passphrase is not None:
            raise ValueError("只有使用 SSH 私钥时才能输入私钥口令。")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("端口必须在 1–65535 之间。")
        object.__setattr__(self, "dataset_root", root)
        object.__setattr__(self, "manifest_path", manifest)
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", int(self.port))
        object.__setattr__(self, "username", username)
        object.__setattr__(self, "remote_workdir", remote_workdir)
        object.__setattr__(self, "password", password)
        object.__setattr__(self, "private_key_path", private_key_path)
        object.__setattr__(self, "private_key_passphrase", passphrase)

    @property
    def authentication_method(self) -> AuthenticationMethod:
        return (
            AuthenticationMethod.PRIVATE_KEY
            if self.private_key_path is not None
            else AuthenticationMethod.PASSWORD
        )

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(
            secret
            for secret in (self.password, self.private_key_passphrase)
            if secret
        )


@dataclass(frozen=True, slots=True)
class UploadProgress:
    phase: UploadPhase
    message: str
    completed_files: int = 0
    total_files: int = 0


@dataclass(frozen=True, slots=True)
class OfflineUploadResult:
    trial_uuid: UUID
    remote_trial_directory: str
    file_count: int
    total_bytes: int
    verified_at_utc_ns: int
    transfer_batch_uuid: UUID
    audit_record_path: Path


@dataclass(frozen=True, slots=True)
class TrialUploadFile:
    local_path: Path
    relative_path: PurePosixPath
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TrialUploadPlan:
    trial_uuid: UUID
    project_uuid: UUID
    project_code: str | None
    subject_uuid: UUID
    subject_code: str | None
    session_uuid: UUID
    trial_directory: Path
    files: tuple[TrialUploadFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


class RemoteUploadSession(Protocol):
    """Small seam used by the real Paramiko/SCP session and fake tests."""

    def ensure_directory(self, remote_path: str) -> None: ...

    def exists(self, remote_path: str) -> bool: ...

    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        *,
        progress: UploadByteProgress | None = None,
    ) -> None: ...

    def remote_sha256(
        self,
        remote_path: str,
        *,
        progress: UploadByteProgress | None = None,
    ) -> str: ...

    def rename(self, source: str, destination: str) -> None: ...

    def remove_file(self, remote_path: str) -> None: ...

    def remove_directory(self, remote_path: str) -> None: ...

    def close(self) -> None: ...


RemoteSessionFactory = Callable[[OfflineUploadRequest], RemoteUploadSession]
ProgressCallback = Callable[[UploadProgress], None]
CancelCheck = Callable[[], bool]
UploadByteProgress = Callable[[int, int], None]


def _require_text(field_name: str, value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空。")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} 不能包含控制字符。")
    return normalized


def validate_remote_directory(value: str) -> str:
    """Return a canonical absolute POSIX path safe for the SCP command path.

    SCP starts a fixed remote ``scp`` process internally.  Restricting every
    user-controlled path segment prevents shell metacharacters from reaching
    that implementation.  SFTP operations themselves do not invoke a shell.
    """

    raw = _require_text("remote directory", value).replace("\\", "/")
    if any(part in {".", ".."} for part in raw.split("/")):
        raise ValueError("远程目录不能包含 . 或 .. 路径段。")
    path = PurePosixPath(raw)
    if not path.is_absolute():
        raise ValueError("远程目录必须是以 / 开头的绝对路径。")
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("远程目录不能包含 . 或 .. 路径段。")
    if any(not _is_safe_remote_segment(part) for part in path.parts[1:]):
        raise ValueError(
            "远程目录仅允许字母、数字、斜杠、下划线、连字符和点。"
        )
    normalized = str(path)
    return normalized.rstrip("/") or "/"


def validate_finalized_trial(manifest_path: str | Path) -> TrialManifest:
    """Load a small Manifest and prove it identifies a published Trial."""

    path = Path(manifest_path).expanduser().resolve()
    if path.name != "manifest.json":
        raise UploadError("INVALID_SELECTION", "请选择 Trial 的 manifest.json。")
    if path_has_unpublished_component(path):
        raise UploadError("ACTIVE_TRIAL", "不能上传正在写入或未完成的 Trial。")
    try:
        manifest = load_manifest(path)
    except Exception as exc:
        raise UploadError("INVALID_MANIFEST", f"Manifest 无法验证：{_safe_exception(exc)}") from exc
    if manifest.state is not TrialState.FINALIZED:
        raise UploadError("NOT_FINALIZED", "仅允许上传已最终化（FINALIZED）的 Trial。")
    return manifest


def _iter_trial_paths(trial_directory: Path) -> Iterator[Path]:
    """Yield regular files without following symlinks or junction-like links."""

    for current_root, directory_names, file_names in os.walk(
        trial_directory, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        for name in tuple(directory_names):
            candidate = current / name
            if _is_link_or_reparse(candidate):
                raise UploadError("UNSAFE_PACKAGE", f"Trial 包含符号链接目录：{name}")
            if name_has_storage_suffix(name):
                raise UploadError("INCOMPLETE_PACKAGE", f"Trial 包含未完成目录：{name}")
        for name in file_names:
            candidate = current / name
            if _is_link_or_reparse(candidate):
                raise UploadError("UNSAFE_PACKAGE", f"Trial 包含符号链接文件：{name}")
            if name_has_storage_suffix(name):
                raise UploadError("INCOMPLETE_PACKAGE", f"Trial 包含未完成文件：{name}")
            if not candidate.is_file():
                raise UploadError("UNSAFE_PACKAGE", f"Trial 包含非普通文件：{name}")
            yield candidate


def _is_link_or_reparse(path: Path) -> bool:
    information = path.lstat()
    file_attributes = int(getattr(information, "st_file_attributes", 0))
    # FILE_ATTRIBUTE_REPARSE_POINT also catches Windows junctions that are not
    # reported by Path.is_symlink() on Python 3.11.
    return stat.S_ISLNK(information.st_mode) or bool(file_attributes & 0x400)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_upload_plan(manifest_path: str | Path) -> TrialUploadPlan:
    """Hash every package file and re-check Manifest Artifact integrity."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = validate_finalized_trial(path)
    trial_directory = path.parent
    if trial_directory.name == ".exo":
        trial_directory = trial_directory.parent
    files: list[TrialUploadFile] = []
    by_relative_path: dict[str, TrialUploadFile] = {}
    for local_path in sorted(_iter_trial_paths(trial_directory)):
        relative = PurePosixPath(local_path.relative_to(trial_directory).as_posix())
        if any(not _is_safe_remote_segment(part) for part in relative.parts):
            raise UploadError(
                "UNSAFE_FILENAME",
                f"Trial 文件名不适合安全 SCP 传输：{relative.as_posix()}",
            )
        item = TrialUploadFile(
            local_path=local_path,
            relative_path=relative,
            size_bytes=local_path.stat().st_size,
            sha256=_sha256_file(local_path),
        )
        files.append(item)
        by_relative_path[relative.as_posix()] = item

    if not files or ".exo/manifest.json" not in by_relative_path:
        raise UploadError("INCOMPLETE_PACKAGE", "Trial 包缺少 .exo/manifest.json。")
    for artifact in manifest.artifacts:
        local = by_relative_path.get(artifact.relative_path)
        if local is None:
            raise UploadError(
                "MISSING_ARTIFACT", f"Manifest 所列 Artifact 不存在：{artifact.relative_path}"
            )
        if local.size_bytes != artifact.size_bytes or local.sha256 != artifact.sha256:
            raise UploadError(
                "LOCAL_INTEGRITY_FAILED",
                f"Artifact 大小或 SHA-256 与 Manifest 不符：{artifact.relative_path}",
            )
    return TrialUploadPlan(
        trial_uuid=manifest.trial_uuid,
        project_uuid=manifest.project_uuid,
        project_code=manifest.project_code,
        subject_uuid=manifest.subject_uuid,
        subject_code=manifest.subject_code,
        session_uuid=manifest.session_uuid,
        trial_directory=trial_directory,
        files=tuple(files),
    )


def build_remote_trial_directory(
    remote_workdir: str, plan: TrialUploadPlan
) -> str:
    """Build the architecture-defined remote hierarchy from Manifest IDs."""

    project_segment = _safe_identity(plan.project_code, str(plan.project_uuid))
    subject_segment = _safe_identity(plan.subject_code, str(plan.subject_uuid))
    return _remote_join(
        validate_remote_directory(remote_workdir),
        project_segment,
        subject_segment,
        str(plan.session_uuid),
        "trials",
        str(plan.trial_uuid),
    )


def _safe_identity(preferred: str | None, fallback: str) -> str:
    candidate = str(preferred).strip() if preferred is not None else ""
    if _is_safe_remote_segment(candidate):
        return candidate
    if not _is_safe_remote_segment(fallback):
        raise UploadError("UNSAFE_REMOTE_PATH", "远程路径包含不安全标识。")
    return fallback


def _write_upload_audit(
    request: OfflineUploadRequest,
    plan: TrialUploadPlan,
    *,
    status: str,
    started_at_utc_ns: int,
    completed_at_utc_ns: int,
    remote_trial_directory: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> Path:
    """Atomically persist a credential-free transfer and file audit record."""

    audit_path = (
        request.dataset_root
        / UPLOAD_AUDIT_DIRECTORY
        / str(plan.trial_uuid)
        / f"{request.transfer_batch_uuid}.json"
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "transfer_batch_uuid": str(request.transfer_batch_uuid),
        "trial_uuid": str(plan.trial_uuid),
        "status": status,
        "started_at_utc_ns": started_at_utc_ns,
        "completed_at_utc_ns": completed_at_utc_ns,
        "remote": {
            "host": request.host,
            "port": request.port,
            "username": request.username,
            "authentication_method": request.authentication_method.value,
            "trial_directory": remote_trial_directory,
        },
        "files": [
            {
                "relative_path": item.relative_path.as_posix(),
                "size_bytes": item.size_bytes,
                "local_sha256": item.sha256,
                "remote_sha256": item.sha256 if status == "VERIFIED" else None,
            }
            for item in plan.files
        ],
        "error": (
            {"code": error_code, "message": error_message}
            if error_code is not None
            else None
        ),
    }
    # Password is intentionally absent by construction. This assertion guards
    # future edits from accidentally adding a credential field. Exception text
    # is separately redacted before it reaches this function.
    def credential_key_present(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).casefold() in {"password", "passphrase"}
                or credential_key_present(child)
                for key, child in value.items()
            )
        if isinstance(value, list):
            return any(credential_key_present(child) for child in value)
        return False

    if credential_key_present(payload):
        raise UploadError("AUDIT_SECRET_GUARD", "凭据安全检查阻止了审计记录写入。")
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_name(f".{audit_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, audit_path)
    finally:
        temporary.unlink(missing_ok=True)
    return audit_path


class ParamikoScpSession:
    """A connected Paramiko/SCP session; it never executes remote commands."""

    def __init__(self, request: OfflineUploadRequest) -> None:
        try:
            import paramiko
            from scp import SCPClient
        except ImportError as exc:  # pragma: no cover - dependency installation failure
            raise UploadError(
                "DEPENDENCY_MISSING", "缺少 Paramiko/scp，请重新执行首次构建脚本。"
            ) from exc

        client = paramiko.SSHClient()
        self._client = client
        self._sftp: Any | None = None
        self._scp: Any | None = None
        self._upload_progress: UploadByteProgress | None = None
        try:
            client.load_system_host_keys()
            system_known_hosts = Path.home() / ".ssh" / "known_hosts"
            if system_known_hosts.is_file():
                client.load_system_host_keys(str(system_known_hosts))
            application_known_hosts = (
                Path.home() / ".exo_collection_system" / "known_hosts"
            )
            if application_known_hosts.is_file():
                client.load_host_keys(str(application_known_hosts))

            class ConfirmFirstUsePolicy(paramiko.MissingHostKeyPolicy):
                def missing_host_key(
                    self, policy_client: Any, hostname: str, key: Any
                ) -> None:
                    fingerprint = "SHA256:" + base64.b64encode(
                        hashlib.sha256(key.asbytes()).digest()
                    ).decode("ascii").rstrip("=")
                    host_key = HostKeyInfo(
                        lookup_hostname=hostname,
                        algorithm=key.get_name(),
                        key_base64=key.get_base64(),
                        sha256_fingerprint=fingerprint,
                    )
                    accepted = request.accepted_host_key
                    if accepted != host_key:
                        raise UnknownHostKeyError(host_key)
                    application_known_hosts.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    policy_client.get_host_keys().add(
                        hostname, key.get_name(), key
                    )
                    policy_client.save_host_keys(str(application_known_hosts))

            # First use requires an explicit round-trip to the UI. Once saved,
            # Paramiko's normal host-key lookup enforces the same key on every
            # later connection and rejects key changes.
            client.set_missing_host_key_policy(ConfirmFirstUsePolicy())
            authentication: dict[str, Any]
            if request.private_key_path is not None:
                authentication = {
                    "key_filename": str(request.private_key_path),
                    "passphrase": request.private_key_passphrase,
                }
            else:
                authentication = {"password": request.password}
            client.connect(
                hostname=request.host,
                port=request.port,
                username=request.username,
                look_for_keys=False,
                allow_agent=False,
                timeout=15.0,
                banner_timeout=15.0,
                auth_timeout=15.0,
                **authentication,
            )
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise UploadError("CONNECTION_FAILED", "SSH 连接未建立。")
            self._sftp = client.open_sftp()
            def scp_progress(
                _filename: bytes,
                total_bytes: int,
                transferred_bytes: int,
                _peername: object,
            ) -> None:
                callback = self._upload_progress
                if callback is not None:
                    callback(int(transferred_bytes), int(total_bytes))

            self._scp = SCPClient(
                transport,
                socket_timeout=30.0,
                progress4=scp_progress,
            )
        except UnknownHostKeyError:
            self.close()
            raise
        except paramiko.BadHostKeyException as exc:
            self.close()
            raise UploadError(
                "HOST_KEY_MISMATCH",
                "SSH 主机密钥与已保存指纹不一致，已拒绝连接。请联系服务器管理员核查。",
            ) from exc
        except paramiko.AuthenticationException as exc:
            self.close()
            raise UploadError(
                "AUTHENTICATION_FAILED",
                "SSH 认证失败，请检查用户名、密码或私钥/口令。",
            ) from exc
        except BaseException:
            self.close()
            raise

    @property
    def sftp(self) -> Any:
        if self._sftp is None:
            raise RuntimeError("SFTP session is closed")
        return self._sftp

    @property
    def scp(self) -> Any:
        if self._scp is None:
            raise RuntimeError("SCP session is closed")
        return self._scp

    def exists(self, remote_path: str) -> bool:
        try:
            self.sftp.stat(remote_path)
        except OSError as exc:
            if isinstance(exc, FileNotFoundError) or getattr(exc, "errno", None) == 2:
                return False
            raise
        return True

    def ensure_directory(self, remote_path: str) -> None:
        path = PurePosixPath(validate_remote_directory(remote_path))
        current = PurePosixPath("/")
        for part in path.parts[1:]:
            current /= part
            current_text = str(current)
            try:
                attributes = self.sftp.stat(current_text)
            except OSError as exc:
                if not (
                    isinstance(exc, FileNotFoundError)
                    or getattr(exc, "errno", None) == 2
                ):
                    raise
                self.sftp.mkdir(current_text)
            else:
                if not stat.S_ISDIR(attributes.st_mode):
                    raise UploadError(
                        "REMOTE_PATH_CONFLICT",
                        f"远程路径已存在但不是目录：{current_text}",
                    )

    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        *,
        progress: UploadByteProgress | None = None,
    ) -> None:
        # All remote segments were checked before this reaches SCPClient.
        self._upload_progress = progress
        try:
            self.scp.put(str(local_path), remote_path=remote_path, recursive=False)
        except BaseException:
            # A progress callback may deliberately raise when Collector starts
            # or the operator cancels. Close the SCP channel immediately so the
            # remote process releases its partial file before SFTP cleanup.
            if self._scp is not None:
                try:
                    self._scp.close()
                finally:
                    self._scp = None
            raise
        finally:
            self._upload_progress = None

    def remote_sha256(
        self,
        remote_path: str,
        *,
        progress: UploadByteProgress | None = None,
    ) -> str:
        digest = hashlib.sha256()
        total_bytes = int(getattr(self.sftp.stat(remote_path), "st_size", 0))
        transferred_bytes = 0
        if progress is not None:
            progress(transferred_bytes, total_bytes)
        with self.sftp.open(remote_path, "rb") as stream:
            while chunk := stream.read(_COPY_BUFFER_SIZE):
                digest.update(chunk)
                transferred_bytes += len(chunk)
                if progress is not None:
                    progress(transferred_bytes, total_bytes)
        return digest.hexdigest()

    def rename(self, source: str, destination: str) -> None:
        self.sftp.rename(source, destination)

    def remove_file(self, remote_path: str) -> None:
        try:
            self.sftp.remove(remote_path)
        except OSError as exc:
            if not (
                isinstance(exc, FileNotFoundError)
                or getattr(exc, "errno", None) == 2
            ):
                raise

    def remove_directory(self, remote_path: str) -> None:
        try:
            self.sftp.rmdir(remote_path)
        except OSError as exc:
            if not (
                isinstance(exc, FileNotFoundError)
                or getattr(exc, "errno", None) == 2
            ):
                raise

    def close(self) -> None:
        if self._scp is not None:
            try:
                self._scp.close()
            finally:
                self._scp = None
        if self._sftp is not None:
            try:
                self._sftp.close()
            finally:
                self._sftp = None
        self._client.close()


def _default_remote_session(request: OfflineUploadRequest) -> RemoteUploadSession:
    return ParamikoScpSession(request)


class SshScpTrialUploader:
    """Upload, verify, then publish one immutable Trial package."""

    def __init__(self, session_factory: RemoteSessionFactory | None = None) -> None:
        self._session_factory = session_factory or _default_remote_session

    def upload(
        self,
        request: OfflineUploadRequest,
        *,
        progress: ProgressCallback | None = None,
        cancelled: CancelCheck | None = None,
    ) -> OfflineUploadResult:
        report = progress or (lambda _update: None)
        is_cancelled = cancelled or (lambda: False)

        started_at_utc_ns = time.time_ns()
        self._guard(request, is_cancelled)
        report(UploadProgress(UploadPhase.VALIDATING, "正在验证本地 Trial 完整性…"))
        plan = build_upload_plan(request.manifest_path)
        self._guard(request, is_cancelled)

        trial_name = str(plan.trial_uuid)
        remote_final = build_remote_trial_directory(request.remote_workdir, plan)
        remote_trials_directory = str(PurePosixPath(remote_final).parent)
        staging_name = f".{trial_name}.partial-{uuid4().hex}"
        remote_staging = _remote_join(remote_trials_directory, staging_name)
        cleanup_files = [
            _remote_join(remote_staging, *item.relative_path.parts)
            for item in plan.files
        ]
        created_directories: set[str] = set()
        session: RemoteUploadSession | None = None

        try:
            report(UploadProgress(UploadPhase.CONNECTING, "正在建立 SSH/SCP 连接…"))
            session = self._session_factory(request)
            self._guard(request, is_cancelled)
            session.ensure_directory(remote_trials_directory)
            if session.exists(remote_final):
                raise UploadError(
                    "REMOTE_EXISTS", f"远程 Trial 目录已存在，为防止覆盖已取消：{remote_final}"
                )
            created_directories.add(remote_staging)
            session.ensure_directory(remote_staging)

            total_files = len(plan.files)
            for index, item in enumerate(plan.files, start=1):
                self._guard(request, is_cancelled)
                parent = _remote_join(
                    remote_staging,
                    *item.relative_path.parts[:-1],
                )
                if parent not in created_directories:
                    created_directories.add(parent)
                    session.ensure_directory(parent)
                remote_file = _remote_join(remote_staging, *item.relative_path.parts)
                report(
                    UploadProgress(
                        UploadPhase.UPLOADING,
                        f"正在上传 {index}/{total_files}：{item.relative_path.as_posix()}",
                        index - 1,
                        total_files,
                    )
                )
                last_guard_ns = 0
                last_report_ns = 0

                def file_progress(
                    transferred_bytes: int,
                    total_bytes: int,
                    *,
                    current_index: int = index,
                    current_item: TrialUploadFile = item,
                ) -> None:
                    nonlocal last_guard_ns
                    nonlocal last_report_ns
                    now_ns = time.perf_counter_ns()
                    # The SCP library invokes this once per transfer buffer.
                    # Bound lock-file reads and IPC polling to 20 Hz while still
                    # making cancellation responsive inside a multi-gigabyte file.
                    if (
                        last_guard_ns == 0
                        or now_ns - last_guard_ns >= 50_000_000
                        or transferred_bytes >= total_bytes
                    ):
                        self._guard(request, is_cancelled)
                        last_guard_ns = now_ns
                    if (
                        last_report_ns == 0
                        or now_ns - last_report_ns >= 250_000_000
                        or transferred_bytes >= total_bytes
                    ):
                        percentage = (
                            100.0
                            if total_bytes <= 0
                            else min(
                                100.0,
                                max(0.0, transferred_bytes * 100.0 / total_bytes),
                            )
                        )
                        report(
                            UploadProgress(
                                UploadPhase.UPLOADING,
                                f"正在上传 {current_index}/{total_files}："
                                f"{current_item.relative_path.as_posix()} · {percentage:.1f}%",
                                current_index - 1,
                                total_files,
                            )
                        )
                        last_report_ns = now_ns

                session.upload_file(
                    item.local_path,
                    remote_file,
                    progress=file_progress,
                )

            for index, item in enumerate(plan.files, start=1):
                self._guard(request, is_cancelled)
                remote_file = _remote_join(remote_staging, *item.relative_path.parts)
                report(
                    UploadProgress(
                        UploadPhase.VERIFYING,
                        f"正在校验 {index}/{total_files}：{item.relative_path.as_posix()}",
                        index - 1,
                        total_files,
                    )
                )
                last_guard_ns = 0
                last_report_ns = 0

                def verification_progress(
                    verified_bytes: int,
                    total_bytes: int,
                    *,
                    current_index: int = index,
                    current_item: TrialUploadFile = item,
                ) -> None:
                    nonlocal last_guard_ns
                    nonlocal last_report_ns
                    now_ns = time.perf_counter_ns()
                    if (
                        last_guard_ns == 0
                        or now_ns - last_guard_ns >= 50_000_000
                        or verified_bytes >= total_bytes
                    ):
                        self._guard(request, is_cancelled)
                        last_guard_ns = now_ns
                    if (
                        last_report_ns == 0
                        or now_ns - last_report_ns >= 250_000_000
                        or verified_bytes >= total_bytes
                    ):
                        percentage = (
                            100.0
                            if total_bytes <= 0
                            else min(
                                100.0,
                                max(0.0, verified_bytes * 100.0 / total_bytes),
                            )
                        )
                        report(
                            UploadProgress(
                                UploadPhase.VERIFYING,
                                f"正在校验 {current_index}/{total_files}："
                                f"{current_item.relative_path.as_posix()} · {percentage:.1f}%",
                                current_index - 1,
                                total_files,
                            )
                        )
                        last_report_ns = now_ns

                remote_digest = session.remote_sha256(
                    remote_file,
                    progress=verification_progress,
                )
                if remote_digest != item.sha256:
                    raise UploadError(
                        "REMOTE_INTEGRITY_FAILED",
                        f"远程 SHA-256 校验失败：{item.relative_path.as_posix()}",
                    )

            self._guard(request, is_cancelled)
            report(UploadProgress(UploadPhase.PUBLISHING, "校验通过，正在发布远程 Trial…"))
            if session.exists(remote_final):
                raise UploadError(
                    "REMOTE_EXISTS", f"远程 Trial 目录在上传期间已被创建：{remote_final}"
                )
            session.rename(remote_staging, remote_final)
            created_directories.clear()
            cleanup_files.clear()
            verified_at_utc_ns = time.time_ns()
            audit_record_path = _write_upload_audit(
                request,
                plan,
                status="VERIFIED",
                started_at_utc_ns=started_at_utc_ns,
                completed_at_utc_ns=verified_at_utc_ns,
                remote_trial_directory=remote_final,
            )
            result = OfflineUploadResult(
                trial_uuid=plan.trial_uuid,
                remote_trial_directory=remote_final,
                file_count=len(plan.files),
                total_bytes=plan.total_bytes,
                verified_at_utc_ns=verified_at_utc_ns,
                transfer_batch_uuid=request.transfer_batch_uuid,
                audit_record_path=audit_record_path,
            )
            report(
                UploadProgress(
                    UploadPhase.COMPLETED,
                    f"上传并校验完成，共 {result.file_count} 个文件。",
                    result.file_count,
                    result.file_count,
                )
            )
            return result
        except UnknownHostKeyError:
            raise
        except UploadError as exc:
            try:
                _write_upload_audit(
                    request,
                    plan,
                    status="FAILED",
                    started_at_utc_ns=started_at_utc_ns,
                    completed_at_utc_ns=time.time_ns(),
                    remote_trial_directory=remote_final,
                    error_code=exc.code,
                    error_message=_safe_exception(exc, *request.secrets),
                )
            except Exception:
                pass
            raise
        except Exception as exc:
            message = _safe_exception(exc, *request.secrets)
            wrapped = UploadError("TRANSFER_FAILED", f"SSH/SCP 上传失败：{message}")
            try:
                _write_upload_audit(
                    request,
                    plan,
                    status="FAILED",
                    started_at_utc_ns=started_at_utc_ns,
                    completed_at_utc_ns=time.time_ns(),
                    remote_trial_directory=remote_final,
                    error_code=wrapped.code,
                    error_message=str(wrapped),
                )
            except Exception:
                pass
            raise wrapped from exc
        finally:
            if session is not None:
                if cleanup_files or created_directories:
                    for remote_file in reversed(cleanup_files):
                        try:
                            session.remove_file(remote_file)
                        except Exception:
                            pass
                    for directory in sorted(
                        created_directories,
                        key=lambda item: len(PurePosixPath(item).parts),
                        reverse=True,
                    ):
                        try:
                            session.remove_directory(directory)
                        except Exception:
                            pass
                try:
                    session.close()
                except Exception:
                    pass

    @staticmethod
    def _guard(request: OfflineUploadRequest, cancelled: CancelCheck) -> None:
        if cancelled():
            raise UploadCancelled()
        if read_activity(request.dataset_root) is not None:
            raise UploadError(
                "COLLECTOR_ACTIVE",
                "检测到 Collector 正在采集，已禁止或中止网络上传。",
            )


def _remote_join(base: str, *parts: str) -> str:
    path = PurePosixPath(validate_remote_directory(base))
    for part in parts:
        if not _is_safe_remote_segment(str(part)):
            raise UploadError("UNSAFE_REMOTE_PATH", "远程路径包含不安全字符。")
        path /= str(part)
    return str(path)


def _safe_exception(exc: BaseException, *secrets: str) -> str:
    message = str(exc).strip() or type(exc).__name__
    for secret in secrets:
        if secret:
            message = message.replace(secret, "<redacted>")
    # Keep UI errors concise and prevent multiline protocol text from becoming
    # an accidental pseudo-log entry.
    return " ".join(message.split())[:500]


class UploadWorkerEventType(StrEnum):
    PROGRESS = "PROGRESS"
    HOST_KEY_REQUIRED = "HOST_KEY_REQUIRED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class UploadWorkerEvent:
    event_type: UploadWorkerEventType
    progress: UploadProgress | None = None
    result: OfflineUploadResult | None = None
    host_key: HostKeyInfo | None = None
    error_code: str | None = None
    message: str | None = None


def _upload_worker_main(command: Connection, events: Connection) -> None:
    """Spawn-worker entry point.  No password is present in process argv."""

    request: OfflineUploadRequest | None = None

    def cancelled() -> bool:
        if not command.poll():
            return False
        try:
            message = command.recv()
        except EOFError:
            return True
        return message == "CANCEL"

    try:
        incoming = command.recv()
        if not isinstance(incoming, OfflineUploadRequest):
            raise UploadError("INVALID_REQUEST", "上传 Worker 收到了无效请求。")
        request = incoming
        uploader = SshScpTrialUploader()
        while True:
            try:
                result = uploader.upload(
                    request,
                    progress=lambda update: events.send(
                        UploadWorkerEvent(
                            UploadWorkerEventType.PROGRESS, progress=update
                        )
                    ),
                    cancelled=cancelled,
                )
                break
            except UnknownHostKeyError as exc:
                events.send(
                    UploadWorkerEvent(
                        UploadWorkerEventType.HOST_KEY_REQUIRED,
                        host_key=exc.host_key,
                        error_code=exc.code,
                        message=str(exc),
                    )
                )
                try:
                    decision = command.recv()
                except EOFError as pipe_error:
                    raise UploadCancelled() from pipe_error
                if decision == "CANCEL":
                    raise UploadCancelled("操作者未确认 SSH 主机指纹，上传已取消。")
                if (
                    not isinstance(decision, tuple)
                    or len(decision) != 2
                    or decision[0] != "TRUST_HOST_KEY"
                    or decision[1] != exc.host_key
                ):
                    raise UploadError("INVALID_HOST_KEY_DECISION", "SSH 主机指纹确认响应无效。")
                request = replace(request, accepted_host_key=exc.host_key)
        events.send(
            UploadWorkerEvent(UploadWorkerEventType.COMPLETED, result=result)
        )
    except UploadError as exc:
        events.send(
            UploadWorkerEvent(
                UploadWorkerEventType.FAILED,
                error_code=exc.code,
                message=_safe_exception(exc, *(request.secrets if request else ())),
            )
        )
    except BaseException as exc:
        events.send(
            UploadWorkerEvent(
                UploadWorkerEventType.FAILED,
                error_code="WORKER_FAILED",
                message=f"上传 Worker 异常：{_safe_exception(exc, *(request.secrets if request else ()))}",
            )
        )
    finally:
        # Dropping references is the strongest portable guarantee available
        # for immutable Python strings; the credential was never persisted.
        request = None
        command.close()
        events.close()


class UploadWorkerHandle:
    """Parent-side controller polled by a Qt timer without blocking the GUI."""

    def __init__(
        self,
        *,
        context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        self._context = context or multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._command: Connection | None = None
        self._events: Connection | None = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode if self._process is not None else None

    def start(self, request: OfflineUploadRequest) -> None:
        if self._process is not None:
            raise RuntimeError("Upload worker has already been started")
        parent_command: Connection | None = None
        child_command: Connection | None = None
        parent_events: Connection | None = None
        child_events: Connection | None = None
        process: multiprocessing.Process | None = None
        try:
            parent_command, child_command = self._context.Pipe(duplex=True)
            parent_events, child_events = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=_upload_worker_main,
                args=(child_command, child_events),
                name="exo-offline-upload-worker",
                daemon=False,
            )
            process.start()
            child_command.close()
            child_command = None
            child_events.close()
            child_events = None
            # Sent only after spawn over an anonymous local pipe.  The secret
            # therefore never appears in process creation arguments/argv.
            parent_command.send(request)
        except BaseException:
            if process is not None:
                try:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2)
                except (AssertionError, OSError, ValueError):
                    pass
                try:
                    process.close()
                except (AssertionError, OSError, ValueError):
                    pass
            for connection in (
                parent_command,
                child_command,
                parent_events,
                child_events,
            ):
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass
            raise
        self._process = process
        self._command = parent_command
        self._events = parent_events

    def request_cancel(self) -> None:
        if self._command is not None and self.is_alive:
            try:
                self._command.send("CANCEL")
            except (BrokenPipeError, EOFError, OSError):
                pass

    def trust_host_key(self, host_key: HostKeyInfo) -> None:
        if self._command is None or not self.is_alive:
            raise RuntimeError("Upload worker is not waiting for a host key decision")
        self._command.send(("TRUST_HOST_KEY", host_key))

    def poll_events(self, limit: int = 100) -> list[UploadWorkerEvent]:
        events: list[UploadWorkerEvent] = []
        connection = self._events
        if connection is None:
            return events
        while len(events) < limit:
            try:
                available = connection.poll()
            except (BrokenPipeError, EOFError, OSError):
                break
            if not available:
                break
            try:
                event = connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                break
            if isinstance(event, UploadWorkerEvent):
                events.append(event)
        return events

    def join(self, timeout: float | None = None) -> int | None:
        if self._process is None:
            return None
        self._process.join(timeout)
        return self._process.exitcode

    def terminate_for_shutdown(self, timeout: float = 2.0) -> int | None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=timeout)
        if process is not None and process.is_alive():
            # Network libraries can be inside native waits. A bounded kill is
            # required before Qt destroys the handle; credentials and remote
            # staging remain protected by the uploader's atomic workflow.
            process.kill()
            process.join(timeout=timeout)
        if process is not None and process.is_alive():
            raise RuntimeError("upload worker did not exit after terminate/kill")
        return process.exitcode if process is not None else None

    def close(self) -> None:
        if self.is_alive:
            raise RuntimeError("Cannot close a running upload worker")
        for connection in (self._command, self._events):
            if connection is not None:
                connection.close()
        if self._process is not None:
            self._process.close()
        self._command = None
        self._events = None
        self._process = None


__all__ = [
    "AuthenticationMethod",
    "HostKeyInfo",
    "OfflineUploadRequest",
    "OfflineUploadResult",
    "ParamikoScpSession",
    "SshScpTrialUploader",
    "TrialUploadFile",
    "TrialUploadPlan",
    "UploadCancelled",
    "UploadError",
    "UploadPhase",
    "UploadProgress",
    "UploadWorkerEvent",
    "UploadWorkerEventType",
    "UploadWorkerHandle",
    "build_remote_trial_directory",
    "build_upload_plan",
    "validate_finalized_trial",
    "validate_remote_directory",
]
