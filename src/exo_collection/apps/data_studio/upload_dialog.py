"""Credential-ephemeral Qt dialogs for manual offline Trial upload."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from exo_collection.configuration import SharedAppSettings

from .credential_store import delete_password, load_password, save_password
from .upload import (
    OfflineUploadRequest,
    RemoteOnlyTrial,
    RemoteTrialStatus,
    RemoteTrialStatusRecord,
    UploadOperation,
    UploadProgress,
    validate_remote_directory,
)


class _EndpointFormWidget(QGroupBox):
    """Reusable SSH/SCP endpoint form shared by upload and dataset-mirror dialogs.

    Owns the fields, restore/collect/validate, and the Windows Credential
    Manager password round-trip. It is deliberately settings-agnostic: the
    caller supplies the endpoint dict to restore and the persistence setter.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        form = QFormLayout(self)
        self.host_edit = QLineEdit()
        self.host_edit.setObjectName("endpoint_host")
        self.host_edit.setPlaceholderText("主机名或 IP（无默认值）")
        form.addRow("主机：", self.host_edit)

        self.port_spin = QSpinBox()
        self.port_spin.setObjectName("endpoint_port")
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        form.addRow("端口：", self.port_spin)

        self.username_edit = QLineEdit()
        self.username_edit.setObjectName("endpoint_username")
        self.username_edit.setPlaceholderText("用户名（无默认值）")
        form.addRow("用户名：", self.username_edit)

        self.remote_workdir_edit = QLineEdit()
        self.remote_workdir_edit.setObjectName("endpoint_remote_workdir")
        self.remote_workdir_edit.setPlaceholderText("/absolute/path/to/data")
        form.addRow("远程 data 根目录：", self.remote_workdir_edit)

        self.authentication_combo = QComboBox()
        self.authentication_combo.setObjectName("endpoint_authentication")
        self.authentication_combo.addItem("密码", "PASSWORD")
        self.authentication_combo.addItem("SSH 私钥", "PRIVATE_KEY")
        self.authentication_combo.currentIndexChanged.connect(
            self.apply_authentication_mode
        )
        form.addRow("认证方式：", self.authentication_combo)

        self.password_edit = QLineEdit()
        self.password_edit.setObjectName("endpoint_password")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("可安全保存到当前 Windows 用户的凭据管理器")
        form.addRow("密码：", self.password_edit)

        self.remember_password_check = QCheckBox("记住密码（保存到 Windows 凭据管理器）")
        self.remember_password_check.setObjectName("endpoint_remember_password")
        self.remember_password_check.setChecked(True)
        form.addRow("", self.remember_password_check)

        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        self.private_key_edit = QLineEdit()
        self.private_key_edit.setObjectName("endpoint_private_key")
        self.private_key_edit.setReadOnly(True)
        self.private_key_edit.setPlaceholderText("选择本地 SSH 私钥文件")
        key_layout.addWidget(self.private_key_edit, 1)
        self.private_key_button = QPushButton("选择…")
        self.private_key_button.setObjectName("endpoint_browse_private_key")
        self.private_key_button.clicked.connect(self.choose_private_key)
        key_layout.addWidget(self.private_key_button)
        form.addRow("SSH 私钥：", key_row)

        self.passphrase_edit = QLineEdit()
        self.passphrase_edit.setObjectName("endpoint_private_key_passphrase")
        self.passphrase_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.passphrase_edit.setPlaceholderText("可选；仅本次使用并立即清空")
        form.addRow("私钥口令：", self.passphrase_edit)

        self.host_edit.editingFinished.connect(self.load_saved_password)
        self.username_edit.editingFinished.connect(self.load_saved_password)
        self.port_spin.valueChanged.connect(self.load_saved_password)

    def restore(self, endpoint: dict) -> None:
        self.host_edit.setText(str(endpoint.get("host", "")))
        self.port_spin.setValue(int(endpoint.get("port", 22)))
        self.username_edit.setText(str(endpoint.get("username", "")))
        self.remote_workdir_edit.setText(str(endpoint.get("remote_workdir", "")))
        self.private_key_edit.setText(str(endpoint.get("private_key_path", "")))
        authentication = str(endpoint.get("authentication", "PASSWORD"))
        index = self.authentication_combo.findData(authentication)
        self.authentication_combo.setCurrentIndex(max(0, index))
        self.remember_password_check.setChecked(
            bool(endpoint.get("remember_password", True))
        )
        if authentication == "PASSWORD":
            self.load_saved_password()

    def collect(self) -> dict:
        return {
            "host": self.host_edit.text().strip(),
            "port": self.port_spin.value(),
            "username": self.username_edit.text().strip(),
            "remote_workdir": self.remote_workdir_edit.text().strip(),
            "authentication": self.authentication_combo.currentData(),
            "private_key_path": self.private_key_edit.text().strip(),
            "remember_password": self.remember_password_check.isChecked(),
        }

    def validate(self) -> str | None:
        """Return a human-readable error, or ``None`` when the form is usable."""

        if not self.host_edit.text().strip():
            return "主机不能为空。"
        if not self.username_edit.text().strip():
            return "用户名不能为空。"
        try:
            validate_remote_directory(self.remote_workdir_edit.text())
        except ValueError as exc:
            return str(exc)
        if (
            self.authentication_combo.currentData() == "PRIVATE_KEY"
            and not self.private_key_edit.text().strip()
        ):
            return "请选择 SSH 私钥文件。"
        return None

    def use_private_key(self) -> bool:
        return self.authentication_combo.currentData() == "PRIVATE_KEY"

    def apply_authentication_mode(self) -> None:
        use_private_key = self.use_private_key()
        self.password_edit.setEnabled(not use_private_key)
        self.private_key_edit.setEnabled(use_private_key)
        self.private_key_button.setEnabled(use_private_key)
        self.passphrase_edit.setEnabled(use_private_key)
        self.remember_password_check.setEnabled(not use_private_key)
        if not use_private_key:
            self.load_saved_password()

    def load_saved_password(self, *_args: object) -> None:
        if self.use_private_key() or not self.remember_password_check.isChecked():
            return
        try:
            password = load_password(
                self.host_edit.text(),
                self.port_spin.value(),
                self.username_edit.text(),
            )
        except RuntimeError as exc:
            self.remember_password_check.setToolTip(str(exc))
            return
        if password is not None:
            self.password_edit.setText(password)
            self.remember_password_check.setToolTip("已从 Windows 凭据管理器加载密码。")

    def choose_private_key(self) -> None:
        selected, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择 SSH 私钥",
            str(Path.home() / ".ssh"),
            "SSH 私钥 (*)",
        )
        if selected:
            self.private_key_edit.setText(selected)

    def clear_secrets(self) -> None:
        self.password_edit.clear()
        self.passphrase_edit.clear()


class OfflineUploadDialog(QDialog):
    """Collect one transfer endpoint without loading or saving profiles."""

    def __init__(
        self,
        manifest_path: Path | Sequence[Path],
        parent: QWidget | None = None,
        *,
        status_only: bool = False,
        settings: SharedAppSettings | None = None,
        download_target_root: Path | None = None,
    ) -> None:
        super().__init__(parent)
        raw_paths = (manifest_path,) if isinstance(manifest_path, Path) else tuple(manifest_path)
        if not raw_paths:
            raise ValueError("至少需要一个 FINALIZED Trial。")
        self._manifest_paths = tuple(Path(path).expanduser().resolve() for path in raw_paths)
        self._status_only = status_only
        self._download_target_root = (
            Path(download_target_root).expanduser().resolve()
            if download_target_root is not None
            else None
        )
        self._download = self._download_target_root is not None
        self._settings = settings if settings is not None else SharedAppSettings()
        self.setWindowTitle(
            "从服务器拉取数据"
            if self._download
            else ("同步云端状态" if status_only else "人工离线 SSH/SCP 上传")
        )
        self.setModal(True)
        self.resize(560, 380)

        outer = QVBoxLayout(self)
        explanation = QLabel(
            (
                "把服务器 data/ 全量拉取到所选本地文件夹：已下载且一致的 Trial 会跳过，"
                "其余覆盖下载（git pull 式），并逐包校验 SHA-256。"
                if self._download
                else (
                    "只读核对本地与云端 data/ 的同路径文件及 SHA-256；不会上传、下载、覆盖或删除文件。"
                    if status_only
                    else
                    "同步所选层级下全部已最终化 Trial，并完整保留它们在本地 data/ 下的相对目录。"
                    "云端已有同内容文件会跳过，缺少文件会补传，云端额外文件不会删除；"
                    "同路径内容冲突时停止且不覆盖。"
                )
            )
            + "勾选记住密码时只保存到 Windows 凭据管理器，不写入配置或日志。"
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        trial_group = QGroupBox("同步范围")
        trial_form = QFormLayout(trial_group)
        trial_path = QLineEdit(
            "全部云端 Trial（全量拉取）"
            if self._download
            else f"{len(self._manifest_paths)} 个 FINALIZED Trial"
        )
        trial_path.setReadOnly(True)
        trial_path.setObjectName("upload_trial_path")
        trial_form.addRow("拉取范围：" if self._download else "本地范围：", trial_path)
        outer.addWidget(trial_group)

        self.endpoint_form = _EndpointFormWidget("SSH/SCP 目标（每次手工输入）", self)
        outer.addWidget(self.endpoint_form)
        self.endpoint_form.restore(self._settings.upload_endpoint)
        self.endpoint_form.apply_authentication_mode()
        # Backward-compatible aliases so callers and tests reach the fields directly.
        for _name in (
            "host_edit",
            "port_spin",
            "username_edit",
            "remote_workdir_edit",
            "authentication_combo",
            "password_edit",
            "remember_password_check",
            "private_key_edit",
            "private_key_button",
            "passphrase_edit",
        ):
            setattr(self, _name, getattr(self.endpoint_form, _name))

        safety = QLabel(
            "首次连接时系统会显示 SSH SHA-256 主机指纹。"
            "请通过独立渠道与服务器管理员核对后再确认。"
        )
        safety.setWordWrap(True)
        safety.setStyleSheet("QLabel { color: #664d03; }")
        outer.addWidget(safety)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        self.start_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.start_button.setText(
            "开始拉取"
            if self._download
            else ("开始同步状态" if status_only else "开始上传")
        )
        self.start_button.setObjectName("start_offline_upload")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.host_edit.setFocus()

    def take_request(self, dataset_root: Path) -> OfflineUploadRequest:
        """Build a request and persist only the operator-approved credentials."""

        password = self.password_edit.text()
        passphrase = self.passphrase_edit.text()
        private_key_path = self.private_key_edit.text().strip()
        use_private_key = self.authentication_combo.currentData() == "PRIVATE_KEY"
        request = OfflineUploadRequest(
            dataset_root=dataset_root,
            manifest_path=self._manifest_paths[0],
            additional_manifest_paths=self._manifest_paths[1:],
            operation=(
                UploadOperation.DOWNLOAD
                if self._download
                else (
                    UploadOperation.SYNC_REMOTE_STATUS
                    if self._status_only
                    else UploadOperation.UPLOAD
                )
            ),
            host=self.host_edit.text(),
            port=self.port_spin.value(),
            username=self.username_edit.text(),
            remote_workdir=self.remote_workdir_edit.text(),
            password=(None if use_private_key else password),
            private_key_path=(Path(private_key_path) if use_private_key else None),
            private_key_passphrase=(passphrase or None if use_private_key else None),
            download_target_root=self._download_target_root,
        )
        self._settings.set_upload_endpoint(
            {
                "host": request.host,
                "port": request.port,
                "username": request.username,
                "remote_workdir": request.remote_workdir,
                "authentication": self.authentication_combo.currentData(),
                "private_key_path": private_key_path,
                "remember_password": self.remember_password_check.isChecked(),
            }
        )
        if not use_private_key and self.remember_password_check.isChecked():
            save_password(request.host, request.port, request.username, password)
        elif not use_private_key:
            delete_password(request.host, request.port, request.username)
        if not self.remember_password_check.isChecked():
            self.endpoint_form.clear_secrets()
        return request

    def reject(self) -> None:
        self.endpoint_form.clear_secrets()
        super().reject()


class DatasetUploadSettingsDialog(QDialog):
    """Configure the separate dataset-mirror server directory (no upload)."""

    def __init__(
        self,
        settings: SharedAppSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings if settings is not None else SharedAppSettings()
        self.setWindowTitle("数据集上传设置")
        self.setModal(True)
        self.resize(560, 360)

        outer = QVBoxLayout(self)
        explanation = QLabel(
            "配置一个独立于「上传/下载」的服务器目录，用于镜像全部「已接收」session"
            "（保留本地文件树结构，增量上传 + 删除不再接收的 session）。\n"
            "勾选记住密码时只保存到 Windows 凭据管理器，不写入配置或日志。"
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        self.endpoint_form = _EndpointFormWidget("数据集服务器目录（独立于上传/下载）", self)
        outer.addWidget(self.endpoint_form)
        self.endpoint_form.restore(self._settings.dataset_upload_endpoint)
        self.endpoint_form.apply_authentication_mode()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        save_button.setText("保存")
        save_button.setObjectName("save_dataset_upload_endpoint")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self._reject)
        outer.addWidget(buttons)
        self.endpoint_form.host_edit.setFocus()

    def _save(self) -> None:
        error = self.endpoint_form.validate()
        if error is not None:
            QMessageBox.warning(self, "数据集上传参数无效", error)
            return
        values = self.endpoint_form.collect()
        self._settings.set_dataset_upload_endpoint(values)
        use_private_key = self.endpoint_form.use_private_key()
        if not use_private_key and values["remember_password"]:
            save_password(
                values["host"],
                values["port"],
                values["username"],
                self.endpoint_form.password_edit.text(),
            )
        elif not use_private_key:
            delete_password(values["host"], values["port"], values["username"])
        self.endpoint_form.clear_secrets()
        self.accept()

    def _reject(self) -> None:
        self.endpoint_form.clear_secrets()
        self.reject()


class SelectiveUploadDialog(QDialog):
    """Let the operator pick specific Trials when the cloud is not a subset."""

    _STATUS_LABELS = {
        RemoteTrialStatus.UPLOADED: "已上传",
        RemoteTrialStatus.NOT_UPLOADED: "未上传",
        RemoteTrialStatus.PARTIAL: "部分缺失",
        RemoteTrialStatus.CONFLICT: "内容冲突",
    }

    def __init__(
        self,
        records: Sequence[RemoteTrialStatusRecord],
        remote_only: Sequence[RemoteOnlyTrial] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._records = tuple(records)
        self.setWindowTitle("选择要上传的 Trial")
        self.setModal(True)
        self.resize(680, 480)

        outer = QVBoxLayout(self)
        explanation = QLabel(
            "云端存在本地没有的数据，无法安全地自动增量更新。\n"
            "请勾选要上传的本地 Trial；只会补传所选 Trial，不会删除或覆盖云端已有内容。"
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        if remote_only:
            warning = QLabel(
                f"⚠ 云端有 {len(remote_only)} 个本地没有的 Trial，"
                "本次操作不会处理它们。"
            )
            warning.setStyleSheet("QLabel { color: #842029; }")
            warning.setWordWrap(True)
            outer.addWidget(warning)

        self.trial_list = QListWidget()
        self.trial_list.setObjectName("selective_upload_list")
        self._checkbox_items: list[tuple[QListWidgetItem, Path]] = []
        for record in self._records:
            status_text = self._STATUS_LABELS.get(record.status, record.status.value)
            item = QListWidgetItem(f"[{status_text}] {record.manifest_path}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(Qt.ItemDataRole.UserRole, str(record.manifest_path))
            item.setCheckState(
                Qt.CheckState.Checked
                if record.status
                in (RemoteTrialStatus.NOT_UPLOADED, RemoteTrialStatus.PARTIAL)
                else Qt.CheckState.Unchecked
            )
            self.trial_list.addItem(item)
            self._checkbox_items.append((item, record.manifest_path))
        outer.addWidget(self.trial_list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        self.upload_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.upload_button.setText("上传所选")
        self.upload_button.setObjectName("confirm_selective_upload")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def selected_manifest_paths(self) -> tuple[Path, ...]:
        selected: list[Path] = []
        for item, path in self._checkbox_items:
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(path)
        return tuple(selected)


class UploadProgressDialog(QDialog):
    """Non-blocking upload status view with a controlled cancel request."""

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("上传进度")
        self.setModal(False)
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        self.status_label = QLabel("正在启动独立上传进程…")
        self.status_label.setObjectName("upload_progress_status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("upload_progress_bar")
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton("取消上传")
        self.cancel_button.setObjectName("cancel_offline_upload")
        self.cancel_button.clicked.connect(self._request_cancel)
        layout.addWidget(self.cancel_button)

    def update_progress(self, progress: UploadProgress) -> None:
        self.status_label.setText(progress.message)
        if progress.total_files > 0:
            self.progress_bar.setRange(0, progress.total_files)
            self.progress_bar.setValue(progress.completed_files)
        else:
            self.progress_bar.setRange(0, 0)

    def waiting_for_host_key(self) -> None:
        self.status_label.setText("等待操作者核对 SSH 主机指纹…")
        self.progress_bar.setRange(0, 0)

    def waiting_for_delete_confirmation(self) -> None:
        self.status_label.setText("等待确认是否删除云端多余文件…")
        self.progress_bar.setRange(0, 0)

    def mark_cancelling(self) -> None:
        self.status_label.setText("正在取消并清理远程临时目录…")
        self.cancel_button.setEnabled(False)

    def mark_finished(self) -> None:
        self.cancel_button.setEnabled(False)

    def _request_cancel(self) -> None:
        self.mark_cancelling()
        self.cancel_requested.emit()

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        if self.cancel_button.isEnabled():
            self._request_cancel()
            event.ignore()  # type: ignore[attr-defined]
        else:
            event.accept()  # type: ignore[attr-defined]


__all__ = ["OfflineUploadDialog", "SelectiveUploadDialog", "UploadProgressDialog"]
