"""Credential-ephemeral Qt dialogs for manual offline Trial upload."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .upload import OfflineUploadRequest, UploadProgress


class OfflineUploadDialog(QDialog):
    """Collect one transfer endpoint without loading or saving profiles."""

    def __init__(self, manifest_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manifest_path = Path(manifest_path).expanduser().resolve()
        self.setWindowTitle("人工离线 SSH/SCP 上传")
        self.setModal(True)
        self.resize(560, 380)

        outer = QVBoxLayout(self)
        explanation = QLabel(
            "仅上传已最终化 Trial。上传在独立进程中执行，远程逐文件 "
            "SHA-256 通过后才发布。密码仅在内存中传递，不会保存。"
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        trial_group = QGroupBox("Trial")
        trial_form = QFormLayout(trial_group)
        trial_dir = self._manifest_path.parent
        if trial_dir.name == ".exo":
            trial_dir = trial_dir.parent
        trial_path = QLineEdit(str(trial_dir))
        trial_path.setReadOnly(True)
        trial_path.setObjectName("upload_trial_path")
        trial_form.addRow("本地数据包：", trial_path)
        outer.addWidget(trial_group)

        endpoint_group = QGroupBox("SSH/SCP 目标（每次手工输入）")
        endpoint_form = QFormLayout(endpoint_group)
        self.host_edit = QLineEdit()
        self.host_edit.setObjectName("upload_host")
        self.host_edit.setPlaceholderText("主机名或 IP（无默认值）")
        endpoint_form.addRow("主机：", self.host_edit)

        self.port_spin = QSpinBox()
        self.port_spin.setObjectName("upload_port")
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        endpoint_form.addRow("端口：", self.port_spin)

        self.username_edit = QLineEdit()
        self.username_edit.setObjectName("upload_username")
        self.username_edit.setPlaceholderText("用户名（无默认值）")
        endpoint_form.addRow("用户名：", self.username_edit)

        self.remote_workdir_edit = QLineEdit()
        self.remote_workdir_edit.setObjectName("upload_remote_workdir")
        self.remote_workdir_edit.setPlaceholderText("/absolute/remote/workdir")
        endpoint_form.addRow("远程工作目录：", self.remote_workdir_edit)

        self.authentication_combo = QComboBox()
        self.authentication_combo.setObjectName("upload_authentication")
        self.authentication_combo.addItem("密码", "PASSWORD")
        self.authentication_combo.addItem("SSH 私钥", "PRIVATE_KEY")
        self.authentication_combo.currentIndexChanged.connect(
            self._apply_authentication_mode
        )
        endpoint_form.addRow("认证方式：", self.authentication_combo)

        self.password_edit = QLineEdit()
        self.password_edit.setObjectName("upload_password")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("仅本次使用，关闭或开始后立即清空")
        endpoint_form.addRow("密码：", self.password_edit)

        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        self.private_key_edit = QLineEdit()
        self.private_key_edit.setObjectName("upload_private_key")
        self.private_key_edit.setReadOnly(True)
        self.private_key_edit.setPlaceholderText("选择本地 SSH 私钥文件")
        key_layout.addWidget(self.private_key_edit, 1)
        self.private_key_button = QPushButton("选择…")
        self.private_key_button.setObjectName("browse_upload_private_key")
        self.private_key_button.clicked.connect(self._choose_private_key)
        key_layout.addWidget(self.private_key_button)
        endpoint_form.addRow("SSH 私钥：", key_row)

        self.passphrase_edit = QLineEdit()
        self.passphrase_edit.setObjectName("upload_private_key_passphrase")
        self.passphrase_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.passphrase_edit.setPlaceholderText("可选；仅本次使用并立即清空")
        endpoint_form.addRow("私钥口令：", self.passphrase_edit)
        outer.addWidget(endpoint_group)
        self._apply_authentication_mode()

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
        self.start_button.setText("开始上传")
        self.start_button.setObjectName("start_offline_upload")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.host_edit.setFocus()

    def take_request(self, dataset_root: Path) -> OfflineUploadRequest:
        """Build the ephemeral request and immediately clear the UI password."""

        password = self.password_edit.text()
        passphrase = self.passphrase_edit.text()
        private_key_path = self.private_key_edit.text().strip()
        use_private_key = self.authentication_combo.currentData() == "PRIVATE_KEY"
        try:
            return OfflineUploadRequest(
                dataset_root=dataset_root,
                manifest_path=self._manifest_path,
                host=self.host_edit.text(),
                port=self.port_spin.value(),
                username=self.username_edit.text(),
                remote_workdir=self.remote_workdir_edit.text(),
                password=(None if use_private_key else password),
                private_key_path=(Path(private_key_path) if use_private_key else None),
                private_key_passphrase=(
                    passphrase or None if use_private_key else None
                ),
            )
        finally:
            self._clear_secrets()

    def _apply_authentication_mode(self) -> None:
        use_private_key = self.authentication_combo.currentData() == "PRIVATE_KEY"
        self.password_edit.setEnabled(not use_private_key)
        self.private_key_edit.setEnabled(use_private_key)
        self.private_key_button.setEnabled(use_private_key)
        self.passphrase_edit.setEnabled(use_private_key)
        self._clear_secrets()

    def _choose_private_key(self) -> None:
        selected, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择 SSH 私钥",
            str(Path.home() / ".ssh"),
            "SSH 私钥 (*)",
        )
        if selected:
            self.private_key_edit.setText(selected)

    def _clear_secrets(self) -> None:
        self.password_edit.clear()
        self.passphrase_edit.clear()

    def reject(self) -> None:
        self._clear_secrets()
        super().reject()


class UploadProgressDialog(QDialog):
    """Non-blocking upload status view with a controlled cancel request."""

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SSH/SCP 上传进度")
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


__all__ = ["OfflineUploadDialog", "UploadProgressDialog"]
