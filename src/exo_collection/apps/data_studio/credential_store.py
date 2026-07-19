"""Native Windows Credential Manager storage for Data Studio SSH passwords."""

from __future__ import annotations

import ctypes
import hashlib
import sys
from ctypes import wintypes


SERVICE_NAME = "ExoCollectionSystem.DataStudio.SSH"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168


class _CredentialW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def credential_account(host: str, port: int, username: str) -> str:
    identity = f"{host.strip().casefold()}:{int(port)}:{username.strip()}"
    if not host.strip() or not username.strip():
        raise ValueError("主机和用户名不能为空。")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{SERVICE_NAME}/{digest}"


def _api() -> tuple[object, object, object, object]:
    if sys.platform != "win32":
        raise RuntimeError("保存 SSH 密码只支持 Windows 凭据管理器。")
    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    cred_write = advapi32.CredWriteW
    cred_write.argtypes = [ctypes.POINTER(_CredentialW), wintypes.DWORD]
    cred_write.restype = wintypes.BOOL
    cred_read = advapi32.CredReadW
    cred_read.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_CredentialW)),
    ]
    cred_read.restype = wintypes.BOOL
    cred_delete = advapi32.CredDeleteW
    cred_delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    cred_delete.restype = wintypes.BOOL
    cred_free = advapi32.CredFree
    cred_free.argtypes = [ctypes.c_void_p]
    cred_free.restype = None
    return cred_write, cred_read, cred_delete, cred_free


def load_password(host: str, port: int, username: str) -> str | None:
    if not host.strip() or not username.strip():
        return None
    _write, read, _delete, free = _api()
    pointer = ctypes.POINTER(_CredentialW)()
    if not read(credential_account(host, port, username), _CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):  # type: ignore[operator]
        error = ctypes.get_last_error()
        if error == _ERROR_NOT_FOUND:
            return None
        raise RuntimeError(f"无法读取 Windows 凭据管理器：{ctypes.WinError(error)}")
    try:
        credential = pointer.contents
        blob = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
        return blob.decode("utf-16-le")
    finally:
        free(pointer)  # type: ignore[operator]


def save_password(host: str, port: int, username: str, password: str) -> None:
    if not password:
        raise ValueError("密码不能为空。")
    write, _read, _delete, _free = _api()
    blob = password.encode("utf-16-le")
    buffer = ctypes.create_string_buffer(blob)
    credential = _CredentialW()
    credential.Type = _CRED_TYPE_GENERIC
    credential.TargetName = credential_account(host, port, username)
    credential.CredentialBlobSize = len(blob)
    credential.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    credential.Persist = _CRED_PERSIST_LOCAL_MACHINE
    credential.UserName = username.strip()
    if not write(ctypes.byref(credential), 0):  # type: ignore[operator]
        error = ctypes.get_last_error()
        raise RuntimeError(f"无法保存到 Windows 凭据管理器：{ctypes.WinError(error)}")


def delete_password(host: str, port: int, username: str) -> None:
    if not host.strip() or not username.strip():
        return
    _write, _read, delete, _free = _api()
    if not delete(credential_account(host, port, username), _CRED_TYPE_GENERIC, 0):  # type: ignore[operator]
        error = ctypes.get_last_error()
        if error != _ERROR_NOT_FOUND:
            raise RuntimeError(f"无法删除 Windows 凭据：{ctypes.WinError(error)}")


__all__ = [
    "SERVICE_NAME",
    "credential_account",
    "delete_password",
    "load_password",
    "save_password",
]
