# Windows installer

安装器应同时包含 `ExoCollector.exe` 与 `ExoDataStudio.exe`。第一里程碑使用 PyInstaller 生成两个独立的本地可执行文件。

在项目根目录运行推荐的统一打包入口：

```powershell
.\Build_Windows.cmd
```

`Build_Windows.cmd` 内部调用 `packaging\build_windows.ps1`，并绕过仅针对该进程的 PowerShell 执行策略限制。底层脚本会检查：

- `.venv\Scripts\python.exe` 是否存在且版本为 Python 3.11；
- PyInstaller 是否已经安装；
- 两个 PyInstaller spec 和最终 exe 是否存在；
- 每个外部构建步骤是否成功退出。

成功后输出：

```text
dist\ExoCollector.exe
dist\ExoDataStudio.exe
```

日常运行时，直接双击项目根目录下的 `Run_ExoCollector.cmd` 或 `Run_ExoDataStudio.cmd`，无需传递任何应用命令行参数。首次使用时在 UI 中选择数据根目录；该选择会持久化为当前 Windows 用户的默认值，后续仍可在 UI 中更换。

当前脚本只构建两个本地可执行入口；正式安装器、代码签名和升级策略在现场发布阶段完善。
