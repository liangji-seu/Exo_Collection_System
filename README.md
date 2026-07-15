# Exo Collection System

外骨骼多模态数据采集系统的第二版实现。一个仓库提供两个桌面应用：

- **Exo Collector**：设备检查、工况锁定、采集、实时预览与 Trial 最终化；
- **Exo Data Studio**：本地数据树、统计、质量审核、离线回放和人工离线上传入口。

两者共享 `exo_collection` 核心包。架构和数据契约以 [ARCHITECTURE.md](ARCHITECTURE.md) 为准。

## 日常运行（推荐，无需命令行参数）

完成一次 Windows 打包后，直接双击项目根目录中的脚本：

- `Run_ExoCollector.cmd`：启动采集端；
- `Run_ExoDataStudio.cmd`：启动数据管理端。

也可以在 PowerShell 中运行相同脚本：

```powershell
.\Run_ExoCollector.cmd
.\Run_ExoDataStudio.cmd
```

脚本会根据自身位置寻找 `dist` 中的程序，因此从其他工作目录运行、或项目路径中包含空格和中文时也不需要修改路径。它们不会向桌面应用传入任何命令行参数。

成品版 `Run_*.cmd` 会异步启动 GUI；如需查看源码日志并让脚本返回应用退出码，请使用下文的 `Run_*_From_Source.cmd`。

首次启动后，在 UI 的“数据根目录”处点击“选择…”指定数据目录。该选择会保存在当前 Windows 用户的应用设置中，后续启动自动作为默认目录；需要更换时仍在 UI 中重新选择。Collector 与 Data Studio 应使用同一个数据根目录。

## Windows 开发环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,packaging]"
```

运行测试：

```powershell
python -m pytest
```

## 从源码运行（开发调试）

无需激活虚拟环境，也无需给应用传参数：

```powershell
.\Run_ExoCollector_From_Source.cmd
.\Run_ExoDataStudio_From_Source.cmd
```

源码脚本会检查 `.venv`、Python 3.11 和必要依赖，并在缺少环境时显示可直接执行的修复命令。源码运行时会保留终端窗口，便于查看调试输出。

## 编译打包 Windows 可执行文件

推荐使用项目根目录的统一入口：

```powershell
.\Build_Windows.cmd
```

该入口内部调用 `packaging\build_windows.ps1`。如需直接调用 PowerShell 脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\build_windows.ps1
```

构建脚本会检查 `.venv` 是否使用 Python 3.11、PyInstaller 是否已安装，以及每一步的退出码。成功后生成：

```text
dist\ExoCollector.exe
dist\ExoDataStudio.exe
```

随后使用本页开头的两个 `Run_*.cmd` 即可日常启动。

采集期间不要运行全盘校验、回放或上传；Data Studio 检测到 Collector
活动租约后会自动进入轻量模式。`.recording` Trial 只能通过显式恢复流程检查，
不会被 Data Studio 当成已最终化数据打开。
