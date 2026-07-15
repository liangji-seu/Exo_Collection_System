# Exo Collection System

外骨骼多模态数据采集系统的第二版实现。一个仓库提供两个桌面应用：

- **Exo Collector**：设备检查、工况锁定、采集、实时预览与 Trial 最终化；
- **Exo Data Studio**：本地数据树、统计、质量审核、离线回放和人工离线上传入口。

两者共享 `exo_collection` 核心包。架构和数据契约以 [ARCHITECTURE.md](ARCHITECTURE.md) 为准。

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

运行模拟采集和两个桌面入口：

```powershell
exo-simulate-trial --data-root .\runtime_data --duration 3
exo-collector --data-root .\runtime_data
exo-data-studio --data-root .\runtime_data
```

无显示器环境可使用 `--smoke-test` 验证两个 UI 能创建并正常退出。

执行完整的“独立 Worker → 四模态模拟采集 → 原子最终化”启动检查：

```powershell
.\.venv\Scripts\python.exe -m exo_collection.apps.collector.main `
  --collect-smoke-test --data-root .\runtime_data --duration 0.5
```

也可以不激活虚拟环境，直接使用模块入口：

```powershell
.\.venv\Scripts\python.exe -m exo_collection.apps.collector.main --data-root .\runtime_data
.\.venv\Scripts\python.exe -m exo_collection.apps.data_studio.main --data-root .\runtime_data
```

构建两个 Windows 可执行入口：

```powershell
.\packaging\build_windows.ps1
```

冻结版启动与采集检查：

```powershell
.\dist\ExoCollector.exe --smoke-test --data-root .\runtime_data
.\dist\ExoDataStudio.exe --smoke-test --data-root .\runtime_data
.\dist\ExoCollector.exe --collect-smoke-test `
  --data-root .\runtime_data --duration 0.5
```

采集期间不要运行全盘校验、回放或上传；Data Studio 检测到 Collector
活动租约后会自动进入轻量模式。`.recording` Trial 只能通过显式恢复流程检查，
不会被 Data Studio 当成已最终化数据打开。
