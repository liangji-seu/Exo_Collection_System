# Windows installer

安装器应同时包含 `ExoCollector.exe` 与 `ExoDataStudio.exe`。第一里程碑使用相邻的 PyInstaller spec 生成两个入口：

```powershell
.\packaging\build_windows.ps1
```

该脚本只构建两个本地可执行入口；正式安装器、代码签名和升级策略在现场发布阶段完善。
