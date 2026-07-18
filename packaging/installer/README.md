# Windows 发布产物

项目使用一个仓库构建两个桌面应用，并将它们放进同一个发布包：

- `ExoCollector.exe`
- `ExoDataStudio.exe`

新电脑首次使用 64 位系统 CPython 3.11 运行零参数入口：

```powershell
python first_time_setup_and_build.py
```

该脚本不创建虚拟环境，按以下顺序执行：

1. 将项目、测试、打包和硬件依赖安装到当前 Windows 用户的系统 Python；
2. 运行完整 `pytest` 测试套件；
3. 使用 `packaging/collector.spec` 构建 Collector；
4. 使用 `packaging/data_studio.spec` 构建 Data Studio；
5. 将两个 EXE 写入 `dist/`。

主要输出为：

```text
dist/ExoCollector.exe
dist/ExoDataStudio.exe
```

日常重新打包时，已安装依赖的电脑可直接运行 `python build_exe.py`。

可选安装器定义位于 `ExoCollectionSystem.iss`。安装 Inno Setup 6 后可用
Inno Setup Compiler 单独编译该文件；`build_exe.py` 当前只生成两个 EXE。

```text
release/ExoCollectionSystem-<version>-Setup.exe
```

构建所需条件：64 位 Windows、64 位系统 CPython 3.11、Git，以及
`.[dev,packaging,hardware]` 依赖。真实硬件版本还必须在构建前安装
Xsens 官方 wheel、Scapy/pyserial 和 Npcap；首次脚本会显式检查这些条件。

当前发布已提供 Raw Ethernet/Npcap 超声、Xsens MTw 和 Teensy 真实
Adapter，但构建/模拟测试通过不等于物理硬件验收。正式现场发布仍需
完成真机连接、长时间压力测试、同步输入硬件接入和代码签名。
