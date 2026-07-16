# Windows 发布产物

项目使用一个仓库构建两个桌面应用，并将它们放进同一个发布包：

- `ExoCollector.exe`
- `ExoDataStudio.exe`

推荐从项目根目录运行零参数入口：

```powershell
.\Build_Windows.cmd
```

默认构建链路严格按以下顺序执行：

1. 运行完整 `pytest` 测试套件；
2. 使用 `packaging/collector.spec` 构建 Collector；
3. 使用 `packaging/data_studio.spec` 构建 Data Studio；
4. 对冻结后的 Collector 和 Data Studio 执行 offscreen UI smoke check，其中
   Collector 验证 spawn 设备预检，Data Studio 验证 Catalog 与管理索引 spawn；
5. 对冻结后的 Collector 执行一次短模拟 Trial，验证 Windows `spawn` Worker；
6. 在 `release/` 生成单一 Windows x64 ZIP、全 payload SHA-256 构建清单和
   ZIP `.sha256` 外部校验文件，并重新打开 ZIP 逐条验证；
7. 若检测到 Inno Setup 6，则额外编译可选安装器。

主要输出为：

```text
release/ExoCollectionSystem-<version>-windows-x64.zip
release/ExoCollectionSystem-<version>-windows-x64.zip.sha256
```

ZIP 内同时包含两个 EXE、两个零参数启动脚本、快速说明和
`BUILD_MANIFEST.json`。清单记录 Git commit/dirty 状态、精确构建环境和全部运行文件
的大小与 SHA-256。即使电脑未安装 Inno Setup，该 ZIP 也是完整的可分发产物。

可选安装器定义位于 `ExoCollectionSystem.iss`。安装 Inno Setup 6 后再次运行
统一构建脚本，将额外得到：

```text
release/ExoCollectionSystem-<version>-Setup.exe
```

默认正式构建还要求可验证的干净 Git checkout；构建前后 HEAD/工作树状态必须
一致。仅开发诊断可显式使用 `-AllowDirtyWorkingTree`，清单将如实标记为 dirty。

构建所需条件：64 位 Windows、64 位 Python 3.11、Git、项目 `.venv`，以及
`.[dev,packaging]` 依赖。新电脑可直接运行 `First_Time_Setup.cmd` 自动建立环境并
执行同一条完整发布链路。

当前发布只验证内置模拟设备和可替换 Adapter 接口；没有真实厂商 SDK、协议和
服务器部署参数，因此不得将构建成功描述为真实硬件已经可用。正式现场发布仍需
补充代码签名、已验证的硬件 Adapter 和实验室部署验收。
