# 系统运行日志

Exo Collector 和 Exo Data Studio 每次启动时都会在本目录创建独立的 UTF-8 日志文件。

- `ExoCollector_日期_时间_pid进程号.log`
- `ExoDataStudio_日期_时间_pid进程号.log`

日志会记录应用启动与退出、设备连接、预览 Worker、Trial、Manifest、告警及未处理异常。密码、Token、API Key 等敏感字段会在写盘前被脱敏。

单个日志文件达到 10 MiB 后自动轮转，历史日志不会在应用启动时自动删除。
