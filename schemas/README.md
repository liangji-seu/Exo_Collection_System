# Schemas

版本化 JSON Schema 放在此目录，并由共享 Pydantic 模型生成后纳入版本控制。

- `manifest-v1.0.0.json`：首个稳定 Manifest 契约；
- `manifest-v1.1.0.json`：增加 `F`/`T` 项目代码、三位受试者编码及结构化实验元数据。

Reader 保持对 `1.0.0` 的向后兼容。已经发布的 Schema 文件不可原地修改；任何不向后兼容的契约变化必须发布新版本。
