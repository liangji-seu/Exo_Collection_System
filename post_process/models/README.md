# models/

放 OpenSim 模型与 marker 配置。**这些文件需要你后续提供官方版本，我不自行从
不明来源下载。**

```
models/
  gait2392_generic.osim        # [BLOCKING] 官方 gait2392 通用模型（用户提供）
  marker_sets/
    hh19_markerset.xml         # [BLOCKING] HH19 → OpenSim MarkerSet 映射
  scale/
    hh19_scale_setup_template.xml  # [BLOCKING] Scale measurement 定义
```

## BLOCKING 说明

1. **gait2392_generic.osim 缺失** → Scale 无法执行。请提供官方模型文件
   （OpenSim 自带 gait2392_simbody.osim，或 Rajagopal 全身模型）。
2. **HH19 → gait2392 的 marker mapping / measurement**：不可仅凭 marker 名字
   相似就自动推导。需要确认哪些 HH19 marker 映射到 gait2392 的哪些 marker，
   以及用哪些静态 marker 定义 measurement（骨段长度）。
3. 提供后把这些路径填进 `configs/pipeline_template.yaml` 的
   `files.generic_model.path` 与 `marker.mapping`。
