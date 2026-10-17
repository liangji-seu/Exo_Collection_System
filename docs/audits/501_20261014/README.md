# Subject 501 file inventory (2026-10-14)

结论：501 共 55 条采集记录，其中 43 条具备动态 C3D + 正式 Gaitway TXT 候选，静态标定有 1 个 C3D；重复/重采记录保留、不去重。文件齐备只说明采集文件已落盘，不等于解算质量合格。

This is a bounded, read-only inventory of `Exo_Collection_Data/501`.

## Scope and method

- Inspected session directory names, `.exo/manifest.json`, `.exo/quality_report.json`, `.exo/configuration_snapshot.json`, and file presence.
- Gaitway TXT is counted only when it is a session-top-level file matching `501_*.txt`; `.exo/warnings.txt` is recorded separately and is never counted as Gaitway data.
- No C3D/TXT/HDF5/BIN signal contents were decoded, solved, or revalidated.
- `manifest.state=FINALIZED` and `quality_report.required_artifacts_complete=true` are reported as acquisition completion evidence. They do not imply analysis readiness.
- `review_status=待确认` is used when the recorded quality grade is not A or when C3D/ultrasound files are absent. Missing formal Gaitway TXT is recorded as a file-level caveat and is not independently treated as invalid.

## Summary

- Date: `2026-10-14`; day folder: `d1`; sessions: **55**.
- Lifecycle: **55 FINALIZED**, **0 FAILED**, **0 ABORT/abnormal termination** in the manifests.
- Quality reports: **55/55 grade C**; all 55 report `required_artifacts_complete=true` and have `.exo/statistics.json` and `.exo/quality_report.json`.
- File presence across 55 sessions: EMG 55, IMU 55, formal Gaitway TXT 43, C3D 51, mocap H5 51, Xingying trigger JSONL 51, ultrasound BIN 13. The 55 `.exo/warnings.txt` files are report sidecars and are excluded from the Gaitway count.
- The formal Gaitway TXT headers provide subject fields for 43/55 sessions: `Patient weight (kg)=80.000`, `Patient height (m)=1.870`, and `Patient sex=M`. These are header fields, not a new signal parse.
- The Collector configuration snapshot stores `experiment_metadata.subject.{age_years,sex,height_cm,weight_kg,leg_length_cm}`, but all five fields are null in all 55 snapshots. Probe mapping/fixation metadata is also unset in all 55 snapshots. The processing code/docs identify the Gaitway TXT header as the fallback source for weight/height.

## Condition interpretation

All recorded conditions are `NOEXO`. Forward walking conditions are `DWALK_0P6_NOEXO` (7), `DWALK_0P8_NOEXO` (5), `DWALK_1P0_NOEXO` (8), plus `STAND_WALK_STAND_NOEXO` (5) and the start baseline (1). Backward walking conditions are `BWALK_0P2_NOEXO` (7), `BWALK_0P3_NOEXO` (7), `BWALK_0P4_NOEXO` (5), plus `STAND_BWALK_STAND_NOEXO` (8). Static calibration is `STATIC_CALIB` (1).

The experiment note supplied for this audit says that BWALK means the subject turns 180 degrees while treadmill direction stays unchanged. This is recorded as context only; no kinematic or signal-direction calculation was performed.

Formal Gaitway coverage by condition is:

| condition | sessions | C3D | formal Gaitway TXT | C3D + Gaitway |
|---|---:|---:|---:|---:|
| BWALK_0P2_NOEXO | 7 | 7 | 7 | 7 |
| BWALK_0P3_NOEXO | 7 | 7 | 5 | 5 |
| BWALK_0P4_NOEXO | 5 | 5 | 5 | 5 |
| DWALK_0P6_NOEXO | 7 | 5 | 5 | 5 |
| DWALK_0P8_NOEXO | 5 | 5 | 5 | 5 |
| DWALK_1P0_NOEXO | 8 | 8 | 6 | 6 |
| STAND_BWALK_STAND_NOEXO | 8 | 8 | 5 | 5 |
| STAND_WALK_STAND_NOEXO | 5 | 5 | 5 | 5 |
| START_STAND_30S_NOEXO | 1 | 0 | 0 | 0 |
| START_WALK_1P0_30S_NOEXO | 1 | 0 | 0 | 0 |
| STATIC_CALIB | 1 | 1 | 0 | 0 |

The 12 sessions without a formal top-level `501_*.txt` are:

- `d1\F_BASE\START_STAND_30S_NOEXO\session1_20261014_084130`
- `d1\F_BASE\START_WALK_1P0_30S_NOEXO\session1_20261014_084216`
- `d1\F_STEADY\BWALK_0P3_NOEXO\session1_20261014_092713`
- `d1\F_STEADY\BWALK_0P3_NOEXO\session2_20261014_092916`
- `d1\F_STEADY\DWALK_0P6_NOEXO\session1_20261014_084332`
- `d1\F_STEADY\DWALK_0P6_NOEXO\session1_20261014_084452`
- `d1\F_STEADY\DWALK_1P0_NOEXO\session2_20261014_085354`
- `d1\F_STEADY\DWALK_1P0_NOEXO\session3_20261014_091440`
- `d1\F_TRANSIENT\STAND_BWALK_STAND_NOEXO\session2_20261014_100214`
- `d1\F_TRANSIENT\STAND_BWALK_STAND_NOEXO\session2_20261014_100354`
- `d1\F_TRANSIENT\STAND_BWALK_STAND_NOEXO\session2_20261014_100424`
- `d1\T\STATIC_CALIB\session1_20261014_084014`

## Quality and file caveats

All 55 quality reports are grade C. The recurring recorded issue codes are EMG sequence continuity gaps (55 sessions), IMU non-finite values (55 sessions), and ultrasound sequence continuity gaps (13 sessions with ultrasound). These are metadata/report findings, not a new signal-quality assessment by this audit.

The CSV lists exact relative paths for sessions missing C3D, formal Gaitway TXT, or ultrasound. The 4 C3D-missing sessions are the two start-baseline sessions and the first two `DWALK_0P6_NOEXO` attempts. Formal Gaitway TXT is missing in 12 sessions, including `STATIC_CALIB`; therefore the static calibration has C3D but no formal Gaitway TXT. C3D+formal-Gaitway intersection is 43 sessions. Ultrasound is present only for 13 sessions; the absence is left as a file-level caveat and is not converted into an invalidity judgment.

Repeated `condition.repeat_index` values identify likely reacquisition/repeat attempts and are preserved in the CSV. They are not deduplicated.

The accompanying `inventory.csv` is the compact row-level manifest/file-presence list; paths are relative to `Exo_Collection_Data/501`.


## 正式 Gaitway TXT 内容初筛补充

一次性读取43个正式TXT的时间、总垂直力和双侧垂直力列；42个通过基本完整性检查，1个异常。这不替代同步、marker、IK或力矩质量验收。

- `d1\F_STEADY\BWALK_0P3_NOEXO\session1_20261014_093056`：58行、0.057 s；SHORT_LT_1S|BILATERAL_FZ_ALL_MISSING|NO_BODYWEIGHT_LOADING。须重新导出完整测力台文件并核对配对。
