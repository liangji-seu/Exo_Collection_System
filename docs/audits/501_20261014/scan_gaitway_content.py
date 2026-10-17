"""Read only time/Fz columns; preserve raw files and enrich the file inventory."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3] / 'Exo_Collection_Data' / '501'
results = []
for path in sorted(ROOT.rglob('501_*.txt')):
    with path.open(encoding='utf-8-sig') as handle:
        header = next(i for i, line in enumerate(handle) if line.startswith('Time (s)\t'))
    names = ['Time (s)', 'GRFz vertical (N)', 'FzL(N)', 'FzR(N)']
    frame = pd.read_csv(path, sep='\t', skiprows=header, usecols=names)
    time = frame[names[0]].dropna()
    duration = float(time.iloc[-1] - time.iloc[0]) if len(time) else 0.0
    fractions = [float(np.isfinite(frame[name]).mean()) for name in names[2:]]
    fz = frame[names[1]]
    flags = []
    if duration < 1.0:
        flags.append('SHORT_LT_1S')
    if fractions == [0.0, 0.0]:
        flags.append('BILATERAL_FZ_ALL_MISSING')
    if not np.isfinite(fz).any() or fz.max() < 100:
        flags.append('NO_BODYWEIGHT_LOADING')
    results.append(dict(session=str(path.parent.relative_to(ROOT)), gaitway_bytes=path.stat().st_size,
                        gaitway_data_rows=len(frame), gaitway_duration_s=duration,
                        gaitway_fz_min_N=float(fz.min()), gaitway_fz_max_N=float(fz.max()),
                        gaitway_left_fz_finite_fraction=fractions[0], gaitway_right_fz_finite_fraction=fractions[1],
                        gaitway_content_status='|'.join(flags) if flags else 'BASIC_CHECK_OK_NOT_SOLVE_QC'))
assert len(results) == 43, f'Unexpected formal TXT count: {len(results)}'
lookup = {row['session']: row for row in results}
inventory = OUT / 'inventory.csv'
with inventory.open(encoding='utf-8-sig', newline='') as handle:
    reader = csv.DictReader(handle)
    fields, rows = list(reader.fieldnames), list(reader)
for key in results[0]:
    if key != 'session' and key not in fields:
        fields.append(key)
for row in rows:
    row.update({key: value for key, value in lookup.get(row['relative_session_path'], {}).items() if key != 'session'})
with inventory.open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(OUT / 'gaitway_content_audit.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
bad = [row for row in results if row['gaitway_content_status'] != 'BASIC_CHECK_OK_NOT_SOLVE_QC']
heading = '\n\n## 正式 Gaitway TXT 内容初筛补充\n'
summary = heading + f'\n一次性读取43个正式TXT的时间、总垂直力和双侧垂直力列；{43-len(bad)}个通过基本完整性检查，{len(bad)}个异常。这不替代同步、marker、IK或力矩质量验收。\n\n'
for row in bad:
    summary += f"- `{row['session']}`：{row['gaitway_data_rows']}行、{row['gaitway_duration_s']} s；{row['gaitway_content_status']}。须重新导出完整测力台文件并核对配对。\n"
readme = OUT / 'README.md'
readme.write_text(readme.read_text(encoding='utf-8').split(heading)[0] + summary, encoding='utf-8')
print(json.dumps(dict(files_scanned=len(results), basic_check_ok=43-len(bad), anomalies=bad), ensure_ascii=False))
