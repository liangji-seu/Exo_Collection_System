"""User-invoked installation from the existing local SDK wheel, without network."""
from pathlib import Path
import struct
import subprocess
import sys

if __name__ == '__main__':
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    arch = 'x64' if struct.calcsize('P') == 8 else 'Win32'
    root = Path(__file__).resolve().parents[3] / 'Exo_data_capture_system/MT SDK/Python' / arch
    wheels = sorted(root.glob(f'xsensdeviceapi-*-{tag}-*.whl'))
    print(f'Python: {sys.executable}\nVersion: {sys.version}\nSDK: {root}', flush=True)
    if not wheels:
        print('找不到与当前 Python 匹配的本地 wheel。请使用 Python 3.11 x64 的 EXO 环境，或安装匹配版本的 Xsens SDK。')
        raise SystemExit(2)
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',str(wheels[-1])],check=True)
    subprocess.run([sys.executable,'-c','import xsensdeviceapi; print("Xsens SDK import OK")'],check=True)
