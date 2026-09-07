"""Independent MTw USB-only logger. No GUI, wireless master, interpolation or cross-device joining."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import queue
import re
import sys
import threading
import time
import traceback
from datetime import datetime

HERE = Path(__file__).resolve().parent
FIELDS = ['arrival_index', 'host_monotonic_ns', 'host_utc_ns', 'elapsed_host_s',
          'packet_counter', 'sample_time_fine', 'has_calibrated', 'has_orientation',
          'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',
          'mag_x', 'mag_y', 'mag_z', 'roll_deg', 'pitch_deg', 'yaw_deg',
          'quat_w', 'quat_x', 'quat_y', 'quat_z']


def normal_id(value):
    result = str(value).strip().upper().removeprefix('0X')
    if not re.fullmatch(r'[0-9A-F]{8,16}', result):
        raise ValueError(f'请填写完整的 8–16 位十六进制 MTw ID：{value!r}')
    return result


def read_config(path):
    c = json.loads(path.read_text(encoding='utf-8-sig'))
    sensors = c.get('sensors', {})
    if not 1 <= len(sensors) <= 3:
        raise ValueError('sensors 必须包含 1–3 颗设备；正式三颗测试请保留三个条目。')
    for name in sensors:
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError('位置名称只允许英文字母、数字和下划线。')
    c['sensors'] = {name: normal_id(did) for name, did in sensors.items()}
    if len(set(c['sensors'].values())) != len(sensors):
        raise ValueError('设备 ID 不可重复。')
    if c.get('sample_rate_hz') != 100:
        raise ValueError('此 demo 固定测试 100 Hz，不自动降频。')
    for key in ('duration_s', 'no_data_timeout_s'):
        if not isinstance(c.get(key), (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} 必须是有限正数。')
    if not isinstance(c.get('queue_capacity'), int) or c['queue_capacity'] < 1:
        raise ValueError('queue_capacity 必须是正整数。')
    return c


def select_ports(xda, sensors, announce=print):
    inventory, direct = [], {}
    ports = xda.XsScanner_scanPorts()
    for index in range(ports.size()):
        p = ports[index]
        did = str(p.deviceId().toXsString()).upper()
        mtw = bool(p.deviceId().isMtw())
        row = {'id': did, 'port': str(p.portName()), 'baudrate': int(p.baudrate()), 'is_mtw': mtw}
        inventory.append(row)
        announce(f'发现 {did}  {row["port"]}  '+('MTw 直连候选' if mtw else '非直连 MTw，忽略'))
        if mtw:
            if did in direct:
                raise RuntimeError(f'扫描出现重复 ID {did}，请检查连接。')
            direct[did] = p
    matched = [(name, direct[did]) for name, did in sensors.items() if did in direct]
    missing = [f'{name}={did}' for name, did in sensors.items() if did not in direct]
    if not matched:
        found = ', '.join(sorted(direct)) if direct else '（无 MTw）'
        raise RuntimeError(
            '未找到任何配置的 USB MTw。'
            + f'已发现 MTw：{found}；配置：'
            + ', '.join(f'{name}={did}' for name, did in sensors.items())
            + '。请拔掉 Dongle，用数据线直连；关闭 Collector/MT Manager，检查驱动和 ID。'
        )
    if missing:
        announce(
            f'自动匹配：找到 {len(matched)}/{len(sensors)} 颗'
            f'（{", ".join(name for name, _ in matched)}），'
            f'缺少 {", ".join(missing)}，仅采集已连接的设备。'
        )
    return matched, inventory


class CounterStats:
    def __init__(self):
        self.last = None
        self.missing = 0
        self.duplicates = 0
        self.reset_or_out_of_order = 0
        self.known = 0

    def observe(self, value):
        if value is None:
            return
        self.known += 1
        if self.last is not None:
            delta = (value-self.last) % 65536
            if delta == 0:
                self.duplicates += 1
            elif delta < 32768:
                self.missing += delta-1
            else:
                self.reset_or_out_of_order += 1
        self.last = value

    def report(self):
        return {'counter_available_rows': self.known,
                'inferred_missing_from_forward_counter_jumps': self.missing if self.known >= 2 else None,
                'duplicate_counters': self.duplicates,
                'counter_reset_or_out_of_order': self.reset_or_out_of_order}


def unpack_packet(packet):
    pc = int(packet.packetCounter()) if packet.containsPacketCounter() else None
    fine = int(packet.sampleTimeFine()) if packet.containsSampleTimeFine() else None
    calibrated = bool(packet.containsCalibratedData())
    orientation = bool(packet.containsOrientation())
    values = [math.nan]*16
    if calibrated:
        for offset, v in ((0,packet.calibratedAcceleration()),
                          (3,packet.calibratedGyroscopeData()), (6,packet.calibratedMagneticField())):
            values[offset:offset+3] = [float(v[i]) for i in range(3)]
    if orientation:
        e = packet.orientationEuler()
        values[9:12] = [float(e.x()),float(e.y()),float(e.z())]
        q = packet.orientationQuaternion()
        values[12:16] = [float(q[i]) for i in range(4)]
    return pc, fine, calibrated, orientation, values


def collect(xda, config, out, announce=print):
    summary = {'status':'STARTING', 'config':config, 'transport':'direct_usb_only',
               'cross_device_hardware_sync_verified':False, 'sensors':{}, 'errors':[],
               'matched_sensor_names':[], 'missing_sensor_names':[],
               'timing_note':'host timestamps are callback arrival times, NOT simultaneous sample times',
               'units':{'acc':'m/s2','gyr':'rad/s','mag':'SDK native a.u.',
                        'euler':'degree','quaternion':'wxyz','sample_time_fine':'raw SDK ticks'}}
    control, opened, callbacks, files = None, [], [], []
    packets = queue.Queue(maxsize=config['queue_capacity'])
    fatal = threading.Event()
    accepting = threading.Event()
    accepting.set()
    start_ns = time.perf_counter_ns()
    stats, writers = {}, {}
    callback_failures = []

    class Callback(xda.XsCallback):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def onLiveDataAvailable(self, device, packet):
            if not accepting.is_set():
                return
            mono, utc = time.perf_counter_ns(), time.time_ns()
            try:
                packets.put_nowait((self.name, mono, utc, xda.XsDataPacket(packet)))
            except Exception as exc:
                callback_failures.append(f'{self.name}: 回调队列/复制失败 {type(exc).__name__}: {exc}')
                fatal.set()

    def save(item):
        name, mono, utc, packet = item
        pc, fine, calibrated, orientation, values = unpack_packet(packet)
        s = stats[name]
        writers[name].writerow([s['rows'],mono,utc,(mono-start_ns)/1e9,pc,fine,
                               int(calibrated),int(orientation),*values])
        s['rows'] += 1
        s['first_ns'] = mono if s['first_ns'] is None else s['first_ns']
        s['last_ns'] = mono
        s['counter'].observe(pc)
        if calibrated and orientation and all(math.isfinite(v) for v in values):
            s['complete_rows'] += 1

    try:
        selected, summary['scan_inventory'] = select_ports(xda,config['sensors'],announce)
        summary['matched_sensor_names'] = [name for name, _ in selected]
        summary['missing_sensor_names'] = [name for name in config['sensors']
                                           if name not in {n for n, _ in selected}]
        control = xda.XsControl_construct()
        if not control:
            raise RuntimeError('XsControl_construct 失败')
        for name,p in selected:
            if not control.openPort(p.portName(),p.baudrate()):
                raise RuntimeError(f'{name} 端口 {p.portName()} 打开失败；可能被其他程序占用。')
            dev = control.device(p.deviceId())
            if not dev:
                raise RuntimeError(f'{name} 设备对象不可用')
            opened.append((name,dev,p))
            if dev.connectivityState() != xda.XCS_PluggedIn:
                raise RuntimeError(f'{name} SDK 未确认有线状态 XCS_PluggedIn，拒绝继续。')
            if not dev.gotoConfig():
                raise RuntimeError(f'{name} 不能进入配置模式')
            dev.setOptions(xda.XSO_Orientation | xda.XSO_Calibrate, 0)
            output = xda.XsOutputConfigurationArray()
            for data_id, freq in (
                (xda.XDI_PacketCounter, 0),
                (xda.XDI_SampleTimeFine, 0),
                (xda.XDI_Acceleration, 100),
                (xda.XDI_RateOfTurn, 100),
                (xda.XDI_MagneticField, 100),
                (xda.XDI_EulerAngles, 100),
            ):
                output.push_back(xda.XsOutputConfiguration(data_id, freq))
            if not dev.setOutputConfiguration(output):
                raise RuntimeError(f'{name} 设置输出配置失败')
            supported_raw = dev.supportedUpdateRates()
            supported = [int(supported_raw[i]) for i in range(supported_raw.size())]
            if 100 not in supported:
                raise RuntimeError(f'{name} 不支持 100 Hz；设备报告：{supported}')
            if not dev.setUpdateRate(100) or int(dev.updateRate()) != 100:
                raise RuntimeError(f'{name} 设置/读回 100 Hz 失败')
            summary['sensors'][name] = {'id':config['sensors'][name], 'port':str(p.portName()),
                'product':str(dev.productCode()), 'supported_rates':supported, 'reported_rate_hz':int(dev.updateRate())}
            s = {'rows':0,'complete_rows':0,'first_ns':None,'last_ns':None,'counter':CounterStats()}
            stats[name] = s
            f = (out/f'{name}_{config["sensors"][name]}.csv').open('w',newline='',encoding='utf-8-sig')
            files.append(f)
            writers[name] = csv.writer(f)
            writers[name].writerow(FIELDS)
            callback = Callback(name)
            callbacks.append((dev,callback))
            dev.addCallbackHandler(callback)
            announce(f'{name} {config["sensors"][name]}：USB 有线确认，100 Hz 配置确认')
        for name,dev,_ in opened:
            if not dev.gotoMeasurement():
                raise RuntimeError(f'{name} 无法进入测量模式')
        start = time.monotonic()
        next_report = start
        announce('采集中；Ctrl+C 停止并保存。三颗分别落盘，不按相同计数器强行对齐。')
        while time.monotonic()-start < config['duration_s']:
            if fatal.is_set():
                raise RuntimeError('回调队列异常，停止采集以避免静默丢数据。')
            try:
                save(packets.get(timeout=.05))
            except queue.Empty:
                pass
            now = time.monotonic()
            for name,s in stats.items():
                if s['last_ns'] is None:
                    silent = now-start
                else:
                    silent = (time.perf_counter_ns()-s['last_ns'])/1e9
                if silent > config['no_data_timeout_s']:
                    raise RuntimeError(f'{name} 超过 {config["no_data_timeout_s"]} 秒无数据；CSV 已保留。')
                if now-start > config['no_data_timeout_s'] and s['complete_rows']==0:
                    raise RuntimeError(f'{name} 有包但没有完整姿态/校准数据，请检查 SDK/固件。')
            if now >= next_report:
                announce(' | '.join(f'{name}: {s["rows"]} 行，计数跳缺 {s["counter"].missing}' for name,s in stats.items()))
                for f in files:
                    f.flush()
                next_report = now+2
        summary['status'] = 'FINISHED'
    except KeyboardInterrupt:
        summary['status'] = 'STOPPED_BY_USER'
    except Exception as exc:
        summary['status'] = 'FAILED'
        summary['errors'].append(str(exc))
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        announce(f'错误：{exc}')
    finally:
        for name,dev,_ in reversed(opened):
            try:
                if not dev.gotoConfig():
                    raise RuntimeError('gotoConfig returned false')
            except Exception as exc:
                summary['errors'].append(f'{name} 停止失败：{exc}')
        accepting.clear()
        for dev,cb in callbacks:
            try:
                dev.removeCallbackHandler(cb)
            except Exception as exc:
                summary['errors'].append(f'移除回调失败：{exc}')
        while not packets.empty():
            try:
                save(packets.get_nowait())
            except Exception as exc:
                summary['errors'].append(f'保存剩余数据失败：{exc}')
                break
        for f in files:
            f.close()
        if control:
            try:
                control.close()
            except Exception as exc:
                summary['errors'].append(f'关闭端口失败：{exc}')
        summary['errors'].extend(callback_failures)
        for name,s in stats.items():
            span = (s['last_ns']-s['first_ns'])/1e9 if s['rows']>1 else 0
            summary['sensors'][name].update({'rows':s['rows'],'complete_rows':s['complete_rows'],
                'arrival_span_s':span,'observed_arrival_rate_hz':(s['rows']-1)/span if span>0 else None,
                **s['counter'].report()})
            observed_rate = summary['sensors'][name]['observed_arrival_rate_hz']
            announce(f'{name} 保存 {s["rows"]} 行，完整信号 {s["complete_rows"]} 行，'
                     f'接收频率估计 {observed_rate if observed_rate is not None else "未知"} Hz；'
                     f'计数跳缺 {s["counter"].report()["inferred_missing_from_forward_counter_jumps"]}')
        if summary['errors']:
            summary['status'] = 'FAILED'
        summary['signal_check'] = 'REVIEW_REQUIRED'
        if len(stats)==len(config['sensors']) and all(s['complete_rows']>0 for s in stats.values()):
            summary['signal_check'] = 'ALL_SENSORS_PRODUCED_DATA_NOT_A_SYNC_VALIDATION'
        if summary['status']=='FINISHED' and any(s['complete_rows']==0 for s in stats.values()):
            summary['status']='FAILED'
            summary['errors'].append('至少一颗设备没有产生完整数据。')
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description='MTw 直接 USB 100 Hz 采集 demo')
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--scan-only',action='store_true',help='仅扫描端口，不进入测量')
    args = parser.parse_args()
    config = read_config(args.config)
    try:
        import xsensdeviceapi as xda
    except ImportError as exc:
        print(f'当前 Python：{sys.executable}\n无法导入 xsensdeviceapi：{exc}\n请先运行 install_sdk.bat，或查看 README.md。')
        return 2
    if args.scan_only:
        select_ports(xda,config['sensors'])
        print('指定设备已找到；此处还没有验证有线状态/100 Hz 输出。')
        return 0
    root = Path(config['output_dir'])
    if not root.is_absolute():
        root = args.config.resolve().parent/root
    root.mkdir(parents=True,exist_ok=True)
    out = root/(datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000000:09d}')
    out.mkdir()
    (out/'config_snapshot.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'run.log').open('w',encoding='utf-8') as log:
        def announce(message):
            print(message,flush=True)
            log.write(f'{datetime.now().isoformat()} {message}\n')
            log.flush()
        announce(f'结果目录：{out}')
        result = collect(xda,config,out,announce)
        announce(f'结束：{result["status"]}。详见 summary.json。')
    return 1 if result['status']=='FAILED' else 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'启动失败：{exc}',file=sys.stderr)
        raise SystemExit(2)
