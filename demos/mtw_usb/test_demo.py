"""Hardware-free checks of discovery, strict transport/rate gates and independent persistence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import run_demo as demo


class FakeArray(list):
    def size(self): return len(self)


class FakeId:
    def __init__(self, text, mtw=True): self.text,self.mtw=text,mtw
    def toXsString(self): return self.text
    def isMtw(self): return self.mtw


class FakePort:
    def __init__(self, did): self.did=did
    def deviceId(self): return FakeId(self.did)
    def portName(self): return 'COM_'+self.did
    def baudrate(self): return 115200


class Packet:
    def __init__(self, counter=0): self.counter=counter
    def containsPacketCounter(self): return True
    def packetCounter(self): return self.counter
    def containsSampleTimeFine(self): return True
    def sampleTimeFine(self): return self.counter*100
    def containsCalibratedData(self): return True
    def containsOrientation(self): return True
    def calibratedAcceleration(self): return [0,0,9.81]
    def calibratedGyroscopeData(self): return [1,2,3]
    def calibratedMagneticField(self): return [.1,.2,.3]
    def orientationEuler(self): return SimpleNamespace(x=lambda:0,y=lambda:0,z=lambda:0)
    def orientationQuaternion(self): return [1,0,0,0]


class Device:
    def __init__(self, did, rate=100, connected=2, count=8):
        self.did,self.rate,self.connected,self.count=did,rate,connected,count
        self.callback=None
    def connectivityState(self): return self.connected
    def gotoConfig(self): return True
    def setOptions(self,*args): pass
    def supportedUpdateRates(self): return FakeArray([self.rate])
    def setUpdateRate(self,rate): return rate==self.rate
    def updateRate(self): return self.rate
    def setOutputConfiguration(self,cfg): return True
    def productCode(self): return 'FAKE_TEST_DEVICE'
    def addCallbackHandler(self,cb): self.callback=cb
    def removeCallbackHandler(self,cb): self.callback=None
    def gotoMeasurement(self):
        for k in range(self.count): self.callback.onLiveDataAvailable(self,Packet(k))
        return True


class FakeOutputConfigArray(list):
    def push_back(self, item): self.append(item)


def fake(devices):
    byid={d.did:d for d in devices}
    control=SimpleNamespace(openPort=lambda *a:True,device=lambda did:byid[did.toXsString()],close=lambda:None)
    return SimpleNamespace(XsScanner_scanPorts=lambda:FakeArray([FakePort(d.did) for d in devices]),
        XsControl_construct=lambda:control,XsCallback=object,XsDataPacket=lambda p:p,
        XCS_PluggedIn=2,XSO_Orientation=1,XSO_Calibrate=2,
        XsOutputConfigurationArray=FakeOutputConfigArray,
        XsOutputConfiguration=lambda data_id, freq: (data_id, freq),
        XDI_PacketCounter=1, XDI_SampleTimeFine=2, XDI_Acceleration=3,
        XDI_RateOfTurn=4, XDI_MagneticField=5, XDI_EulerAngles=6)


class Tests(unittest.TestCase):
    def config(self):
        return {'sensors':{'left_leg':'10B42626','right_leg':'10B4260D','pelvis':'10B4261F'},
                'sample_rate_hz':100,'duration_s':.02,'no_data_timeout_s':.1,'queue_capacity':100,'output_dir':'output'}

    def execute(self, devices, config=None):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            result=demo.collect(fake(devices),config or self.config(),p,lambda s:None)
            disk=json.loads((p/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result,disk)
            rows={}
            for f in p.glob('*.csv'):
                with f.open(encoding='utf-8-sig') as stream:
                    rows[f.name]=list(csv.DictReader(stream))
        return result,rows

    def test_three_independent_streams(self):
        cfg=self.config()
        result,rows=self.execute([Device(did,count=n) for did,n in zip(cfg['sensors'].values(),[8,3,6])])
        self.assertEqual(result['status'],'FINISHED')
        self.assertEqual(sorted(map(len,rows.values())),[3,6,8])
        self.assertFalse(result['cross_device_hardware_sync_verified'])

    def test_partial_match_auto_detects_subset(self):
        # 只插了一颗（left_leg），应自动识别为这三颗之一并只采这一颗，而不是报缺。
        result,rows=self.execute([Device('10B42626')])
        self.assertEqual(result['status'],'FINISHED')
        self.assertEqual(result['matched_sensor_names'],['left_leg'])
        self.assertEqual(set(result['missing_sensor_names']),{'right_leg','pelvis'})
        self.assertEqual(list(rows.keys()),['left_leg_10B42626.csv'])

    def test_no_configured_sensor_found(self):
        # 插的是三颗之外的陌生 MTw，一颗都不匹配，才报错。
        result,_=self.execute([Device('FFFFFFFF')])
        self.assertEqual(result['status'],'FAILED')
        self.assertIn('未找到任何配置',result['errors'][0])

    def test_wireless_rejected(self):
        result,_=self.execute([Device(did,connected=3) for did in self.config()['sensors'].values()])
        self.assertEqual(result['status'],'FAILED')
        self.assertIn('XCS_PluggedIn',result['errors'][0])

    def test_wrong_rate_rejected(self):
        result,_=self.execute([Device(did,rate=80) for did in self.config()['sensors'].values()])
        self.assertEqual(result['status'],'FAILED')
        self.assertIn('不支持 100',result['errors'][0])

    def test_queue_overflow_fails_loudly(self):
        cfg=self.config();cfg['queue_capacity']=1
        result,_=self.execute([Device(did) for did in cfg['sensors'].values()],cfg)
        self.assertEqual(result['status'],'FAILED')
        self.assertTrue(any('回调' in e for e in result['errors']))

    def test_no_data(self):
        cfg=self.config();cfg['duration_s']=.12;cfg['no_data_timeout_s']=.02
        result,_=self.execute([Device(did,count=0) for did in cfg['sensors'].values()],cfg)
        self.assertEqual(result['status'],'FAILED')
        self.assertTrue(any('无数据' in e for e in result['errors']))

    def test_counter_wrap_and_initial_offset(self):
        s=demo.CounterStats()
        for v in [65534,65535,0,2,2,1]:s.observe(v)
        self.assertEqual(s.missing,1)
        self.assertEqual(s.duplicates,1)
        self.assertEqual(s.reset_or_out_of_order,1)
        self.assertIsNone(demo.CounterStats().report()['inferred_missing_from_forward_counter_jumps'])

    def test_validate_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';c=self.config();c['sensors']['pelvis']='10B42626'
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):demo.read_config(p)


if __name__=='__main__':unittest.main()
