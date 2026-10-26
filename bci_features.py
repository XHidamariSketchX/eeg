"""LSL metadata and sampling-independent features for device metrics."""
import threading
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

LABELS = ['theta_percent','alpha_percent','beta_percent','wear_state','packet_counter','received_lsl_time']


def stream_names(demo=False):
    prefix = 'Gemini-Demo' if demo else 'Gemini'
    return prefix+'-Metrics', prefix+'-TaskMarkers'


class MetricsOutlet:
    def __init__(self, demo=False):
        name, _ = stream_names(demo)
        info = StreamInfo(name, 'DeviceMetrics', 6, 0, 'double64', name)
        info.desc().append_child_value('description', 'Device band percentages; irregular host arrival timestamps, not EEG voltage samples.')
        channels = info.desc().append_child('channels')
        for label in LABELS:
            channels.append_child('channel').append_child_value('label', label)
        self.outlet = StreamOutlet(info)

    def push(self, values):
        stamp = local_clock()
        self.outlet.push_sample([*values[15:18], values[2], values[0], stamp], timestamp=stamp)


class MetricFeatures:
    def __init__(self, window_seconds=8):
        self.window_seconds = window_seconds
        self.lock = threading.Lock()
        self.last_valid = 0
        self.last_status = '等待完整片段'
        self.accepted = self.rejected = 0

    def ProcessFeatures(self, epoch, sr, target=None):
        # sr is an upstream buffer capacity hint, NOT a device sampling rate.
        x = np.asarray(epoch, dtype=float)
        reason = None
        if x.ndim != 2 or x.shape[1] != 6 or len(x) < 5 or not np.isfinite(x).all():
            reason = '片段不足或数据无效'
        elif np.any(x[:,3] != 1):
            reason = '佩戴异常'
        elif np.any(x[:,:3]<0) or np.any(x[:,:3]>100):
            reason = '波段百分比超出范围'
        elif np.any(np.diff(x[:,5])<0) or np.any(np.diff(x[:,5])>.5):
            reason = '片段时间不连续'
        elif x[-1,5]-x[0,5] < self.window_seconds*.75:
            reason = '片段时间覆盖不足'
        elif np.any(np.diff(x[:,4]) % 256 != 1):
            reason = '片段包序号不连续'
        with self.lock:
            if reason:
                self.rejected += 1
                self.last_valid = 0
                self.last_status = '跳过：'+reason
                return None
            self.accepted += 1
            self.last_valid = time.monotonic()
            self.last_status = '有效指标片段'
        metrics = x[:,:3]
        return np.concatenate([metrics.mean(axis=0), metrics.std(axis=0),
                               np.median(metrics,axis=0),metrics[-1]-metrics[0]])

    def snapshot(self):
        with self.lock:
            return self.last_valid, self.last_status, self.accepted, self.rejected
