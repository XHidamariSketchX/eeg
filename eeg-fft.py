"""双通道输出序列频谱。默认使用 cycles/sample；--fs 50 显式假设输出为 50 Hz。"""
import argparse
from collections import deque
import math
import threading
import time

import numpy as np
import serial


class LineDecoder:
    def __init__(self):
        self.pending = bytearray()
        self.invalid = 0

    def feed(self, chunk):
        self.pending.extend(chunk)
        rows = []
        while b'\n' in self.pending:
            raw, _, self.pending = self.pending.partition(b'\n')
            if not raw.strip():
                continue
            try:
                fields = raw.decode('ascii').strip().split(',')
                if len(fields) != 21:
                    raise ValueError('expected 21 columns')
                values = [float(v) for v in fields]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError('nonfinite input')
                if values[0] != int(values[0]) or not 0 <= values[0] <= 255:
                    raise ValueError('invalid packet counter')
                if values[2] != int(values[2]):
                    raise ValueError('invalid wear state')
                rows.append(values)
            except (ValueError, UnicodeDecodeError):
                self.invalid += 1
        if len(self.pending) > 65536:
            self.pending.clear()
            self.invalid += 1
        return rows


class Acquisition:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.samples = deque(maxlen=args.rows)
        self.arrivals = deque(maxlen=10000)
        self.latest = None
        self.last_time = None
        self.last_counter = None
        self.total = self.invalid = self.discontinuities = 0
        self.state = '等待连接'

    def accept(self, values, stamp):
        with self.lock:
            counter = int(values[0])
            if self.last_counter is not None and (counter - self.last_counter) % 256 != 1:
                self.samples.clear()
                self.discontinuities += 1
            if self.last_time is not None and stamp - self.last_time > 1:
                self.samples.clear()
            self.last_counter, self.last_time = counter, stamp
            self.latest = values
            self.total += 1
            self.arrivals.append(stamp)
            if values[2] == self.args.wear_ok:
                self.samples.append(values[6:8])
            else:
                self.samples.clear()

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            while self.arrivals and self.arrivals[0] < now - 5:
                self.arrivals.popleft()
            if self.last_time is None or now - self.last_time > 1:
                self.samples.clear()
            span = now - self.arrivals[0] if len(self.arrivals) > 1 else 0
            rate = (len(self.arrivals) - 1) / span if span > 0 else 0
            return (list(self.samples), self.latest, rate, self.total, self.invalid,
                    self.discontinuities, self.state, self.last_time)

    def run(self):
        try:
            if self.args.demo:
                with self.lock:
                    self.state = '模拟数据 · 50 点/秒'
                started = time.monotonic()
                counter = 0
                while not self.stop.wait(0.02):
                    # 模拟信号按样本序号生成，已知频率为 10 Hz 和 6 Hz。
                    t = counter / 50
                    v = [0.] * 21
                    v[0], v[2] = counter % 256, self.args.wear_ok
                    v[6:8] = [50 * math.sin(2 * math.pi * 10 * t), 35 * math.sin(2 * math.pi * 6 * t)]
                    v[19:21] = [50 + 10 * math.sin((time.monotonic()-started)/3), 60]
                    self.accept(v, time.monotonic())
                    counter += 1
                return
            with serial.Serial(self.args.port, self.args.baud, timeout=0.1) as ser:
                decoder = LineDecoder()
                with self.lock:
                    self.state = f'已连接 {self.args.port}'
                while not self.stop.is_set():
                    chunk = ser.read(ser.in_waiting or 1)
                    stamp = time.monotonic()
                    for values in decoder.feed(chunk):
                        self.accept(values, stamp)
                    with self.lock:
                        if decoder.invalid != self.invalid:
                            self.samples.clear()
                        self.invalid = decoder.invalid
        except (serial.SerialException, OSError) as exc:
            with self.lock:
                self.state = f'串口错误：{exc}'
            print(self.state, flush=True)
        finally:
            with self.lock:
                self.samples.clear()
            self.stop.set()


def estimate_psd(samples, fs=1.0):
    from scipy.signal import welch
    array = np.asarray(samples, dtype=float)
    return welch(array, fs=fs, nperseg=min(128, len(array)), axis=0, detrend='constant')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--rows', type=int, default=200)
    parser.add_argument('--wear-ok', type=int, default=1, help='正常佩戴状态，默认按现有录制设为 1')
    parser.add_argument('--fs', type=float, help='明确指定的输出采样率假设；不是硬件采样率')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args(argv)
    if args.rows < 32 or args.baud <= 0 or (args.fs is not None and (not math.isfinite(args.fs) or args.fs <= 0)):
        parser.error('rows 至少 32，波特率和 fs 必须为有限正数')

    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(bottom=0.25, top=0.82, wspace=0.3)
    fig.suptitle('双通道输出序列分析' + (' · 模拟数据' if args.demo else ''))
    fs = args.fs if args.fs is not None else 1.0
    ax1.set_title(f'Welch 功率谱 · 假设输出 {fs:g} Hz' if args.fs else 'Welch 功率谱 · 输出采样率未知')
    ax1.set_xlabel('频率 (Hz，基于指定假设)' if args.fs else '归一化频率 (cycles/sample)')
    ax1.set_ylabel('输出值²/Hz' if args.fs else '输出值²/(cycles/sample)')
    ax1.set_xlim(0, fs / 2)
    ax1.grid(alpha=0.3)
    lines = [ax1.plot([], [], label=f'ADC{i+1} 滤波值')[0] for i in range(2)]
    ax1.legend()
    bars = ax2.bar(['设备专注度', '设备放松度'], [0, 0], color=['orange', 'teal'])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('设备评分原值')
    ax2.set_title('设备指标 · 等待数据')
    ax2.grid(axis='y', alpha=0.3)
    status = fig.text(0.06, 0.10, '', fontsize=10)
    fig.text(0.06, 0.035, '接收速率不等于设备采样率；此图用于输出序列探索，不作情绪诊断。', fontsize=10)
    acquisition = Acquisition(args)
    worker = threading.Thread(target=acquisition.run, daemon=True)
    worker.start()
    print('默认正常佩戴状态=1；频谱不会自动把接收速率当作采样率。关闭窗口或 Ctrl+C 退出。', flush=True)
    last_report = 0
    try:
        while plt.fignum_exists(fig.number):
            samples, latest, rate, total, invalid, jumps, state, last_time = acquisition.snapshot()
            fresh = last_time is not None and time.monotonic() - last_time <= 1 and not acquisition.stop.is_set()
            wear = latest[2] if latest else None
            ready = fresh and wear == args.wear_ok and len(samples) == args.rows
            note = '积累数据' if fresh else '等待数据 / 连接中断'
            if fresh and wear != args.wear_ok:
                note = f'佩戴状态 {wear:g}：暂停频谱'
            if ready:
                frequencies, psd = estimate_psd(samples, fs)
                if np.all(np.var(samples, axis=0) > 1e-12):
                    for i, line in enumerate(lines):
                        line.set_data(frequencies, psd[:, i])
                    ax1.relim()
                    ax1.autoscale_view(scalex=False)
                    note = '频谱已更新（输出序列）'
                else:
                    ready = False
                    note = '通道恒定或全零：暂停频谱'
            if not ready:
                for line in lines:
                    line.set_data([], [])
            for bar, value in zip(bars, latest[19:21] if latest else [0, 0]):
                bar.set_height(value)
                bar.set_alpha(1 if fresh and wear == args.wear_ok else 0.25)
            ax2.set_title('设备指标' + (f' · {latest[19]:.2f} / {latest[20]:.2f}' if latest else ' · 等待数据')
                          + ('（当前有效接收）' if fresh and wear == args.wear_ok else '（无有效实时数据）'))
            message = (f'{state} | 接收 {rate:.2f} 行/秒 | 总计 {total} | 无效 {invalid} | 序号不连续 {jumps}\n'
                       f'{note} | 窗口 {len(samples)}/{args.rows} 点')
            status.set_text(message)
            if time.monotonic() - last_report >= 2:
                print(message.replace('\n', ' | '), flush=True)
                last_report = time.monotonic()
            fig.canvas.draw_idle()
            plt.pause(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        acquisition.stop.set()
        worker.join(timeout=2)
        plt.close(fig)
        print('已停止采集并释放串口。', flush=True)


if __name__ == '__main__':
    main()
