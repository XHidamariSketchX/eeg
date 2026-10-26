"""本地实时 EEG 监视器：python eeg_viewer.py [--demo]。"""
import argparse
from collections import deque
import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import serial
from pylsl import StreamInfo, StreamOutlet
from eeg import parse_line, SERIAL_PORT, BAUD_RATE
from recording import Recorder
from bci_features import MetricsOutlet

BG = '#101824'
PANEL = '#192536'
TEXT = '#e5edf7'
MUTED = '#92a4ba'
COLORS = ('#53d5d0', '#a99bff')


class Viewer:
    def __init__(self, root, demo=False):
        self.root = root
        self.demo = demo
        self.events = queue.Queue(maxsize=4096)
        self.stop = threading.Event()
        self.worker = None
        self.metrics_outlet = None
        self.points = deque(maxlen=10000)
        self.arrivals = deque(maxlen=2000)
        self.last_values = None
        self.count = self.bad = 0
        self.state = '未连接'
        self.recorder = Recorder()
        self.connected_port = SERIAL_PORT
        self.connected_baud = BAUD_RATE
        root.title('EEG · 双通道实时监视' + (' · 模拟数据' if demo else ''))
        root.geometry('1120x800')
        root.minsize(900, 720)
        root.configure(bg=BG)
        header = tk.Frame(root, bg=BG)
        header.pack(fill='x', padx=24, pady=(18, 10))
        tk.Label(header, text='EEG / 实时监视', bg=BG, fg=TEXT,
                 font=('Microsoft YaHei UI', 22, 'bold')).pack(side='left')
        self.port = tk.StringVar(value=SERIAL_PORT)
        self.baud = tk.StringVar(value=str(BAUD_RATE))
        for title, variable, width in [('串口', self.port, 8), ('波特率', self.baud, 9)]:
            tk.Label(header, text=title, bg=BG, fg=MUTED).pack(side='left', padx=(18, 5))
            ttk.Entry(header, textvariable=variable, width=width).pack(side='left')
        self.button = ttk.Button(header, text='连接', command=self.toggle)
        self.button.pack(side='right')
        self.status = tk.Label(root, text='等待连接', bg=BG, fg=MUTED, anchor='w')
        self.status.pack(fill='x', padx=24)
        recording_bar = tk.Frame(root, bg=BG)
        recording_bar.pack(fill='x', padx=24, pady=(8, 0))
        self.record_button = ttk.Button(recording_bar, text='开始录制', command=self.toggle_recording)
        self.record_button.pack(side='left')
        self.record_status = tk.Label(recording_bar, text='未录制 · 保存全部 21 列与接收时间',
                                      bg=BG, fg=MUTED, anchor='w', wraplength=720, justify='left')
        self.record_status.pack(side='left', padx=12, fill='x', expand=True)
        cards = tk.Frame(root, bg=BG)
        cards.pack(fill='x', padx=24, pady=14)
        self.cards = {}
        for name in ['电池电压', '佩戴检测原值', '专注度', '放松度']:
            card = tk.Frame(cards, bg=PANEL)
            card.pack(side='left', expand=True, fill='both', padx=4)
            tk.Label(card, text=name, bg=PANEL, fg=MUTED, font=('Microsoft YaHei UI', 10)).pack(pady=(10, 0))
            value = tk.Label(card, text='—', bg=PANEL, fg=TEXT, font=('Microsoft YaHei UI', 23, 'bold'))
            value.pack(pady=(2, 10))
            self.cards[name] = value
        controls = tk.Frame(root, bg=BG)
        controls.pack(fill='x', padx=24)
        self.mode = tk.StringVar(value='滤波数据（第 7、8 列）')
        ttk.Combobox(controls, textvariable=self.mode, state='readonly', width=26,
                     values=['滤波数据（第 7、8 列）', '原始 ADC（第 5、6 列）']).pack(side='left')
        tk.Label(controls, text='最近 10 秒 · 自动纵轴 · 数值单位未确认', bg=BG, fg=MUTED).pack(side='right')
        plots = tk.Frame(root, bg=BG)
        plots.pack(fill='both', expand=True, padx=24, pady=8)
        plots.columnconfigure(0, weight=1, uniform='plots')
        plots.columnconfigure(1, weight=1, uniform='plots')
        plots.rowconfigure(1, weight=1)
        tk.Label(plots, text='双通道脑电', bg=BG, fg=TEXT, anchor='w').grid(row=0, column=0, sticky='ew')
        tk.Label(plots, text='θ / α / β · 设备指标趋势（非电压波形）', bg=BG,
                 fg=TEXT, anchor='w').grid(row=0, column=1, sticky='ew', padx=(12, 0))
        self.canvas = tk.Canvas(plots, bg=BG, highlightthickness=0, width=400)
        self.canvas.grid(row=1, column=0, sticky='nsew')
        self.band_canvas = tk.Canvas(plots, bg=BG, highlightthickness=0, width=400)
        self.band_canvas.grid(row=1, column=1, sticky='nsew', padx=(12, 0))
        self.bands = tk.Label(root, text='波段指标：等待数据', bg=PANEL, fg=TEXT,
                              font=('Microsoft YaHei UI', 12), pady=14)
        self.bands.pack(fill='x', padx=24)
        tk.Label(root, text='时间轴按主机接收时间绘制；串口批量到达可能造成时间抖动。波段与评分显示设备原值。',
                 bg=BG, fg=MUTED, anchor='w').pack(fill='x', padx=24, pady=12)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(50, self.update)

    def toggle(self):
        if self.worker and self.worker.is_alive():
            self.stop.set()
            self.button.configure(state='disabled', text='断开中…')
            return
        try:
            baud = int(self.baud.get())
            if baud <= 0:
                raise ValueError
        except ValueError:
            self.state = '请输入有效波特率'
            return
        self.points.clear()
        self.recorder.stop()
        self.arrivals.clear()
        self.last_values = None
        self.count = self.bad = 0
        self.stop.clear()
        self.state = '连接中'
        self.button.configure(text='断开')
        self.connected_port = self.port.get().strip()
        self.connected_baud = baud
        self.worker = threading.Thread(target=self.read, args=(self.connected_port, baud), daemon=True)
        self.worker.start()

    def toggle_recording(self):
        active, _, _, _ = self.recorder.snapshot()
        if active:
            self.recorder.stop()
        elif self.worker and self.worker.is_alive() and not self.stop.is_set():
            try:
                self.recorder.start(self.connected_port, self.connected_baud, self.demo)
            except OSError as exc:
                self.recorder.error = f'无法开始录制：{exc}'

    def received_sample(self, values):
        stamp, wall_time = time.monotonic(), time.time()
        self.recorder.write(stamp, wall_time, values)
        if self.metrics_outlet is not None:
            self.metrics_outlet.push(values)
        self.emit('sample', (stamp, values))

    def emit(self, kind, payload):
        while not self.stop.is_set():
            try:
                self.events.put((kind, payload), timeout=0.1)
                return
            except queue.Full:
                pass

    def read(self, port, baud):
        try:
            self.metrics_outlet = MetricsOutlet(self.demo)
            if self.demo:
                self.emit('state', '模拟数据 · 未连接设备')
                started = time.monotonic()
                demo_counter = 0
                while not self.stop.wait(0.02):
                    t = time.monotonic() - started
                    v = [0.] * 21
                    v[0:4] = [demo_counter % 256, 3.82, 1, 12.5]
                    demo_counter += 1
                    v[4:8] = [1000 + 80 * math.sin(t * 11), 700 + 60 * math.sin(t * 13),
                               70 * math.sin(t * 11) + 12 * math.sin(t * 33), 55 * math.sin(t * 13)]
                    v[14:21] = [20, 15 + 5 * math.sin(t / 2), 35 + 9 * math.sin(t / 3),
                                25 + 7 * math.cos(t / 2.5), 5,
                                50 + 20 * math.sin(t / 4), 60 + 15 * math.cos(t / 5)]
                    self.received_sample(v)
                return
            with serial.Serial(port, baud, timeout=0.2) as ser:
                info = StreamInfo('Gemini-EEG', 'EEG', 2, 0, 'double64', f'gemini_csv_{port}_7_8')
                channels = info.desc().append_child('channels')
                for label in ('ADC1_filtered', 'ADC2_filtered'):
                    channels.append_child('channel').append_child_value('label', label)
                outlet = StreamOutlet(info)
                self.emit('state', f'已连接 {port} · LSL: Gemini-EEG（滤波两路）')
                pending = bytearray()
                while not self.stop.is_set():
                    pending.extend(ser.read(ser.in_waiting or 1))
                    while b'\n' in pending:
                        raw, _, pending = pending.partition(b'\n')
                        if not raw.strip():
                            continue
                        try:
                            values = parse_line(raw)
                        except (ValueError, UnicodeDecodeError):
                            self.emit('bad', None)
                            continue
                        self.received_sample(values)
                        outlet.push_sample(values[6:8])
                    if len(pending) > 65536:
                        pending.clear()
                        self.emit('bad', None)
        except Exception as exc:
            self.emit('state', f'连接失败或中断：{exc}；请关闭占用串口的其他程序')
        finally:
            self.metrics_outlet = None
            self.recorder.stop()

    def update(self):
        for _ in range(4096):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'sample':
                stamp, values = payload
                self.points.append(payload)
                self.arrivals.append(stamp)
                self.last_values = values
                self.count += 1
            elif kind == 'bad':
                self.bad += 1
            else:
                self.state = payload
        now = time.monotonic()
        while self.arrivals and self.arrivals[0] < now - 2:
            self.arrivals.popleft()
        alive = self.worker is not None and self.worker.is_alive()
        active, rows, path, error = self.recorder.snapshot()
        self.record_button.configure(text='停止录制' if active else '开始录制',
                                     state='normal' if active or (alive and not self.stop.is_set()) else 'disabled')
        if error:
            self.record_status.configure(text=error, fg='#ff8f8f')
        elif path:
            self.record_status.configure(text=f'{"录制中" if active else "已保存"} · {rows} 行 · {path}',
                                         fg='#f5bd69' if active else MUTED)
        if not alive:
            self.button.configure(text='连接', state='normal')
            if self.stop.is_set():
                self.state = '已断开'
        stale = not self.points or now - self.points[-1][0] > 2
        suffix = ' · 等待有效数据' if stale and alive else ''
        self.status.configure(text=f'{self.state}{suffix} | {len(self.arrivals) / 2:.1f} 行/秒 | 收到 {self.count} 行 | 无效 {self.bad} 行')
        v = self.last_values
        if v is not None:
            for name, value in zip(self.cards, [f'{v[1]:.2f} V', f'{v[2]:g}', f'{v[19]:.2f}', f'{v[20]:.2f}']):
                self.cards[name].configure(text=value, fg=MUTED if stale or not alive else TEXT)
            self.bands.configure(text='设备波段指标   ' + '     '.join(f'{name}  {value:.2f}' for name, value in zip(['δ', 'θ', 'α', 'β', 'γ'], v[14:19])))
        else:
            for card in self.cards.values():
                card.configure(text='—')
            self.bands.configure(text='波段指标：等待数据')
        self.draw(now)
        self.root.after(50, self.update)

    def draw(self, now):
        offset = 6 if self.mode.get().startswith('滤波') else 4
        self.draw_panels(self.canvas, now, [
            (offset, 'ADC1', COLORS[0]), (offset + 1, 'ADC2', COLORS[1])])
        self.draw_panels(self.band_canvas, now, [
            (15, 'θ · 4–8 Hz', '#53d5d0'),
            (16, 'α · 8–13 Hz', '#a99bff'),
            (17, 'β · 13–30 Hz', '#f5bd69')])

    def draw_panels(self, c, now, series):
        c.delete('all')
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return
        visible = [(t, v) for t, v in self.points if t >= now - 10]
        for index, (column, label, color) in enumerate(series):
            top = index * h / len(series) + 30
            bottom = (index + 1) * h / len(series) - 28
            left, right = 76, w - 18
            c.create_text(8, top - 15, text=label, fill=color, anchor='w', font=('Segoe UI', 11, 'bold'))
            values = [v[column] for _, v in visible]
            if values:
                c.create_text(right, top - 15, text=f'{values[-1]:.2f}', fill=color, anchor='e')
            lo, hi = (min(values), max(values)) if values else (-1, 1)
            margin = max((hi - lo) * 0.12, 1)
            lo, hi = lo - margin, hi + margin
            for tick in range(3):
                y = top + (bottom - top) * tick / 2
                c.create_line(left, y, right, y, fill='#2a394c')
                c.create_text(left - 8, y, text=f'{hi - (hi - lo) * tick / 2:.1f}', fill=MUTED, anchor='e')
            for tick in range(6):
                x = left + (right - left) * tick / 5
                c.create_line(x, top, x, bottom, fill='#243245')
                c.create_text(x, bottom + 14, text=f'{-10 + tick * 2} s', fill=MUTED)
            coords = []
            for t, v in visible:
                coords.extend([left + (t - now + 10) / 10 * (right - left),
                               bottom - (v[column] - lo) / (hi - lo) * (bottom - top)])
            if len(coords) >= 4:
                c.create_line(*coords, fill=color, width=1.6)
            else:
                c.create_text((left + right) / 2, (top + bottom) / 2, text='等待数据', fill=MUTED)

    def close(self):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=1)
        self.recorder.stop()
        self.root.destroy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='显示明确标记的模拟数据，不打开串口')
    args = parser.parse_args()
    window = tk.Tk()
    app = Viewer(window, args.demo)
    if args.demo:
        app.toggle()
    window.mainloop()
