"""Thread-safe recording of received device rows, with explicit host timestamps."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from eeg import FIELD_LABELS


class Recorder:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else Path(__file__).resolve().parent / 'recordings'
        self.lock = threading.Lock()
        self.file = None
        self.path = None
        self.rows = 0
        self.error = None

    def start(self, port, baud, demo=False):
        with self.lock:
            if self.file is not None:
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            name = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.path = self.directory / f'{"demo" if demo else "eeg"}_{name}.csv'
            self.rows = 0
            self.error = None
            self.started = time.monotonic()
            self.last_flush = self.started
            self.metadata = {
                'format_version': 1, 'demo': demo, 'port': port, 'baud_rate': baud,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'timestamp_meaning': 'Host time when a complete valid CSV line is parsed; not device sampling time.',
                'hardware_sample_rate_hz_vendor': 250,
                'export_sample_rate_hz': None,
                'columns': ['received_utc', 'elapsed_seconds', 'host_monotonic_seconds', *FIELD_LABELS],
            }
            self.file = self.path.open('x', newline='', encoding='utf-8-sig')
            try:
                self.writer = csv.writer(self.file)
                self.writer.writerow(self.metadata['columns'])
                self.file.flush()
                self._save_metadata()
            except Exception:
                self.file.close()
                self.file = None
                raise

    def _save_metadata(self):
        self.path.with_suffix('.json').write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    def write(self, stamp, wall_time, values):
        with self.lock:
            if self.file is None or stamp < self.started:
                return
            try:
                self.writer.writerow([
                    datetime.fromtimestamp(wall_time, timezone.utc).isoformat(timespec='microseconds'),
                    f'{stamp - self.started:.6f}', f'{stamp:.6f}', *values,
                ])
                self.rows += 1
                if stamp - self.last_flush >= 1:
                    self.file.flush()
                    self.last_flush = stamp
            except OSError as exc:
                self.error = f'录制写入失败：{exc}'
                self._stop_locked()

    def _stop_locked(self):
        if self.file is None:
            return
        handle, self.file = self.file, None
        try:
            handle.close()
        except OSError as exc:
            self.error = f'文件保存失败：{exc}'
        self.metadata.update(
            stopped_utc=datetime.now(timezone.utc).isoformat(),
            duration_seconds=time.monotonic() - self.started,
            rows=self.rows, error=self.error,
        )
        try:
            self._save_metadata()
        except OSError as exc:
            self.error = f'录制说明文件保存失败：{exc}'

    def stop(self):
        with self.lock:
            self._stop_locked()

    def snapshot(self):
        with self.lock:
            return self.file is not None, self.rows, self.path, self.error
