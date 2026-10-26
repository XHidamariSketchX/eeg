import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

from recording import Recorder
from eeg import FIELD_LABELS


class RecordingTests(unittest.TestCase):
    def test_round_trip_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(directory)
            recorder.start('COM3', 115200)
            first_path = recorder.path
            values = [float(i) for i in range(21)]
            recorder.write(recorder.started - 1, time.time(), values)
            for _ in range(3):
                recorder.write(time.monotonic(), time.time(), values)
            recorder.stop()
            recorder.stop()
            with first_path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual([float(rows[0][field]) for field in FIELD_LABELS], values)
            self.assertGreaterEqual(float(rows[0]['elapsed_seconds']), 0)
            self.assertTrue(rows[0]['received_utc'].endswith('+00:00'))
            metadata = json.loads(first_path.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['rows'], 3)
            self.assertIsNone(metadata['export_sample_rate_hz'])
            recorder.start('COM3', 115200, demo=True)
            self.assertNotEqual(recorder.path, first_path)
            self.assertTrue(recorder.path.name.startswith('demo_'))
            recorder.stop()
            self.assertTrue(first_path.exists())

    def test_write_failure_stops_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(directory)
            recorder.start('COM3', 115200)
            recorder.writer = Mock()
            recorder.writer.writerow.side_effect = OSError('disk full')
            recorder.write(time.monotonic(), time.time(), [0.] * 21)
            active, rows, _, error = recorder.snapshot()
            self.assertFalse(active)
            self.assertEqual(rows, 0)
            self.assertIn('disk full', error)

    def test_invalid_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'file'
            path.write_text('occupied')
            recorder = Recorder(path)
            with self.assertRaises(OSError):
                recorder.start('COM3', 115200)
            self.assertFalse(recorder.snapshot()[0])


if __name__ == '__main__':
    unittest.main()
