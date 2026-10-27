"""Run with: python -m unittest discover -s tests -v."""
import tempfile
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import main


class CsvSupportTests(unittest.TestCase):
    def read_text(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'recording.csv'
            path.write_text(text, encoding='utf-8-sig')
            return main.read_recording(path)

    def test_counter_and_missing_time_use_250_hz(self):
        signals, headers, _ = self.read_text('t,adc1,adc2\n1,10,0\n2,20,0\n3,30,0\n')
        self.assertEqual([h['label'] for h in headers], ['adc1', 'adc2'])
        self.assertEqual(headers[0]['sample_rate'], 250)
        np.testing.assert_array_equal(signals[0], [10, 20, 30])
        self.assertEqual(self.read_text('CH1\n1\n2\n')[1][0]['sample_rate'], 250)

    def test_seconds_do_not_change_rate_or_samples(self):
        signals, headers, info = self.read_text('time_s,CH1_raw_uV\n4,1\n4.003,9\n4.010,3\n')
        self.assertAlmostEqual(headers[0]['sample_rate'], 250)
        np.testing.assert_array_equal(signals[0], [1, 9, 3])
        self.assertEqual(info['record_duration'], 3 / 250)
        self.assertEqual(headers[0]['dimension'], 'uV')

    def test_clock_reset_preserves_order(self):
        signals, headers, info = self.read_text('time_s,CH1\n0,1\n0.004,2\n0,3\n0.004,4\n')
        self.assertAlmostEqual(headers[0]['sample_rate'], 250)
        np.testing.assert_array_equal(signals[0], [1, 2, 3, 4])
        self.assertTrue(info['notes'])

    def test_invalid_inputs(self):
        for text in ('adc1\nNaN\n1\n',
                     'adc1,adc2\n1,2\n3\n', 'time_s\n0\n1\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read_text(text)
        signals, headers, _ = self.read_text('time_s,CH1\n0,1\n0,2\n')
        self.assertEqual(headers[0]['sample_rate'], 250)
        np.testing.assert_array_equal(signals[0], [1, 2])

    def test_streamlit_csv_analysis(self):
        from streamlit.testing.v1 import AppTest
        for path in sorted(Path('raw_data').glob('*.csv')):
            upload = io.BytesIO(path.read_bytes())
            upload.name = path.name
            with self.subTest(path=path), patch('streamlit.file_uploader', return_value=upload):
                app = AppTest.from_file('app.py').run(timeout=30)
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertIn('adc1' if path.name.startswith('raw') else 'CH1_raw_uV', app.selectbox[0].value)
                self.assertIn('250.0 Hz', app.selectbox[0].value)
                app.button[0].click().run(timeout=30)
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertTrue(app.success)
                main.plt.close('all')

    def test_repository_recordings_and_edf_bdf_compatibility(self):
        csv_files = sorted(Path('raw_data').glob('*.csv'))
        if not csv_files:
            self.skipTest('Local recordings unavailable')
        edf_files = main.list_edf_files()
        representatives = [next(p for p in edf_files if p.suffix.lower() == suffix)
                           for suffix in ('.edf', '.bdf')]
        for path in csv_files + representatives:
            with self.subTest(path=path):
                signals, headers, info = main.read_recording(path)
                self.assertTrue(all(np.isfinite(x).all() for x in signals))
                if path.suffix.lower() == '.csv':
                    self.assertTrue(all(h['sample_rate'] == 250 for h in headers))
                    self.assertEqual(info['record_duration'], len(signals[0]) / 250)
                    if path.name.startswith('eeg_processed'):
                        self.assertEqual(info['record_duration'], 368)
                if path.suffix.lower() != '.csv':
                    old_signals, old_headers, old_info = main.read_edf(path)
                    self.assertEqual(headers, old_headers)
                    self.assertEqual(info, old_info)
                    for old, new in zip(old_signals, signals):
                        np.testing.assert_array_equal(old, new)
                with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.show'), patch('builtins.print'):
                    main.analyze_edf(path, channel_index=0, window_start=0, show_fft=True, show_welch=True)
                main.plt.close('all')


if __name__ == '__main__':
    unittest.main()
