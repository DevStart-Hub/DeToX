"""Storage integration tests: python tests/test_hdf5_recording.py.

Execute the actual Base.py storage methods without importing PsychoPy/Tobii.
Tests include buffer saving and event matching with deterministic fake inputs,
not hardware streaming, real PsychoPy windows, or callback concurrency.
"""

import ast
from datetime import datetime
import os
from pathlib import Path
import runpy
import tempfile
import threading
import time
from collections import deque
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import tables

from hdf5_fixtures import fake_batch


ROOT = Path(__file__).resolve().parents[1]
cfg = SimpleNamespace(**runpy.run_path(str(ROOT / 'DeToX' / 'ETSettings.py')))

# Compile the real coordinate converter too; pixel conversion needs only NumPy.
coords_source = ROOT / 'DeToX' / 'Coords.py'
coords_tree = ast.parse(coords_source.read_text(encoding='utf-8'))
converter = next(node for node in coords_tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'get_psychopy_pos')
coords_namespace = {'np': np}
exec(compile(ast.Module(body=[converter], type_ignores=[]), str(coords_source), 'exec'), coords_namespace)
Coords = SimpleNamespace(get_psychopy_pos=coords_namespace['get_psychopy_pos'])

# Compile the selected production functions unchanged, not copies of their logic.
source = ROOT / 'DeToX' / 'Base.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
tracker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ETracker')
names = {'start_recording', '_adapt_gaze_data', '_check_continuity',
         '_prepare_recording', '_prepare_gaze_schema', '_save_hdf5_data', '_save_csv_data',
         '_on_gaze_data', 'record_event', '_check_gaze_samples',
         'save_data', 'stop_recording', 'stop_live_monitor'}
methods = [node for node in tracker.body if isinstance(node, ast.FunctionDef) and node.name in names]
namespace = dict(globals())
exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
StorageHarness = type('StorageHarness', (), {name: namespace[name] for name in names})


def controller(raw):
    instance = StorageHarness()
    instance.raw_format = raw
    instance.simulate = True
    instance.fps = 120
    instance.win = SimpleNamespace(size=[1920, 1080], units='pix')
    instance.coordinate_units = 'tobii' if raw else 'pix'
    instance.first_timestamp = None
    instance.live_monitor = None
    instance.verbose = False
    return instance


def fake_sdk_sample(timestamp, device_timestamp):
    sample = {'system_time_stamp': timestamp, 'device_time_stamp': device_timestamp}
    for eye in ('left', 'right'):
        sample[f'{eye}_gaze_point_on_display_area'] = (0.25, 0.50)
        sample[f'{eye}_gaze_point_in_user_coordinate_system'] = (1.0, 2.0, 3.0)
        sample[f'{eye}_gaze_origin_in_user_coordinate_system'] = (4.0, 5.0, 6.0)
        sample[f'{eye}_pupil_diameter'] = 3.0
        for field in ('gaze_point_validity', 'gaze_origin_validity', 'pupil_validity'):
            sample[f'{eye}_{field}'] = 1
    return sample


class RecordingStorageTests(unittest.TestCase):
    def test_optional_trackbox_columns_are_selected_once_for_csv_and_hdf5(self):
        field = 'left_gaze_origin_in_trackbox_coordinate_system'
        columns = cfg.RawDataColumns.OPTIONAL_FIELDS[field]
        for extension in ('csv', 'h5'):
            for available in (False, True):
                with self.subTest(extension=extension, available=available):
                    instance = controller(True)
                    instance._prepare_recording(str(Path(self.temp.name) / f'optional-{available}.{extension}'))
                    first = fake_sdk_sample(1000, 1000)
                    if available:
                        # NaN is an unavailable measurement, not an unavailable field.
                        first[field] = (np.nan, np.nan, np.nan)
                    original_order = None
                    for sample in (first, fake_sdk_sample(2000, 2000)):
                        gaze, _ = instance._adapt_gaze_data(pd.DataFrame([dict(sample, Events='')]), None)
                        if original_order is None:
                            original_order = list(instance.recording_columns)
                        else:
                            self.assertEqual(instance.recording_columns, original_order)
                        if extension == 'csv':
                            instance._save_csv_data(gaze)
                        else:
                            instance._save_hdf5_data(gaze, None)
                    saved = (pd.read_csv(instance.filename) if extension == 'csv'
                             else pd.read_hdf(instance.filename, key='gaze'))
                    self.assertEqual(len(saved), 2)
                    self.assertEqual(list(saved.columns), original_order)
                    for column in columns:
                        self.assertEqual(column in saved.columns, available)
                        if available:
                            self.assertTrue(saved[column].isna().all())
                    right = cfg.RawDataColumns.OPTIONAL_FIELDS[
                        'right_gaze_origin_in_trackbox_coordinate_system']
                    self.assertFalse(any(column in saved.columns for column in right))

    def test_trackbox_values_and_nans_are_preserved_and_schema_resets(self):
        instance = controller(True)
        field = 'left_gaze_origin_in_trackbox_coordinate_system'
        columns = cfg.RawDataColumns.OPTIONAL_FIELDS[field]
        for recording, available in enumerate((True, False)):
            instance._prepare_recording(str(Path(self.temp.name) / f'reset-{recording}.h5'))
            sample = fake_sdk_sample(1000, 1000)
            sample['left_pupil_diameter'] = np.nan
            sample['left_pupil_validity'] = 0
            if available:
                sample[field] = (.2, .4, np.nan)
            gaze, _ = instance._adapt_gaze_data(pd.DataFrame([dict(sample, Events='')]), None)
            instance._save_hdf5_data(gaze, None)
            saved = pd.read_hdf(instance.filename, key='gaze')
            self.assertTrue(np.isnan(saved['left_pupil_diameter'].iloc[0]))
            self.assertEqual(saved['left_pupil_validity'].iloc[0], 0)
            self.assertEqual(columns[0] in saved.columns, available)
            if available:
                self.assertEqual(saved[columns[0]].iloc[0], .2)
                self.assertEqual(saved[columns[1]].iloc[0], .4)
                self.assertTrue(np.isnan(saved[columns[2]].iloc[0]))

    def test_late_optional_fields_do_not_change_selected_schema(self):
        instance = controller(True)
        instance._prepare_recording(str(Path(self.temp.name) / 'late.h5'))
        sample = fake_sdk_sample(1000, 1000)
        instance._adapt_gaze_data(pd.DataFrame([dict(sample, Events='')]), None)
        selected = list(instance.recording_columns)
        sample['left_gaze_origin_in_trackbox_coordinate_system'] = (.1, .2, .3)
        gaze, _ = instance._adapt_gaze_data(pd.DataFrame([dict(sample, Events='')]), None)
        self.assertEqual(list(gaze.columns), selected)

    def test_missing_optional_field_in_one_row_becomes_nan(self):
        instance = controller(True)
        instance._prepare_recording(str(Path(self.temp.name) / 'mixed.h5'))
        first = fake_sdk_sample(1000, 1000)
        second = fake_sdk_sample(2000, 2000)
        field = 'left_gaze_origin_in_trackbox_coordinate_system'
        first[field] = (.1, .2, .3)
        batch = pd.DataFrame([dict(first, Events=''), dict(second, Events='')])
        gaze, _ = instance._adapt_gaze_data(batch, None)
        instance._save_hdf5_data(gaze, None)
        saved = pd.read_hdf(instance.filename, key='gaze')
        columns = cfg.RawDataColumns.OPTIONAL_FIELDS[field]
        np.testing.assert_allclose(saved.loc[0, columns].to_numpy(dtype=float), [.1, .2, .3])
        self.assertTrue(saved.loc[1, columns].isna().all())

    def test_empty_raw_recording_has_no_continuity_samples(self):
        instance = controller(True)
        instance._prepare_recording(str(Path(self.temp.name) / 'empty-raw.h5'))
        self.assertIsNone(instance._check_continuity())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='detox-recording-')
        self.addCleanup(self.temp.cleanup)

    def test_setup_creates_empty_tables_with_metadata(self):
        for raw in (False, True):
            with self.subTest(raw=raw):
                instance = controller(raw)
                instance._prepare_recording(str(Path(self.temp.name) / f'{raw}.h5'))
                if raw:
                    with tables.open_file(instance.filename, 'r') as file:
                        self.assertNotIn('gaze', file.root)
                    # Raw schema becomes known at the first saved batch.
                    gaze, _ = fake_batch(raw)
                    instance._prepare_gaze_schema(gaze)
                with tables.open_file(instance.filename, 'r') as file:
                    self.assertEqual(file.root.gaze.nrows, 0)
                    self.assertEqual(file.root.events.nrows, 0)
                    columns = cfg.RawDataColumns if raw else cfg.SimplifiedDataColumns
                    self.assertEqual(file.root.gaze.dtype, columns.get_hdf5_dtype())
                    self.assertEqual(file.root.events.dtype, cfg.EventDataColumns.get_hdf5_dtype(raw))
                    self.assertEqual(file.root._v_attrs.eyetracker_model, 'Simulation')
                    self.assertEqual(file.root._v_attrs.relative_timestamps, str(not raw))
                    self.assertEqual(file.root._v_attrs.timestamp_units, 'us' if raw else 'ms')
                    self.assertEqual(file.root._v_attrs.coordinate_units, 'tobii' if raw else 'pix')
                self.assertTrue(pd.read_hdf(instance.filename, key='gaze').empty)
                self.assertTrue(pd.read_hdf(instance.filename, key='events').empty)

    def test_repeated_saves_and_clean_event_columns(self):
        for raw in (False, True):
            with self.subTest(raw=raw):
                instance = controller(raw)
                instance._prepare_recording(str(Path(self.temp.name) / f'{raw}.h5'))
                first, _ = fake_batch(raw)
                instance._save_hdf5_data(first, None)
                self.assertTrue(pd.read_hdf(instance.filename, key='events').empty)

                for start, label in ((4, 'trial_start'), (8, 'trial_end')):
                    gaze, events = fake_batch(raw, start=start, label=label)
                    events['_target_idx'] = 3
                    instance._save_hdf5_data(gaze[gaze.columns[::-1]], events)
                    self.assertEqual(events['_target_idx'].tolist(), [3])
                    self.assertEqual(gaze['Events'].iloc[-1], label)

                saved = pd.read_hdf(instance.filename, key='gaze')
                events = pd.read_hdf(instance.filename, key='events')
                timestamp = 'system_time_stamp' if raw else 'TimeStamp'
                self.assertEqual(len(saved), 12)
                interval = 10_000 if raw else 10
                self.assertEqual(saved[timestamp].tolist(), [i * interval for i in range(12)])
                self.assertEqual(saved['Events'].iloc[-1], 'trial_end')
                self.assertEqual(list(events.columns), [timestamp, 'Events'])
                self.assertEqual(events[timestamp].tolist(), [7 * interval, 11 * interval])
                self.assertEqual(events['Events'].tolist(), ['trial_start', 'trial_end'])

    def test_invalid_event_batch_does_not_append_gaze(self):
        instance = controller(False)
        instance._prepare_recording(str(Path(self.temp.name) / 'invalid.h5'))
        gaze, events = fake_batch(False, label='trial_start')
        events = events.drop(columns=['TimeStamp'])
        with self.assertRaises(KeyError):
            instance._save_hdf5_data(gaze, events)
        self.assertTrue(pd.read_hdf(instance.filename, key='gaze').empty)
        self.assertTrue(pd.read_hdf(instance.filename, key='events').empty)

    def test_csv_setup_and_append_are_unchanged(self):
        for raw in (False, True):
            with self.subTest(raw=raw):
                instance = controller(raw)
                instance._prepare_recording(str(Path(self.temp.name) / f'{raw}.csv'))
                gaze, _ = fake_batch(raw)
                instance._save_csv_data(gaze)
                saved = pd.read_csv(instance.filename)
                self.assertEqual(list(saved.columns), list(gaze.columns))
                self.assertEqual(len(saved), len(gaze))

    def test_timestamp_policy_through_adaptation_and_saving(self):
        origin = 9_876_543_210_123
        samples = []
        for offset in (0, 10_000, 20_000):
            sample = {'system_time_stamp': origin + offset,
                      'device_time_stamp': 1_234_567 + offset, 'Events': ''}
            for eye in ('left', 'right'):
                sample[f'{eye}_gaze_point_on_display_area'] = (0.25, 0.50)
                sample[f'{eye}_gaze_point_in_user_coordinate_system'] = (1.0, 2.0, 3.0)
                sample[f'{eye}_gaze_origin_in_user_coordinate_system'] = (4.0, 5.0, 6.0)
                sample[f'{eye}_pupil_diameter'] = 3.0
                for field in ('gaze_point_validity', 'gaze_origin_validity', 'pupil_validity'):
                    sample[f'{eye}_{field}'] = 1
            samples.append(sample)

        for raw in (False, True):
            with self.subTest(raw=raw):
                instance = controller(raw)
                instance.fps = 100
                instance._prepare_recording(str(Path(self.temp.name) / f'timing-{raw}.h5'))
                for batch, event_time in ((samples[:2], origin + 7), (samples[2:], origin + 20_007)):
                    events = pd.DataFrame([{'system_time_stamp': event_time, 'Events': 'event'}])
                    gaze, events = instance._adapt_gaze_data(pd.DataFrame(batch), events)
                    instance._save_hdf5_data(gaze, events)

                gaze = pd.read_hdf(instance.filename, key='gaze')
                events = pd.read_hdf(instance.filename, key='events')
                if raw:
                    self.assertEqual(gaze['system_time_stamp'].tolist(), [origin, origin + 10_000, origin + 20_000])
                    self.assertEqual(gaze['device_time_stamp'].tolist(), [1_234_567, 1_244_567, 1_254_567])
                    self.assertEqual(events['system_time_stamp'].tolist(), [origin + 7, origin + 20_007])
                    self.assertIsNone(instance.first_timestamp)
                    self.assertEqual(gaze['left_gaze_point_on_display_area_x'].tolist(), [0.25] * 3)
                else:
                    self.assertEqual(gaze['TimeStamp'].tolist(), [0, 10, 20])
                    self.assertEqual(events['TimeStamp'].tolist(), [0, 20])
                    self.assertEqual(gaze['Left_X'].tolist(), [-480.0] * 3)
                    self.assertEqual(gaze['Left_Y'].tolist(), [0.0] * 3)
                self.assertEqual(instance._check_continuity()['dropped_samples'], 0)

    def test_start_resets_origin_and_enforces_format_policy(self):
        for raw in (False, True):
            with self.subTest(raw=raw):
                instance = controller(raw)
                instance.recording = False
                instance.gaze_data = deque()
                instance.first_timestamp = 999
                instance.experiment_clock = SimpleNamespace(reset=lambda: None)
                instance._get_info = lambda moment: None
                instance._simulate_data_loop = lambda data_type: None
                if raw:
                    with self.assertWarnsRegex(UserWarning, 'coordinate_units is ignored'):
                        instance.start_recording(
                            str(Path(self.temp.name) / f'start-{raw}.h5'),
                            raw_format=True, coordinate_units='pix',
                        )
                    self.assertEqual(instance.coordinate_units, 'tobii')
                else:
                    instance.start_recording(
                        str(Path(self.temp.name) / f'start-{raw}.h5'),
                    )
                    self.assertEqual(instance.coordinate_units, 'pix')
                instance._simulation_thread.join()
                self.assertIsNone(instance.first_timestamp)
                with tables.open_file(instance.filename, 'r') as file:
                    self.assertEqual(file.root._v_attrs.relative_timestamps, str(not raw))
                    self.assertEqual(file.root._v_attrs.timestamp_units, 'us' if raw else 'ms')

    def test_removed_timestamp_option_is_not_accepted(self):
        with self.assertRaisesRegex(TypeError, 'relative_timestamps'):
            controller(False).start_recording(relative_timestamps=True)

    def test_simplified_format_rejects_tobii_units(self):
        instance = controller(False)
        instance.recording = False
        with self.assertRaisesRegex(ValueError, 'simplified format only'):
            instance.start_recording(coordinate_units='tobii')

    def test_gaze_synchronization_is_bounded(self):
        instance = controller(False)
        instance._buf_lock = threading.Lock()
        instance.gaze_data = deque()
        instance.event_data = deque()
        clock = SimpleNamespace(now=0.0)

        def wait(seconds):
            clock.now += seconds

        with patch.dict(namespace, {
            'time': SimpleNamespace(monotonic=lambda: clock.now),
            'core': SimpleNamespace(wait=wait),
        }):
            self.assertTrue(instance._check_gaze_samples())
            instance.event_data.append({'system_time_stamp': 100})
            with self.assertWarnsRegex(UserWarning, '100 ms'):
                self.assertFalse(instance._check_gaze_samples())
            self.assertAlmostEqual(clock.now, 0.100)

            instance.gaze_data.append({'system_time_stamp': 100})
            self.assertTrue(instance._check_gaze_samples())
            self.assertAlmostEqual(clock.now, 0.100)

    def test_full_buffer_save_and_stop_workflow(self):
        origin = 9_876_543_210_123
        for raw in (False, True):
            with self.subTest(raw=raw):
                instance = controller(raw)
                instance.fps = 100
                instance.recording = False
                instance.gaze_data = deque()
                instance.event_data = deque()
                instance._buf_lock = threading.Lock()
                instance.gaze_contingent_buffer = None
                instance.experiment_clock = SimpleNamespace(reset=lambda: None, getTime=lambda: 1.0)
                instance._get_info = lambda moment: None
                instance._simulate_data_loop = lambda data_type: None
                instance._unsubscribe_warnings = lambda: None
                instance.eyetracker = SimpleNamespace(unsubscribe_from=lambda *args: None)

                clock = SimpleNamespace(timestamp=origin)
                sdk = SimpleNamespace(
                    get_system_time_stamp=lambda: clock.timestamp,
                    EYETRACKER_GAZE_DATA='fake_gaze_stream',
                )

                def unexpected_wait(seconds):
                    self.fail('Fake gaze should already have caught up to every event')

                core = SimpleNamespace(getTime=lambda: 1.0, wait=unexpected_wait)
                with patch.dict(namespace, {'core': core, 'tr': sdk, 'NicePrint': lambda *args, **kwargs: None}):
                    instance.start_recording(
                        str(Path(self.temp.name) / f'workflow-{raw}.h5'),
                        raw_format=raw,
                    )
                    instance._simulation_thread.join()
                    # Feed the actual callback/event methods with fake SDK input.
                    instance.simulate = False

                    batches = [
                        ([0, 10_000, 20_000], [(1_000, 'start'), (2_000, 'stim')]),
                        ([30_000, 40_000], [(35_000, 'mid')]),
                        ([50_000, 60_000], [(55_000, 'end')]),
                    ]
                    for index, (offsets, events) in enumerate(batches):
                        for offset in offsets:
                            instance._on_gaze_data(fake_sdk_sample(origin + offset, 1_234_567 + offset))
                        for offset, label in events:
                            clock.timestamp = origin + offset
                            instance.record_event(label)

                        if index < 2:
                            instance.save_data()
                            self.assertTrue(instance.recording)
                            self.assertFalse(instance.gaze_data)
                            self.assertFalse(instance.event_data)
                            self.assertEqual(len(pd.read_hdf(instance.filename, key='gaze')), 3 if index == 0 else 5)
                            # Saving empty buffers must not duplicate rows.
                            instance.save_data()
                        else:
                            instance.stop_recording(data_check=True)

                self.assertFalse(instance.recording)
                self.assertFalse(instance.gaze_data)
                self.assertFalse(instance.event_data)
                gaze = pd.read_hdf(instance.filename, key='gaze')
                events = pd.read_hdf(instance.filename, key='events')
                timestamp = 'system_time_stamp' if raw else 'TimeStamp'
                self.assertEqual(gaze['Events'].tolist(), ['', 'start | stim', '', '', 'mid', '', 'end'])
                x_column = 'left_gaze_point_on_display_area_x' if raw else 'Left_X'
                self.assertEqual(gaze[x_column].tolist(), [0.25 if raw else -480.0] * 7)
                self.assertEqual(events['Events'].tolist(), ['start', 'stim', 'mid', 'end'])
                self.assertEqual(list(events.columns), [timestamp, 'Events'])
                expected_gaze = [origin + i * 10_000 for i in range(7)] if raw else list(range(0, 70, 10))
                expected_events = [origin + i for i in (1_000, 2_000, 35_000, 55_000)] if raw else [1, 2, 35, 55]
                self.assertEqual(gaze[timestamp].tolist(), expected_gaze)
                self.assertEqual(events[timestamp].tolist(), expected_events)
                if raw:
                    self.assertEqual(gaze['device_time_stamp'].tolist(), [1_234_567 + i * 10_000 for i in range(7)])
                self.assertEqual(instance._check_continuity(), {'total_samples': 7, 'dropped_samples': 0})


if __name__ == '__main__':
    unittest.main(verbosity=2)
