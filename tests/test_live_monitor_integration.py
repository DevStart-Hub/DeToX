"""Controller and launcher contracts, without PsychoPy, hardware or windows."""

import ast
from collections import deque
import importlib.util
from pathlib import Path
import sys
import json
import struct
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings


ROOT = Path(__file__).resolve().parents[1]


def controller_fixture():
    """Compile the actual controller, following the existing isolation tests."""
    path = ROOT / 'DeToX' / 'Base.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'ETracker')
    namespace = {'__name__': 'monitor_test_package.Base', '__package__': 'monitor_test_package',
                 'warnings': warnings, 'NicePrint': Mock(),
                 'cfg': SimpleNamespace(colors=SimpleNamespace(left_eye=(0, 255, 0),
                                                               right_eye=(255, 0, 0)))}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    controller = namespace['ETracker'].__new__(namespace['ETracker'])
    controller.live_monitor = None
    controller.recording = False
    controller.simulate = False
    controller.verbose = False
    controller.eyetracker = SimpleNamespace(address='test-uri')
    controller.win = SimpleNamespace(screen=0)
    controller._buf_lock = threading.Lock()
    controller.gaze_data = deque()
    controller.gaze_contingent_buffer = deque(maxlen=10)
    return controller


class ControllerMonitorTests(unittest.TestCase):
    def setUp(self):
        self.controller = controller_fixture()
        self.launcher = Mock()
        self.geometry = Mock(return_value=(0, 0, 800, 600))
        modules = {
            'monitor_test_package': ModuleType('monitor_test_package'),
            'monitor_test_package.gui': ModuleType('monitor_test_package.gui'),
            'monitor_test_package.gui.LiveMonitor': SimpleNamespace(LiveMonitor=self.launcher),
            'monitor_test_package.gui.TrackBox': SimpleNamespace(get_screen_rect=self.geometry),
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_enable_before_and_during_recording_passes_address_and_display_options(self):
        for recording in (False, True):
            self.controller.recording = recording
            self.controller.live_monitor = None
            self.controller.enable_live_monitor(screen=1, size=(500, 350), update_rate=10,
                                                 show_images=True, reverse_eye=True)
            kwargs = self.launcher.call_args.kwargs
            self.assertEqual(kwargs['tracker_address'], 'test-uri')
            self.assertEqual(kwargs['size'], (500, 350))
            self.assertEqual(kwargs['update_rate'], 10)
            self.assertTrue(kwargs['show_images'])
            self.assertTrue(kwargs['reverse_eye'])
            self.assertEqual(kwargs['screen_rect'], (0, 0, 800, 600))
            self.assertEqual(self.controller.recording, recording)

    def test_repeated_enable_keeps_live_child_and_explicit_enable_restarts_dead_child(self):
        old = Mock()
        old.is_alive.return_value = True
        self.controller.live_monitor = old
        self.controller.enable_live_monitor()
        self.launcher.assert_not_called()
        old.is_alive.return_value = False
        self.controller.enable_live_monitor()
        old.stop.assert_called_once()
        self.launcher.assert_called_once()

    def test_launch_failure_warns_without_affecting_recording(self):
        self.controller.recording = True
        self.launcher.side_effect = OSError('Cannot launch')
        with self.assertWarnsRegex(UserWarning, 'Cannot launch'):
            self.controller.enable_live_monitor()
        self.assertTrue(self.controller.recording)
        self.assertIsNone(self.controller.live_monitor)

    def test_simulation_selects_packet_mode(self):
        self.controller.simulate = True
        self.controller.enable_live_monitor()
        self.assertIsNone(self.launcher.call_args.kwargs['tracker_address'])
        self.assertFalse(self.launcher.call_args.kwargs['reverse_eye'])

    def test_real_gaze_callback_never_calls_monitor(self):
        monitor = Mock()
        monitor.push.side_effect = RuntimeError('Must never be called')
        self.controller.live_monitor = monitor
        sample = {'left_gaze_point_on_display_area': (0.5, 0.5)}
        self.controller._on_gaze_data(sample)
        self.assertEqual(list(self.controller.gaze_data), [sample])
        self.assertEqual(list(self.controller.gaze_contingent_buffer), [[(0.5, 0.5), None]])
        monitor.push.assert_not_called()

    def test_shutdown_cleans_monitor_even_if_recording_save_fails(self):
        self.controller.recording = True
        self.controller.stop_recording = Mock(side_effect=RuntimeError('Save failed'))
        monitor = Mock()
        self.controller.live_monitor = monitor
        with self.assertRaisesRegex(RuntimeError, 'Save failed'):
            self.controller._close()
        monitor.stop.assert_called_once()

    def test_stop_failure_does_not_escape_and_repeated_stop_is_safe(self):
        monitor = Mock()
        monitor.stop.side_effect = OSError('Stop failed')
        self.controller.live_monitor = monitor
        with self.assertWarnsRegex(UserWarning, 'Stop failed'):
            self.controller.stop_live_monitor()
        self.controller.stop_live_monitor()
        self.assertIsNone(self.controller.live_monitor)
        monitor.stop.assert_called_once()

    def prepare_recording_stop(self):
        self.controller.recording = True
        self.controller.simulate = True
        self.controller._check_gaze_samples = Mock()
        self.controller._stop_simulation = None
        self.controller._simulation_thread = None
        self.controller.experiment_clock = SimpleNamespace(getTime=lambda: 1)
        self.controller.filename = 'not-written.h5'
        self.controller.save_data = Mock()
        self.controller.live_monitor = Mock()

    def test_stop_recording_keeps_monitor_when_requested(self):
        self.prepare_recording_stop()
        self.controller.stop_recording(data_check=False, stop_monitor=False)
        self.controller.save_data.assert_called_once()
        self.controller.live_monitor.stop.assert_not_called()
        self.assertFalse(self.controller.recording)

    def test_stop_recording_closes_monitor_by_default_after_saving(self):
        self.prepare_recording_stop()
        monitor = self.controller.live_monitor
        calls = Mock()
        calls.attach_mock(self.controller.save_data, 'save')
        calls.attach_mock(monitor.stop, 'stop')
        self.controller.stop_recording(data_check=False)
        self.assertEqual([call[0] for call in calls.mock_calls], ['save', 'stop'])
        self.assertIsNone(self.controller.live_monitor)
        self.assertFalse(self.controller.recording)

    def test_save_failure_still_respects_monitor_flag(self):
        for stop_monitor in (True, False):
            with self.subTest(stop_monitor=stop_monitor):
                self.prepare_recording_stop()
                monitor = self.controller.live_monitor
                self.controller.save_data.side_effect = RuntimeError('Save failed')
                with self.assertRaisesRegex(RuntimeError, 'Save failed'):
                    self.controller.stop_recording(data_check=False, stop_monitor=stop_monitor)
                self.assertEqual(monitor.stop.call_count, int(stop_monitor))

    def test_shutdown_closes_monitor_left_open_after_recording(self):
        self.prepare_recording_stop()
        monitor = self.controller.live_monitor
        self.controller.stop_recording(data_check=False, stop_monitor=False)
        self.controller._close()
        monitor.stop.assert_called_once()

    def test_monitor_cleanup_failure_does_not_prevent_save(self):
        self.prepare_recording_stop()
        self.controller.live_monitor.stop.side_effect = OSError('Stop failed')
        with self.assertWarnsRegex(UserWarning, 'Stop failed'):
            self.controller.stop_recording(data_check=False)
        self.controller.save_data.assert_called_once()


def load_launcher():
    """Load launcher files directly, without importing the DeToX package."""
    path = ROOT / 'DeToX' / 'gui'
    lifecycle_spec = importlib.util.spec_from_file_location('ProcessLifecycle', path / 'ProcessLifecycle.py')
    lifecycle = importlib.util.module_from_spec(lifecycle_spec)
    lifecycle_spec.loader.exec_module(lifecycle)
    spec = importlib.util.spec_from_file_location('live_monitor_target', path / 'LiveMonitor.py')
    launcher = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'ProcessLifecycle': lifecycle}):
        spec.loader.exec_module(launcher)
    return launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.module = load_launcher()
        self.process = Mock(pid=123)
        self.process.poll.return_value = None
        self.job = Mock()
        self.patches = [patch.object(self.module.sys, 'platform', 'win32'),
                        patch.object(self.module, 'WindowsProcessJob', return_value=self.job),
                        patch.object(self.module.subprocess, 'Popen', return_value=self.process),
                        patch.object(self.module.threading, 'Thread')]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_direct_launch_assigns_ownership_before_releasing_gate(self):
        calls = Mock()
        calls.attach_mock(self.job.assign, 'assign')
        calls.attach_mock(self.process.stdin.write, 'write')
        monitor = self.module.LiveMonitor(tracker_address='uri', size=(500, 350),
                                          show_images=True, reverse_eye=True)
        self.assertEqual([call[0] for call in calls.mock_calls], ['assign', 'write'])
        config = json.loads(self.module.subprocess.Popen.call_args.args[0][-1])
        self.assertEqual(config['tracker_address'], 'uri')
        self.assertTrue(config['show_images'])
        self.assertTrue(config['reverse_eye'])
        self.assertTrue(config['wait_for_start'])
        monitor.push((0.5,) * 3, None)
        self.process.stdin.write.assert_called_once_with(b'start\n')

    def test_simulation_push_only_updates_a_bounded_slot(self):
        monitor = self.module.LiveMonitor()
        self.process.stdin.write.reset_mock()
        monitor.push((0.5,) * 3, (0.6,) * 3)
        self.assertEqual(len(monitor._packet), struct.calcsize('6f'))
        self.process.stdin.write.assert_not_called()
        self.assertTrue(monitor._packet_ready.is_set())

    def test_stop_returns_without_waiting_and_is_idempotent(self):
        monitor = self.module.LiveMonitor(tracker_address='uri')
        monitor.stop()
        monitor.stop()
        self.process.stdin.close.assert_not_called()
        self.process.wait.assert_not_called()
        self.assertFalse(monitor.is_alive())

    def test_failed_ownership_setup_stops_child_without_releasing_start_gate(self):
        self.job.assign.side_effect = OSError('Cannot own child')
        with self.assertRaisesRegex(OSError, 'Cannot own child'):
            self.module.LiveMonitor(tracker_address='uri')
        self.process.stdin.write.assert_not_called()
        self.module.threading.Thread.assert_called_with(
            target=self.module.LiveMonitor._close_input, args=(self.process,), daemon=True)
        self.job.close.assert_called_once()

    def test_reaper_kills_unresponsive_child_and_releases_job(self):
        timeout = self.module.subprocess.TimeoutExpired('monitor', 1)
        self.process.wait.side_effect = [timeout, timeout, 0]
        self.module.LiveMonitor._reap(self.process, self.job)
        self.process.terminate.assert_called_once()
        self.process.kill.assert_called_once()
        self.job.close.assert_called_once()


class CleanupTests(unittest.TestCase):
    def test_blocked_input_close_does_not_prevent_bounded_termination(self):
        module = load_launcher()
        process, job = Mock(), Mock()
        close_started = threading.Event()
        child_terminated = threading.Event()
        close_finished = threading.Event()

        def close_input():
            close_started.set()
            # Model Windows close waiting for a writer blocked by a frozen child.
            child_terminated.wait(timeout=5)
            close_finished.set()

        def wait_for_exit(timeout):
            self.assertTrue(close_started.wait(timeout=1))
            if not child_terminated.is_set():
                raise module.subprocess.TimeoutExpired('monitor', timeout)
            return 0

        process.stdin.close.side_effect = close_input
        process.wait.side_effect = wait_for_exit
        process.terminate.side_effect = child_terminated.set
        try:
            module.LiveMonitor._reap(process, job)
            process.terminate.assert_called_once()
            self.assertTrue(close_finished.wait(timeout=1))
            job.close.assert_called_once()
        finally:
            child_terminated.set()


if __name__ == '__main__':
    unittest.main()
