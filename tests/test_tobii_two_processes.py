"""Hardware-free checks for the standalone probe's reporting and cleanup only."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


source = Path(__file__).resolve().parents[1] / 'examples' / 'tobii_two_processes.py'
spec = importlib.util.spec_from_file_location('tobii_two_processes', source)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class TwoProcessProbeTests(unittest.TestCase):
    def test_counter_counts_packets_even_without_valid_eyes(self):
        counter = probe.SampleCounter()
        counter.callback({'left_user_position_validity': 0})
        counter.callback({})
        self.assertEqual(counter.read(), 2)

    def test_duration_rejects_nonfinite_and_out_of_range_values(self):
        self.assertEqual(probe.duration('2.5'), 2.5)
        for value in ('nan', 'inf', '0', '-1', '31'):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    probe.duration(value)

    def test_receive_rejects_timeout_eof_and_worker_error(self):
        reports = Mock()
        reports.poll.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'Timed out'):
            probe.receive(reports, 1)
        reports.poll.return_value = True
        reports.recv.side_effect = EOFError
        with self.assertRaisesRegex(RuntimeError, 'without a report'):
            probe.receive(reports, 1)
        reports.recv.side_effect = None
        reports.recv.return_value = ('error', 'connection refused')
        with self.assertRaisesRegex(RuntimeError, 'connection refused'):
            probe.receive(reports, 1)

    def test_worker_reports_samples_and_unsubscribes(self):
        tracker = Mock()
        tracker.subscribe_to.side_effect = lambda stream, callback, **kw: callback({})
        sdk = SimpleNamespace(EyeTracker=Mock(return_value=tracker),
                              EYETRACKER_USER_POSITION_GUIDE='position')
        reports = Mock()
        reports.recv.return_value = 'stop'
        with patch.dict('sys.modules', {'tobii_research': sdk}):
            probe.position_worker('tracker-uri', reports)
        sdk.EyeTracker.assert_called_once_with('tracker-uri')
        self.assertEqual(reports.send.call_args_list[0].args[0][0], 'ready')
        self.assertEqual(reports.send.call_args_list[1].args[0], ('samples', 1))
        callback = tracker.subscribe_to.call_args.args[1]
        tracker.unsubscribe_from.assert_called_once_with('position', callback)
        reports.close.assert_called_once()

    def test_worker_reports_cleanup_failure_instead_of_success(self):
        tracker = Mock()
        tracker.unsubscribe_from.side_effect = RuntimeError('disconnect failed')
        sdk = SimpleNamespace(EyeTracker=Mock(return_value=tracker),
                              EYETRACKER_USER_POSITION_GUIDE='position')
        reports = Mock()
        with patch.dict('sys.modules', {'tobii_research': sdk}):
            probe.position_worker('tracker-uri', reports)
        kind, message = reports.send.call_args.args[0]
        self.assertEqual(kind, 'error')
        self.assertIn('disconnect failed', message)

    def test_missing_hardware_has_actionable_error(self):
        sdk = SimpleNamespace(find_all_eyetrackers=lambda: ())
        with patch.dict('sys.modules', {'tobii_research': sdk}):
            with self.assertRaisesRegex(RuntimeError, 'Found 0 trackers'):
                probe.run_probe(None, 1)


if __name__ == '__main__':
    unittest.main()
