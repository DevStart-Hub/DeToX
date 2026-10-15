"""Hardware-free regression checks for the positioning/recording boundary.

Load the controller class without importing PsychoPy or the Tobii SDK; only
the external display and tracker interfaces used by these tests are stubbed.
Run with: python -m unittest discover -s tests -p test_positioning_isolation.py
"""

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock


class PositioningIsolationTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'DeToX' / 'Base.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        controller_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'ETracker'
        )
        self.namespace = {
            'time': time,
            'NicePrint': Mock(),
            'visual': SimpleNamespace(MovieStim=type('MovieStim', (), {}),
                                      Rect=Mock(), Circle=Mock()),
            'core': SimpleNamespace(wait=Mock()),
            'event': SimpleNamespace(getKeys=Mock(return_value=['space'])),
            'tr': SimpleNamespace(EYETRACKER_USER_POSITION_GUIDE='position'),
            'cfg': SimpleNamespace(colors=SimpleNamespace(left_eye=(0, 255, 0),
                                                          right_eye=(255, 0, 0))),
            'Coords': SimpleNamespace(get_tobii_pos=lambda win, pos: (0.5, 0.5)),
        }
        module = ast.Module(body=[controller_class], type_ignores=[])
        exec(compile(module, str(source), 'exec'), self.namespace)
        self.controller = self.namespace['ETracker'].__new__(self.namespace['ETracker'])
        self.controller._buf_lock = threading.Lock()
        self.controller._position_data = deque(maxlen=1)
        self.controller.gaze_data = deque([{'recorded': True}])
        self.controller.gaze_contingent_buffer = deque([['left', 'right']])
        self.controller.live_monitor = Mock()
        self.controller.recording = False
        self.controller.simulate = False
        self.controller.verbose = False
        self.controller.win = Mock()
        self.controller.eyetracker = Mock()

    def assert_gaze_untouched(self):
        self.assertEqual(list(self.controller.gaze_data), [{'recorded': True}])
        self.assertEqual(list(self.controller.gaze_contingent_buffer), [['left', 'right']])
        self.controller.live_monitor.push.assert_not_called()

    def test_position_packets_keep_only_latest_without_changing_gaze(self):
        self.controller._on_user_position({'old': True})
        latest = {'latest': True}
        self.controller._on_user_position(latest)
        self.assertEqual(list(self.controller._position_data), [latest])
        self.assert_gaze_untouched()

    def test_simulated_position_uses_the_separate_state(self):
        self.controller.mouse = Mock()
        self.controller.mouse.getWheelRel.return_value = (0, 0)
        self.controller.mouse.getPos.return_value = (0, 0)
        self.controller._simulate_user_position_guide()
        self.assertEqual(len(self.controller._position_data), 1)
        self.assertIn('left_user_position', self.controller._position_data[0])
        self.assert_gaze_untouched()

    def test_recording_guard_runs_before_any_display_or_subscription(self):
        self.controller.recording = True
        for overlay in (False, True):
            with self.assertRaisesRegex(RuntimeError, 'active recording'):
                self.controller.show_status(overlay=overlay)
        self.namespace['visual'].Rect.assert_not_called()
        self.namespace['NicePrint'].assert_not_called()
        self.controller.eyetracker.subscribe_to.assert_not_called()
        self.assertTrue(self.controller.recording)
        self.assert_gaze_untouched()

    def test_integrated_positioning_clears_stale_state_and_cleans_up(self):
        self.controller._on_user_position({'stale': True})

        def subscribe(stream, callback, as_dictionary):
            self.assertFalse(self.controller._position_data)
            self.assertEqual(callback, self.controller._on_user_position)
            callback({'left_user_position_validity': 0,
                      'right_user_position_validity': 0})

        self.controller.eyetracker.subscribe_to.side_effect = subscribe
        self.controller.show_status(video_help=False)
        self.controller.eyetracker.unsubscribe_from.assert_called_once_with(
            'position', self.controller._on_user_position
        )
        self.assert_gaze_untouched()

    def test_integrated_positioning_unsubscribes_after_display_failure(self):
        self.namespace['event'].getKeys.side_effect = RuntimeError('display failed')
        with self.assertRaisesRegex(RuntimeError, 'display failed'):
            self.controller.show_status(video_help=False)
        self.controller.eyetracker.unsubscribe_from.assert_called_once_with(
            'position', self.controller._on_user_position
        )
        self.assert_gaze_untouched()


if __name__ == '__main__':
    unittest.main()
