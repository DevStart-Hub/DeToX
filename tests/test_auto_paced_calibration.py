"""Hardware-free calibration checks: python -m unittest discover -s tests -p "test_*.py"."""

import importlib
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


def load_modules():
    """Load real DeToX code with unavailable hardware/GUI dependencies stubbed."""
    # Keep real extension modules loaded after the temporary module patch ends.
    for name in ('numpy', 'PIL.Image', 'PIL.ImageDraw'):
        importlib.import_module(name)
    package = types.ModuleType('DeToX')
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'DeToX')]
    dependencies = {
        'DeToX': package,
        'psychopy': Mock(),
        'psychopy.tools': Mock(),
        'psychopy.tools.monitorunittools': Mock(),
        'tobii_research': Mock(),
        'tables': Mock(),
        'pandas': Mock(),
    }
    with patch.dict(sys.modules, dependencies):
        calibration = importlib.import_module('DeToX.Calibration')
        base = importlib.import_module('DeToX.Base')
    return calibration, base


calibration, base = load_modules()


class AutoPacedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.time = 0.0
        self.frames = []
        self.current_target = None
        self.collected = []
        self.session = calibration.MouseCalibrationSession.__new__(
            calibration.MouseCalibrationSession
        )
        self.session.win = Mock()
        self.session.audio = Mock(isPlaying=False)
        self.session.stim_objects = [Mock(), Mock(), Mock()]
        self.session.remaining_points = [0, 1, 2]
        self.points = [(0, 0), (-0.5, 0.5), (0.5, -0.5)]
        self.session._draw_calibration_border = Mock()
        self.session._clear_collected_data = Mock()
        self.session._animate = Mock(side_effect=self.animate)
        self.session._collect_data_at_point = Mock(side_effect=self.collect)
        self.session.win.clearBuffer.side_effect = self.clear_frame
        self.session.win.flip.side_effect = self.flip

        test = self

        class Clock:
            def __init__(self):
                self.reset()

            def reset(self):
                self.start = test.time

            def getTime(self):
                return test.time - self.start

        self.clock_patch = patch.object(calibration.core, 'Clock', Clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)

    def clear_frame(self):
        self.current_target = None

    def animate(self, stim, clock, point_idx, phase='attention'):
        self.current_target = point_idx

    def flip(self):
        self.time += 0.1
        self.frames.append(self.current_target)

    def collect(self, target, point_idx, **kwargs):
        # Collection must never happen before the correct target is displayed.
        self.assertEqual(self.frames[-1], point_idx)
        self.collected.append(point_idx)
        return True

    def run_collection(self, keys, presentation_order=(1, 2, 3)):
        with patch.object(calibration.event, 'getKeys', side_effect=keys):
            return self.session._present_and_collect_calibration_points(
                self.points,
                pacing='auto',
                presentation_order=presentation_order,
                num_samples=5,
            )

    def test_automatic_points_require_one_space_each_and_ignore_numbers(self):
        keys = [
            [],           # Show the first target.
            ['1', 'space'],  # Number is ignored; collect the first target.
            [],           # Consume result and show the next target.
            ['space'],    # Collect the second target.
            [],           # Consume result and show the third target.
            ['space'],    # Collect the third target.
            [],           # Consume result and finish the pass.
        ]
        self.assertTrue(self.run_collection(keys))
        self.assertEqual(self.collected, [0, 1, 2])
        self.assertNotIn(None, self.frames)  # No blank interval.
        self.assertEqual(self.session.remaining_points, [0, 1, 2])
        self.session._collect_data_at_point.assert_called_with(
            self.points[2], 2, num_samples=5
        )

    def test_retry_subset_uses_the_requested_order_without_a_blank_gap(self):
        self.session.remaining_points = [2, 0]
        keys = [[], ['space'], [], ['space'], []]
        self.assertTrue(self.run_collection(keys))
        self.assertEqual(self.collected, [0, 2])
        self.assertNotIn(None, self.frames)
        self.assertEqual(self.frames[-1], 2)

    def test_escape_while_waiting_stops_audio_and_clears_data(self):
        self.assertFalse(self.run_collection([[], ['escape', 'space']]))
        self.assertEqual(self.collected, [])
        self.session.audio.stop.assert_called_once()
        self.session._clear_collected_data.assert_called_once()

    def test_escape_pressed_during_collection_is_not_discarded(self):
        self.assertFalse(self.run_collection([[], ['space'], ['escape']]))
        self.assertEqual(self.collected, [0])
        self.session._clear_collected_data.assert_called_once()

    def test_escape_after_success_stops_before_next_collection(self):
        keys = [[], ['space'], [], ['escape']]
        self.assertFalse(self.run_collection(keys))
        self.assertEqual(self.collected, [0])
        self.assertNotIn(None, self.frames)

    def test_manual_default_still_allows_recollection_and_enter_to_finish(self):
        keys = [
            ['1'],       # Show point 1.
            ['space'],   # Collect it.
            [],          # Consume result; target disappears.
            ['1'],       # Show point 1 again.
            ['space'],   # Recollect it.
            [],          # Consume result.
            ['return'],  # Show results.
        ]
        with patch.object(calibration.event, 'getKeys', side_effect=keys):
            self.assertTrue(
                self.session._present_and_collect_calibration_points(
                    self.points,
                    pacing='manual',
                )
            )
        self.assertEqual(self.collected, [0, 0])

    def test_activate_target_sets_position_orientation_and_audio(self):
        animation_clock = Mock()
        self.session._activate_calibration_target(
            1,
            self.points[1],
            animation_clock,
        )
        stim = self.session.stim_objects[1]
        stim.setPos.assert_called_once_with(self.points[1])
        stim.setOri.assert_called_once_with(0)
        animation_clock.reset.assert_called_once_with()
        self.session.audio.play.assert_called_once_with()
        self.session.audio.setVolume.assert_called_once_with(1.0)

    def test_failed_collection_keeps_same_automatic_point_for_retry(self):
        def fail_then_succeed(target, point_idx, **kwargs):
            self.assertEqual(self.frames[-1], point_idx)
            self.collected.append(point_idx)
            return len(self.collected) > 1

        self.session.remaining_points = [0]
        self.session._collect_data_at_point = Mock(
            side_effect=fail_then_succeed
        )
        keys = [[], ['space'], [], ['space'], []]
        self.assertTrue(
            self.run_collection(keys, presentation_order=(1,))
        )
        self.assertEqual(self.collected, [0, 0])

    def test_temporary_adapters_consume_collection_result_once(self):
        self.session._collect_data_at_point = Mock(return_value=True)
        self.session._start_point_data_collection(0, self.points[0])
        self.assertTrue(self.session._update_point_data_collection())
        self.assertIsNone(self.session._update_point_data_collection())

    def test_auto_instructions_use_space_or_escape_and_do_not_mention_gap(self):
        self.session.show_message_and_wait = Mock(return_value=['space'])
        self.assertTrue(self.session._show_auto_paced_instructions('Look at the target'))
        args, kwargs = self.session.show_message_and_wait.call_args
        self.assertIn('Look at the target', args[0])
        self.assertNotIn('gap', args[0].lower())
        self.assertNotIn('blank', args[0].lower())
        self.assertEqual(kwargs['key_list'], ['space', 'escape'])
        self.session.show_message_and_wait.return_value = ['escape']
        self.assertFalse(self.session._show_auto_paced_instructions('Look at the target'))

    def test_auto_order_is_followed(self):
        keys = [[], ['space'], [], ['space'], [], ['space'], []]
        self.assertTrue(
            self.run_collection(keys, presentation_order=(3, 1, 2))
        )
        self.assertEqual(self.collected, [2, 0, 1])

    def test_random_order_shuffles_pending_points(self):
        keys = [[], ['space'], [], ['space'], [], ['space'], []]
        with patch.object(
            calibration.random,
            'shuffle',
            side_effect=lambda points: points.reverse(),
        ):
            with patch.object(calibration.event, 'getKeys', side_effect=keys):
                self.assertTrue(
                    self.session._present_and_collect_calibration_points(
                        self.points,
                        pacing='auto',
                        num_samples=5,
                    )
                )
        self.assertEqual(self.collected, [2, 1, 0])

    def test_tobii_startup_escape_does_not_enter_calibration_mode(self):
        session = calibration.TobiiCalibrationSession.__new__(calibration.TobiiCalibrationSession)
        session.calibration = Mock()
        session._show_auto_paced_instructions = Mock(return_value=False)
        self.assertFalse(session.run(self.points, pacing='auto'))
        session.calibration.enter_calibration_mode.assert_not_called()

    def test_tobii_session_error_still_leaves_calibration_mode(self):
        session = calibration.TobiiCalibrationSession.__new__(calibration.TobiiCalibrationSession)
        session.calibration = Mock()
        session._show_auto_paced_instructions = Mock(return_value=True)
        session._run_calibration_session = Mock(side_effect=RuntimeError('collection failed'))
        with self.assertRaisesRegex(RuntimeError, 'collection failed'):
            session.run(self.points, pacing='auto')
        session.calibration.enter_calibration_mode.assert_called_once()
        session.calibration.leave_calibration_mode.assert_called_once()

    def test_mouse_run_collects_initial_retry_and_restart_passes(self):
        session = self.session
        session.calibration_points = self.points
        session.calibration_data = {}
        session._show_auto_paced_instructions = Mock(return_value=True)
        session.check_points = Mock()
        session._prepare_session = Mock()
        session._show_results = Mock()
        session._selection_phase = Mock(side_effect=[[2, 0], None, []])
        with patch.object(calibration.event, 'getKeys', side_effect=[[], ['space'], []] * 40):
            self.assertTrue(
                session.run(
                    self.points, pacing='auto', presentation_order=(1, 2, 3)
                )
            )
        self.assertEqual(self.collected, [0, 1, 2, 0, 2, 0, 1, 2])

    def test_tobii_run_collects_retry_points_and_leaves_calibration_mode(self):
        session = calibration.TobiiCalibrationSession.__new__(calibration.TobiiCalibrationSession)
        session.__dict__.update(self.session.__dict__)
        session.calibration_points = self.points
        session.calstim_units = 'height'
        session.calibration = Mock()
        session.calibration.compute_and_apply.return_value.status = calibration.tr.CALIBRATION_STATUS_SUCCESS
        session._show_auto_paced_instructions = Mock(return_value=True)
        session.check_points = Mock()
        session._prepare_session = Mock()
        session._show_calibration_result = Mock()
        session._discard_phase = Mock()
        session._selection_phase = Mock(side_effect=[[2, 0], []])
        with patch.object(calibration, 'get_tobii_pos', return_value=self.points):
            with patch.object(calibration.event, 'getKeys', side_effect=[[], ['space'], []] * 30):
                self.assertTrue(
                    session.run(
                        self.points, pacing='auto', presentation_order=(1, 2, 3)
                    )
                )
        self.assertEqual(self.collected, [0, 1, 2, 0, 2])
        session.calibration.enter_calibration_mode.assert_called_once()
        session.calibration.leave_calibration_mode.assert_called_once()


class CalibrationAnimationTests(unittest.TestCase):
    def setUp(self):
        self.session = calibration.MouseCalibrationSession.__new__(
            calibration.MouseCalibrationSession
        )
        self.session.win = Mock()
        self.session.stim_size = 'big'
        self.session.anim_type = 'zoom'
        self.session.collection_anim_type = 'zoom'
        self.stim = Mock(units='height')
        self.session._draw_calibration_border = Mock()
        self.clock = Mock()
        self.clock.getTime.return_value = 0.0

    def animate(self, phase):
        with patch.object(
            calibration,
            'convert_height_to_units',
            side_effect=lambda win, value, target_units: value,
        ):
            self.session._animate(self.stim, self.clock, 0, phase)

    def test_zoom_starts_at_maximum_and_reaches_minimum(self):
        size, rotation = self.session._zoom_animation(
            0.0, min_size=0.05, max_size=0.16
        )
        self.assertAlmostEqual(size, 0.16)
        self.assertEqual(rotation, 0.0)

        half_cycle = math.pi / calibration.cfg.animation.zoom_speed
        size, rotation = self.session._zoom_animation(
            half_cycle, min_size=0.05, max_size=0.16
        )
        self.assertAlmostEqual(size, 0.05)
        self.assertEqual(rotation, 0.0)

    def test_animate_selects_attention_and_collection_zoom_bounds(self):
        self.animate('attention')
        self.assertAlmostEqual(
            self.stim.setSize.call_args.args[0],
            calibration.cfg.animation.attention_zoom_max_size_big,
        )
        self.assertEqual(self.stim.setOri.call_args.args[0], 0.0)

        self.stim.reset_mock()
        self.animate('collection')
        self.assertAlmostEqual(
            self.stim.setSize.call_args.args[0],
            calibration.cfg.animation.collection_zoom_max_size_big,
        )
        self.assertEqual(self.stim.setOri.call_args.args[0], 0.0)

    def test_trill_uses_fixed_size_and_bounded_rotation_with_pauses(self):
        rotations = [
            self.session._trill_animation(index / 1000, 0.14, 20)[1]
            for index in range(1100)
        ]
        self.assertGreaterEqual(max(abs(value) for value in rotations), 18)
        self.assertLessEqual(max(abs(value) for value in rotations), 20)

        size, rotation = self.session._trill_animation(0.0, 0.14, 20)
        self.assertEqual(size, 0.14)
        self.assertEqual(rotation, 0.0)

        size, rotation = self.session._trill_animation(1.2, 0.14, 20)
        self.assertEqual(size, 0.14)
        self.assertEqual(rotation, 0.0)

    def test_collection_trill_can_rotate_more_than_attention(self):
        self.session.anim_type = 'trill'
        self.session.collection_anim_type = 'trill'
        self.clock.getTime.return_value = 0.25

        self.animate('attention')
        attention_rotation = self.stim.setOri.call_args.args[0]

        self.stim.reset_mock()
        self.animate('collection')
        collection_rotation = self.stim.setOri.call_args.args[0]

        self.assertGreater(abs(collection_rotation), abs(attention_rotation))
        self.assertAlmostEqual(
            self.stim.setSize.call_args.args[0],
            calibration.cfg.animation.collection_trill_size_big,
        )

    def test_unknown_animation_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'animation phase'):
            self.session._animate(self.stim, self.clock, 0, 'invalid')


class CalibrationApiTests(unittest.TestCase):
    def setUp(self):
        self.controller = base.ETracker.__new__(base.ETracker)
        self.controller.win = Mock()
        self.controller.mouse = Mock()
        self.controller.simulate = True
        self.controller.verbose = False

    def test_new_options_reach_both_session_types(self):
        for simulate, name in [(True, 'MouseCalibrationSession'), (False, 'TobiiCalibrationSession')]:
            with self.subTest(simulate=simulate), patch.object(base, name) as session:
                self.controller.simulate = simulate
                self.controller.calibration = Mock()
                session.return_value.run.return_value = True
                self.assertTrue(self.controller.calibrate(
                    infant_stims=False, audio=False, shuffle=False,
                    anim_type='trill', collection_anim_type='zoom',
                    pacing='auto', presentation_order=[3, 1, 2, 4, 5],
                ))
                session.return_value.run.assert_called_once_with(
                    base.cfg.calibration.points_5,
                    pacing='auto',
                    presentation_order=(3, 1, 2, 4, 5),
                )
                self.assertEqual(
                    session.call_args.kwargs['collection_anim_type'], 'zoom'
                )

    def test_default_is_manual_with_random_auto_order(self):
        with patch.object(base, 'MouseCalibrationSession') as session:
            self.controller.calibrate(infant_stims=False, audio=False)
            session.return_value.run.assert_called_once_with(
                base.cfg.calibration.points_5,
                pacing='manual',
                presentation_order='random',
            )

    def test_manual_pacing_ignores_presentation_order(self):
        with patch.object(base, 'MouseCalibrationSession') as session:
            self.assertTrue(self.controller.calibrate(
                infant_stims=False, audio=False, pacing='manual',
                presentation_order=['not', 'used'],
            ))
            session.return_value.run.assert_called_once_with(
                base.cfg.calibration.points_5,
                pacing='manual',
                presentation_order='random',
            )

    def test_invalid_pacing_and_animation_types_are_rejected(self):
        cases = [
            ({'pacing': 'sometimes'}, 'pacing'),
            ({'anim_type': 'spin'}, 'anim_type'),
            ({'collection_anim_type': 'spin'}, 'collection_anim_type'),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.controller.calibrate(**kwargs)

    def test_presentation_order_must_be_a_complete_1_based_permutation(self):
        invalid_orders = [
            [1, 2, 3, 4],
            [1, 2, 3, 4, 6],
            [1, 2, 3, 4, 4],
            [1, 2, 3, 4, True],
            [1, 2, 3, 4, '5'],
            None,
        ]
        for presentation_order in invalid_orders:
            with self.subTest(presentation_order=presentation_order):
                with self.assertRaisesRegex(ValueError, 'presentation_order'):
                    self.controller.calibrate(
                        pacing='auto',
                        presentation_order=presentation_order,
                    )


if __name__ == '__main__':
    unittest.main()
