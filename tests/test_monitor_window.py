"""Check subprocess data flow without opening Tk or connecting to hardware."""

import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, PropertyMock, patch


class FakeView:
    def __init__(self, config, show_images=False):
        self.config = config
        self.show_images = show_images
        self.eye_img_size = 100
        self.left_img_label = Mock()
        self.right_img_label = Mock()
        self.root = Mock()
        self.root.winfo_width.return_value = 200
        self.root.winfo_height.return_value = 200
        self.root.winfo_screenwidth.return_value = 800
        self.root.winfo_screenheight.return_value = 600
        self.update_eye_positions = Mock()

    def hide_image_panels(self):
        self.show_images = False


source = Path(__file__).resolve().parents[1] / 'DeToX' / 'gui' / 'MonitorWindow.py'
spec = importlib.util.spec_from_file_location('monitor_window_test_target', source)
module = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'TrackBox': SimpleNamespace(TrackBoxView=FakeView)}):
    spec.loader.exec_module(module)


class MonitorWindowTests(unittest.TestCase):
    def make_window(self, **config):
        with patch.object(module.MonitorWindow, '_start_listener'):
            return module.MonitorWindow(config)

    def test_monitor_defaults_to_resizable_and_passes_initial_size(self):
        config = {'size': (500, 350)}
        with patch.object(module.MonitorWindow, '_start_listener'):
            window = module.MonitorWindow(config)
        self.assertTrue(window.config['resizable'])
        self.assertEqual(window.config['size'], (500, 350))
        window.root.geometry.assert_called_once_with('500x350+150+125')
        self.assertNotIn('resizable', config)
        self.assertFalse(self.make_window(resizable=False).config['resizable'])

    def test_callback_keeps_latest_only_and_respects_validity(self):
        window = self.make_window()
        window._on_user_position({'left_user_position_validity': 1,
                                 'left_user_position': (0.2, 0.3, 0.4)})
        self.assertEqual(window._latest_positions, (0.2, 0.3, 0.4, None, None, None))
        window._on_user_position({})
        self.assertEqual(window._latest_positions, (None,) * 6)
        window.update_eye_positions.assert_not_called()

    def test_refresh_draws_then_clears_stale_feedback(self):
        window = self.make_window(update_rate=10)
        with patch.object(module.time, 'monotonic', return_value=10):
            window._store_positions((0.5,) * 6)
            window._refresh()
        window.update_eye_positions.assert_called_with(*((0.5,) * 6))
        window.root.after.assert_called_with(100, window._refresh)
        with patch.object(module.time, 'monotonic', return_value=12):
            window._refresh()
        window.update_eye_positions.assert_called_with(*((None,) * 6))

    def test_direct_input_eof_requests_shutdown_without_tk_calls(self):
        window = self.make_window(tracker_address='uri')
        window.root.reset_mock()
        with patch.object(module.os, 'read', return_value=b''):
            window._read_stdin()
        self.assertTrue(window._stop_requested.is_set())
        self.assertEqual(window.root.mock_calls, [])
        window._refresh()
        window.root.quit.assert_called_once()

    def test_legacy_packet_mode_keeps_latest_packet_and_invalid_eye(self):
        window = self.make_window()
        packet = struct.pack('6f', 0.5, 0.5, 0.5, float('nan'), float('nan'), float('nan'))
        with patch.object(module.os, 'read', side_effect=[packet[:8], packet[8:], b'']):
            window._read_stdin()
        self.assertEqual(window._latest_positions, (0.5, 0.5, 0.5, None, None, None))

    def test_run_uses_own_connection_and_cleans_up_after_gui_failure(self):
        window = self.make_window(tracker_address='uri')
        tracker = Mock()
        sdk = SimpleNamespace(EyeTracker=Mock(return_value=tracker),
                              EYETRACKER_USER_POSITION_GUIDE='position')
        window.root.mainloop.side_effect = RuntimeError('GUI failed')
        with patch.dict('sys.modules', {'tobii_research': sdk}):
            with self.assertRaisesRegex(RuntimeError, 'GUI failed'):
                window.run()
        sdk.EyeTracker.assert_called_once_with('uri')
        tracker.subscribe_to.assert_called_once_with('position', window._on_user_position,
                                                     as_dictionary=True)
        tracker.unsubscribe_from.assert_called_once_with('position', window._on_user_position)
        window.root.destroy.assert_called_once()

    def test_connection_failure_still_destroys_window(self):
        window = self.make_window(tracker_address='uri')
        sdk = SimpleNamespace(EyeTracker=Mock(side_effect=RuntimeError('No tracker')))
        with patch.dict('sys.modules', {'tobii_research': sdk}):
            with self.assertRaisesRegex(RuntimeError, 'No tracker'):
                window.run()
        window.root.destroy.assert_called_once()

    def test_invalid_timing_is_rejected_before_window_creation(self):
        for config in ({'update_rate': 0}, {'update_rate': float('nan')},
                       {'stale_timeout': -1}, {'stale_timeout': float('inf')}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.make_window(**config)

    def make_image_window(self, **config):
        image = MagicMock()
        image_tk = Mock()
        with patch.dict('sys.modules', {'PIL': SimpleNamespace(Image=image, ImageTk=image_tk)}):
            window = self.make_window(show_images=True, tracker_address='uri', **config)
        return window, image, image_tk

    def test_images_route_by_camera_not_region_and_callbacks_do_not_draw(self):
        window, _, _ = self.make_image_window()
        window.left_img_label.reset_mock()
        window.right_img_label.reset_mock()
        window._on_eye_image({'camera_id': 0, 'region_id': 0, 'image_data': b'old'})
        window._on_eye_image({'camera_id': 0, 'region_id': 0, 'image_data': b'new'})
        window._on_eye_image({'camera_id': 1, 'region_id': 0, 'image_data': b'other'})
        window._on_eye_image({'camera_id': 9, 'image_data': b'ignored'})
        self.assertEqual({key: frame[0] for key, frame in window._latest_images.items()},
                         {(0, 0): b'new', (1, 0): b'other'})
        window.left_img_label.config.assert_not_called()
        window.right_img_label.config.assert_not_called()

    def test_image_refresh_consumes_frames_once_and_clears_stale_photos(self):
        window, image, image_tk = self.make_image_window()
        with patch.object(module.time, 'monotonic', return_value=10):
            window._on_eye_image({'camera_id': 0, 'image_data': b'encoded'})
            window._refresh_images()
            self.assertIn(0, window._photos)
            window._refresh_images()
        image.open.assert_called_once()
        self.assertEqual(window._latest_images[(0, 0)][0], b'encoded')
        with patch.object(module.time, 'monotonic', return_value=12):
            window._refresh_images()
        self.assertNotIn(0, window._photos)
        window.left_img_label.config.assert_called_with(image=window._placeholder_photo)

    def test_regions_do_not_overwrite_each_other_and_panels_prefer_different_regions(self):
        window, image, _ = self.make_image_window()
        for camera in (0, 1):
            for region in (0, 1):
                window._on_eye_image({'camera_id': camera, 'region_id': region,
                                      'image_data': bytes([camera, region])})
        self.assertEqual(len(window._latest_images), 4)
        window._refresh_images()
        self.assertIs(window._image_labels[1], window.left_img_label)
        self.assertIs(window._image_labels[0], window.right_img_label)
        decoded = [call.args[0].getvalue() for call in image.open.call_args_list]
        self.assertEqual(decoded, [b'\x01\x01', b'\x00\x00'])

    def test_missing_camera_never_reuses_other_cameras_image(self):
        window, image, _ = self.make_image_window()
        window._on_eye_image({'camera_id': 0, 'region_id': 0, 'image_data': b'camera0'})
        window._refresh_images()
        self.assertIn(0, window._photos)
        self.assertNotIn(1, window._photos)
        window.left_img_label.config.assert_called_with(image=window._placeholder_photo)
        image.open.assert_called_once()

    def test_reverse_eye_swaps_only_panel_mapping_not_sources_or_position_data(self):
        window, _, _ = self.make_image_window(reverse_eye=True)
        window._on_eye_image({'camera_id': 0, 'region_id': 0, 'image_data': b'camera0'})
        window._on_eye_image({'camera_id': 1, 'region_id': 1, 'image_data': b'camera1'})
        window._store_positions((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        window._refresh()
        self.assertIs(window._image_labels[0], window.left_img_label)
        self.assertIs(window._image_labels[1], window.right_img_label)
        self.assertEqual({key: frame[0] for key, frame in window._latest_images.items()},
                         {(0, 0): b'camera0', (1, 1): b'camera1'})
        window.update_eye_positions.assert_called_with(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    def test_reverse_eye_has_no_effect_when_images_are_disabled(self):
        window = self.make_window(reverse_eye=True)
        self.assertFalse(window.show_images)
        self.assertFalse(hasattr(window, '_image_labels'))

    def test_region_buffer_is_bounded(self):
        window, _, _ = self.make_image_window()
        with patch.object(module.time, 'monotonic', side_effect=range(10)):
            for region in range(10):
                window._on_eye_image({'camera_id': 0, 'region_id': region, 'image_data': b'frame'})
        self.assertEqual(len(window._latest_images), 4)
        self.assertEqual(set(window._latest_images), {(0, region) for region in range(6, 10)})

    def test_bad_image_is_nonfatal_and_warns_only_once(self):
        window, image, _ = self.make_image_window()
        image.open.side_effect = OSError('Bad frame')
        with patch.object(module.sys, 'stderr') as stderr, \
                patch.object(module.time, 'monotonic', return_value=10):
            for _ in range(2):
                window._on_eye_image({'camera_id': 0, 'image_data': b'bad'})
                window._refresh_images()
        self.assertEqual(sum('Bad frame' in str(call) for call in stderr.write.call_args_list), 1)
        self.assertTrue(window.show_images)
        image.open.side_effect = None
        with patch.object(module.time, 'monotonic', return_value=11):
            window._on_eye_image({'camera_id': 0, 'image_data': b'valid'})
            window._refresh_images()
        self.assertIn(0, window._photos)

    def test_unavailable_image_stream_does_not_disable_position_or_cleanup(self):
        window, _, _ = self.make_image_window()
        tracker = Mock()
        tracker.device_capabilities = ('eye_images',)
        def subscribe(stream, callback, **kwargs):
            if stream == 'images':
                raise RuntimeError('Images unsupported')
        tracker.subscribe_to.side_effect = subscribe
        sdk = SimpleNamespace(EyeTracker=Mock(return_value=tracker),
                              EYETRACKER_USER_POSITION_GUIDE='position', EYETRACKER_EYE_IMAGES='images',
                              CAPABILITY_HAS_EYE_IMAGES='eye_images')
        with patch.dict('sys.modules', {'tobii_research': sdk}), patch.object(module.sys, 'stderr') as stderr:
            window.run()
        self.assertFalse(window.show_images)
        self.assertEqual(sum('Continuing with position feedback only' in str(call)
                             for call in stderr.write.call_args_list), 1)
        window.root.mainloop.assert_called_once()
        tracker.unsubscribe_from.assert_called_once_with('position', window._on_user_position)

    def test_both_streams_cleanup_even_if_image_unsubscribe_fails(self):
        window, _, _ = self.make_image_window()
        tracker = Mock()
        tracker.device_capabilities = ('eye_images',)
        def unsubscribe(stream, callback):
            if stream == 'images':
                raise RuntimeError('Image cleanup failed')
        tracker.unsubscribe_from.side_effect = unsubscribe
        sdk = SimpleNamespace(EyeTracker=Mock(return_value=tracker),
                              EYETRACKER_USER_POSITION_GUIDE='position', EYETRACKER_EYE_IMAGES='images',
                              CAPABILITY_HAS_EYE_IMAGES='eye_images')
        with patch.dict('sys.modules', {'tobii_research': sdk}), patch.object(module.sys, 'stderr'):
            window.run()
        self.assertEqual(tracker.unsubscribe_from.call_count, 2)
        window.root.destroy.assert_called_once()

    def test_unsupported_images_skip_subscription_hide_panels_and_warn_once(self):
        window, _, _ = self.make_image_window()
        tracker = Mock(device_capabilities=())
        sdk = SimpleNamespace(EyeTracker=Mock(return_value=tracker),
                              EYETRACKER_USER_POSITION_GUIDE='position',
                              CAPABILITY_HAS_EYE_IMAGES='eye_images')
        with patch.dict('sys.modules', {'tobii_research': sdk}), patch.object(module.sys, 'stderr') as stderr:
            window.run()
            window._disable_images('Repeated failure')
        self.assertFalse(window.show_images)
        tracker.subscribe_to.assert_called_once_with('position', window._on_user_position,
                                                     as_dictionary=True)
        window.root.mainloop.assert_called_once()
        tracker.unsubscribe_from.assert_called_once_with('position', window._on_user_position)
        self.assertEqual(sum('Continuing with position feedback only' in str(call)
                             for call in stderr.write.call_args_list), 1)

    def test_images_disabled_by_choice_do_not_query_capabilities_or_warn(self):
        window = self.make_window(tracker_address='uri', show_images=False)
        tracker = Mock()
        sdk = SimpleNamespace(EyeTracker=Mock(return_value=tracker),
                              EYETRACKER_USER_POSITION_GUIDE='position')
        with patch.dict('sys.modules', {'tobii_research': sdk}), patch.object(module.sys, 'stderr') as stderr:
            window.run()
        stderr.write.assert_not_called()
        self.assertNotIn('device_capabilities', tracker._mock_children)

    def test_failed_capability_lookup_keeps_position_feedback_running(self):
        window, _, _ = self.make_image_window()
        tracker = Mock()
        type(tracker).device_capabilities = PropertyMock(side_effect=RuntimeError('Lookup failed'))
        sdk = SimpleNamespace(EyeTracker=Mock(return_value=tracker),
                              EYETRACKER_USER_POSITION_GUIDE='position',
                              CAPABILITY_HAS_EYE_IMAGES='eye_images')
        with patch.dict('sys.modules', {'tobii_research': sdk}), patch.object(module.sys, 'stderr') as stderr:
            window.run()
        self.assertFalse(window.show_images)
        window.root.mainloop.assert_called_once()
        tracker.subscribe_to.assert_called_once_with('position', window._on_user_position,
                                                     as_dictionary=True)
        tracker.unsubscribe_from.assert_called_once_with('position', window._on_user_position)
        self.assertEqual(sum('Lookup failed' in str(call) for call in stderr.write.call_args_list), 1)

    def test_missing_pillow_and_simulation_keep_position_only_mode(self):
        with patch.dict('sys.modules', {'PIL': None}), patch.object(module.sys, 'stderr'):
            window = self.make_window(show_images=True, tracker_address='uri')
        self.assertFalse(window.show_images)
        self.assertFalse(self.make_window(show_images=True).show_images)


if __name__ == '__main__':
    unittest.main()
