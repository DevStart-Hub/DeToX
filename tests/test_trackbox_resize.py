"""Hardware-free sizing checks; no Tk windows are created."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


source = Path(__file__).resolve().parents[1] / 'DeToX' / 'gui' / 'TrackBox.py'
spec = importlib.util.spec_from_file_location('trackbox_resize_target', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrackBoxResizeTests(unittest.TestCase):
    def test_unavailable_images_remove_panels_and_release_layout_minimum(self):
        view = module.TrackBoxView.__new__(module.TrackBoxView)
        view._image_panels = [Mock(), Mock()]
        view._position_panel = Mock()
        view.root = Mock()
        view.root.winfo_reqwidth.return_value = 400
        view.root.winfo_reqheight.return_value = 300
        view.show_images = True
        view.hide_image_panels()
        for panel in view._image_panels:
            panel.grid_remove.assert_called_once()
        view._position_panel.grid_configure.assert_called_once_with(padx=0)
        view.root.minsize.assert_called_once_with(400, 300)
        self.assertFalse(view.show_images)

    def make_feedback_view(self):
        view = module.TrackBoxView.__new__(module.TrackBoxView)
        view._draw_eye = Mock()
        view.left_dot, view.right_dot = 'left', 'right'
        view.tracking_label = Mock()
        view.status_label = Mock()
        view.z_canvas = Mock()
        view.z_marker = 'marker'
        view.scale = 1
        view.bar_left, view.bar_top, view.bar_bottom = 0, 5, 19
        view.z_bar_width = 400
        return view

    def test_tracking_count_and_distance_instructions(self):
        view = self.make_feedback_view()
        for distance, instruction in ((0.1, 'Move farther away'),
                                      (0.5, 'Good position'), (0.9, 'Move closer')):
            with self.subTest(distance=distance):
                view.update_eye_positions(.4, .5, distance, .6, .5, distance)
                self.assertEqual(view.tracking_label.config.call_args.kwargs['text'],
                                 '2 / 2 eyes detected')
                self.assertEqual(view.status_label.config.call_args.kwargs['text'], instruction)
                view.z_canvas.itemconfig.assert_called_with('marker', state='normal')
        view.update_eye_positions(.4, .5, .5, None, None, None)
        self.assertEqual(view.tracking_label.config.call_args.kwargs['text'], '1 / 2 eyes detected')

    def test_missing_feedback_hides_distance_marker_instead_of_retaining_old_value(self):
        view = self.make_feedback_view()
        view.update_eye_positions(.4, .5, .5, .6, .5, .5)
        view.update_eye_positions(None, None, None, float('nan'), float('nan'), float('nan'))
        view.z_canvas.itemconfig.assert_called_with('marker', state='hidden')
        self.assertEqual(view.tracking_label.config.call_args.kwargs['text'], '0 / 2 eyes detected')
        self.assertEqual(view.status_label.config.call_args.kwargs['text'], 'Waiting for position')

    def test_invalid_initial_size_is_rejected_before_creating_tk(self):
        for size in ((0, 100), (-1, 200), (300,), '300x200', (300.5, 200), (True, 100)):
            with self.subTest(size=size), patch.object(module.tk, 'Tk') as root:
                with self.assertRaises(ValueError):
                    module.TrackBoxView({'size': size})
                root.assert_not_called()

    def test_trackbox_resize_updates_guides_and_redraws_current_eyes(self):
        view = module.TrackBoxView.__new__(module.TrackBoxView)
        view.canvas = Mock()
        view.vertical_guide = 'vertical'
        view.horizontal_guide = 'horizontal'
        view._last_eye_positions = (0.5,) * 6
        view.update_eye_positions = Mock()
        view._resize_trackbox(SimpleNamespace(width=502, height=302))
        self.assertEqual((view.trackbox_width, view.trackbox_height), (500, 300))
        view.canvas.coords.assert_any_call('vertical', 250, 0, 250, 300)
        view.canvas.coords.assert_any_call('horizontal', 0, 150, 500, 150)
        view.update_eye_positions.assert_called_once_with(*((0.5,) * 6))

    def test_distance_resize_rebuilds_gradient_only_when_width_changes(self):
        view = module.TrackBoxView.__new__(module.TrackBoxView)
        view.z_bar_width = 150
        view._last_eye_positions = (None,) * 6
        view._draw_distance_bar = Mock()
        view.update_eye_positions = Mock()
        view._resize_distance_bar(SimpleNamespace(width=500))
        self.assertEqual(view.z_bar_width, 500)
        view._draw_distance_bar.assert_called_once()
        view._resize_distance_bar(SimpleNamespace(width=500))
        view._draw_distance_bar.assert_called_once()
        self.assertEqual(view.update_eye_positions.call_count, 2)


if __name__ == '__main__':
    unittest.main()
