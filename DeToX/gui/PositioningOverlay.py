"""Experimental participant positioning overlay shown before recording.

The overlay API and platform behavior may change; validate it on the intended
tracker/display setup before use.

``PositioningOverlay`` is a Tkinter window that floats over the PsychoPy
window during ``ETracker.show_status(overlay=True)``. It shows the Tobii
user-position guide (track box + distance bar) and, when the tracker supports
it, live eye-camera images.

It runs inside the main PsychoPy process and is refreshed by the frame loop
of ``show_status`` - it does not own its own event loop.
"""

import io
import threading
import tkinter as tk

import tobii_research as tr
from PIL import Image, ImageTk

from .TrackBox import TrackBoxView, get_screen_rect


try:
    from DeToX import ETSettings as cfg
    DEFAULT_LEFT_COLOR = cfg.colors.left_eye[:3]
    DEFAULT_RIGHT_COLOR = cfg.colors.right_eye[:3]
except ImportError:
    DEFAULT_LEFT_COLOR = (100, 200, 255)
    DEFAULT_RIGHT_COLOR = (255, 100, 120)


def _get_psychopy_window_geometry(win):
    """Return PsychoPy window geometry as x, y, width, height."""
    try:
        win_w, win_h = win.size
        if hasattr(win, 'pos') and win.pos is not None:
            return (int(win.pos[0]), int(win.pos[1]), int(win_w), int(win_h))

        if getattr(win, '_isFullScr', False):
            screen_num = getattr(win, 'screen', 0) or 0
            rect = get_screen_rect(screen_num)
            if rect is not None:
                return rect
            return (0, 0, int(win_w), int(win_h))
    except Exception:
        pass

    return (0, 0, 800, 600)


class PositioningOverlay(TrackBoxView):
    """
    Experimental Tkinter overlay for positioning with Tobii eye images.

    The GUI subscribes to Tobii user-position data and, when available, eye-image
    frames. It is updated synchronously from ETracker.show_status().
    """

    def __init__(self, eyetracker, win=None, scale=1.0, position='top-center', offset_y=0,
                 left_eye_color=None, right_eye_color=None):
        config = {
            'scale': scale,
            'left_eye_color': left_eye_color or DEFAULT_LEFT_COLOR,
            'right_eye_color': right_eye_color or DEFAULT_RIGHT_COLOR
        }

        super().__init__(config, show_images=True)

        self.tracker = eyetracker
        self.psychopy_win = win
        self.position = position
        self.offset_y = offset_y

        self.lock = threading.Lock()
        self.latest_position = None
        self.eye_image_0 = None
        self.eye_image_1 = None
        self.running = True
        self.pressed_keys = []
        self._position_subscribed = False
        self._eye_images_subscribed = False

        self.root.bind("<Key>", self._on_key)

        placeholder_pil = Image.new('L', (self.eye_img_size, self.eye_img_size), color=17)
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_pil)
        self.left_img_label.config(image=self.placeholder_photo)
        self.right_img_label.config(image=self.placeholder_photo)

        self.root.update_idletasks()
        self._position_window()
        self.root.deiconify()
        self.root.lift()
        self._subscribe()

    def _position_window(self):
        """Position the overlay relative to the PsychoPy window."""
        if self.psychopy_win:
            win_x, win_y, win_w, _ = _get_psychopy_window_geometry(self.psychopy_win)
        else:
            win_x, win_y, win_w = 0, 0, self.root.winfo_screenwidth()

        tk_w = self.root.winfo_width()

        if self.position == 'top-left':
            x = win_x + 30
        elif self.position == 'top-right':
            x = win_x + win_w - tk_w - 30
        else:
            x = win_x + (win_w - tk_w) // 2

        y = win_y + self.offset_y
        self.root.geometry(f"+{x}+{y}")

    def _on_position(self, data):
        """Store the latest Tobii user-position packet."""
        with self.lock:
            self.latest_position = data

    def _on_eye_image(self, data):
        """Store the latest Tobii eye-image packet."""
        img_bytes = data.get('image_data')
        region_id = data.get('region_id')
        if img_bytes:
            with self.lock:
                if region_id == 0:
                    self.eye_image_0 = img_bytes
                elif region_id == 1:
                    self.eye_image_1 = img_bytes

    def _subscribe(self):
        """Subscribe to Tobii position data and optional eye images."""
        self.tracker.subscribe_to(
            tr.EYETRACKER_USER_POSITION_GUIDE,
            self._on_position,
            as_dictionary=True
        )
        self._position_subscribed = True

        try:
            self.tracker.subscribe_to(
                tr.EYETRACKER_EYE_IMAGES,
                self._on_eye_image,
                as_dictionary=True
            )
            self._eye_images_subscribed = True
        except Exception as exc:
            print(f"Warning: eye images unavailable for overlay GUI: {exc}")

    def _unsubscribe(self):
        """Unsubscribe from streams that were successfully subscribed."""
        if not hasattr(self, 'tracker') or self.tracker is None:
            return

        if self._position_subscribed:
            try:
                self.tracker.unsubscribe_from(
                    tr.EYETRACKER_USER_POSITION_GUIDE,
                    self._on_position
                )
            except Exception as exc:
                print(f"Warning: failed to unsubscribe from position guide: {exc}")
            self._position_subscribed = False

        if self._eye_images_subscribed:
            try:
                self.tracker.unsubscribe_from(
                    tr.EYETRACKER_EYE_IMAGES,
                    self._on_eye_image
                )
            except Exception as exc:
                print(f"Warning: failed to unsubscribe from eye images: {exc}")
            self._eye_images_subscribed = False

    def update(self):
        """Pump latest Tobii data into the overlay and process Tk events."""
        if not self.running:
            return

        try:
            with self.lock:
                pos_data = self.latest_position
                img_0, img_1 = self.eye_image_0, self.eye_image_1
                # Consume the stored frames so each new image is decoded once.
                self.eye_image_0 = None
                self.eye_image_1 = None

            if pos_data:
                lv = pos_data.get("left_user_position_validity", 0)
                rv = pos_data.get("right_user_position_validity", 0)
                lx, ly, lz = pos_data.get("left_user_position", (None, None, None)) if lv else (None, None, None)
                rx, ry, rz = pos_data.get("right_user_position", (None, None, None)) if rv else (None, None, None)
                self.update_eye_positions(lx, ly, lz, rx, ry, rz)

            if img_1:
                self._update_eye_image(self.left_img_label, img_1, 'left')
            if img_0:
                self._update_eye_image(self.right_img_label, img_0, 'right')

            self.root.update()
        except tk.TclError:
            self.running = False

    def _update_eye_image(self, label, img_bytes, which):
        """Decode one eye image and assign it to a Tk label."""
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).resize(
                (self.eye_img_size, self.eye_img_size),
                Image.NEAREST
            )
            photo = ImageTk.PhotoImage(pil_img)
            label.config(image=photo)
            if which == 'left':
                self.left_photo = photo
            else:
                self.right_photo = photo
        except Exception:
            pass

    def close(self):
        """Close the overlay window and unsubscribe from Tobii streams."""
        self.running = False
        self._unsubscribe()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def is_running(self):
        """
        Return whether the overlay window is still running.

        Returns
        -------
        bool
            True while the Tk window is available for updates.
        """
        return self.running

    def _on_key(self, event):
        """Capture Tk key presses for show_status()."""
        self.pressed_keys.append(event.keysym.lower())

    def get_keys(self):
        """
        Return and clear key presses captured by the overlay.

        Returns
        -------
        list of str
            Lowercase Tk key names collected since the previous call.
        """
        keys = self.pressed_keys.copy()
        self.pressed_keys.clear()
        return keys
