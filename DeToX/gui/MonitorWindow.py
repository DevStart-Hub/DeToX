"""Experimental window shown by the detached live-monitor subprocess.

With ``tracker_address`` in its JSON configuration, this process connects
directly to Tobii for position feedback and optional camera images. Without an
address it receives simulated positions through binary stdin packets.

Callbacks and the stdin reader store data only. A Tk timer updates the display
on the main thread. No DeToX controller, recording buffer or PsychoPy is needed.

This file is both a module and a script: run directly, it expects a JSON
config as its single argument.
"""

import io
import json
import math
import os
import struct
import sys
import threading
import time

try:
    from .TrackBox import TrackBoxView
except ImportError:
    from TrackBox import TrackBoxView


# Wire format for packets sent by LiveMonitor.py (keep in sync).
PACKET_FORMAT = '6f'
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)


def _get_screen_geometry(root):
    """Fallback monitor geometry from Tk (primary screen only)."""
    try:
        total_w, total_h = root.winfo_screenwidth(), root.winfo_screenheight()
        return (0, 0, total_w, total_h)
    except Exception:
        return (0, 0, 800, 600)


class MonitorWindow(TrackBoxView):
    """
    Experimental Tkinter window that displays live eye position.

    Parameters
    ----------
    config : dict
        Display configuration accepted by TrackBoxView. ``tracker_address``
        selects a direct Tobii connection; omit it for legacy binary stdin.
        ``update_rate`` sets the display refresh rate in Hz (default 20).
        ``stale_timeout`` hides feedback after this many seconds without new
        samples (default 1). Both values must be finite and greater than zero.
        ``report_ready=True`` prints a JSON startup report for test launchers.
        When run as a script, ``wait_for_start=True`` waits for a ``start`` line
        on stdin before creating the window or connecting to the tracker.
        ``size=(width, height)`` sets the initial content size in pixels.
        ``resizable`` allows dragging the window edges (default True).
        ``show_images=True`` adds the latest image from each camera (default
        False; real hardware only). Requires Pillow. Unsupported image streams
        or missing Pillow leave position feedback working without camera images.
        Unsupported images or failed image subscriptions hide both image panels
        and print one warning to stderr. Images disabled by choice do not warn.
        Individual unreadable frames are cleared and skipped, with one decoding
        warning; later valid frames can still be displayed.
        ``reverse_eye=True`` swaps the image panels. By default camera 1 is on
        the left and camera 0 on the right. Source-region selection is unchanged;
        position dots and recorded data are not swapped or mirrored.
    margin : int, optional
        Minimum screen-edge margin in pixels. Default is 40.

    Notes
    -----
    This is feedback, not a recorder. Only the latest position is retained.
    The process does not change calibration or tracker settings. Closing stdin
    requests shutdown in both modes. A frozen process requires its launcher
    to enforce termination; pipe EOF alone is not a hard-crash guarantee.
    """

    def __init__(self, config, margin=40):
        config = dict(config)
        config.setdefault('resizable', True)
        rate = float(config.get('update_rate', 20))
        self.stale_timeout = float(config.get('stale_timeout', 1))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('update_rate must be finite and greater than zero')
        if not math.isfinite(self.stale_timeout) or self.stale_timeout <= 0:
            raise ValueError('stale_timeout must be finite and greater than zero')
        self.refresh_ms = max(1, round(1000 / rate))
        self._sample_lock = threading.Lock()
        self._latest_positions = (None,) * 6
        self._last_sample_time = None
        self._stop_requested = threading.Event()
        self._latest_images = {}  # (camera, region) -> (encoded bytes, receive time)
        self._displayed_frames = {}
        self._photos = {}
        self._image_warning_reported = False
        self._image_backend = None
        show_images = bool(config.get('show_images') and config.get('tracker_address'))
        if show_images:
            try:
                from PIL import Image, ImageTk
                self._image_backend = (Image, ImageTk)
            except ImportError as exc:
                print(f'[Eye Monitor] Images unavailable: {exc}', file=sys.stderr)
                show_images = False
        super().__init__(config, show_images=show_images)
        if show_images:
            left_camera, right_camera = (1, 0)
            if config.get('reverse_eye', False):
                left_camera, right_camera = (0, 1)
            self._image_labels = {left_camera: self.left_img_label,
                                  right_camera: self.right_img_label}
            Image, ImageTk = self._image_backend
            placeholder = Image.new('L', (self.eye_img_size, self.eye_img_size), color=17)
            self._placeholder_photo = ImageTk.PhotoImage(placeholder, master=self.root)
            self.left_img_label.config(image=self._placeholder_photo)
            self.right_img_label.config(image=self._placeholder_photo)

        self.root.update_idletasks()
        tk_w = self.root.winfo_width()
        tk_h = self.root.winfo_height()
        # A withdrawn window may still report its old dimensions. Preserve the
        # requested startup size when adding the screen-position coordinates.
        if self.config.get('size') is not None:
            tk_w, tk_h = self.config['size']

        screen_rect = self.config.get('screen_rect')
        if screen_rect:
            sx, sy, sw, sh = screen_rect
        else:
            sx, sy, sw, sh = _get_screen_geometry(self.root)

        x = sx + max(margin, (sw - tk_w) // 2)
        y = sy + max(margin, (sh - tk_h) // 2)

        self.root.geometry(f"{tk_w}x{tk_h}+{x}+{y}")
        self.root.deiconify()
        self.root.protocol('WM_DELETE_WINDOW', self.root.quit)
        self._start_listener()

    def _start_listener(self):
        """Start a daemon reader; it never calls Tk methods."""
        thread = threading.Thread(target=self._read_stdin, daemon=True)
        thread.start()

    def _read_stdin(self):
        """Watch parent EOF, or receive legacy packets, without touching Tk."""
        try:
            # Raw reads avoid holding Python's buffered stdin lock when a user
            # closes the window while this daemon thread is waiting for input.
            input_fd = sys.stdin.fileno()
            if self.config.get('tracker_address'):
                # Direct mode carries no samples through the parent pipe.
                while os.read(input_fd, 1):
                    pass
                return
            buf = b''
            while True:
                chunk = os.read(input_fd, PACKET_SIZE - len(buf))
                if not chunk:
                    break
                buf += chunk
                if len(buf) == PACKET_SIZE:
                    values = struct.unpack(PACKET_FORMAT, buf)
                    self._store_positions(tuple(None if math.isnan(v) else v for v in values))
                    buf = b''
        except (OSError, ValueError) as exc:
            print(f'Monitor input closed: {exc}', file=sys.stderr)
        finally:
            self._stop_requested.set()

    def _store_positions(self, positions):
        """Replace the latest six coordinates under a short, child-local lock."""
        with self._sample_lock:
            self._latest_positions = positions
            self._last_sample_time = time.monotonic()

    def _on_user_position(self, sample):
        """
        Store one Tobii position-guide sample without drawing or doing I/O.

        Parameters
        ----------
        sample : dict
            SDK dictionary containing left/right user-position tuples and their
            validity flags. Invalid eyes are represented by three None values.
        """
        left = (None, None, None)
        right = (None, None, None)
        if sample.get('left_user_position_validity'):
            left = tuple(sample['left_user_position'])
        if sample.get('right_user_position_validity'):
            right = tuple(sample['right_user_position'])
        self._store_positions(left + right)

    def _refresh(self):
        """Draw the latest feedback and schedule another tick on the Tk thread."""
        if self._stop_requested.is_set():
            self.root.quit()
            return
        with self._sample_lock:
            positions = self._latest_positions
            sample_time = self._last_sample_time
        if sample_time is None or time.monotonic() - sample_time > self.stale_timeout:
            positions = (None,) * 6
        self.update_eye_positions(*positions)
        if self.show_images:
            self._refresh_images()
        self.root.after(self.refresh_ms, self._refresh)

    def _on_eye_image(self, sample):
        """
        Retain the latest image per camera/region pair, without drawing or I/O.

        Parameters
        ----------
        sample : dict
            Tobii dictionary containing camera_id, region_id and image_data.
            Keep regions separate so two crops from one camera cannot overwrite
            each other. At most four regions per camera are retained.
        """
        camera = sample.get('camera_id')
        region = sample.get('region_id', 0)
        image = sample.get('image_data')
        if camera not in (0, 1) or not isinstance(region, int) or region < 0 or not image:
            return
        key = (camera, region)
        with self._sample_lock:
            camera_keys = [source for source in self._latest_images if source[0] == camera]
            if key not in self._latest_images and len(camera_keys) >= 4:
                oldest = min(camera_keys, key=lambda source: self._latest_images[source][1])
                del self._latest_images[oldest]
            self._latest_images[key] = (image, time.monotonic())

    def _refresh_images(self):
        """Keep panels on distinct cameras and prefer region 0/1 respectively.

        If a preferred region is unavailable, use another fresh region from that
        same camera, never copy the other camera's image. These defaults are not
        an anatomical left/right claim.
        """
        with self._sample_lock:
            images = dict(self._latest_images)
        now = time.monotonic()
        for camera, label in self._image_labels.items():
            sources = [key for key, (_, received_at) in images.items()
                       if key[0] == camera and now - received_at <= self.stale_timeout]
            if not sources:
                label.config(image=self._placeholder_photo)
                self._photos.pop(camera, None)
                self._displayed_frames.pop(camera, None)
                continue
            preferred = (camera, camera)  # Camera 0 / region 0; camera 1 / region 1.
            source_key = preferred if preferred in sources else min(sources)
            encoded_image, received_at = images[source_key]
            version = (source_key, received_at)
            if self._displayed_frames.get(camera) == version:
                continue
            self._displayed_frames[camera] = version
            try:
                photo = self._create_photo(encoded_image)
                label.config(image=photo)
                self._photos[camera] = photo  # Tk needs a live Python reference.
            except Exception as exc:
                label.config(image=self._placeholder_photo)
                self._photos.pop(camera, None)
                if not self._image_warning_reported:
                    print(f'[Eye Monitor] Could not display camera image: {exc}', file=sys.stderr)
                    self._image_warning_reported = True

    def _create_photo(self, encoded_image):
        """Decode and center one camera frame without changing its aspect ratio."""
        Image, ImageTk = self._image_backend
        size = self.eye_img_size
        with Image.open(io.BytesIO(encoded_image)) as source:
            frame = source.convert('RGB')
            frame.thumbnail((size, size))
        panel = Image.new('RGB', (size, size), color=(17, 17, 17))
        panel.paste(frame, ((size - frame.width) // 2, (size - frame.height) // 2))
        return ImageTk.PhotoImage(panel, master=self.root)

    def _subscribe_images(self, tracker, sdk):
        """Try optional images; a capability or subscription failure keeps positions working."""
        if not self.show_images:
            return False
        try:
            if sdk.CAPABILITY_HAS_EYE_IMAGES not in tracker.device_capabilities:
                self._disable_images('This tracker does not support eye images.')
                return False
            tracker.subscribe_to(sdk.EYETRACKER_EYE_IMAGES,
                                 self._on_eye_image, as_dictionary=True)
        except Exception as exc:
            self._disable_images(f'Could not enable eye images: {exc}')
            return False
        return True

    def run(self):
        """
        Run feedback until the window closes or the parent closes stdin.

        Notes
        -----
        A direct SDK connection is created here, inside this process. Shutdown
        unsubscribes only this process's position/image callbacks and destroys Tk,
        including when connection setup or the event loop raises an exception.
        Required connection/position errors propagate to the script entry point,
        which exits nonzero. Optional image failures keep position feedback running.
        """
        tracker = None
        position_subscribed = False
        images_subscribed = False
        try:
            address = self.config.get('tracker_address')
            if address and not self._stop_requested.is_set():
                import tobii_research as tr
                tracker = tr.EyeTracker(address)
                tracker.subscribe_to(tr.EYETRACKER_USER_POSITION_GUIDE,
                                     self._on_user_position, as_dictionary=True)
                position_subscribed = True
                images_subscribed = self._subscribe_images(tracker, tr)
            self._refresh()
            if self.config.get('report_ready'):
                print(json.dumps({'ready': True, 'direct': position_subscribed,
                                  'images': images_subscribed}), flush=True)
            if not self._stop_requested.is_set():
                self.root.mainloop()
        finally:
            self._stop_requested.set()
            try:
                if images_subscribed:
                    try:
                        tracker.unsubscribe_from(tr.EYETRACKER_EYE_IMAGES, self._on_eye_image)
                    except Exception as exc:
                        print(f'[Eye Monitor] Image unsubscribe failed: {exc}', file=sys.stderr)
                if position_subscribed:
                    tracker.unsubscribe_from(tr.EYETRACKER_USER_POSITION_GUIDE,
                                             self._on_user_position)
            finally:
                self.root.destroy()

    def _disable_images(self, reason):
        """Hide unavailable camera panels and warn once, preserving position feedback."""
        if not self.show_images:
            return
        self.hide_image_panels()
        print(f'[Eye Monitor] Warning: {reason} Continuing with position feedback only.',
              file=sys.stderr)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError('Expected one JSON configuration argument')
        monitor_config = json.loads(sys.argv[1])
        # Optional startup gate: the launcher assigns OS ownership first.
        if monitor_config.get('wait_for_start'):
            # Do not prefetch simulation bytes into Python's buffered stdin:
            # the background reader uses raw os.read after this exact token.
            token = b''
            while len(token) < len(b'start\n'):
                chunk = os.read(sys.stdin.fileno(), len(b'start\n') - len(token))
                if not chunk:
                    break
                token += chunk
            if token != b'start\n':
                raise SystemExit(0)
        gui = MonitorWindow(monitor_config)
        gui.run()
    except Exception as exc:
        print(f"Monitor failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
