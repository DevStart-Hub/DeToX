"""Experimental client that drives the detached live eye-position monitor.

The monitor API and platform behavior may change; validate it on the intended
tracker/display setup before use.

``LiveMonitor`` launches ``MonitorWindow.py`` in a separate process and
streams throttled, non-blocking eye-position packets to it over stdin.

Running the window in its own process keeps Tkinter isolated from the
PsychoPy process, so a busy or frozen monitor can never stall data
collection.
"""

import json
import os
import struct
import subprocess
import sys
import time


# Wire format for packets sent to MonitorWindow.py (keep in sync).
PACKET_FORMAT = '6f'


class LiveMonitor:
    """
    Experimental client for a detached live eye-position monitor subprocess.

    The parent process sends throttled binary eye-position packets to the child
    process over stdin, keeping Tkinter isolated from the PsychoPy process.
    """

    def __init__(self, scale=1.0, screen=0, screen_rect=None, update_rate=20,
                 left_eye_color=None, right_eye_color=None,
                 display_size_mm=(555.0, 312.0), approximate=False):
        config = {
            'scale': scale,
            'screen': screen,
            'screen_rect': list(screen_rect) if screen_rect else None,
            'status_text': 'Gaze appears when recording starts',
            'left_eye_color': list(left_eye_color or (0, 212, 255)),
            'right_eye_color': list(right_eye_color or (255, 110, 199)),
            'approximate': approximate
        }

        monitor_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'MonitorWindow.py'
        )

        self.process = subprocess.Popen(
            [sys.executable, monitor_script, json.dumps(config)],
            stdin=subprocess.PIPE,
            bufsize=0
        )
        try:
            os.set_blocking(self.process.stdin.fileno(), False)
        except (OSError, AttributeError):
            pass
        self._alive = True
        self._update_interval = 1.0 / update_rate if update_rate > 0 else 0
        self._last_update_time = 0
        self._display_size_mm = display_size_mm

    def push_origin(self, gaze_data):
        """
        Send one gaze sample's eye origins to the monitor.

        Origins are in the user coordinate system (millimetres). They are mapped
        onto the monitor's normalized 0-1 view: x and y use the display size, and
        z uses the tracker's 50-80 cm operating range. This mapping is approximate
        because the SDK exposes no track-box coordinate system.

        Parameters
        ----------
        gaze_data : dict
            Tobii gaze sample (``as_dictionary=True``) with the per-eye
            ``*_gaze_origin_in_user_coordinate_system`` and validity fields.
        """
        width_mm, height_mm = self._display_size_mm

        def normalized(origin, valid):
            if not valid or origin is None:
                return None
            x, y, z = origin
            return (0.5 - x / width_mm, 0.5 - y / height_mm, (z - 500.0) / 300.0)

        self.push(
            normalized(gaze_data.get('left_gaze_origin_in_user_coordinate_system'),
                       gaze_data.get('left_gaze_origin_validity', 0)),
            normalized(gaze_data.get('right_gaze_origin_in_user_coordinate_system'),
                       gaze_data.get('right_gaze_origin_validity', 0)),
        )

    def push(self, left_eye, right_eye):
        """
        Send one eye-position update to the monitor.

        Parameters
        ----------
        left_eye, right_eye : tuple or None
            Normalized (0-1) coordinates as (x, y, z). None marks an invalid eye.

        Notes
        -----
        This never blocks the caller. If the monitor is busy and cannot accept
        the update immediately, the update is dropped and the next one is sent.
        """
        if not self._alive:
            return

        current_time = time.time()
        if current_time - self._last_update_time < self._update_interval:
            return

        self._last_update_time = current_time

        try:
            lx, ly, lz = (float(x) for x in left_eye) if left_eye else (float('nan'), float('nan'), float('nan'))
            rx, ry, rz = (float(x) for x in right_eye) if right_eye else (float('nan'), float('nan'), float('nan'))
            packet = struct.pack(PACKET_FORMAT, lx, ly, lz, rx, ry, rz)
            sent = self.process.stdin.write(packet)
            if sent is None:
                return
        except BlockingIOError:
            return
        except (BrokenPipeError, OSError, ValueError, AttributeError):
            self._alive = False

    def stop(self):
        """Stop the monitor subprocess, preferring graceful EOF shutdown."""
        self._alive = False

        if not self.process:
            return

        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass

        try:
            self.process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
