"""Launch and own the optional monitor without using the real gaze callback.

Real hardware is read directly by the child. Mouse simulation uses a bounded
latest-packet slot and a background writer; it never writes a pipe in push().
"""

import json
import math
import os
import struct
import subprocess
import sys
import threading
import time

try:
    from .ProcessLifecycle import WindowsProcessJob
except ImportError:
    from ProcessLifecycle import WindowsProcessJob


PACKET_FORMAT = '6f'


class LiveMonitor:
    """
    Launch the independent eye-position subprocess.

    Parameters
    ----------
    scale : float, optional
        Scale for fonts, padding and eye dots. Default is 1.
    screen : int, optional
        Screen index, used with the pixel rectangle resolved by DeToX.
    screen_rect : tuple or None, optional
        Target display rectangle as (x, y, width, height).
    update_rate : float, optional
        Maximum feedback refresh rate in Hz. Default is 20.
    left_eye_color, right_eye_color : tuple or None, optional
        RGB eye colors, with components from 0 to 255.
    tracker_address : str or None, optional
        Tracker URI for the child's own SDK connection. None selects simulation
        packets; no SDK object is shared between processes.
    size : tuple or None, optional
        Initial content size in pixels as (width, height).
    resizable : bool, optional
        Allow resizing by dragging the window edges. Default is True.
    show_images : bool, optional
        Show optional camera images in the child. Default is False. Real tracker
        only; unsupported or failed image streams hide the image panels and
        print one warning, without disabling position feedback.
    reverse_eye : bool, optional
        Swap camera panel positions when images are enabled. Default is False.
        Does not change position dots or gaze data.

    Notes
    -----
    Startup does not wait for the GUI or SDK to become ready. Child diagnostics
    inherit stderr. Windows ownership is established before initialization;
    other platforms have cooperative EOF cleanup only. Stop runs bounded process
    cleanup in a background thread unless wait=True is explicitly requested.
    """

    def __init__(self, scale=1.0, screen=0, screen_rect=None, update_rate=20,
                 left_eye_color=None, right_eye_color=None, tracker_address=None,
                 size=None, resizable=True, show_images=False, reverse_eye=False):
        if not math.isfinite(update_rate) or update_rate <= 0:
            raise ValueError('update_rate must be finite and greater than zero')
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be finite and greater than zero')
        if size is not None:
            if (not isinstance(size, (list, tuple)) or len(size) != 2
                    or any(type(value) is not int or value <= 0 for value in size)):
                raise ValueError('size must contain two positive integers')
        self.process = None
        self._job = None
        self._direct = tracker_address is not None
        self._stopped = threading.Event()
        self._packet_ready = threading.Event()
        self._packet_lock = threading.Lock()
        self._packet = None
        self._update_interval = 1 / update_rate
        self._last_update_time = 0
        config = {
            'scale': scale, 'screen': screen,
            'screen_rect': list(screen_rect) if screen_rect else None,
            'update_rate': update_rate, 'tracker_address': tracker_address,
            'size': size, 'resizable': resizable, 'wait_for_start': True,
            'show_images': show_images,
            'reverse_eye': reverse_eye,
            'status_text': 'Waiting for eyes' if self._direct else 'Waiting for simulated gaze',
            'left_eye_color': list(left_eye_color or (0, 212, 255)),
            'right_eye_color': list(right_eye_color or (255, 110, 199)),
        }
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MonitorWindow.py')
        try:
            if sys.platform == 'win32':
                self._job = WindowsProcessJob()
            self.process = subprocess.Popen(
                [sys.executable, script, json.dumps(config)],
                stdin=subprocess.PIPE, bufsize=0)
            if self._job is not None:
                self._job.assign(self.process.pid)
            self.process.stdin.write(b'start\n')
            if not self._direct:
                threading.Thread(target=self._write_packets, daemon=True).start()
        except Exception:
            self.stop(wait=True)
            raise

    def is_alive(self):
        """Return whether the launched process is still running, not SDK readiness."""
        return self.process is not None and self.process.poll() is None

    def push(self, left_eye, right_eye):
        """
        Offer simulated eye coordinates without waiting on a pipe.

        Parameters
        ----------
        left_eye, right_eye : tuple or None
            Normalized (x, y, z) coordinates. None represents an invalid eye.

        Notes
        -----
        Ignored in direct hardware mode. Only one pending packet is retained;
        newer simulation feedback replaces older feedback while the child is busy.
        """
        if self._direct or self._stopped.is_set():
            return
        now = time.monotonic()
        if now - self._last_update_time < self._update_interval:
            return
        self._last_update_time = now
        left = left_eye if left_eye is not None else (float('nan'),) * 3
        right = right_eye if right_eye is not None else (float('nan'),) * 3
        packet = struct.pack(PACKET_FORMAT, *left, *right)
        with self._packet_lock:
            self._packet = packet
        self._packet_ready.set()

    def _write_packets(self):
        """Perform simulation-only pipe writes away from acquisition threads."""
        process = self.process
        try:
            while not self._stopped.is_set():
                self._packet_ready.wait()
                with self._packet_lock:
                    packet = self._packet
                    self._packet = None
                    self._packet_ready.clear()
                if packet is None or self._stopped.is_set():
                    continue
                # Raw writes avoid holding a Python file lock during shutdown.
                remaining = packet
                while remaining and not self._stopped.is_set():
                    written = os.write(process.stdin.fileno(), remaining)
                    if written == 0:
                        raise BrokenPipeError('Monitor input closed')
                    remaining = remaining[written:]
        except (OSError, ValueError):
            self._stopped.set()

    def stop(self, wait=False):
        """
        Request shutdown and reap the child without delaying recording or saving.

        Parameters
        ----------
        wait : bool, optional
            Wait for bounded cleanup instead of starting a background reaper.
            Default is False. Safe to call repeatedly.
        """
        self._stopped.set()
        self._packet_ready.set()
        process, job = self.process, self._job
        self.process = None
        self._job = None
        if process is None:
            if job is not None:
                job.close()
            return
        if wait:
            self._reap(process, job)
        else:
            threading.Thread(target=self._reap, args=(process, job), daemon=True).start()

    @staticmethod
    def _close_input(process):
        """Request EOF away from the caller; a blocked writer may delay close on Windows."""
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _reap(process, job):
        """Try EOF first, then terminate/kill; always release Windows ownership."""
        # Closing a Windows pipe may wait for a blocked writer. Keep termination
        # independent of that close, so a frozen reader cannot stall shutdown.
        try:
            threading.Thread(target=LiveMonitor._close_input, args=(process,), daemon=True).start()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f'[Eye Monitor] Cleanup failed: {exc}', file=sys.stderr)
        finally:
            if job is not None:
                try:
                    job.close()
                except OSError as exc:
                    print(f'[Eye Monitor] Ownership cleanup failed: {exc}', file=sys.stderr)
