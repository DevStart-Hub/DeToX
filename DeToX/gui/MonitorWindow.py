"""Experimental window shown by the detached live-monitor subprocess.

``MonitorWindow`` is a ``TrackBoxView`` that receives small binary
eye-position packets on stdin and updates the track-box display. It is
launched as a standalone script by ``LiveMonitor`` (one process for the whole
monitor), which keeps Tkinter's event loop out of the PsychoPy process.

This file is both a module and a script: run directly, it expects a JSON
config as its single argument.
"""

import json
import math
import struct
import sys
import threading

try:
    from .TrackBox import TrackBoxView
except ImportError:
    from TrackBox import TrackBoxView


# Wire format for packets sent by LiveMonitor.py (keep in sync).
PACKET_SIZE = 24
PACKET_FORMAT = '6f'


def _get_screen_geometry(root, screen_num):
    """Fallback monitor geometry from Tk (primary screen only)."""
    try:
        total_w, total_h = root.winfo_screenwidth(), root.winfo_screenheight()
        return (0, 0, total_w, total_h)
    except Exception:
        return (0, 0, 800, 600)


class MonitorWindow(TrackBoxView):
    """
    Experimental Tkinter window that displays live eye position.

    It receives binary eye-position packets on stdin and updates the shared
    TrackBoxView display. Runs inside its own subprocess.
    """

    def __init__(self, config, margin=40):
        self.margin = margin
        super().__init__(config, show_images=False)

        self.root.update_idletasks()
        tk_w = self.root.winfo_width()
        tk_h = self.root.winfo_height()

        screen_rect = self.config.get('screen_rect')
        if screen_rect:
            sx, sy, sw, sh = screen_rect
        else:
            sx, sy, sw, sh = _get_screen_geometry(
                self.root, self.config.get('screen', 0)
            )

        x = sx + max(margin, (sw - tk_w) // 2)
        y = sy + max(margin, (sh - tk_h) // 2)

        self.root.geometry(f"{tk_w}x{tk_h}+{x}+{y}")
        self.root.deiconify()
        self._start_listener()

    def _start_listener(self):
        """Start the stdin listener thread."""
        thread = threading.Thread(target=self._read_stdin, daemon=True)
        thread.start()

    def _read_stdin(self):
        """Read binary eye-position packets from stdin."""
        buf = b''

        while True:
            try:
                chunk = sys.stdin.buffer.read(PACKET_SIZE - len(buf))
                if not chunk:
                    break

                buf += chunk

                while len(buf) >= PACKET_SIZE:
                    pkt, buf = buf[:PACKET_SIZE], buf[PACKET_SIZE:]
                    lx, ly, lz, rx, ry, rz = struct.unpack(PACKET_FORMAT, pkt)
                    self.root.after(
                        0,
                        self.update_eye_positions,
                        None if math.isnan(lx) else lx,
                        None if math.isnan(ly) else ly,
                        None if math.isnan(lz) else lz,
                        None if math.isnan(rx) else rx,
                        None if math.isnan(ry) else ry,
                        None if math.isnan(rz) else rz
                    )
            except Exception:
                break

        try:
            self.root.after(0, self.root.quit)
        except Exception:
            pass

    def run(self):
        """Run the subprocess Tk event loop."""
        self.root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            monitor_config = json.loads(sys.argv[1])
            gui = MonitorWindow(monitor_config)
            gui.run()
        except Exception as exc:
            print(f"Monitor failed: {exc}", file=sys.stderr)
