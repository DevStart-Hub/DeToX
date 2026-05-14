# LiveMonitor.py
import sys
import struct
import math
import threading
import subprocess
import json
import os
import time

# --- THE FIX ---
try:
    # This works when imported normally by your main experiment script (Base.py)
    from .BaseEyeGUI import CoreEyeGUI
except ImportError:
    # This works when it runs as a detached subprocess in the background
    from BaseEyeGUI import CoreEyeGUI
# ---------------

PACKET_SIZE = 24
PACKET_FORMAT = '6f'

def _get_screen_geometry(root, screen_num):
    """Determine coordinate offset for the requested monitor."""
    try:
        total_w, total_h = root.winfo_screenwidth(), root.winfo_screenheight()
        if screen_num == 0: return (0, 0, total_w, total_h)
        return (1920 * screen_num, 0, 1920, 1080)
    except Exception:
        return (0, 0, 800, 600)

class EyeMonitorGUI(CoreEyeGUI):
    """Server class: Runs entirely within the subprocess."""
    def __init__(self, config):
        super().__init__(config, show_images=False)
        
        # Force layout calculation before getting dimensions
        self.root.update_idletasks()
        
        # Get actual window size (now accurate after update_idletasks)
        tk_w = self.root.winfo_width()
        tk_h = self.root.winfo_height()
        
        # Position on requested screen
        screen_num = self.config.get('screen', 0)
        screen_x, screen_y, _, _ = _get_screen_geometry(self.root, screen_num)
        
        self.root.geometry(f"{tk_w}x{tk_h}+{screen_x + 50}+{screen_y + 50}")
        
        self.root.deiconify()  # Show window after positioning
        
        self._start_listener()
        
    def _start_listener(self):
        thread = threading.Thread(target=self._read_stdin, daemon=True)
        thread.start()
        
    def _read_stdin(self):
        while True:
            try:
                data = sys.stdin.buffer.read(PACKET_SIZE)
                if len(data) < PACKET_SIZE: break
                lx, ly, lz, rx, ry, rz = struct.unpack(PACKET_FORMAT, data)
                self.root.after(0, self.update_eye_positions, 
                                None if math.isnan(lx) else lx, None if math.isnan(ly) else ly, None if math.isnan(lz) else lz,
                                None if math.isnan(rx) else rx, None if math.isnan(ry) else ry, None if math.isnan(rz) else rz)
            except Exception:
                break
        self.root.after(0, self.root.quit)

    def run(self):
        self.root.mainloop()

class LiveMonitor:
    """Client class: instantiated by main experiment script."""
    def __init__(self, scale=1.0, screen=0, update_rate=20, left_eye_color=None, right_eye_color=None):
        config = {
            'scale': scale,
            'screen': screen,
            'left_eye_color': list(left_eye_color or (0, 212, 255)),
            'right_eye_color': list(right_eye_color or (255, 110, 199))
        }
        self.process = subprocess.Popen([sys.executable, __file__, json.dumps(config)], stdin=subprocess.PIPE)
        self._alive = True
        
        # Throttling Variables
        self._update_interval = 1.0 / update_rate if update_rate > 0 else 0
        self._last_update_time = 0
        
    def push(self, left_eye, right_eye):
        if not self._alive: return
        
        # Throttle the pipe I/O
        current_time = time.time()
        if current_time - self._last_update_time < self._update_interval:
            return  # Skip this frame to save I/O overhead
            
        self._last_update_time = current_time
        
        try:
            lx, ly, lz = (float(x) for x in left_eye) if left_eye else (float('nan'), float('nan'), float('nan'))
            rx, ry, rz = (float(x) for x in right_eye) if right_eye else (float('nan'), float('nan'), float('nan'))
            self.process.stdin.write(struct.pack(PACKET_FORMAT, lx, ly, lz, rx, ry, rz))
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._alive = False

    def is_alive(self) -> bool:
        """Return True if the monitor subprocess is still running."""
        if not self._alive:
            return False
        if self.process is not None and self.process.poll() is not None:
            self._alive = False
            return False
        return True

    def stop(self):
        self._alive = False
        if self.process:
            self.process.terminate()
            try: self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired: self.process.kill()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
            gui = EyeMonitorGUI(config)
            gui.run()
        except Exception as e:
            print(f"Monitor failed: {e}", file=sys.stderr)