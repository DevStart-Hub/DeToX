"""
LiveMonitor - Detached Process Version (Binary Protocol)
========================================================

A cross-platform eye position monitor that runs as a completely independent
subprocess using efficient binary communication.

Architecture
------------
This module implements a high-performance "Client-Server" model using 
binary standard input/output pipes:

1.  **Client (`LiveMonitor` class):**
    Launches the subprocess and sends eye data as fixed-size 24-byte packets.
    It uses Python's `struct` module to pack 6 floats (Left X, Y, Z, Right X, Y, Z).

2.  **Server (`EyeMonitorGUI` class):**
    Runs in the separate process. It hosts a Tkinter window and a background 
    thread that constantly reads 24-byte chunks from `stdin`.

Performance Note
----------------
Binary encoding is significantly faster than JSON for high-frequency data (120Hz+).
- Binary: 24 bytes per frame.
- JSON:   ~60-80 bytes per frame (plus parsing overhead).
"""

import sys
import struct
import math
import threading
import tkinter as tk
import subprocess
import json
import os

# --- Constants ---

BASE_WIDTH = 220
BASE_HEIGHT = 300
TRACKBOX_RATIO = 1.33

# Protocol definition: 6 floats (lx, ly, lz, rx, ry, rz)
# '6f' means 6 standard 4-byte floats. 6 * 4 = 24 bytes total.
PACKET_SIZE = 24
PACKET_FORMAT = '6f'


# --- Helper Functions ---

def _interpolate_color(color1, color2, t):
    """
    Linearly interpolate between two RGB colors.
    
    Parameters
    ----------
    color1, color2 : tuple
        (r, g, b) tuples for start and end colors.
    t : float
        Interpolation factor (0.0 to 1.0).
    """
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    return f'#{r:02x}{g:02x}{b:02x}'


def _get_screen_geometry(root, screen_num):
    """
    Determine the coordinate offset for the requested monitor.
    
    Attempts to use 'screeninfo' if available, otherwise falls back to a 
    standard 1920x1080 offset assumption.
    """
    try:
        total_width = root.winfo_screenwidth()
        total_height = root.winfo_screenheight()
        
        if screen_num == 0:
            return (0, 0, total_width, total_height)
        
        # Try to use external library for accurate multi-monitor data
        try:
            from screeninfo import get_monitors
            monitors = list(get_monitors())
            if screen_num < len(monitors):
                m = monitors[screen_num]
                return (m.x, m.y, m.width, m.height)
        except ImportError:
            pass
        
        # Fallback: Assume monitor is to the right
        return (1920 * screen_num, 0, 1920, 1080)
            
    except Exception:
        return (0, 0, 800, 600)


# --- Server Logic (GUI Process) ---

class EyeMonitorGUI:
    """
    Tkinter-based eye position monitor window.
    
    This class manages the GUI elements and the background data listener.
    It runs entirely within the subprocess launched by the main script.
    """
    
    # Pre-defined colors for the distance bar gradient
    RED = (180, 40, 40)
    YELLOW = (180, 180, 40)
    GREEN = (40, 160, 40)
    
    def __init__(self, config):
        """
        Initialize the GUI with configuration passed from the parent process.
        
        Parameters
        ----------
        config : dict
            Dictionary containing 'scale', 'screen', colors, etc.
        """
        self.config = config
        self.scale = config.get('scale', 1.0)
        
        # Parse custom eye colors from config
        left_color = config.get('left_eye_color', [0, 212, 255])
        right_color = config.get('right_eye_color', [255, 110, 199])
        
        # Convert RGB lists to hex strings for Tkinter
        self.left_hex = f'#{left_color[0]:02x}{left_color[1]:02x}{left_color[2]:02x}'
        self.right_hex = f'#{right_color[0]:02x}{right_color[1]:02x}{right_color[2]:02x}'
        
        # Calculate scaled dimensions
        self.win_width = int(BASE_WIDTH * self.scale)
        self.win_height = int(BASE_HEIGHT * self.scale)
        self.dot_radius = max(5, int(7 * self.scale))
        
        # Initialize UI Components
        self._create_window()
        self._create_trackbox()
        self._create_distance_bar()
        self._create_status_label()
        
        # Start the background thread for data reception
        self._start_listener()
    
    def _create_window(self):
        """Setup the main Tkinter window properties."""
        self.root = tk.Tk()
        self.root.title("DeToX Eye Monitor")
        self.root.attributes("-topmost", True)  # Keep on top of other windows
        self.root.configure(bg='#1a1a1a')
        self.root.resizable(False, False)
              
        # Position the window on the requested screen
        screen_num = self.config.get('screen', 0)
        screen_x, screen_y, _, _ = _get_screen_geometry(self.root, screen_num)
        
        # Offset by 50px to avoid hiding behind OS taskbars
        self.root.geometry(f"{self.win_width}x{self.win_height}+{screen_x + 50}+{screen_y + 50}")
    
    def _create_trackbox(self):
        """Setup the X/Y coordinate visualization area."""
        padding = int(20 * self.scale)
        available_width = self.win_width - (2 * padding)
        available_height = int(self.win_height * 0.55)
        
        # Calculate dimensions while maintaining Aspect Ratio (4:3)
        if available_width / available_height > TRACKBOX_RATIO:
            self.canvas_height = available_height
            self.canvas_width = int(self.canvas_height * TRACKBOX_RATIO)
        else:
            self.canvas_width = available_width
            self.canvas_height = int(self.canvas_width / TRACKBOX_RATIO)
        
        # Main drawing canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=self.canvas_width, 
            height=self.canvas_height,
            bg='#111111', 
            highlightthickness=1, 
            highlightbackground="#333333"
        )
        self.canvas.pack(pady=(int(15 * self.scale), int(10 * self.scale)))
        
        # Draw reference crosshair
        cx, cy = self.canvas_width // 2, self.canvas_height // 2
        self.canvas.create_line(cx, 0, cx, self.canvas_height, fill="#333333", dash=(2, 2))
        self.canvas.create_line(0, cy, self.canvas_width, cy, fill="#333333", dash=(2, 2))
        
        # Initialize Eye Dots (Hidden by default)
        self.left_dot = self.canvas.create_oval(0, 0, 0, 0, fill=self.left_hex, 
                                                 outline="white", state='hidden')
        self.right_dot = self.canvas.create_oval(0, 0, 0, 0, fill=self.right_hex, 
                                                  outline="white", state='hidden')
        
        tk.Label(self.root, text="Track Box", fg="#666666", bg='#1a1a1a',
                 font=("Arial", int(9 * self.scale))).pack()
    
    def _create_distance_bar(self):
        """Setup the Z-axis distance indicator."""
        padding = int(20 * self.scale)
        available_width = self.win_width - (2 * padding)
        
        self.z_bar_width = min(available_width, int(200 * self.scale))
        z_bar_height = int(35 * self.scale)
        
        self.z_canvas = tk.Canvas(
            self.root, 
            width=self.z_bar_width + 20, 
            height=z_bar_height,
            bg='#1a1a1a', 
            highlightthickness=0
        )
        self.z_canvas.pack(pady=(int(15 * self.scale), int(5 * self.scale)))
        
        # Define bar geometry
        self.bar_left = 10
        self.bar_right = self.bar_left + self.z_bar_width
        self.bar_top = int(12 * self.scale)
        self.bar_bottom = int(28 * self.scale)
        self.marker_top = int(7 * self.scale)
        self.marker_bottom = int(33 * self.scale)
        
        # Draw the color gradient
        self._draw_gradient()
        
        # Initialize the moving marker
        z_center = self.bar_left + self.z_bar_width // 2
        self.z_marker = self.z_canvas.create_rectangle(
            z_center - 3, self.marker_top, z_center + 3, self.marker_bottom,
            fill="white", outline="black"
        )
        
        # Add labels "Near" and "Far"
        label_y = (self.bar_top + self.bar_bottom) // 2
        self.z_canvas.create_text(self.bar_left + 10, label_y, text="Near",
                                  fill="#ffffff", font=("Arial", int(8 * self.scale)), anchor="w")
        self.z_canvas.create_text(self.bar_right - 10, label_y, text="Far",
                                  fill="#ffffff", font=("Arial", int(8 * self.scale)), anchor="e")
    
    def _draw_gradient(self):
        """Draw the Red-Yellow-Green-Yellow-Red gradient line by line."""
        for i in range(self.z_bar_width):
            t = i / self.z_bar_width
            
            # Interpolate color based on position (0.0 to 1.0)
            if t < 0.3:
                color = _interpolate_color(self.RED, self.YELLOW, t / 0.3)
            elif t < 0.5:
                color = _interpolate_color(self.YELLOW, self.GREEN, (t - 0.3) / 0.2)
            elif t < 0.7:
                color = _interpolate_color(self.GREEN, self.YELLOW, (t - 0.5) / 0.2)
            else:
                color = _interpolate_color(self.YELLOW, self.RED, (t - 0.7) / 0.3)
            
            self.z_canvas.create_line(
                self.bar_left + i, self.bar_top,
                self.bar_left + i, self.bar_bottom,
                fill=color
            )
    
    def _create_status_label(self):
        """Setup the text label for status messages."""
        self.status_label = tk.Label(
            self.root, 
            text="Waiting...", 
            fg="#666666", 
            bg='#1a1a1a',
            font=("Arial", int(11 * self.scale))
        )
        self.status_label.pack(pady=5)
    
    def _start_listener(self):
        """Spawn the background thread to handle IO."""
        thread = threading.Thread(target=self._read_stdin, daemon=True)
        thread.start()
    
    def _read_stdin(self):
        """
        Background Thread: Reads binary data from stdin.
        
        This loop constantly tries to read 24 bytes (PACKET_SIZE) from
        the input buffer. If successful, it unpacks the floats and
        schedules a GUI update on the main thread.
        """
        while True:
            try:
                # Read exactly 24 bytes
                data = sys.stdin.buffer.read(PACKET_SIZE)
                
                # If read returns 0 bytes or incomplete data, the parent process closed
                if len(data) < PACKET_SIZE:
                    break
                
                # Unpack binary data into 6 floats
                lx, ly, lz, rx, ry, rz = struct.unpack(PACKET_FORMAT, data)
                
                # Convert NaNs to None for easier handling in GUI logic
                left = None if math.isnan(lx) else (lx, ly, lz)
                right = None if math.isnan(rx) else (rx, ry, rz)
                
                # Schedule update on Main Thread (Tkinter is not thread-safe)
                self.root.after(0, self._update_display, left, right)
                
            except struct.error:
                continue # Skip malformed packets
            except Exception:
                break # Stop on IO errors
        
        # If loop breaks, close the window
        self.root.after(0, self.root.quit)
    
    def _update_display(self, left, right):
        """Main update routine called by the listener thread."""
        lx, ly, lz = left if left else (None, None, None)
        rx, ry, rz = right if right else (None, None, None)
        
        # Update X/Y positions
        self._draw_eye(self.left_dot, lx, ly)
        self._draw_eye(self.right_dot, rx, ry)
        
        # Update Z-bar (average of both eyes)
        self._update_distance(lz, rz)
    
    def _draw_eye(self, dot, x, y):
        """Update position of a single eye dot."""
        if x is not None:
            try:
                # Mirror X axis: Tracker 0 is Left, but on screen 0 is Left edge.
                # Usually eye trackers mirror this, but we mirror again to make it intuitive (mirror-like)
                # or keep it consistent with the trackbox coordinate system.
                # Here we assume (1-x) creates a "mirror" effect suitable for the user.
                px = (1 - float(x)) * self.canvas_width
                py = float(y) * self.canvas_height
                
                # Clamp coordinates to stay inside canvas
                px = max(self.dot_radius, min(self.canvas_width - self.dot_radius, px))
                py = max(self.dot_radius, min(self.canvas_height - self.dot_radius, py))
                
                # Move the dot
                self.canvas.coords(
                    dot,
                    px - self.dot_radius, py - self.dot_radius,
                    px + self.dot_radius, py + self.dot_radius
                )
                self.canvas.itemconfig(dot, state='normal')
            except (ValueError, TypeError):
                self.canvas.itemconfig(dot, state='hidden')
        else:
            self.canvas.itemconfig(dot, state='hidden')
    
    def _update_distance(self, lz, rz):
        """Calculate average distance and update Z-bar marker."""
        valid_z = []
        for z in [lz, rz]:
            if z is not None:
                try:
                    valid_z.append(float(z))
                except (ValueError, TypeError):
                    pass
        
        if valid_z:
            avg_z = sum(valid_z) / len(valid_z)
            
            # Map normalized Z (0.0-1.0) to pixel width
            z_px = self.bar_left + max(0, min(1, avg_z)) * self.z_bar_width
            
            # Move marker
            self.z_canvas.coords(
                self.z_marker,
                z_px - 3, self.marker_top,
                z_px + 3, self.marker_bottom
            )
            
            # Update status text based on distance
            if 0.3 <= avg_z <= 0.7:
                self.status_label.config(text="Position: Good ✓", fg="#00ff00")
            elif avg_z < 0.3:
                self.status_label.config(text="Too Close", fg="#ffaa00")
            else:
                self.status_label.config(text="Too Far", fg="#ffaa00")
        else:
            # No eyes detected
            self.status_label.config(text="Eyes Not Detected", fg="#ff4444")
            self.canvas.itemconfig(self.left_dot, state='hidden')
            self.canvas.itemconfig(self.right_dot, state='hidden')
    
    def run(self):
        """Start the main event loop (Blocking)."""
        self.root.mainloop()


# --- Client Logic (Main Process) ---

class LiveMonitor:
    """
    Controller that launches and communicates with the monitor subprocess.
    
    This class is instantiated by the main experiment script. It handles
    serializing data into binary format and writing it to the subprocess's pipe.
    """
    
    def __init__(self, scale=1.0, screen=0, update_rate=5, 
                 left_eye_color=None, right_eye_color=None):
        """
        Initialize and launch the monitor subprocess.
        """
        # Set default colors if not provided
        if left_eye_color is None:
            left_eye_color = (0, 212, 255)
        if right_eye_color is None:
            right_eye_color = (255, 110, 199)
        
        # Configuration dict to be passed as JSON arg to subprocess
        config = {
            'scale': scale,
            'screen': screen,
            'left_eye_color': list(left_eye_color),
            'right_eye_color': list(right_eye_color)
        }
        
        # Launch this file as a standalone script
        # stdin=subprocess.PIPE allows us to write binary data to it
        self.process = subprocess.Popen(
            [sys.executable, __file__, json.dumps(config)],
            stdin=subprocess.PIPE
        )
        self._alive = True
    
    def push(self, left_eye, right_eye):
        """
        Serialize eye coordinates and send to the monitor.
        
        Parameters
        ----------
        left_eye, right_eye : tuple or None
            (x, y, z) coordinates.
        """
        if not self._alive:
            return
        
        try:
            # Prepare data, converting missing eyes to NaN
            if left_eye:
                lx, ly, lz = float(left_eye[0]), float(left_eye[1]), float(left_eye[2])
            else:
                lx = ly = lz = float('nan')
            
            if right_eye:
                rx, ry, rz = float(right_eye[0]), float(right_eye[1]), float(right_eye[2])
            else:
                rx = ry = rz = float('nan')
            
            # Pack into 24 bytes (6 floats)
            data = struct.pack(PACKET_FORMAT, lx, ly, lz, rx, ry, rz)
            
            # Write to pipe
            self.process.stdin.write(data)
            self.process.stdin.flush()
            
        except (BrokenPipeError, OSError, ValueError):
            # Process died or pipe broke
            self._alive = False
    
    def stop(self):
        """Terminate the monitor subprocess gracefully."""
        self._alive = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
    
    def is_alive(self):
        """Check if the subprocess is still running."""
        return self._alive and self.process.poll() is None


# --- Entry Point ---

if __name__ == "__main__":
    """
    Main Execution Block.
    
    This block only runs when the file is executed as a script (i.e., inside
    the subprocess created by `LiveMonitor`). It reads the configuration JSON
    from command line arguments and starts the GUI.
    """
    if len(sys.argv) > 1:
        try:
            # 1. Parse config JSON from command line
            config = json.loads(sys.argv[1])
            
            # 2. Initialize GUI
            gui = EyeMonitorGUI(config)
            
            # 3. Start Event Loop (blocks until window closes)
            gui.run()
            
        except Exception as e:
            print(f"Monitor failed: {e}", file=sys.stderr)