"""
EyePositionGUI - Non-blocking Tkinter eye position display.

This module provides a Tkinter window that shows:
- Track box with eye position dots
- Distance bar
- Live eye camera images

Designed to run alongside PsychoPy using update() instead of mainloop().
Positions itself relative to a PsychoPy window.

Usage
-----
```python
from EyePositionGUI import EyePositionGUI

# Create GUI positioned on top of PsychoPy window
gui = EyePositionGUI(eyetracker, win=psychopy_win, scale=1.0)

# In your PsychoPy loop:
while running:
    gui.update()          # Update Tkinter window
    video.draw()          # Draw PsychoPy video
    win.flip()            # Flip PsychoPy window
    
    if event.getKeys(['space']):
        running = False

# Cleanup
gui.close()
```
"""

import tkinter as tk
from PIL import Image, ImageTk
import io
import threading
import tobii_research as tr

# Try to import DeToX config, fallback to defaults
try:
    from DeToX import ETSettings as cfg
    DEFAULT_LEFT_COLOR = cfg.colors.left_eye[:3]
    DEFAULT_RIGHT_COLOR = cfg.colors.right_eye[:3]
except ImportError:
    DEFAULT_LEFT_COLOR = (100, 200, 255)
    DEFAULT_RIGHT_COLOR = (255, 100, 120)


def _interpolate_color(color1, color2, t):
    """Linearly interpolate between two RGB colors."""
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    return f'#{r:02x}{g:02x}{b:02x}'


def _get_psychopy_window_geometry(win):
    """
    Get PsychoPy window position and size in screen coordinates.
    
    Returns
    -------
    tuple
        (x, y, width, height) of the PsychoPy window
    """
    try:
        # Get window size
        win_width, win_height = win.size
        
        # Check for Retina scaling on macOS
        try:
            if hasattr(win, 'useRetina') and win.useRetina:
                # Size is in retina pixels, position is in points
                pass  # Position should still be correct
        except:
            pass
        
        # Get window position
        # PsychoPy pos is (0,0) for center of screen in fullscreen
        # For non-fullscreen, we need to get actual window position
        
        if win._isFullScr:
            # Fullscreen: window covers the whole monitor
            # Get monitor info
            try:
                from psychopy import monitors
                # Try to get monitor position for multi-monitor setups
                if hasattr(win, 'screen') and win.screen is not None:
                    screen_num = win.screen
                else:
                    screen_num = 0
                
                # Try screeninfo for accurate multi-monitor positions
                try:
                    from screeninfo import get_monitors
                    monitors_list = list(get_monitors())
                    if screen_num < len(monitors_list):
                        m = monitors_list[screen_num]
                        return (m.x, m.y, m.width, m.height)
                except ImportError:
                    pass
                
                # Fallback: assume standard monitor arrangement
                return (1920 * screen_num, 0, win_width, win_height)
                
            except Exception:
                return (0, 0, win_width, win_height)
        else:
            # Windowed mode: get actual window position
            # PsychoPy stores window position differently depending on backend
            try:
                # Try to get from pyglet window
                if hasattr(win, '_win') and hasattr(win._win, 'get_location'):
                    x, y = win._win.get_location()
                    return (x, y, win_width, win_height)
            except:
                pass
            
            try:
                # Try pos attribute (some PsychoPy versions)
                if hasattr(win, 'pos') and win.pos is not None:
                    x, y = win.pos
                    return (int(x), int(y), win_width, win_height)
            except:
                pass
            
            # Fallback: assume centered on primary monitor
            try:
                import tkinter as tk
                temp_root = tk.Tk()
                temp_root.withdraw()
                screen_w = temp_root.winfo_screenwidth()
                screen_h = temp_root.winfo_screenheight()
                temp_root.destroy()
                
                x = (screen_w - win_width) // 2
                y = (screen_h - win_height) // 2
                return (x, y, win_width, win_height)
            except:
                return (0, 0, win_width, win_height)
                
    except Exception as e:
        print(f"Warning: Could not get PsychoPy window geometry: {e}")
        return (0, 0, 800, 600)


class EyePositionGUI:
    """
    Non-blocking Tkinter window showing eye position and camera images.
    
    Designed to run alongside PsychoPy - use update() instead of mainloop().
    Positions itself relative to a PsychoPy window (top-center by default).
    
    Parameters
    ----------
    eyetracker : tobii_research.EyeTracker
        Connected Tobii eye tracker object.
    win : psychopy.visual.Window, optional
        PsychoPy window to position relative to. If provided, the GUI
        will appear at top-center of this window. If None, uses screen 0.
    scale : float, optional
        Scale factor for UI elements. Default 1.0.
    position : str, optional
        Position relative to PsychoPy window: 'top-center', 'top-left', 'top-right'.
        Default 'top-center'.
    offset_y : int, optional
        Vertical offset from top of window in pixels. Default 30.
    left_eye_color : tuple, optional
        RGB color for left eye. Defaults to cfg.colors.left_eye.
    right_eye_color : tuple, optional
        RGB color for right eye. Defaults to cfg.colors.right_eye.
    """
    
    RED = (180, 40, 40)
    YELLOW = (180, 180, 40)
    GREEN = (40, 160, 40)
    
    def __init__(self, eyetracker, win=None, scale=1.0, position='top-center',
                 offset_y=30, left_eye_color=None, right_eye_color=None):
        
        self.tracker = eyetracker
        self.psychopy_win = win
        self.scale = scale
        self.position = position
        self.offset_y = offset_y
        
        # Use cfg colors as defaults
        if left_eye_color is None:
            left_eye_color = DEFAULT_LEFT_COLOR
        if right_eye_color is None:
            right_eye_color = DEFAULT_RIGHT_COLOR
        
        # Colors
        self.left_hex = f'#{left_eye_color[0]:02x}{left_eye_color[1]:02x}{left_eye_color[2]:02x}'
        self.right_hex = f'#{right_eye_color[0]:02x}{right_eye_color[1]:02x}{right_eye_color[2]:02x}'
        
        # Thread-safe storage
        self.lock = threading.Lock()
        self.latest_position = None
        self.eye_image_0 = None  # region_id 0 (right eye)
        self.eye_image_1 = None  # region_id 1 (left eye)
        
        # Scaled dimensions
        self.eye_img_size = int(100 * self.scale)
        self.trackbox_width = int(150 * self.scale)
        self.trackbox_height = int(110 * self.scale)
        self.dot_radius = max(4, int(6 * self.scale))
        
        # State
        self.running = True
        self.status_text = "Waiting..."
        self.status_color = "#666666"
        
        # PhotoImage references
        self.left_photo = None
        self.right_photo = None
        self.placeholder_photo = None
        
        # Build UI
        self._create_window()
        self._create_layout()
        
        # Subscribe to eye tracker
        self._subscribe()
    
    def _create_window(self):
        """Setup Tkinter window."""
        self.root = tk.Tk()
        self.root.title("Eye Position")
        self.root.attributes("-topmost", True)
        self.root.configure(bg='#1a1a1a')
        self.root.resizable(False, False)
        self.root.overrideredirect(False)  # Keep window decorations minimal
        
        # Create placeholder image
        placeholder_pil = Image.new('L', (self.eye_img_size, self.eye_img_size), color=17)
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_pil)
    
    def _create_layout(self):
        """Create compact horizontal layout."""
        # Scaled values
        pad = int(10 * self.scale)
        font_small = ("Arial", max(7, int(7 * self.scale)))
        font_label = ("Arial", max(9, int(9 * self.scale)), "bold")
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(padx=pad, pady=pad)
        
        # === LEFT EYE ===
        left_frame = tk.Frame(main_frame, bg='#1a1a1a')
        left_frame.pack(side=tk.LEFT, padx=int(8 * self.scale))
        
        tk.Label(left_frame, text="L", fg=self.left_hex, bg='#1a1a1a',
                 font=font_label).pack()
        
        self.left_img_label = tk.Label(
            left_frame,
            bg='#111111',
            image=self.placeholder_photo,
            highlightthickness=1,
            highlightbackground="#333333"
        )
        self.left_img_label.pack()
        
        # === CENTER ===
        center_frame = tk.Frame(main_frame, bg='#1a1a1a')
        center_frame.pack(side=tk.LEFT, padx=int(10 * self.scale))
        
        # Track box
        self.canvas = tk.Canvas(
            center_frame,
            width=self.trackbox_width,
            height=self.trackbox_height,
            bg='#111111',
            highlightthickness=1,
            highlightbackground="#333333"
        )
        self.canvas.pack()
        
        # Crosshair
        cx, cy = self.trackbox_width // 2, self.trackbox_height // 2
        self.canvas.create_line(cx, 0, cx, self.trackbox_height, fill="#333333", dash=(2, 2))
        self.canvas.create_line(0, cy, self.trackbox_width, cy, fill="#333333", dash=(2, 2))
        
        # Eye dots
        self.left_dot = self.canvas.create_oval(0, 0, 0, 0, fill=self.left_hex,
                                                 outline="white", state='hidden')
        self.right_dot = self.canvas.create_oval(0, 0, 0, 0, fill=self.right_hex,
                                                  outline="white", state='hidden')
        
        # Distance bar
        self.z_bar_width = self.trackbox_width
        z_bar_height = int(20 * self.scale)
        self.z_canvas = tk.Canvas(center_frame, width=self.z_bar_width, height=z_bar_height,
                                   bg='#1a1a1a', highlightthickness=0)
        self.z_canvas.pack(pady=(int(5 * self.scale), 0))
        
        self.bar_left = 0
        self.bar_top = int(4 * self.scale)
        self.bar_bottom = int(16 * self.scale)
        
        # Gradient
        for i in range(self.z_bar_width):
            t = i / self.z_bar_width
            if t < 0.3:
                color = _interpolate_color(self.RED, self.YELLOW, t / 0.3)
            elif t < 0.5:
                color = _interpolate_color(self.YELLOW, self.GREEN, (t - 0.3) / 0.2)
            elif t < 0.7:
                color = _interpolate_color(self.GREEN, self.YELLOW, (t - 0.5) / 0.2)
            else:
                color = _interpolate_color(self.YELLOW, self.RED, (t - 0.7) / 0.3)
            self.z_canvas.create_line(self.bar_left + i, self.bar_top,
                                      self.bar_left + i, self.bar_bottom, fill=color)
        
        # Marker
        z_center = self.z_bar_width // 2
        marker_half = max(2, int(3 * self.scale))
        self.z_marker = self.z_canvas.create_rectangle(
            z_center - marker_half, self.bar_top - 2,
            z_center + marker_half, self.bar_bottom + 2,
            fill="white", outline="black"
        )
        
        # Status
        self.status_label = tk.Label(center_frame, text="Waiting...", fg="#666666",
                                      bg='#1a1a1a', font=font_small)
        self.status_label.pack(pady=(int(3 * self.scale), 0))
        
        # === RIGHT EYE ===
        right_frame = tk.Frame(main_frame, bg='#1a1a1a')
        right_frame.pack(side=tk.LEFT, padx=int(8 * self.scale))
        
        tk.Label(right_frame, text="R", fg=self.right_hex, bg='#1a1a1a',
                 font=font_label).pack()
        
        self.right_img_label = tk.Label(
            right_frame,
            bg='#111111',
            image=self.placeholder_photo,
            highlightthickness=1,
            highlightbackground="#333333"
        )
        self.right_img_label.pack()
        
        # Position window after layout is complete
        self.root.update_idletasks()
        self._position_window()
    
    def _position_window(self):
        """Position window relative to PsychoPy window."""
        # Get PsychoPy window geometry
        if self.psychopy_win is not None:
            win_x, win_y, win_w, win_h = _get_psychopy_window_geometry(self.psychopy_win)
        else:
            # Fallback to primary screen
            win_x, win_y = 0, 0
            win_w = self.root.winfo_screenwidth()
            win_h = self.root.winfo_screenheight()
        
        # Get this Tkinter window size
        tk_w = self.root.winfo_width()
        tk_h = self.root.winfo_height()
        
        # Calculate position based on alignment
        if self.position == 'top-center':
            x = win_x + (win_w - tk_w) // 2
            y = win_y + self.offset_y
        elif self.position == 'top-left':
            x = win_x + 30
            y = win_y + self.offset_y
        elif self.position == 'top-right':
            x = win_x + win_w - tk_w - 30
            y = win_y + self.offset_y
        else:
            # Default: top-center
            x = win_x + (win_w - tk_w) // 2
            y = win_y + self.offset_y
        
        self.root.geometry(f"+{x}+{y}")
    
    # --- Callbacks ---
    
    def _on_position(self, data):
        with self.lock:
            self.latest_position = data
    
    def _on_eye_image(self, data):
        img_bytes = data.get('image_data')
        region_id = data.get('region_id')
        if img_bytes:
            with self.lock:
                if region_id == 0:
                    self.eye_image_0 = img_bytes
                elif region_id == 1:
                    self.eye_image_1 = img_bytes
    
    def _on_gaze(self, data):
        pass
    
    def _subscribe(self):
        """Subscribe to eye tracker streams."""
        self.tracker.subscribe_to(
            tr.EYETRACKER_USER_POSITION_GUIDE,
            self._on_position,
            as_dictionary=True
        )
        self.tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA,
            self._on_gaze,
            as_dictionary=True
        )
        self.tracker.subscribe_to(
            tr.EYETRACKER_EYE_IMAGES,
            self._on_eye_image,
            as_dictionary=True
        )
    
    def _unsubscribe(self):
        """Unsubscribe from eye tracker streams."""
        try:
            self.tracker.unsubscribe_from(tr.EYETRACKER_USER_POSITION_GUIDE, self._on_position)
            self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._on_gaze)
            self.tracker.unsubscribe_from(tr.EYETRACKER_EYE_IMAGES, self._on_eye_image)
        except:
            pass
    
    # --- Display Update ---
    
    def update(self):
        """
        Update the display. Call this in your main loop.
        
        Non-blocking - returns immediately after processing pending events.
        """
        if not self.running:
            return
        
        try:
            with self.lock:
                pos_data = self.latest_position
                img_0 = self.eye_image_0
                img_1 = self.eye_image_1
            
            if pos_data:
                self._update_trackbox(pos_data)
            
            if img_1:
                self._update_eye_image(self.left_img_label, img_1, 'left')
            if img_0:
                self._update_eye_image(self.right_img_label, img_0, 'right')
            
            # Process Tkinter events (non-blocking)
            self.root.update()
            
        except tk.TclError:
            # Window was closed
            self.running = False
        except Exception as e:
            self.status_label.config(text="Error", fg="#ff4444")
    
    def _update_trackbox(self, data):
        """Update track box and distance bar."""
        lv = data.get("left_user_position_validity", 0)
        rv = data.get("right_user_position_validity", 0)
        lx, ly, lz = data.get("left_user_position", (None, None, None))
        rx, ry, rz = data.get("right_user_position", (None, None, None))
        
        # Left dot
        if lv and lx is not None:
            self._draw_dot(self.left_dot, lx, ly)
        else:
            self.canvas.itemconfig(self.left_dot, state='hidden')
        
        # Right dot
        if rv and rx is not None:
            self._draw_dot(self.right_dot, rx, ry)
        else:
            self.canvas.itemconfig(self.right_dot, state='hidden')
        
        # Distance
        valid_z = []
        if lv and lz is not None:
            valid_z.append(lz)
        if rv and rz is not None:
            valid_z.append(rz)
        
        if valid_z:
            avg_z = sum(valid_z) / len(valid_z)
            z_px = max(0, min(1, avg_z)) * self.z_bar_width
            marker_half = max(2, int(3 * self.scale))
            self.z_canvas.coords(self.z_marker,
                                 z_px - marker_half, self.bar_top - 2,
                                 z_px + marker_half, self.bar_bottom + 2)
            
            if 0.3 <= avg_z <= 0.7:
                self.status_label.config(text="Good ✓", fg="#00ff00")
            elif avg_z < 0.3:
                self.status_label.config(text="Too Close", fg="#ffaa00")
            else:
                self.status_label.config(text="Too Far", fg="#ffaa00")
        else:
            self.status_label.config(text="No Eyes", fg="#ff4444")
    
    def _draw_dot(self, dot, x, y):
        """Update eye dot position."""
        px = (1 - float(x)) * self.trackbox_width
        py = float(y) * self.trackbox_height
        px = max(self.dot_radius, min(self.trackbox_width - self.dot_radius, px))
        py = max(self.dot_radius, min(self.trackbox_height - self.dot_radius, py))
        
        self.canvas.coords(dot, px - self.dot_radius, py - self.dot_radius,
                           px + self.dot_radius, py + self.dot_radius)
        self.canvas.itemconfig(dot, state='normal')
    
    def _update_eye_image(self, label, img_bytes, which):
        """Update eye image."""
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = pil_img.resize((self.eye_img_size, self.eye_img_size), Image.NEAREST)
        
        photo = ImageTk.PhotoImage(pil_img)
        label.config(image=photo)
        
        if which == 'left':
            self.left_photo = photo
        else:
            self.right_photo = photo
    
    # --- Control ---
    
    def close(self):
        """Clean up and close the window."""
        self.running = False
        self._unsubscribe()
        try:
            self.root.destroy()
        except:
            pass
    
    def is_running(self):
        """Check if window is still open."""
        return self.running


# --- Standalone test ---

if __name__ == "__main__":
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description="EyePositionGUI test")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--position", default="top-center", 
                        choices=["top-center", "top-left", "top-right"])
    args = parser.parse_args()
    
    print(f"EyePositionGUI - scale={args.scale}, position={args.position}")
    print("Press Ctrl+C to exit")
    
    trackers = tr.find_all_eyetrackers()
    if not trackers:
        print("No eye tracker found!")
        exit(1)
    
    tracker = trackers[0]
    print(f"Connected to: {tracker.model}")
    
    # Create GUI (no PsychoPy window - will use screen center)
    gui = EyePositionGUI(tracker, win=None, scale=args.scale, position=args.position)
    
    # Main loop (simulating what would happen with PsychoPy)
    try:
        while gui.is_running():
            gui.update()
            time.sleep(0.016)  # ~60fps
    except KeyboardInterrupt:
        print("\nStopped")
    
    gui.close()
    print("Done")