"""
LiveMonitor - Real-time Eye Position Visualization
===================================================

A cross-platform monitor window that displays the participant's eye position
relative to the eye tracker's track box. Runs in a separate process to avoid
interfering with PsychoPy's OpenGL rendering.

Architecture
------------
The monitor uses multiprocessing for cross-platform compatibility:

    ┌─────────────────────────────────────────┐
    │            MAIN PROCESS                 │
    │  ┌─────────────┐    ┌────────────────┐  │
    │  │  PsychoPy   │    │   ETracker     │  │
    │  │  Experiment │    │   Callback     │  │
    │  │  (OpenGL)   │    │                │  │
    │  └─────────────┘    └───────┬────────┘  │
    │                             │           │
    │                      monitor.push()     │
    └─────────────────────────────┼───────────┘
                                  │
                       multiprocessing.Queue
                          (thread-safe)
                                  │
    ┌─────────────────────────────┼───────────┐
    │         MONITOR PROCESS     ▼           │
    │  ┌───────────────────────────────────┐  │
    │  │     Tkinter GUI (CPU rendering)   │  │
    │  │                                   │  │
    │  │   ┌─────────────────────────┐     │  │
    │  │   │    Track Box View       │     │  │
    │  │   │    (X/Y position)       │     │  │
    │  │   └─────────────────────────┘     │  │
    │  │   ┌─────────────────────────┐     │  │
    │  │   │    Distance Bar         │     │  │
    │  │   │    (Z position)         │     │  │
    │  │   └─────────────────────────┘     │  │
    │  └───────────────────────────────────┘  │
    └─────────────────────────────────────────┘

Why Multiprocessing?
--------------------
- macOS requires all GUI operations on the main thread
- A separate process has its own main thread, so Tkinter works correctly
- Windows and Linux also work reliably with this approach
- CPU-based rendering (Tkinter) doesn't compete with GPU (PsychoPy)

Coordinate System
-----------------
Uses Tobii's Track Box Coordinate System (normalized 0-1):
- X: 0 = left edge of track box, 1 = right edge
- Y: 0 = top edge of track box, 1 = bottom edge  
- Z: 0 = closest valid distance, 1 = farthest valid distance
- (0.5, 0.5, 0.5) = center of the valid tracking volume

Usage
-----
```python
from LiveMonitor import LiveMonitor

# In ETracker.__init__:
self.live_monitor = LiveMonitor()

# In gaze callback:
def _on_gaze_data(self, gaze_data):
    # ... existing logic ...
    
    if self.live_monitor is not None:
        left = gaze_data.get('left_gaze_origin_in_trackbox_coordinate_system')
        right = gaze_data.get('right_gaze_origin_in_trackbox_coordinate_system')
        left_valid = gaze_data.get('left_gaze_origin_validity', 0)
        right_valid = gaze_data.get('right_gaze_origin_validity', 0)
        
        self.live_monitor.push(
            left if left_valid else None,
            right if right_valid else None
        )

# When done:
self.live_monitor.stop()
```
"""

import multiprocessing as mp

# IMPORTANT: Set spawn method for macOS compatibility
# Must be called before any other multiprocessing usage
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import numpy as np


# =============================================================================
# MONITOR PROCESS FUNCTION
# =============================================================================

def _monitor_process(queue, stop_event):
    """
    Main function that runs in the separate monitor process.
    
    This function initializes Tkinter and runs its event loop.
    
    Parameters
    ----------
    queue : multiprocessing.Queue
        Thread-safe queue for receiving eye position data from main process.
        Each item is a tuple: (left_eye_pos, right_eye_pos) where each pos
        is either (x, y, z) in trackbox coordinates or None if invalid.
    stop_event : multiprocessing.Event
        Signal to gracefully shut down the monitor window.
    """
    import tkinter as tk
    
    # =========================================================================
    # WINDOW SETUP
    # =========================================================================
    
    root = tk.Tk()
    root.title("DeToX Eye Monitor")
    root.attributes("-topmost", True)      # Always visible above other windows
    root.geometry("220x300+50+50")         # Size and position (top-left)
    root.configure(bg='#1a1a1a')           # Dark background
    root.resizable(False, False)           # Fixed size
    
    # Handle window close button
    def on_close():
        stop_event.set()
        root.quit()
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    # =========================================================================
    # TRACK BOX CANVAS (Top-down X/Y view)
    # =========================================================================
    # Shows eye positions as if looking down at the participant from above.
    # The view is MIRRORED so movements feel natural (move right → dot goes right)
    
    canvas = tk.Canvas(
        root, 
        width=200, 
        height=150, 
        bg='#111111',
        highlightthickness=1, 
        highlightbackground="#333333"
    )
    canvas.pack(pady=(15, 10))
    
    # Reference crosshair (center of track box)
    canvas.create_line(100, 0, 100, 150, fill="#333333", dash=(2, 2))  # Vertical
    canvas.create_line(0, 75, 200, 75, fill="#333333", dash=(2, 2))    # Horizontal
    
    # Eye position indicators (initially hidden)
    left_dot = canvas.create_oval(0, 0, 14, 14, fill="#00d4ff", outline="white", state='hidden')
    right_dot = canvas.create_oval(0, 0, 14, 14, fill="#ff6ec7", outline="white", state='hidden')
    
    # Labels
    tk.Label(root, text="Track Box (top view)", fg="#666666", bg='#1a1a1a', 
             font=("Arial", 9)).pack()
    
    # =========================================================================
    # DISTANCE BAR (Z-axis / depth)
    # =========================================================================
    # Shows how far the participant is from the eye tracker.
    # Green zone indicates optimal tracking distance.
    
    z_canvas = tk.Canvas(root, width=200, height=35, bg='#1a1a1a', highlightthickness=0)
    z_canvas.pack(pady=(15, 5))
    
    # Background track
    z_canvas.create_rectangle(10, 12, 190, 28, fill="#333333", outline="")
    
    # Optimal zone (roughly 50-80cm for most Tobii trackers)
    # In trackbox coords: ~0.3 to ~0.7 maps to this range
    z_canvas.create_rectangle(55, 12, 135, 28, fill="#006600", outline="")
    
    # Distance marker (white bar)
    z_marker = z_canvas.create_rectangle(98, 7, 102, 33, fill="white", outline="#000000")
    
    # Labels for distance bar
    z_canvas.create_text(10, 20, text="Near", fill="#666666", font=("Arial", 8), anchor="w")
    z_canvas.create_text(190, 20, text="Far", fill="#666666", font=("Arial", 8), anchor="e")
    
    # =========================================================================
    # STATUS DISPLAY
    # =========================================================================
    
    status_label = tk.Label(
        root, 
        text="Waiting for data...", 
        fg="#666666", 
        bg='#1a1a1a', 
        font=("Arial", 11)
    )
    status_label.pack(pady=(10, 5))
    
    # =========================================================================
    # UPDATE LOOP
    # =========================================================================
    
    def update_display():
        """
        Drains the queue and updates the display.
        
        Called periodically by Tkinter's event loop. Collects all pending
        samples, averages them for stability, and updates the visual elements.
        """
        # Check for stop signal
        if stop_event.is_set():
            root.quit()
            return
        
        # ---------------------------------------------------------------------
        # DRAIN QUEUE (collect all pending samples)
        # ---------------------------------------------------------------------
        samples = []
        while True:
            try:
                samples.append(queue.get_nowait())
            except:
                break
        
        # Average only the last N samples (e.g., ~100ms worth at 120Hz)
        MAX_SAMPLES = 12
        if samples:
            samples = samples[-MAX_SAMPLES:]
            
            # Separate left and right eye data (ONLY ONCE)
            left_samples = [s[0] for s in samples if s[0] is not None]
            right_samples = [s[1] for s in samples if s[1] is not None]
            
            # Average for stability (reduces jitter)
            left_avg = tuple(np.mean(left_samples, axis=0)) if left_samples else None
            right_avg = tuple(np.mean(right_samples, axis=0)) if right_samples else None
    
            
            # -----------------------------------------------------------------
            # UPDATE TRACK BOX CANVAS
            # -----------------------------------------------------------------
            
            def update_eye_dot(pos, dot):
                """Map trackbox coords (0-1) to canvas pixels and update dot."""
                if pos is None:
                    canvas.itemconfig(dot, state='hidden')
                    return None
                
                x, y, z = pos
                
                # Check for NaN values
                if np.isnan(x) or np.isnan(y):
                    canvas.itemconfig(dot, state='hidden')
                    return None
                
                # Map to canvas coordinates
                # Canvas is 200x150 pixels
                # X: 0→0px, 1→200px (left to right)
                # Y: 0→0px, 1→150px, but we invert so 0=top, 1=bottom displays correctly
                px = x * 200
                py = y * 150  # Tobii Y: 0=top, 1=bottom (matches canvas)
                
                # Clamp to canvas bounds
                px = max(7, min(193, px))
                py = max(7, min(143, py))
                
                # Update dot position (centered on coordinates)
                canvas.coords(dot, px - 7, py - 7, px + 7, py + 7)
                canvas.itemconfig(dot, state='normal')
                
                return z  # Return Z for distance calculation
            
            left_z = update_eye_dot(left_avg, left_dot)
            right_z = update_eye_dot(right_avg, right_dot)
            
            # -----------------------------------------------------------------
            # UPDATE DISTANCE BAR
            # -----------------------------------------------------------------
            
            z_values = [z for z in [left_z, right_z] if z is not None and not np.isnan(z)]
            
            if z_values:
                avg_z = np.mean(z_values)  # 0-1 normalized
                
                # Map to distance bar (10px to 190px range)
                z_px = 10 + (avg_z * 180)
                z_px = max(10, min(190, z_px))
                
                # Update marker position
                z_canvas.coords(z_marker, z_px - 3, 7, z_px + 3, 33)
                
                # Update status with visual feedback
                # Optimal range is roughly 0.3-0.7 in trackbox coords
                if 0.3 <= avg_z <= 0.7:
                    status_label.config(text="Position: Good ✓", fg="#00ff00")
                elif avg_z < 0.3:
                    status_label.config(text="Too Close", fg="#ffaa00")
                else:
                    status_label.config(text="Too Far", fg="#ffaa00")
            else:
                status_label.config(text="Eyes Not Detected", fg="#ff4444")
                canvas.itemconfig(left_dot, state='hidden')
                canvas.itemconfig(right_dot, state='hidden')
        
        # Schedule next update (5 Hz refresh rate)
        root.after(200, update_display)
    
    # =========================================================================
    # START EVENT LOOP
    # =========================================================================
    
    update_display()  # Start the update cycle
    root.mainloop()   # Blocks until window closes


# =============================================================================
# LIVEMONITOR CLASS (Interface for main process)
# =============================================================================

class LiveMonitor:
    """
    Cross-platform eye position monitor using multiprocessing.
    
    Creates a separate process with a Tkinter window that displays real-time
    eye position data. Communication happens through a thread-safe queue.
    
    Attributes
    ----------
    queue : multiprocessing.Queue
        Queue for sending eye position data to the monitor process.
    process : multiprocessing.Process
        The monitor process running Tkinter.
    stop_event : multiprocessing.Event
        Signal for graceful shutdown.
    
    Examples
    --------
    >>> monitor = LiveMonitor()
    >>> 
    >>> # In your gaze callback:
    >>> monitor.push(left_eye_tuple, right_eye_tuple)
    >>> 
    >>> # When finished:
    >>> monitor.stop()
    """
    
    def __init__(self):
        """
        Initialize and start the monitor in a separate process.
        
        The monitor window will appear immediately and begin waiting for data.
        """
        self.queue = mp.Queue()
        self.stop_event = mp.Event()
        
        self.process = mp.Process(
            target=_monitor_process,
            args=(self.queue, self.stop_event),
            daemon=True  # Automatically terminates when main process exits
        )
        self.process.start()
    
    def push(self, left_eye, right_eye):
        """
        Send eye position data to the monitor.
        
        Call this from your gaze data callback. The method is non-blocking
        and thread-safe.
        
        Parameters
        ----------
        left_eye : tuple of float or None
            Left eye position in trackbox coordinates (x, y, z) where each
            value is normalized 0-1. Pass None if left eye is not valid.
        right_eye : tuple of float or None
            Right eye position in trackbox coordinates (x, y, z) where each
            value is normalized 0-1. Pass None if right eye is not valid.
        
        Notes
        -----
        Trackbox coordinates are:
        - X: 0 = left edge, 1 = right edge
        - Y: 0 = top edge, 1 = bottom edge
        - Z: 0 = closest, 1 = farthest
        """
        try:
            self.queue.put_nowait((left_eye, right_eye))
        except:
            pass  # Queue full, drop sample (monitor will catch up)
    
    def stop(self):
        """
        Gracefully shut down the monitor.
        
        Signals the monitor process to close its window and terminate.
        Waits up to 1 second for clean shutdown before forcing termination.
        """
        self.stop_event.set()
        self.process.join(timeout=1.0)
        
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.5)
    
    def is_alive(self):
        """
        Check if the monitor is still running.
        
        Returns
        -------
        bool
            True if the monitor process is running, False otherwise.
        """
        return self.process.is_alive()

