# BaseEyeGUI.py
import tkinter as tk
import math

class CoreEyeGUI:
    """
    Shared base class for eye tracking GUI elements.
    Builds the Track Box, Distance Bar, and handles coordinate mapping.
    """
    RED = (180, 40, 40)
    YELLOW = (180, 180, 40)
    GREEN = (40, 160, 40)
    
    def __init__(self, config, show_images=False):
        self.config = config
        self.scale = config.get('scale', 1.0)
        self.show_images = show_images
        
        # Colors
        left_color = config.get('left_eye_color', [100, 200, 255])
        right_color = config.get('right_eye_color', [255, 100, 120])
        self.left_hex = f'#{left_color[0]:02x}{left_color[1]:02x}{left_color[2]:02x}'
        self.right_hex = f'#{right_color[0]:02x}{right_color[1]:02x}{right_color[2]:02x}'
        
        # Dimensions
        self.trackbox_width = int(150 * self.scale)
        self.trackbox_height = int(110 * self.scale)
        self.dot_radius = max(4, int(6 * self.scale))
        self.eye_img_size = int(100 * self.scale) if show_images else 0
        
        self.root = tk.Tk()
        self.root.withdraw() # hide initially until positioned by subclass

        self.root.title("Eye Position")
        self.root.attributes("-topmost", True)
        self.root.configure(bg='#1a1a1a')
        self.root.resizable(False, False)
        
        self._create_layout()

    def _interpolate_color(self, color1, color2, t):
        """Linearly interpolate between two RGB colors."""
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        return f'#{r:02x}{g:02x}{b:02x}'

    def _create_layout(self):
        """Creates the shared layout."""
        pad = int(10 * self.scale)
        font_small = ("Arial", max(7, int(7 * self.scale)))
        font_label = ("Arial", max(9, int(9 * self.scale)), "bold")
        
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(padx=pad, pady=pad)
        
        # --- Optional Left Eye Image ---
        if self.show_images:
            left_frame = tk.Frame(main_frame, bg='#1a1a1a')
            left_frame.pack(side=tk.LEFT, padx=int(8 * self.scale))
            tk.Label(left_frame, text="L", fg=self.left_hex, bg='#1a1a1a', font=font_label).pack()
            self.left_img_label = tk.Label(left_frame, bg='#111111', highlightthickness=1, highlightbackground="#333333")
            self.left_img_label.pack()

        # --- Center Track Box & Z-Bar ---
        center_frame = tk.Frame(main_frame, bg='#1a1a1a')
        center_frame.pack(side=tk.LEFT, padx=int(10 * self.scale))
        
        self.canvas = tk.Canvas(center_frame, width=self.trackbox_width, height=self.trackbox_height,
                                bg='#111111', highlightthickness=1, highlightbackground="#333333")
        self.canvas.pack()
        
        cx, cy = self.trackbox_width // 2, self.trackbox_height // 2
        self.canvas.create_line(cx, 0, cx, self.trackbox_height, fill="#333333", dash=(2, 2))
        self.canvas.create_line(0, cy, self.trackbox_width, cy, fill="#333333", dash=(2, 2))
        
        self.left_dot = self.canvas.create_oval(0, 0, 0, 0, fill=self.left_hex, outline="white", state='hidden')
        self.right_dot = self.canvas.create_oval(0, 0, 0, 0, fill=self.right_hex, outline="white", state='hidden')
        
        self.z_bar_width = self.trackbox_width
        z_bar_height = int(20 * self.scale)
        self.z_canvas = tk.Canvas(center_frame, width=self.z_bar_width, height=z_bar_height, bg='#1a1a1a', highlightthickness=0)
        self.z_canvas.pack(pady=(int(5 * self.scale), 0))
        
        self.bar_left = 0
        self.bar_top = int(4 * self.scale)
        self.bar_bottom = int(16 * self.scale)
        
        for i in range(self.z_bar_width):
            t = i / self.z_bar_width
            if t < 0.3: color = self._interpolate_color(self.RED, self.YELLOW, t / 0.3)
            elif t < 0.5: color = self._interpolate_color(self.YELLOW, self.GREEN, (t - 0.3) / 0.2)
            elif t < 0.7: color = self._interpolate_color(self.GREEN, self.YELLOW, (t - 0.5) / 0.2)
            else: color = self._interpolate_color(self.YELLOW, self.RED, (t - 0.7) / 0.3)
            self.z_canvas.create_line(self.bar_left + i, self.bar_top, self.bar_left + i, self.bar_bottom, fill=color)
        
        z_center = self.z_bar_width // 2
        marker_half = max(2, int(3 * self.scale))
        self.z_marker = self.z_canvas.create_rectangle(z_center - marker_half, self.bar_top - 2,
                                                       z_center + marker_half, self.bar_bottom + 2, fill="white", outline="black")
        
        self.status_label = tk.Label(center_frame, text="Waiting...", fg="#666666", bg='#1a1a1a', font=font_small)
        self.status_label.pack(pady=(int(3 * self.scale), 0))

        # --- Optional Right Eye Image ---
        if self.show_images:
            right_frame = tk.Frame(main_frame, bg='#1a1a1a')
            right_frame.pack(side=tk.LEFT, padx=int(8 * self.scale))
            tk.Label(right_frame, text="R", fg=self.right_hex, bg='#1a1a1a', font=font_label).pack()
            self.right_img_label = tk.Label(right_frame, bg='#111111', highlightthickness=1, highlightbackground="#333333")
            self.right_img_label.pack()

    def update_eye_positions(self, lx, ly, lz, rx, ry, rz):
        """Shared logic to update dots and distance bar."""
        self._draw_eye(self.left_dot, lx, ly)
        self._draw_eye(self.right_dot, rx, ry)
        
        valid_z = [z for z in (lz, rz) if z is not None and not math.isnan(z)]
        if valid_z:
            avg_z = sum(valid_z) / len(valid_z)
            z_px = self.bar_left + max(0, min(1, avg_z)) * self.z_bar_width
            marker_half = max(2, int(3 * self.scale))
            self.z_canvas.coords(self.z_marker, z_px - marker_half, self.bar_top - 2, z_px + marker_half, self.bar_bottom + 2)
            
            if 0.3 <= avg_z <= 0.7: self.status_label.config(text="Good ✓", fg="#00ff00")
            elif avg_z < 0.3: self.status_label.config(text="Too Close", fg="#ffaa00")
            else: self.status_label.config(text="Too Far", fg="#ffaa00")
        else:
            self.status_label.config(text="No Eyes", fg="#ff4444")

    def _draw_eye(self, dot, x, y):
        """Draws single dot clamped to trackbox limits."""
        if x is not None and not math.isnan(x):
            px = (1 - float(x)) * self.trackbox_width
            py = float(y) * self.trackbox_height
            px = max(self.dot_radius, min(self.trackbox_width - self.dot_radius, px))
            py = max(self.dot_radius, min(self.trackbox_height - self.dot_radius, py))
            self.canvas.coords(dot, px - self.dot_radius, py - self.dot_radius, px + self.dot_radius, py + self.dot_radius)
            self.canvas.itemconfig(dot, state='normal')
        else:
            self.canvas.itemconfig(dot, state='hidden')