"""Experimental shared Tkinter building blocks for DeToX's eye-position windows.

This file has no workflow of its own. It provides the pieces that the other
GUI windows reuse:

- ``TrackBoxView``: a small Tk widget that draws the participant track box,
  the left/right eye dots, and the distance bar.
- ``get_screen_rect``: a helper that finds a monitor's pixel geometry (using
  pyglet, which PsychoPy already depends on) so a Tk window can be placed on
  a chosen screen.
"""

import math
import tkinter as tk


def get_screen_rect(screen):
    """
    Return (x, y, width, height) for a monitor, or None if unavailable.

    Uses pyglet (which PsychoPy already depends on) so that screen numbering
    matches PsychoPy's own screen indices.

    Parameters
    ----------
    screen : int
        Zero-based monitor index, as passed to PsychoPy's ``screen`` argument.

    Returns
    -------
    tuple or None
        Monitor geometry as (x, y, width, height) in pixels, or None if the
        monitor could not be resolved.
    """
    try:
        import pyglet
        screens = pyglet.canvas.get_display().get_screens()
        if 0 <= screen < len(screens):
            monitor = screens[screen]
            return (monitor.x, monitor.y, monitor.width, monitor.height)
    except Exception:
        pass
    return None


class TrackBoxView:
    """
    Experimental shared Tkinter layout for eye-position displays.

    Builds the track box, distance bar, and coordinate mapping used by both
    the positioning overlay and the detached live-monitor window.

    Parameters
    ----------
    config : dict
        Display options. ``scale`` controls font sizes, spacing and eye-dot
        radius. Optional ``size=(width, height)`` sets the initial window content
        size in pixels, excluding the title bar. ``resizable`` allows dragging
        window edges (default False). The natural layout size is the minimum;
        smaller requested sizes are clamped by Tk.
    show_images : bool, optional
        Include eye-image panels. Default is False. Image panels keep their
        original size when the central position display is resized.
    """

    BACKGROUND = '#14191f'
    SURFACE = '#1c242d'
    BORDER = '#34414f'
    TEXT = '#edf2f6'
    MUTED = '#abb9c7'
    RED = (219, 111, 116)
    YELLOW = (219, 176, 107)
    GREEN = (80, 194, 158)

    def __init__(self, config, show_images=False):
        self.config = config
        self.scale = config.get('scale', 1.0)
        self.show_images = show_images
        self._last_eye_positions = (None,) * 6
        self._image_panels = []
        size = config.get('size')
        if size is not None:
            if (not isinstance(size, (tuple, list)) or len(size) != 2
                    or any(type(value) is not int or value <= 0 for value in size)):
                raise ValueError('size must contain two positive integers: (width, height)')

        left_color = config.get('left_eye_color', [100, 200, 255])
        right_color = config.get('right_eye_color', [255, 100, 120])
        self.left_hex = f'#{left_color[0]:02x}{left_color[1]:02x}{left_color[2]:02x}'
        self.right_hex = f'#{right_color[0]:02x}{right_color[1]:02x}{right_color[2]:02x}'

        self.trackbox_width = int(240 * self.scale)
        self.trackbox_height = int(176 * self.scale)
        self.dot_radius = max(5, int(7 * self.scale))
        self.eye_img_size = int(144 * self.scale) if show_images else 0

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Eye Position")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=self.BACKGROUND)
        resizable = config.get('resizable', False)
        self.root.resizable(resizable, resizable)

        self._create_layout()
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())
        if size is not None:
            self.root.geometry(f'{size[0]}x{size[1]}')

    def _interpolate_color(self, color1, color2, t):
        """Linearly interpolate between two RGB colors."""
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        return f'#{r:02x}{g:02x}{b:02x}'

    def _create_layout(self):
        """Build a readable header, three aligned views, and one distance strip."""
        pad = max(8, int(16 * self.scale))
        gap = max(6, int(12 * self.scale))
        font_small = ('Segoe UI', max(10, int(10 * self.scale)))
        font_label = ('Segoe UI', max(11, int(11 * self.scale)), 'bold')
        font_title = ('Segoe UI', max(16, int(16 * self.scale)), 'bold')

        main = tk.Frame(self.root, bg=self.BACKGROUND)
        main.pack(padx=pad, pady=pad, fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        header = tk.Frame(main, bg=self.BACKGROUND)
        header.grid(row=0, column=0, sticky='ew', pady=(0, gap))
        tk.Label(header, text='Eye position', font=font_title, fg=self.TEXT,
                 bg=self.BACKGROUND).pack(side=tk.LEFT)
        self.tracking_label = tk.Label(header, text='Waiting for eyes', font=font_label,
                                       fg=self.MUTED, bg=self.BACKGROUND)
        self.tracking_label.pack(side=tk.RIGHT)

        views = tk.Frame(main, bg=self.BACKGROUND)
        views.grid(row=1, column=0, sticky='nsew')
        views.rowconfigure(0, weight=1)
        center_column = 1 if self.show_images else 0
        views.columnconfigure(center_column, weight=1)

        if self.show_images:
            self.left_img_label = self._create_image_panel(
                views, 0, 'LEFT VIEW', self.left_hex, font_label)

        center = tk.Frame(views, bg=self.BACKGROUND)
        self._position_panel = center
        center.grid(row=0, column=center_column, sticky='nsew',
                    padx=(gap, gap) if self.show_images else 0)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(1, weight=1)
        tk.Label(center, text='POSITION', font=font_label, fg=self.MUTED,
                 bg=self.BACKGROUND).grid(row=0, column=0, sticky='w', pady=(0, 8))
        self.canvas = tk.Canvas(center, width=self.trackbox_width, height=self.trackbox_height,
                                bg=self.SURFACE, highlightthickness=1,
                                highlightbackground=self.BORDER)
        self.canvas.grid(row=1, column=0, sticky='nsew')
        cx, cy = self.trackbox_width / 2, self.trackbox_height / 2
        self.vertical_guide = self.canvas.create_line(
            cx, 0, cx, self.trackbox_height, fill=self.BORDER, dash=(3, 5))
        self.horizontal_guide = self.canvas.create_line(
            0, cy, self.trackbox_width, cy, fill=self.BORDER, dash=(3, 5))
        self.left_dot = self.canvas.create_oval(
            0, 0, 0, 0, fill=self.left_hex, outline=self.TEXT, width=1, state='hidden')
        self.right_dot = self.canvas.create_oval(
            0, 0, 0, 0, fill=self.right_hex, outline=self.TEXT, width=1, state='hidden')
        legend = tk.Frame(center, bg=self.BACKGROUND)
        legend.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        tk.Label(legend, text='● Left eye', font=font_small, fg=self.left_hex,
                 bg=self.BACKGROUND).pack(side=tk.LEFT)
        tk.Label(legend, text='Right eye ●', font=font_small, fg=self.right_hex,
                 bg=self.BACKGROUND).pack(side=tk.RIGHT)

        if self.show_images:
            self.right_img_label = self._create_image_panel(
                views, 2, 'RIGHT VIEW', self.right_hex, font_label)

        distance = tk.Frame(main, bg=self.BACKGROUND)
        distance.grid(row=2, column=0, sticky='ew', pady=(gap, 0))
        distance.columnconfigure(0, weight=1)
        distance.columnconfigure(1, weight=1)
        tk.Label(distance, text='DISTANCE', font=font_label, fg=self.MUTED,
                 bg=self.BACKGROUND).grid(row=0, column=0, sticky='w')
        self.status_label = tk.Label(distance, text=self.config.get('status_text', 'Waiting'),
                                     font=font_label, fg=self.MUTED, bg=self.BACKGROUND)
        self.status_label.grid(row=0, column=1, sticky='e')
        self.z_bar_width = self.trackbox_width
        self.bar_left = 0
        self.bar_top = max(4, int(5 * self.scale))
        self.bar_bottom = max(16, int(19 * self.scale))
        self.z_canvas = tk.Canvas(distance, width=self.z_bar_width,
                                  height=max(24, int(26 * self.scale)),
                                  bg=self.BACKGROUND, highlightthickness=0)
        self.z_canvas.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(5, 0))
        self._draw_distance_bar()
        self.z_marker = self.z_canvas.create_rectangle(
            0, 0, 0, 0, fill=self.BACKGROUND, outline=self.TEXT, width=2, state='hidden')
        endpoints = tk.Frame(distance, bg=self.BACKGROUND)
        endpoints.grid(row=2, column=0, columnspan=2, sticky='ew')
        endpoints.columnconfigure(1, weight=1)
        for column, text in enumerate(('Near', 'Preferred range', 'Far')):
            tk.Label(endpoints, text=text, font=font_small, fg=self.MUTED,
                     bg=self.BACKGROUND).grid(row=0, column=column,
                                             sticky='w' if column == 0 else 'e' if column == 2 else '')
        self.canvas.bind('<Configure>', self._resize_trackbox)
        self.z_canvas.bind('<Configure>', self._resize_distance_bar)

    def _create_image_panel(self, parent, column, heading, color, font_label):
        """Create one fixed-size camera panel, returning its image label."""
        panel = tk.Frame(parent, bg=self.BACKGROUND)
        self._image_panels.append(panel)
        panel.grid(row=0, column=column, sticky='ns')
        tk.Label(panel, text=heading, fg=color, bg=self.BACKGROUND,
                 font=font_label).pack(anchor='w', pady=(0, 8))
        image_box = tk.Frame(panel, width=self.eye_img_size + 2, height=self.eye_img_size + 2,
                             bg=self.SURFACE, highlightthickness=1,
                             highlightbackground=self.BORDER)
        image_box.pack()
        image_box.pack_propagate(False)
        image = tk.Label(image_box, bg=self.SURFACE, borderwidth=0)
        image.pack(fill=tk.BOTH, expand=True)
        return image

    def hide_image_panels(self):
        """Use the position-only layout when optional camera feedback is unavailable."""
        for panel in self._image_panels:
            panel.grid_remove()
        self._position_panel.grid_configure(padx=0)
        self.show_images = False
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())

    def _draw_distance_bar(self):
        """Rebuild the colored background when the distance canvas width changes."""
        self.z_canvas.delete('gradient')
        for i in range(self.z_bar_width):
            t = i / self.z_bar_width
            if t < 0.3:
                color = self._interpolate_color(self.RED, self.YELLOW, t / 0.3)
            elif t < 0.5:
                color = self._interpolate_color(self.YELLOW, self.GREEN, (t - 0.3) / 0.2)
            elif t < 0.7:
                color = self._interpolate_color(self.GREEN, self.YELLOW, (t - 0.5) / 0.2)
            else:
                color = self._interpolate_color(self.YELLOW, self.RED, (t - 0.7) / 0.3)
            self.z_canvas.create_line(i, self.bar_top, i, self.bar_bottom,
                                      fill=color, tags='gradient')
        self.z_canvas.tag_lower('gradient')

    def _resize_trackbox(self, event):
        """Update coordinate mapping and guides to match the actual canvas size."""
        self.trackbox_width = max(1, event.width - 2)
        self.trackbox_height = max(1, event.height - 2)
        cx = self.trackbox_width / 2
        cy = self.trackbox_height / 2
        self.canvas.coords(self.vertical_guide, cx, 0, cx, self.trackbox_height)
        self.canvas.coords(self.horizontal_guide, 0, cy, self.trackbox_width, cy)
        self.update_eye_positions(*self._last_eye_positions)

    def _resize_distance_bar(self, event):
        """Stretch the distance gradient and reposition its current marker."""
        width = max(1, event.width)
        if width != self.z_bar_width:
            self.z_bar_width = width
            self._draw_distance_bar()
        self.update_eye_positions(*self._last_eye_positions)

    def update_eye_positions(self, lx, ly, lz, rx, ry, rz):
        """
        Update eye dots and distance marker.

        Parameters
        ----------
        lx, ly, lz, rx, ry, rz : float or None
            Track-box coordinates for the left and right eyes. Missing eyes are
            represented by None or NaN values.
        """
        self._last_eye_positions = (lx, ly, lz, rx, ry, rz)
        self._draw_eye(self.left_dot, lx, ly)
        self._draw_eye(self.right_dot, rx, ry)
        eye_count = sum(x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
                        for x, y in ((lx, ly), (rx, ry)))
        self.tracking_label.config(text=f'{eye_count} / 2 eyes detected',
                                    fg=self.TEXT if eye_count else self.MUTED)

        valid_z = [z for z in (lz, rz) if z is not None and not math.isnan(z)]
        if valid_z:
            self.z_canvas.itemconfig(self.z_marker, state='normal')
            avg_z = sum(valid_z) / len(valid_z)
            z_px = self.bar_left + max(0, min(1, avg_z)) * self.z_bar_width
            marker_half = max(2, int(3 * self.scale))
            self.z_canvas.coords(
                self.z_marker,
                z_px - marker_half,
                self.bar_top - 2,
                z_px + marker_half,
                self.bar_bottom + 2
            )

            if 0.3 <= avg_z <= 0.7:
                self.status_label.config(text="Good position", fg='#50c29e')
            elif avg_z < 0.3:
                self.status_label.config(text="Move farther away", fg='#dbb06b')
            else:
                self.status_label.config(text="Move closer", fg='#dbb06b')
        else:
            self.z_canvas.itemconfig(self.z_marker, state='hidden')
            self.status_label.config(text="Waiting for position", fg=self.MUTED)

    def _draw_eye(self, dot, x, y):
        """Draw one eye dot clamped to the track-box limits."""
        if x is not None and y is not None and not math.isnan(x) and not math.isnan(y):
            px = (1 - float(x)) * self.trackbox_width
            py = float(y) * self.trackbox_height
            px = max(self.dot_radius, min(self.trackbox_width - self.dot_radius, px))
            py = max(self.dot_radius, min(self.trackbox_height - self.dot_radius, py))
            self.canvas.coords(
                dot,
                px - self.dot_radius,
                py - self.dot_radius,
                px + self.dot_radius,
                py + self.dot_radius
            )
            self.canvas.itemconfig(dot, state='normal')
        else:
            self.canvas.itemconfig(dot, state='hidden')
