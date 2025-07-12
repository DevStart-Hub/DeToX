# calibration_config.py
# =====================
# Centralized configuration for all default settings and dictionaries
# for calibration sessions (Tobii and simulation).
#
# Edit these here, or override in your main script:
#   import calibration_config
#   calibration_config.DEFAULT_HIGHLIGHT_SIZE["pix"] = 60.0

# ---------------------------------------------------------
# 1. Key mapping: which key selects which calibration point
#    (Makes it easy to use different keyboards, or change which key triggers each point.)
NUMKEY_DICT = {
    "0": -1, "num_0": -1,
    "1": 0,  "num_1": 0,
    "2": 1,  "num_2": 1,
    "3": 2,  "num_3": 2,
    "4": 3,  "num_4": 3,
    "5": 4,  "num_5": 4,
    "6": 5,  "num_6": 5,
    "7": 6,  "num_7": 6,
    "8": 7,  "num_8": 7,
    "9": 8,  "num_9": 8,
}
# Example: calibration_config.NUMKEY_DICT["1"] = 3  # Now '1' selects point index 3

# ---------------------------------------------------------
# 2. Animation settings for calibration stimuli
#    (Controls how calibration targets animate, and timing of focus collection.)
ANIMATION_SETTINGS = {
    'animation_speed': 1.0,
    'target_min': 0.2,
    'focus_time': 0.5,
}
# Example: calibration_config.ANIMATION_SETTINGS['animation_speed'] = 2.0

# ---------------------------------------------------------
# 3. Colors for all visual elements (lines, dots, highlights, etc.)
CALIBRATION_COLORS = {
    "left_eye":      (0, 255, 0, 255),       # Green (Tobii left eye)
    "right_eye":     (255, 0, 0, 255),       # Red   (Tobii right eye)
    "mouse":         (255, 128, 0, 255),     # Orange (simulated mouse sample)
    "target_outline": (0, 0, 0, 255),        # Black outline for calibration targets
    "highlight":     (255, 255, 0, 255),     # Yellow highlight for selected points
    # Add more (e.g. "background", "text") if needed!
}
# Example: calibration_config.CALIBRATION_COLORS["highlight"] = (0,255,255,255)

# ---------------------------------------------------------
# 4. Size of highlight circles (for retry/selection)
DEFAULT_HIGHLIGHT_SIZE = {
    "norm": 0.08,
    "height": 0.04,
    "pix": 40.0,
    "degFlatPos": 1.0,
    "deg": 1.0,
    "degFlat": 1.0,
    "cm": 1.0,
}
# Example: calibration_config.DEFAULT_HIGHLIGHT_SIZE["pix"] = 60.0

# ---------------------------------------------------------
# 5. Width of lines (highlight, marker, etc)
DEFAULT_LINE_WIDTH = {
    "norm": 4,
    "height": 3,
    "pix": 6,
    "degFlatPos": 2,
    "deg": 2,
    "degFlat": 2,
    "cm": 2,
}
# Example: calibration_config.DEFAULT_LINE_WIDTH["pix"] = 12

# ---------------------------------------------------------
# 6. Size of small marker circles (for already-collected points)
DEFAULT_MARKER_SIZE = {
    "norm": 0.02,
    "height": 0.01,
    "pix": 10.0,
    "degFlatPos": 0.25,
    "deg": 0.25,
    "degFlat": 0.25,
    "cm": 0.25,
}
# Example: calibration_config.DEFAULT_MARKER_SIZE["height"] = 0.02

# ---------------------------------------------------------
# 7. Thickness of the border shown during calibration mode (red frame)
DEFAULT_BORDER_THICKNESS = {
    "norm": 0.01,
    "height": 0.005,
    "pix": 3.0,
    "degFlatPos": 0.1,
    "deg": 0.1,
    "degFlat": 0.1,
    "cm": 0.1,
}
# Example: calibration_config.DEFAULT_BORDER_THICKNESS["norm"] = 0.02

# ---------------------------------------------------------
# 8. Width of lines in calibration result plots (accuracy lines)
DEFAULT_PLOT_LINE_WIDTH = {
    "norm": 2,
    "height": 2,
    "pix": 3,
    "degFlatPos": 1,
    "deg": 1,
    "degFlat": 1,
    "cm": 1,
}
# Example: calibration_config.DEFAULT_PLOT_LINE_WIDTH["pix"] = 6

# ---------------------------------------------------------
# 9. Base text height (controls all text scaling, by units)
DEFAULT_TEXT_HEIGHT = {
    "norm": 0.05,
    "height": 0.025,
    "pix": 20.0,
    "degFlatPos": 0.5,
    "deg": 0.5,
    "degFlat": 0.5,
    "cm": 0.5,
}
# Example: calibration_config.DEFAULT_TEXT_HEIGHT["deg"] = 1.0

# ---------------------------------------------------------
# 10. Font size multipliers (relative to DEFAULT_TEXT_HEIGHT)
FONT_SIZE_MULTIPLIERS = {
    "instruction_text": 1.5,
    "message_text": 1.3,
    "title_text": 1.4,
}
# Example: calibration_config.FONT_SIZE_MULTIPLIERS["instruction_text"] = 2.0

# ===== END OF CONFIG =====
# Edit values here, or override them in your main experiment script!
